from copy import deepcopy

import numpy as np
import pytest
import torch


def test_skipping_loss_readback_preserves_updates(njp):
    # 两个分支使用相同参数和采样序列，单独验证关闭损失回读不改变学习结果。
    reference = njp.DDPGAgent(35, 2, hidden_dim=16)
    optimized = deepcopy(reference)
    replay = njp.ReplayBuffer(32, 35, 2)
    rng = np.random.RandomState(18)
    for _ in range(32):
        replay.push(rng.randn(35), rng.randn(2), rng.randn(), rng.randn(35), False)
    for seed in range(5):
        np.random.seed(seed)
        losses = reference.update(replay, 8)
        np.random.seed(seed)
        assert optimized.update(replay, 8, return_losses=False) is None
        assert np.isfinite(losses).all()
    for name in ("actor", "actor_target", "critic", "critic_target"):
        for key, tensor in getattr(reference, name).state_dict().items():
            assert torch.equal(getattr(optimized, name).state_dict()[key], tensor)


@pytest.mark.parametrize("noise_std", [0.0, 0.1])
def test_batch_actions_match_original_sampling(njp, noise_std):
    agent = njp.DDPGAgent(35, 2, hidden_dim=16)
    observations = np.random.RandomState(17).randn(40, 35).astype(np.float32)
    np.random.seed(19)
    expected = []
    # 独立保留原逐个体前向计算和噪声采样顺序。
    with torch.no_grad():
        for observation in observations:
            action = agent.actor(torch.as_tensor(observation).unsqueeze(0)).numpy()[0]
            if noise_std:
                action += np.random.normal(0.0, noise_std, size=action.shape)
            expected.append(np.clip(action, -1.0, 1.0).astype(np.float32))
    np.random.seed(19)
    actual = agent.act_batch(observations, noise_std=noise_std)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (40, 2)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("fused_adam", [False, True])
def test_cpu_adam_preserves_legacy_constructor(njp, monkeypatch, fused_adam):
    original_adam = torch.optim.Adam
    calls = []

    # 旧版 Adam 不接受 fused；CPU 即使传 --fused_adam 也应保持默认调用。
    def legacy_adam(parameters, lr):
        calls.append(lr)
        return original_adam(parameters, lr=lr)

    monkeypatch.setattr(torch.optim, "Adam", legacy_adam)
    agent = njp.DDPGAgent(35, 2, hidden_dim=16, fused_adam=fused_adam)
    assert calls == [1e-4, 1e-3]
    assert not agent.fused_adam


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is needed to test fused Adam")
def test_fused_adam_matches_default_updates(njp):
    original = njp.DDPGAgent(35, 2, hidden_dim=16, device="cuda")
    fused = njp.DDPGAgent(35, 2, hidden_dim=16, device="cuda", fused_adam=True)
    names = ("actor", "actor_target", "critic", "critic_target")
    for name in names:
        getattr(fused, name).load_state_dict(getattr(original, name).state_dict())
    replay = njp.ReplayBuffer(32, 35, 2)
    rng = np.random.RandomState(27)
    for _ in range(32):
        replay.push(rng.randn(35), rng.randn(2), rng.randn(), rng.randn(35), False)
    for seed in range(3):
        np.random.seed(seed)
        original.update(replay, 16, return_losses=False)
        np.random.seed(seed)
        fused.update(replay, 16, return_losses=False)
    for name in names:
        for key, expected in getattr(original, name).state_dict().items():
            torch.testing.assert_close(getattr(fused, name).state_dict()[key], expected,
                                       rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is needed to request fused Adam")
def test_unsupported_fused_adam_has_clear_error(njp, monkeypatch):
    def legacy_adam(parameters, lr):
        pytest.fail("Unsupported fused Adam must be rejected before creating an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", legacy_adam)
    with pytest.raises(ValueError, match="does not support --fused_adam"):
        njp.DDPGAgent(35, 2, hidden_dim=16, device="cuda", fused_adam=True)


def test_headless_training_batches_both_species(njp, monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    make_env = njp.make_env
    shapes = []
    act_batch = njp.DDPGAgent.act_batch

    def headless_env():
        env = make_env()

        def unexpected_render(*args, **kwargs):
            pytest.fail("render_interval=0 must disable rendering, including episode 20")

        env.render = unexpected_render
        return env

    def record_batch(self, observations, **kwargs):
        shapes.append(observations.shape)
        return act_batch(self, observations, **kwargs)

    monkeypatch.setattr(njp, "make_env", headless_env)
    monkeypatch.setattr(njp.DDPGAgent, "act_batch", record_batch)
    output = tmp_path / "policies.pt"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--episodes", "20", "--episode_length", "1", "--start_steps", "0",
        "--updates_per_step", "0", "--hidden_dim", "16", "--buffer_size", "128",
        "--render_interval", "0", "--model_path", str(output),
    ])
    njp.train(njp.get_args())
    assert shapes == [(3, 35), (10, 35)] * 20
    saved = torch.load(output, weights_only=True)
    assert "predator_actor" in saved and "prey_actor" in saved
