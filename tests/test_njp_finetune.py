import importlib
from pathlib import Path
import sys

import gym
from gym.wrappers import PredatorPreySwarmCustomizer
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def njp():
    # 示例使用 agent/train 等顶层模块名；测试前后隔离导入，避免与其他示例串扰。
    directory = Path(__file__).resolve().parents[1] / "example_algorithm_NJP"
    original_path = list(sys.path)
    names = ("agent", "arguments", "custom_env", "networks", "replay_buffer", "train")
    original_modules = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        yield importlib.import_module("train")
    finally:
        sys.path[:] = original_path
        for name, module in original_modules.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


@pytest.mark.parametrize("half_width", [1, 2])
def test_spawn_uses_full_map(half_width):
    np.random.seed(7)
    env = PredatorPreySwarmCustomizer(gym.make("PredatorPreySwarm-v0"), {"L": half_width})
    try:
        env.reset()
        margin = np.max(env.unwrapped._size)
        assert np.all(np.abs(env.p) <= half_width - margin)
        if half_width == 2:
            assert np.any(np.abs(env.p) > 1)
    finally:
        env.close()


def test_loaded_policy_updates_target_without_changing_critic(njp):
    """检查仅加载策略的边界，并验证批量动作与直接网络推理一致。"""
    source = njp.DDPGAgent(35, 2, hidden_dim=16)
    destination = njp.DDPGAgent(35, 2, hidden_dim=16)
    critic_before = {key: value.clone() for key, value in destination.critic.state_dict().items()}
    destination.load_policy(source.actor.state_dict())
    for key, value in source.actor.state_dict().items():
        assert torch.equal(destination.actor.state_dict()[key], value)
        assert torch.equal(destination.actor_target.state_dict()[key], value)
    for key, value in critic_before.items():
        assert torch.equal(destination.critic.state_dict()[key], value)

    observations = np.random.randn(40, 35).astype(np.float32)
    actions = destination.act_batch(observations)
    with torch.no_grad():
        expected = source.actor(torch.as_tensor(observations)).numpy()
    np.testing.assert_allclose(actions, expected, atol=1e-6)
    assert actions.shape == (40, 2)
    assert actions.dtype == np.float32
    np.testing.assert_allclose(destination.act(observations[0]), expected[0], atol=1e-6)


@pytest.mark.parametrize("previous_training, previous_steps, previous_episodes", [
    ({}, 0, 0),
    ({"environment_steps": 6000, "episodes": 50}, 6000, 50),
    ({"environment_steps": 54000, "episodes": 450,
      "cumulative_environment_steps": 60000, "cumulative_episodes": 500}, 60000, 500),
])
def test_finetune_warmup_uses_saved_policy(njp, monkeypatch, tmp_path, previous_training, previous_steps, previous_episodes):
    # 只训练采样阶段，确认旧检查点的策略、累计计数和源文件均被正确保留。
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source = {}
    for species in ("predator", "prey"):
        source[species + "_actor"] = njp.DDPGAgent(35, 2, hidden_dim=16).actor.state_dict()
    source["training"] = previous_training
    checkpoint = tmp_path / "source.pt"
    output = tmp_path / "output.pt"
    torch.save(source, checkpoint)
    original_bytes = checkpoint.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--init_model", str(checkpoint), "--model_path", str(output),
        "--episodes", "2", "--episode_length", "2", "--hidden_dim", "16",
        "--buffer_size", "128", "--noise", "0", "--noise_min", "0",
        "--checkpoint_interval", "1",
    ])
    buffers = []
    saved_steps = []
    original_buffer = njp.ReplayBuffer
    original_save = torch.save

    def record_save(checkpoint, path):
        assert str(path).endswith(".tmp")
        saved_steps.append(checkpoint["training"]["cumulative_environment_steps"])
        original_save(checkpoint, path)

    class RecordingBuffer(original_buffer):
        def __init__(self, *args):
            super().__init__(*args)
            buffers.append(self)

    monkeypatch.setattr(njp, "ReplayBuffer", RecordingBuffer)
    monkeypatch.setattr(torch, "save", record_save)
    njp.train(njp.get_args())
    saved = torch.load(output, weights_only=True)
    assert saved["training"]["environment_steps"] == 4
    assert saved["training"]["previous_environment_steps"] == previous_steps
    assert saved["training"]["cumulative_environment_steps"] == previous_steps + 4
    assert saved["training"]["episodes"] == 2
    assert saved["training"]["cumulative_episodes"] == previous_episodes + 2
    assert saved_steps == [previous_steps + 2, previous_steps + 4]
    assert not Path(str(output) + ".tmp").exists()
    assert saved["training"]["environment"]["L"] == 2
    assert saved["training"]["environment"]["n_p"] == 12
    assert saved["training"]["environment"]["n_e"] == 40
    assert saved["training"]["rendered_episodes"] == []
    assert checkpoint.read_bytes() == original_bytes
    for species, buffer in zip(("predator", "prey"), buffers):
        policy = njp.DDPGAgent(35, 2, hidden_dim=16)
        policy.load_policy(source[species + "_actor"])
        np.testing.assert_allclose(buffer.actions[0], policy.act(buffer.obs[0]), atol=1e-6)
        for key, value in source[species + "_actor"].items():
            assert torch.equal(saved[species + "_actor"][key], value)


def test_full_training_state_preserves_next_update(njp, tmp_path):
    # 写入超过容量的样本，让缓冲发生回绕；恢复后固定采样种子比较下一次更新。
    agent = njp.DDPGAgent(35, 2, hidden_dim=16, gamma=0.93, tau=0.02)
    replay = njp.ReplayBuffer(8, 35, 2)
    rng = np.random.RandomState(81)
    for _ in range(12):
        replay.push(rng.randn(35), rng.randn(2), rng.randn(), rng.randn(35), False)
    agent.update(replay, 4)
    path = tmp_path / "training.pt"
    torch.save({"agent": agent.state_dict(), "replay": replay.state_dict()}, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored_agent = njp.DDPGAgent(35, 2, hidden_dim=16)
    restored_replay = njp.ReplayBuffer(8, 35, 2)
    restored_agent.load_state_dict(state["agent"])
    restored_replay.load_state_dict(state["replay"])
    assert restored_replay.size == replay.size == 8
    assert restored_replay.ptr == replay.ptr == 4
    assert restored_agent.gamma == agent.gamma
    assert restored_agent.tau == agent.tau
    for name in ("obs", "actions", "rewards", "next_obs", "dones"):
        np.testing.assert_array_equal(getattr(restored_replay, name), getattr(replay, name))
    np.random.seed(92)
    expected = agent.update(replay, 4)
    np.random.seed(92)
    actual = restored_agent.update(restored_replay, 4)
    assert actual == expected
    for name in ("actor", "actor_target", "critic", "critic_target"):
        for key, tensor in getattr(agent, name).state_dict().items():
            assert torch.equal(getattr(restored_agent, name).state_dict()[key], tensor)


def test_skipping_loss_readback_preserves_updates(njp):
    # 两个分支使用相同参数和采样序列，单独验证关闭损失回读不改变学习结果。
    from copy import deepcopy
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is needed to test fused Adam")
def test_fused_adam_can_resume_default_and_fused_states(njp):
    from copy import deepcopy
    original = njp.DDPGAgent(35, 2, hidden_dim=16, device="cuda")
    replay = njp.ReplayBuffer(32, 35, 2)
    rng = np.random.RandomState(27)
    for _ in range(32):
        replay.push(rng.randn(35), rng.randn(2), rng.randn(), rng.randn(35), False)
    original.update(replay, 16, return_losses=False)
    fused = njp.DDPGAgent(35, 2, hidden_dim=16, device="cuda", fused_adam=True)
    # state_dict() 引用实时优化器张量；深拷贝以隔离两组 Adam 动量，再比较下一次更新。
    fused.load_state_dict(deepcopy(original.state_dict()))
    np.random.seed(11)
    original.update(replay, 16, return_losses=False)
    np.random.seed(11)
    fused.update(replay, 16, return_losses=False)
    for name in ("actor", "actor_target", "critic", "critic_target"):
        for key, expected in getattr(original, name).state_dict().items():
            torch.testing.assert_close(getattr(fused, name).state_dict()[key], expected,
                                       rtol=1e-5, atol=1e-6)
    original.load_state_dict(fused.state_dict())
    original.update(replay, 16, return_losses=False)
    assert original.actor_opt.param_groups[0]["fused"] is None
    assert fused.actor_opt.param_groups[0]["fused"] is True
    assert all(torch.isfinite(p).all() for p in original.actor.parameters())


def test_exact_step_limit_headless_and_resume(njp, monkeypatch, tmp_path):
    # 从累计第 19 回合开始，覆盖第 20 回合的渲染边界和最后不足一整回合的步数。
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_make_env = njp.make_env

    def headless_env():
        env = original_make_env()

        def unexpected_render(*args, **kwargs):
            pytest.fail("Headless training must not render, including episode 20")

        env.render = unexpected_render
        return env

    monkeypatch.setattr(njp, "make_env", headless_env)
    source = {
        species + "_actor": njp.DDPGAgent(35, 2, hidden_dim=16).actor.state_dict()
        for species in ("predator", "prey")
    }
    source["training"] = {"cumulative_environment_steps": 199995, "cumulative_episodes": 19}
    initial = tmp_path / "initial.pt"
    output = tmp_path / "200000.pt"
    torch.save(source, initial)
    common = ["--hidden_dim", "16", "--buffer_size", "128", "--batch_size", "4",
              "--render_interval", "0", "--noise", "0.05", "--noise_min", "0.05"]
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--init_model", str(initial), "--model_path", str(output),
        "--episodes", "3", "--episode_length", "2", "--total_steps", "5",
        "--start_steps", "0", *common,
    ])
    njp.train(njp.get_args())
    saved = torch.load(output, weights_only=True)
    metadata = saved["training"]
    assert metadata["environment_steps"] == 5
    assert metadata["cumulative_environment_steps"] == 200000
    assert metadata["cumulative_episodes"] == 22
    assert metadata["episodes"] == 3
    assert metadata["last_episode_steps"] == 1
    assert metadata["rendered_episodes"] == []
    assert metadata["restored_training_state"] is False
    resumed_outputs = []
    # 改变启动种子后，完整检查点仍应恢复相同随机状态，得到相同的后续策略。
    for seed in (7, 81):
        resumed = tmp_path / ("resumed_" + str(seed) + ".pt")
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--init_model", str(output), "--model_path", str(resumed),
            "--episodes", "1", "--episode_length", "1", "--total_steps", "1",
            "--start_steps", "1000", "--seed", str(seed), *common,
        ])
        njp.train(njp.get_args())
        resumed_outputs.append(torch.load(resumed, weights_only=True))
    for checkpoint in resumed_outputs:
        assert checkpoint["training"]["cumulative_environment_steps"] == 200001
        assert checkpoint["training"]["restored_training_state"] is True
        assert checkpoint["training"]["rendered_episodes"] == []
        for species in ("predator", "prey"):
            state = checkpoint["training_state"][species + "_agent"]
            assert next(iter(state["critic_opt"]["state"].values()))["step"].item() == 6
            for key, tensor in checkpoint[species + "_actor"].items():
                assert torch.equal(resumed_outputs[0][species + "_actor"][key], tensor)
