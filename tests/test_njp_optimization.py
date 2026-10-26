from copy import deepcopy
from types import MethodType

import numpy as np
import pytest
import torch

from tests.njp_reference import assert_exact, soft_update, update_tensors


def replay(njp):
    buffer = njp.ReplayBuffer(512, 35, 2)
    rng = np.random.RandomState(28)
    for index in range(512):
        buffer.push(rng.randn(35), rng.randn(2), rng.randn(), rng.randn(35), index % 17 == 0)
    return buffer


def reference(njp, device):
    agent = njp.DDPGAgent(35, 2, device=device, fused_adam=device == 'cuda')
    agent._soft_update = MethodType(soft_update, agent)
    agent._update_tensors = MethodType(update_tensors, agent)
    return agent


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mode', ['original_repeat', 'foreach', 'freeze', 'combined', 'graph'])
def test_hundred_updates_exact(njp, device, mode):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    if mode == 'graph' and device == 'cpu':
        pytest.skip('Graph requires CUDA')
    torch.manual_seed(38)
    baseline = reference(njp, device)
    candidate = njp.DDPGAgent(35, 2, device=device, fused_adam=device == 'cuda', cuda_graph=mode == 'graph')
    candidate.load_state_dict(deepcopy(baseline.state_dict()))
    if mode in ('original_repeat', 'freeze'):
        candidate._soft_update = MethodType(soft_update, candidate)
    if mode in ('original_repeat', 'foreach'):
        candidate._update_tensors = MethodType(update_tensors, candidate)
    buffer = replay(njp)
    for index in range(100):
        np.random.seed(index)
        expected = baseline.update(buffer, 256)
        expected_rng = np.random.get_state()
        np.random.seed(index)
        before_torch_rng = torch.get_rng_state().clone()
        actual = candidate.update(buffer, 256)
        assert actual == expected, (mode, index, actual, expected)
        assert_exact(candidate.state_dict(), baseline.state_dict())
        assert torch.equal(before_torch_rng, torch.get_rng_state())
        for left, right in zip(np.random.get_state(), expected_rng):
            np.testing.assert_array_equal(left, right)
    if mode == 'graph':
        assert candidate.cuda_graph_status == 'active'
        assert candidate.graph_capture_seconds > 0


def test_graph_requires_cuda_and_fused(njp):
    with pytest.raises(ValueError, match='requires CUDA and fused_adam'):
        njp.DDPGAgent(35, 2, cuda_graph=True)


def test_critic_grad_flags_restored_on_exception(njp, monkeypatch):
    agent = njp.DDPGAgent(35, 2)
    def fail(*args, **kwargs):
        raise RuntimeError('actor failed')
    monkeypatch.setattr(agent.actor, 'forward', fail)
    with pytest.raises(RuntimeError, match='actor failed'):
        agent.update(replay(njp), 4)
    assert all(parameter.requires_grad for parameter in agent.critic.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('failure_stage', ['allocation', 'during_warmup', 'before_capture', 'during_capture'])
def test_graph_resume_shape_change_and_capture_failure(njp, monkeypatch, failure_stage):
    buffer = replay(njp)
    eager = reference(njp, 'cuda')
    graph = njp.DDPGAgent(35, 2, device='cuda', fused_adam=True, cuda_graph=True)
    graph.load_state_dict(deepcopy(eager.state_dict()))
    for index, size in enumerate((256, 16, 256)):
        np.random.seed(index)
        eager.update(buffer, size, return_losses=False)
        np.random.seed(index)
        graph.update(buffer, size, return_losses=False)
        assert_exact(graph.state_dict(), eager.state_dict())
        assert graph.cuda_graph_status == 'active'
    saved = deepcopy(graph.state_dict())
    graph.load_state_dict(deepcopy(saved))
    assert graph.cuda_graph_status == 'pending'
    restored = reference(njp, 'cuda')
    restored.load_state_dict(deepcopy(saved))
    assert restored.actor_opt.param_groups[0]['capturable'] is False
    np.random.seed(9)
    graph.update(buffer, 256)
    np.random.seed(9)
    restored.update(buffer, 256)
    assert_exact(graph.state_dict(), restored.state_dict())

    graph.load_state_dict(deepcopy(restored.state_dict()))
    if failure_stage in ('allocation', 'before_capture'):
        def fail(*args, **kwargs):
            raise RuntimeError('injected graph preparation failure')
        monkeypatch.setattr(torch.cuda, 'CUDAGraph' if failure_stage == 'allocation' else 'graph', fail)
    else:
        original_update = graph._update_tensors
        calls = []
        def fail_inside(batch):
            losses = original_update(batch)
            calls.append(1)
            if len(calls) == (2 if failure_stage == 'during_warmup' else 4):
                raise RuntimeError('injected failure during capture')
            return losses
        monkeypatch.setattr(graph, '_update_tensors', fail_inside)
    np.random.seed(10)
    with pytest.warns(RuntimeWarning, match='restored training state'):
        graph.update(buffer, 256)
    np.random.seed(10)
    restored.update(buffer, 256)
    assert graph.cuda_graph_status == 'fallback'
    assert_exact(graph.state_dict(), restored.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_graph_replay_error_is_not_retried(njp, monkeypatch):
    graph = njp.DDPGAgent(35, 2, device='cuda', fused_adam=True, cuda_graph=True)
    buffer = replay(njp)
    graph.update(buffer, 256)
    before = deepcopy(graph.state_dict())
    def fail(*args):
        raise RuntimeError('injected replay failure')
    monkeypatch.setattr(graph._graph_runner, 'replay', fail)
    with pytest.raises(RuntimeError, match='injected replay failure'):
        graph.update(buffer, 256)
    assert_exact(graph.state_dict(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_failed_graph_rollback_stops_training(njp, monkeypatch):
    graph = njp.DDPGAgent(35, 2, device='cuda', fused_adam=True, cuda_graph=True)
    def fail_capture(*args, **kwargs):
        raise RuntimeError('capture failed')
    def fail_restore(*args, **kwargs):
        raise RuntimeError('rollback failed')
    monkeypatch.setattr(torch.cuda, 'graph', fail_capture)
    monkeypatch.setattr(graph, 'load_state_dict', fail_restore)
    with pytest.raises(RuntimeError, match='rollback failed'):
        graph.update(replay(njp), 256)


def test_graph_cli_switches_and_validation(njp, monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'argv', ['train.py', '--cuda_graph', '--no_cuda_graph'])
    assert njp.get_args().cuda_graph is False
    monkeypatch.setattr(sys, 'argv', ['train.py', '--cuda_graph'])
    with pytest.raises(ValueError, match='requires CUDA and --fused_adam'):
        njp.train(njp.get_args())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.usefixtures("njp_training_env")
def test_graph_training_checkpoints_and_bidirectional_resume(njp, monkeypatch, tmp_path):
    import sys
    initial = tmp_path / 'initial.pt'
    torch.save({name + '_actor': njp.DDPGAgent(35, 2, hidden_dim=16).actor.state_dict()
                for name in ('predator', 'prey')}, initial)
    outputs = []
    for graph in (False, True):
        output = tmp_path / ('initial_graph.pt' if graph else 'initial_eager.pt')
        monkeypatch.setattr(sys, 'argv', [
            'train.py', '--init_model', str(initial), '--model_path', str(output),
            '--hidden_dim', '16', '--buffer_size', '128', '--batch_size', '4',
            '--episodes', '2', '--episode_length', '3', '--total_steps', '5',
            '--start_steps', '0', '--noise', '.05', '--noise_min', '.05',
            '--render_interval', '0', '--fused_adam',
            '--cuda_graph' if graph else '--no_cuda_graph'])
        njp.train(njp.get_args())
        saved = torch.load(output, map_location='cpu', weights_only=True)
        assert saved['training']['cumulative_environment_steps'] == 5
        assert saved['training']['cuda_graph_status']['predator'] == ('active' if graph else 'disabled')
        outputs.append((output, saved))
    assert_exact(outputs[0][1]['training_state'], outputs[1][1]['training_state'])
    resumed = []
    # Resume eager with graphs, and graphs with eager, from equivalent saved states.
    for index, (source, _) in enumerate(outputs):
        output = tmp_path / ('resumed_{}.pt'.format(index))
        monkeypatch.setattr(sys, 'argv', [
            'train.py', '--init_model', str(source), '--model_path', str(output),
            '--hidden_dim', '16', '--buffer_size', '128', '--batch_size', '4',
            '--episodes', '1', '--episode_length', '2', '--total_steps', '2',
            '--render_interval', '0', '--noise', '.05', '--noise_min', '.05', '--fused_adam',
            '--cuda_graph' if index == 0 else '--no_cuda_graph'])
        njp.train(njp.get_args())
        saved = torch.load(output, map_location='cpu', weights_only=True)
        assert saved['training']['cumulative_environment_steps'] == 7
        assert saved['training']['restored_training_state'] is True
        for name in ('predator', 'prey'):
            state = saved['training_state'][name + '_agent']
            for opt in ('actor_opt', 'critic_opt'):
                assert all(t['step'].item() == 7 for t in state[opt]['state'].values())
        resumed.append(saved)
    assert_exact(resumed[0]['training_state'], resumed[1]['training_state'])
