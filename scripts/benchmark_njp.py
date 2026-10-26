"""Compare full NJP training steps against a Git revision, in fresh processes.

Run from the repository root; see example_algorithm_NJP/PERFORMANCE.md.
No checkpoints, experiment configuration, or running processes are modified.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'example_algorithm_NJP'))

import gym
import numpy as np
import torch
from agent import DDPGAgent
from custom_env import MyAction, MyObs, MyReward, predator_prey_distances
from replay_buffer import ReplayBuffer


def baseline_module(name, path, revision):
    source = subprocess.check_output(['git', 'show', revision + ':' + path], cwd=ROOT)
    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / 'baseline.py'
        filename.write_bytes(source)
        spec = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def fingerprint(value):
    """Hash values, omitting only the deliberately different Adam backend flags."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype.str)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                if key not in ('fused', 'foreach', 'capturable'):
                    visit(key)
                    visit(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode() + b'\0')

    visit(value)
    return digest.hexdigest()


class Runner:
    def __init__(self, args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        parameters = json.loads((ROOT / 'example_algorithm_NJP/custom_param.json').read_text())
        self.episode_length = 120
        if args.scenario == 'swarm':
            parameters.update(L=4, n_p=12, n_e=40, angle_p_max=.05, angle_e_max=.05)
            self.episode_length = 1000
        self.parameters = parameters
        with contextlib.redirect_stdout(io.StringIO()):
            self.env = gym.make('PredatorPreySwarm-v0')
            for wrapper in (MyReward, MyAction, MyObs):
                self.env = wrapper(self.env, parameters)
        agent_class = DDPGAgent
        kwargs = {'cuda_graph': args.mode == 'graph'}
        if args.mode == 'baseline':
            agent_class = baseline_module('njp_baseline_agent', 'example_algorithm_NJP/agent.py',
                                          args.baseline_ref).DDPGAgent
            env_class = baseline_module('gym.envs.pps._benchmark_baseline', 'gym/envs/pps/pps.py',
                                        args.baseline_ref).PredatorPreySwarmEnv
            self.env.unwrapped.__class__ = env_class
            kwargs = {}
        self.env.seed(args.seed)
        self.obs = self.env.reset()
        self.names = ('predator', 'prey')
        obs_dim = self.obs['predator'].shape[1]
        self.agents = [agent_class(obs_dim, 2, hidden_dim=128, gamma=.95, tau=.01,
                                  device='cuda', fused_adam=True, **kwargs) for _ in self.names]
        self.buffers = [ReplayBuffer(500000, obs_dim, 2) for _ in self.names]
        self.index = 0
        self.updates = [0, 0]
        self.metrics = dict(predator_reward=0., prey_reward=0., min_distance=float('inf'), caught=False)

    def step(self):
        if self.index and self.index % self.episode_length == 0:
            self.obs = self.env.reset()
        actions = {name: agent.act_batch(self.obs[name], noise_std=.05).T
                   for name, agent in zip(self.names, self.agents)}
        obs, rewards, _, _ = self.env.step(actions)
        distances = predator_prey_distances(self.env)
        for name, replay in zip(self.names, self.buffers):
            for index in range(self.obs[name].shape[0]):
                replay.push(self.obs[name][index], actions[name][:, index], rewards[name][index],
                            obs[name][index], False)
            self.metrics[name + '_reward'] += float(np.mean(rewards[name]))
        self.metrics['min_distance'] = min(self.metrics['min_distance'], float(np.min(distances)))
        self.metrics['caught'] |= bool(np.min(distances) <= self.env.size_p + self.env.size_e)
        self.obs = obs
        for index, (agent, replay) in enumerate(zip(self.agents, self.buffers)):
            if len(replay) >= 256:
                agent.update(replay, 256, return_losses=False)
                self.updates[index] += 1
        self.index += 1

    def state(self):
        physical = {name: getattr(self.env.unwrapped, name) for name in (
            '_p', '_dp', '_ddp', '_energy', '_theta', '_heading', '_p_traj',
            '_d_b2b_center', 'd_b2b_edge', '_is_collide_b2b')}
        return dict(agents=[agent.state_dict() for agent in self.agents],
                    replay=[buffer.state_dict() for buffer in self.buffers],
                    observations=self.obs, physical=physical, metrics=self.metrics, updates=self.updates,
                    numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
                    env_rng=self.env.unwrapped.np_random.get_state())


def trial(args):
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this three-way benchmark')
    runner = Runner(args)
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.warmup):
            runner.step()
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - started
        before = list(runner.updates)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.steps):
            runner.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert all(after - prior == args.steps for after, prior in zip(runner.updates, before)), \
            'Increase warmup: both agents must update at every timed step'
        if args.mode == 'graph':
            assert all(agent.cuda_graph_status == 'active' for agent in runner.agents)
        row = dict(mode=args.mode, scenario=args.scenario, seed=args.seed,
                   steps_per_second=args.steps / elapsed, timed_seconds=elapsed,
                   warmup_seconds=warmup_seconds, updates_per_agent=args.steps,
                   capture_seconds=sum(getattr(agent, 'graph_capture_seconds', 0.) for agent in runner.agents),
                   torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                   parameters=runner.parameters, episode_length=runner.episode_length)
        row['state_sha256'] = fingerprint(runner.state())
        return row
    finally:
        runner.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--output', type=Path, default=Path('/tmp/njp-benchmark.json'))
    parser.add_argument('--scenario', choices=['default', 'swarm'], default='default')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=171)
    parser.add_argument('--mode', choices=['baseline', 'eager', 'graph'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.repeats) <= 0:
        parser.error('steps, warmup and repeats must be positive')
    args.baseline_ref = subprocess.check_output(
        ['git', 'rev-parse', args.baseline_ref + '^{commit}'], cwd=ROOT, text=True).strip()
    if args.mode:
        print(json.dumps(trial(args)))
        return
    result = dict(baseline_ref=args.baseline_ref, scenario=args.scenario, warmup_steps=args.warmup,
                  timed_steps=args.steps, repeats=args.repeats, batch_size=256, hidden_dim=128,
                  buffer_capacity=500000, noise=.05, torch_threads=1,
                  gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                  torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__, trials=[])
    modes = ['baseline', 'eager', 'graph']
    for repeat in range(args.repeats):
        offset = repeat % len(modes)
        rows = []
        for mode in modes[offset:] + modes[:offset]:
            command = [sys.executable, str(Path(__file__).resolve()), '--mode', mode,
                       '--baseline-ref', args.baseline_ref, '--scenario', args.scenario,
                       '--seed', str(args.seed + repeat), '--warmup', str(args.warmup), '--steps', str(args.steps)]
            output = subprocess.check_output(command, text=True, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            row = json.loads(output.strip().splitlines()[-1])
            rows.append(row)
            result['trials'].append(row)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print('{} repeat {}: {:.1f} steps/s'.format(mode, repeat + 1, row['steps_per_second']), flush=True)
        assert len({row['state_sha256'] for row in rows}) == 1, 'Training states differ across modes'
    result['exact_final_states'] = True
    result['summary'] = {}
    for mode in modes:
        rows = [row for row in result['trials'] if row['mode'] == mode]
        rates = [row['steps_per_second'] for row in rows]
        result['summary'][mode] = dict(median_steps_per_second=statistics.median(rates),
                                      min_steps_per_second=min(rates), max_steps_per_second=max(rates),
                                      median_capture_seconds=statistics.median(row['capture_seconds'] for row in rows),
                                      median_torch_peak_allocated_mib=statistics.median(row['torch_peak_allocated_mib'] for row in rows))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
