from copy import deepcopy
from types import MethodType

import gym
from gym.envs.pps.putils import get_focused, make_periodic
from gym.wrappers import PredatorPreySwarmCustomizer
import numpy as np
import pytest


def reference_observation(env):
    """使用批处理前的逐个体计算方式，独立检查邻居筛选和观测布局。"""
    obs = np.zeros(env.observation_space.shape)
    motion = env._dp if env._dynamics_mode == "Cartesian" else env._heading
    populations = (slice(0, env._n_p), slice(env._n_p, env._n_pe))
    groups = ((env._FoV_p, (env._topo_n_p2p, env._topo_n_p2e), env.max_energy_p),
              (env._FoV_e, (env._topo_n_e2p, env._topo_n_e2e), env.max_energy_e))
    for species, (observers, (field_of_view, widths, max_energy)) in enumerate(zip(populations, groups)):
        for i in range(observers.start, observers.stop):
            positions, velocities = [env._p[:, [i]]], [env._dp[:, [i]]]
            for neighbor_species, (neighbors, width) in enumerate(zip(populations, widths)):
                rel_pos = env._p[:, neighbors] - env._p[:, [i]]
                if env._is_periodic:
                    rel_pos = make_periodic(rel_pos, env._L)
                rel_vel = motion[:, neighbors] - motion[:, [i]]
                pos, vel = get_focused(rel_pos, rel_vel, field_of_view, width, species == neighbor_species)
                positions.append(pos)
                velocities.append(vel)
            features = np.concatenate((np.concatenate(positions, axis=1),
                                       np.concatenate(velocities, axis=1)), axis=0).T.reshape(-1)
            obs[:features.size, i] = features
            obs[features.size, i] = 2 / max_energy * env._energy[0, i] - 1
            if env._dynamics_mode == "Polar":
                obs[features.size + 1:features.size + 3, i] = env._heading[:, i]
    return obs


@pytest.mark.parametrize("counts", [(1, 1), (3, 10), (12, 40), (0, 3), (3, 0)])
@pytest.mark.parametrize("dynamics", ["Cartesian", "Polar"])
@pytest.mark.parametrize("periodic", [False, True])
def test_batched_observation_boundaries(counts, dynamics, periodic):
    np.random.seed(37)
    env = PredatorPreySwarmCustomizer(gym.make("PredatorPreySwarm-v0"), {
        "L": 2, "n_p": counts[0], "n_e": counts[1],
        "dynamics_mode": dynamics, "is_periodic": periodic,
        "FoV_p": 0.125, "FoV_e": 0,
        "topo_n_p2p": 8, "topo_n_p2e": 1,
        "topo_n_e2p": 0, "topo_n_e2e": 10,
    })
    try:
        env.reset()
        base = env.unwrapped
        # 同时覆盖距离并列、个体重合、视野精确边界和周期地图接缝。
        special = np.array([[0, 0, 0.125, -0.125, 1.99, -1.99],
                            [0, 0, 0, 0, 0, 0]])
        length = min(base._n_pe, special.shape[1])
        base._p[:, :length] = special[:, :length]
        rng = np.random.RandomState(71)
        base._dp[:] = rng.randn(*base._dp.shape)
        base._energy[:] = rng.uniform(0, 1000, base._energy.shape)
        expected = reference_observation(base)
        actual = base._get_obs()
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == env.observation_space.shape
    finally:
        env.close()


@pytest.mark.parametrize("dynamics", ["Cartesian", "Polar"])
@pytest.mark.parametrize("periodic", [False, True])
def test_batched_observation_trajectory(dynamics, periodic):
    np.random.seed(31)
    env = PredatorPreySwarmCustomizer(gym.make("PredatorPreySwarm-v0"), {
        "L": 2, "n_p": 12, "n_e": 40,
        "dynamics_mode": dynamics, "is_periodic": periodic,
    })
    try:
        env.reset()
        reference = deepcopy(env.unwrapped)
        # 只替换参考环境的观测实现，以逐步精确比较检验批处理未改变完整轨迹。
        reference._get_obs = MethodType(reference_observation, reference)
        rng = np.random.RandomState(53)
        for _ in range(30):
            action = rng.uniform(-1, 1, env.action_space.shape)
            actual = env.step(action.copy())
            expected = reference.step(action.copy())
            for current, original in zip(actual, expected):
                np.testing.assert_array_equal(current, original)
    finally:
        env.close()
