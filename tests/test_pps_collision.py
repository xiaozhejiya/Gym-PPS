from copy import deepcopy
from types import MethodType, SimpleNamespace

import gym
from gym.envs.pps.pps import PredatorPreySwarmEnv
from gym.envs.pps.putils import make_periodic
from gym.utils import get_dist_b2b
from gym.wrappers import PredatorPreySwarmCustomizer
import numpy as np
import pytest


def reference_collision_force(env):
    """保留原始逐粒子对实现，作为批量碰撞计算的独立回归基准。"""
    forces = np.zeros((2 * env._n_peo, env._n_peo))
    for i in range(env._n_peo):
        for j in range(i):
            delta = env._p[:, j] - env._p[:, i]
            if env._is_periodic:
                delta = make_periodic(delta, env._L)
            direction = delta / env._d_b2b_center[i, j]
            forces[2 * i:2 * (i + 1), j] = (
                env._is_collide_b2b[i, j] * env.d_b2b_edge[i, j]
                * env._k_ball * (-direction)
            )
            forces[2 * j:2 * (j + 1), i] = -forces[2 * i:2 * (i + 1), j]
    return forces.sum(axis=1, keepdims=True).reshape(env._n_peo, 2).T


@pytest.mark.parametrize("count", [1, 2, 13, 52])
@pytest.mark.parametrize("periodic", [False, True])
def test_collision_force_matches_original(count, periodic):
    rng = np.random.RandomState(19)
    positions = rng.uniform(-2, 2, (2, count))
    if count >= 2:
        positions[:, 0] = [0, 0]
        positions[:, 1] = [0.02, 0.01]
    if count >= 4:
        positions[:, -2] = [-1.985, 0.5]
        positions[:, -1] = [1.985, 0.5]
    sizes = rng.uniform(0.035, 0.2, count)
    pair_sizes = sizes[:, None] + sizes[None, :]
    np.fill_diagonal(pair_sizes, 0)
    center, edge, colliding = get_dist_b2b(positions, 2, periodic, pair_sizes)
    env = SimpleNamespace(
        _p=positions, _n_peo=count, _L=2, _is_periodic=periodic,
        _d_b2b_center=center, d_b2b_edge=edge,
        _is_collide_b2b=colliding, _k_ball=50,
    )
    actual = PredatorPreySwarmEnv._get_ball_collision_force(env)
    np.testing.assert_array_equal(actual, reference_collision_force(env))
    np.testing.assert_allclose(actual.sum(axis=1), 0, atol=1e-12)
    assert np.isfinite(actual).all()
    if count >= 2:
        assert np.any(actual != 0)


@pytest.mark.parametrize("dynamics", ["Cartesian", "Polar"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("obstacles", [0, 2])
def test_trajectory_matches_original(dynamics, periodic, obstacles):
    np.random.seed(23)
    env = PredatorPreySwarmCustomizer(gym.make("PredatorPreySwarm-v0"), {
        "L": 2, "n_p": 12, "n_e": 40,
        "dynamics_mode": dynamics, "is_periodic": periodic,
        "pursuer_strategy": "input", "escaper_strategy": "input",
    })
    try:
        # 模拟器没有障碍物数量的公开设置入口，测试直接修改后重新初始化。
        env.unwrapped._n_o = obstacles
        env.__reinit__()
        env.reset()
        assert env.unwrapped._n_peo == 52 + obstacles
        # 人工布置接触粒子和地图接缝两侧的粒子，尽早覆盖碰撞与周期边界情况。
        env.unwrapped._p[:, :2] = [[0, 0.02], [0, 0.01]]
        env.unwrapped._p[:, 2:4] = [[-1.985, 1.985], [0.5, 0.5]]
        reference = deepcopy(env.unwrapped)
        reference._get_ball_collision_force = MethodType(reference_collision_force, reference)
        rng = np.random.RandomState(41)
        for _ in range(25):
            action = rng.uniform(-1, 1, (2, env.unwrapped._n_peo))
            action[:, 52:] = 0
            # step 会原地缩放动作；两套环境各传一份副本，避免测试输入相互污染。
            actual = env.step(action.copy())
            expected = reference.step(action.copy())
            for actual_value, expected_value in zip(actual, expected):
                np.testing.assert_array_equal(actual_value, expected_value)
            for name in ("_p", "_dp", "_energy"):
                np.testing.assert_array_equal(getattr(env.unwrapped, name), getattr(reference, name))
    finally:
        env.close()
