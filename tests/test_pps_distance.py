from copy import deepcopy

import gym
from gym.envs.pps.putils import get_dist_b2b_batch
from gym.utils import get_dist_b2b
from gym.wrappers import PredatorPreySwarmCustomizer
import numpy as np
import pytest


@pytest.mark.parametrize('half_width', [1, 2, 4])
@pytest.mark.parametrize('periodic', [False, True])
def test_exact_contact_and_adjacent_floats(half_width, periodic):
    # Origin-relative coordinates avoid rounding away the nextafter perturbation
    # when adding a small gap to a much larger boundary coordinate.
    contact = np.float64(.125)
    sizes = np.array([[0., contact], [contact, 0.]])
    for gap, collision in ((np.nextafter(contact, 0.), True),
                           (contact, False),
                           (np.nextafter(contact, np.inf), False)):
        positions = np.array([[0., gap], [0., 0.]])
        actual = get_dist_b2b_batch(positions, half_width, periodic, sizes)
        expected = get_dist_b2b(positions, half_width, periodic, sizes)
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)
        assert actual[0][0, 1] == gap
        assert actual[1][0, 1] == abs(gap - contact)
        assert bool(actual[2][0, 1]) is collision


@pytest.mark.parametrize('half_width', [1, 2, 4])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('count', [1, 2, 13, 52, 54])
def test_distances_match_native_exactly(half_width, periodic, count):
    for seed in range(10):
        rng = np.random.RandomState(seed)
        p = rng.uniform(-half_width, half_width, (2, count))
        radii = rng.uniform(.035, .2, count)
        sizes = radii[:, None] + radii[None, :]
        np.fill_diagonal(sizes, 0)
        if count > 1:
            # Coincident centers, exact contact, adjacent floats, and periodic seams.
            gap = [0., sizes[0, 1], np.nextafter(sizes[0, 1], 0),
                   np.nextafter(sizes[0, 1], np.inf), 2 * half_width][seed % 5]
            p[:, 0] = [-half_width, 0]
            p[:, 1] = [-half_width + gap, 0]
        for actual, expected in zip(get_dist_b2b_batch(p, half_width, periodic, sizes),
                                    get_dist_b2b(p, half_width, periodic, sizes)):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('half_width', [1, 2, 4])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('dynamics', ['Polar', 'Cartesian'])
@pytest.mark.parametrize('obstacles', [0, 2])
def test_distance_trajectory_exact(half_width, periodic, dynamics, obstacles, monkeypatch):
    import gym.envs.pps.pps as pps
    np.random.seed(9)
    env = PredatorPreySwarmCustomizer(gym.make('PredatorPreySwarm-v0'), {
        'L': half_width, 'n_p': 12, 'n_e': 40, 'is_periodic': periodic,
        'dynamics_mode': dynamics, 'pursuer_strategy': 'input', 'escaper_strategy': 'input'})
    try:
        env.unwrapped._n_o = obstacles
        env.__reinit__()
        env.reset()
        old = deepcopy(env.unwrapped)
        # Repeat reset with the same RNG, as rejection sampling also depends on distances.
        random_state = np.random.get_state()
        actual = env.reset()
        np.random.set_state(random_state)
        with monkeypatch.context() as patch:
            patch.setattr(pps, 'get_dist_b2b_batch', get_dist_b2b)
            expected = old.reset()
        np.testing.assert_array_equal(actual, expected)
        rng = np.random.RandomState(21)
        for _ in range(100):
            action = rng.uniform(-1, 1, (2, env.unwrapped._n_peo))
            action[:, 52:] = 0
            actual = env.step(action.copy())
            with monkeypatch.context() as patch:
                patch.setattr(pps, 'get_dist_b2b_batch', get_dist_b2b)
                expected = old.step(action.copy())
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            for name in ('_p', '_dp', '_ddp', '_energy', '_d_b2b_center', 'd_b2b_edge', '_is_collide_b2b'):
                np.testing.assert_array_equal(getattr(env.unwrapped, name), getattr(old, name))
    finally:
        env.close()
