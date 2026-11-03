"""Persistent, policy-independent supervision for first-contact terminal values."""
import itertools
import numpy as np
import torch


def cube_rotations():
    rotations = []
    for axes in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            matrix = np.eye(3, dtype=np.float32)[list(axes)] * np.asarray(signs, np.float32)[:, None]
            if np.linalg.det(matrix) > 0:
                rotations.append(matrix)
    return np.asarray(rotations)


ROTATIONS = cube_rotations()


def augment_terminal(observations, rng, arena_L):
    rotation = ROTATIONS[rng.randint(len(ROTATIONS), size=len(observations))]
    result = observations.copy()
    for region in (slice(0, 6), slice(6, -4), slice(-3, None)):
        vectors = result[:, region].reshape(len(result), -1, 3)
        result[:, region] = np.einsum('bij,bkj->bki', rotation, vectors).reshape(len(result), -1)
    shift = rng.uniform(-arena_L, arena_L, size=(len(result), 3)).astype(np.float32)
    result[:, :3] = (result[:, :3] + shift + arena_L) % (2 * arena_L) - arena_L
    return result


class TerminalCalibration:
    """Small frozen terminal set; its -1 labels remain valid as the Actor changes.

    This does not contain Monte Carlo policy returns. Randomness is private and
    checkpointed; sample() runs outside graph capture, like ordinary replay.
    """
    def __init__(self, observations, arena_L, seed=700001):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != 178 or not len(observations):
            raise ValueError('Terminal calibration requires nonempty 178D prey observations')
        if not np.isfinite(observations).all() or not np.isfinite(arena_L) or arena_L <= 0:
            raise ValueError('Terminal calibration inputs must be finite with positive arena_L')
        if np.any(np.linalg.norm(observations[:, 6:9], axis=1) >= .095):
            raise ValueError('Terminal calibration observations must have a contact within 0.095')
        self.observations = observations.copy()
        self.arena_L = float(arena_L)
        self._rng = np.random.RandomState(seed)

    def sample(self, count):
        ids = self._rng.randint(len(self.observations), size=count)
        obs = augment_terminal(self.observations[ids], self._rng, self.arena_L)
        action = self._rng.uniform(-1, 1, (count, 4)).astype(np.float32)
        return obs, action, np.full((count, 1), -1, dtype=np.float32)

    def state_dict(self):
        rng = self._rng.get_state()
        return dict(layout='periodic-178d-first-contact-terminal-v1', arena_L=self.arena_L,
            observations=torch.from_numpy(self.observations.copy()),
            rng=(rng[0], rng[1].tolist(), *rng[2:]))

    @classmethod
    def from_state_dict(cls, state):
        if state.get('layout') != 'periodic-178d-first-contact-terminal-v1':
            raise ValueError('Unknown terminal calibration layout')
        result = cls(state['observations'].cpu().numpy(), state['arena_L'])
        rng = state['rng']
        result._rng.set_state((rng[0], np.asarray(rng[1], np.uint32), *rng[2:]))
        return result


class NearThreatReplay:
    """Live one-step transitions near a predator; no frozen policy-return labels."""
    def __init__(self, radius=.5, capacity=30000, seed=700003):
        from replay_buffer import ReplayBuffer
        if not np.isfinite(radius) or radius <= .095:
            raise ValueError('Near-threat radius must exceed the capture radius')
        self.radius = float(radius)
        self.buffer = ReplayBuffer(capacity, 178, 4, seed=seed)

    def push_batch(self, obs, actions, rewards, next_obs, dones):
        mask = np.linalg.norm(obs[:, 6:9], axis=1) < self.radius
        dones = np.broadcast_to(dones, (len(obs),)) if np.ndim(dones) == 0 else np.asarray(dones).reshape(-1)
        self.buffer.push_batch(obs[mask], actions[mask], np.asarray(rewards).reshape(-1)[mask],
                               next_obs[mask], dones[mask])

    def state_dict(self):
        return dict(layout='near-threat-one-step-v1', radius=self.radius, replay=self.buffer.state_dict())

    @classmethod
    def from_state_dict(cls, state):
        if state.get('layout') != 'near-threat-one-step-v1':
            raise ValueError('Unknown near-threat replay layout')
        result = cls(state['radius'], state['replay']['capacity'])
        result.buffer.load_state_dict(state['replay'])
        return result
