"""Isolated configuration for the confined-space experiment in NJP section 4.4."""
import json
from pathlib import Path

from gym import spaces
from gym.envs.pps.pps import PredatorPreySwarmEnv
import numpy as np

from custom_env import MyAction, MyObs, MyReward, split_obs


CONFIG_PATH = Path(__file__).with_name("experiments") / "paper_swirl.json"


def settings(arena_L=None):
    config = json.loads(CONFIG_PATH.read_text())
    if arena_L is not None:
        diagonal = float(2 * float(arena_L) * np.sqrt(2))
        if not np.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("arena_L must be finite and positive")
        config["environment"].update(L=float(arena_L), FoV_p=diagonal, FoV_e=diagonal)
    return config


def defaults(evaluation=False):
    config = settings()
    if not evaluation:
        return config["training"]
    return {"hidden_dim": config["training"]["hidden_dim"],
            "hidden_layers": config["training"]["hidden_layers"],
            "episodes": config["evaluation"]["episodes"],
            "episode_length": config["evaluation"]["episode_length"],
            "model_path": config["training"]["model_path"]}


class PaperSwirlEnv(PredatorPreySwarmEnv):
    def __init__(self, reward_weights):
        super().__init__()
        self.paper_rewards = dict(reward_weights)

    def _set_nearest_pursuer_action(self, actions):
        nearest = np.argmin(self._d_b2b_center[:self._n_p, self._n_p:self._n_pe], axis=1)
        delta = self._p[:, self._n_p + nearest] - self._p[:, :self._n_p]
        bearing = np.arctan2(delta[1], delta[0])
        error = (bearing - self._theta[0, :self._n_p] + np.pi) % (2 * np.pi) - np.pi
        error[np.all(delta == 0, axis=0)] = 0
        actions[0, :self._n_p] = np.clip(error, -self._angle_p_max, self._angle_p_max)
        actions[1, :self._n_p] = self._linAcc_p_max

    def _limit_velocity(self):
        for group, limit in ((slice(0, self._n_p), self._linVel_p_max),
                             (slice(self._n_p, self._n_pe), self._linVel_e_max)):
            velocity = self._dp[:, group]
            speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
            fast = speed > limit
            velocity[:, fast] *= limit / speed[fast]

    def _get_reward(self, actions):
        contacts = self._is_collide_b2b[self._n_p:self._n_pe, :self._n_p]
        reward = np.concatenate((contacts.sum(axis=0), -contacts.sum(axis=1))).astype(float)[None, :]
        reward *= self.paper_rewards["contact"]
        if self._penalize_control_effort:
            # The engine has already scaled actions into force and radians per
            # step. The paper's angular effort uses radians per second.
            reward -= (self.paper_rewards["forward_effort"] * np.abs(actions[[1], :self._n_pe])
                       + self.paper_rewards["angular_effort"] * np.abs(actions[[0], :self._n_pe] / self._dt))
        if self._penalize_collide_walls and not self._is_periodic:
            reward -= self.paper_rewards["wall"] * self.is_collide_b2w[:, :self._n_pe].sum(axis=0, keepdims=True)
        return reward


class PaperObservation(MyObs):
    def __reinit__(self):
        super().__reinit__()
        features = self.env.observation_space.shape[0] - 1
        self.observation_space = spaces.Dict({
            "predator": spaces.Box(-np.inf, np.inf, (self.env.n_p, features), dtype=np.float32),
            "prey": spaces.Box(-np.inf, np.inf, (self.env.n_e, features), dtype=np.float32),
        })

    def observation(self, observation):
        # Native Polar layout ends in [energy, heading_x, heading_y].
        energy = observation.shape[0] - 3
        # Paper order: own position/velocity/heading, then neighbor features.
        return split_obs(np.concatenate((observation[:4], observation[energy + 1:],
                                         observation[4:energy]), axis=0), self.env)


def make_env(evaluation=False, arena_L=None):
    config = settings(arena_L=arena_L)
    params = dict(config["environment"])
    if evaluation:
        params["n_e"] = config["evaluation"]["n_e"]
    env = PaperSwirlEnv(config["rewards"])
    return PaperObservation(MyAction(MyReward(env, params), params), params)
