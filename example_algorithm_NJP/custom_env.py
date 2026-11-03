import gym
from gym import spaces
import numpy as np
from gym.wrappers import PredatorPreySwarmCustomizer
from gym.wrappers import CustomObservation, CustomReward, CustomAction

"""在此脚本中定义观测和奖励。
可使用以下环境属性构造观测或奖励函数：
self.env.p、dp、ddp、theta、heading、d_b2b_center、is_collide_b2b、energy。
"""


def split_obs(obs, env):
    return {
        "predator": obs[:, :env.n_p].T.astype(np.float32),
        "prey": obs[:, env.n_p:env.n_pe].T.astype(np.float32),
    }


def build_action(predator_actions, prey_actions, env):
    action = np.zeros(env.unwrapped.action_space.shape, dtype=np.float32)
    action[:, :env.n_p] = predator_actions
    action[:, env.n_p:env.n_pe] = prey_actions
    return action


def split_rewards(reward, env):
    reward = reward.reshape(-1)
    return {
        "predator": reward[:env.n_p].astype(np.float32),
        "prey": reward[env.n_p:env.n_pe].astype(np.float32),
    }


def predator_prey_pair_distances(env):
    rel = env.p[:, env.n_p:env.n_pe][:, None, :] - env.p[:, :env.n_p][:, :, None]
    if env.is_periodic:
        rel[rel > env.L] -= 2 * env.L
        rel[rel < -env.L] += 2 * env.L
    distances = np.linalg.norm(rel, axis=0).astype(np.float32)
    if getattr(env.unwrapped, "prey_death_enabled", False):
        distances[:, ~env.unwrapped.alive_mask[env.n_p:env.n_pe]] = np.inf
    return distances


def predator_prey_distances(env):
    return np.min(predator_prey_pair_distances(env), axis=1).astype(np.float32)


def prey_predator_distances(env):
    return np.min(predator_prey_pair_distances(env), axis=0).astype(np.float32)


class MyObs(CustomObservation):

    def __init__(self, env, args):
        super().__init__(env, args)
        self.__reinit__()

    def __reinit__(self):
        super().__reinit__()
        self.observation_space = spaces.Dict(
            {
                "predator": spaces.Box(
                    shape=(self.env.n_p, self.env.observation_space.shape[0]),
                    low=-np.inf,
                    high=np.inf,
                    dtype=np.float32,
                ),
                "prey": spaces.Box(
                    shape=(self.env.n_e, self.env.observation_space.shape[0]),
                    low=-np.inf,
                    high=np.inf,
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, obs):
        return split_obs(obs, self.env)
        

class MyReward(CustomReward):
    
    def reward(self, observation, reward, action):
        return split_rewards(reward, self.env)


class MyAction(CustomAction):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__reinit__()

    def __reinit__(self):
        super().__reinit__()
        action_dim = self.env.action_space.shape[0]
        self.action_space = spaces.Dict(
            {
                "predator": spaces.Box(shape=(action_dim, self.env.n_p), low=-1.0, high=1.0, dtype=np.float32),
                "prey": spaces.Box(shape=(action_dim, self.env.n_e), low=-1.0, high=1.0, dtype=np.float32),
            }
        )

    def action(self, action):
        return build_action(action["predator"], action["prey"], self.env)

    def reverse_action(self, action):
        return {
            "predator": action[:, :self.env.n_p],
            "prey": action[:, self.env.n_p:self.env.n_pe],
        }
    


class MyEnv(PredatorPreySwarmCustomizer):
    def __init__(self, env, args):
        super().__init__(env, args)

    def compute_speed(self):
        if self.env.dp.shape[0] == 2:
            speed = np.sqrt(self.env.dp[[0],:]**2 + self.env.dp[[1],:]**2)
        else:
            speed = np.linalg.norm(self.env.dp, axis=0, keepdims=True)
        return speed
    
    def myfunc(self):
        # 在此定义自定义函数
        # 在此添加自定义代码
        pass
