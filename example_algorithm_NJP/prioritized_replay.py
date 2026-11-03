"""按 TD 误差比例回放，混合均匀采样并进行重要性修正。"""
import numpy as np
import torch

from replay_buffer import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    prioritized = True

    def __init__(self, capacity, obs_dim, act_dim, seed=1, alpha=.6, beta=.4,
                 beta_steps=200000, uniform_mix=.1, epsilon=1e-6):
        if int(capacity) <= 0:
            raise ValueError("经验池容量必须为正数")
        if not (np.isfinite(alpha) and 0 <= alpha <= 1
                and np.isfinite(beta) and 0 <= beta <= 1
                and np.isfinite(uniform_mix) and 0 <= uniform_mix <= 1
                and np.isfinite(epsilon) and epsilon > 0 and beta_steps > 0):
            raise ValueError("优先回放参数无效")
        super().__init__(capacity, obs_dim, act_dim, seed=1 if seed is None else seed)
        self.alpha, self.beta_start = float(alpha), float(beta)
        self.beta_steps, self.uniform_mix = int(beta_steps), float(uniform_mix)
        self.epsilon = float(epsilon)
        self._leaf_count = 1 << (self.capacity - 1).bit_length()
        self._sum = np.zeros(2 * self._leaf_count, dtype=np.float64)
        self._min = np.full(2 * self._leaf_count, np.inf, dtype=np.float64)
        self._priorities = np.zeros(self.capacity, dtype=np.float64)
        self.max_priority = 1.
        self.sample_calls = self.sampled_transitions = self.sampled_deaths = 0

    def _set_leaves(self, indices):
        indices = np.unique(np.asarray(indices, dtype=np.int64))
        if not len(indices):
            return
        nodes = self._leaf_count + indices
        values = self._priorities[indices] ** self.alpha
        self._sum[nodes], self._min[nodes] = values, values
        while nodes[0] > 1:
            nodes = np.unique(nodes // 2)
            self._sum[nodes] = self._sum[2 * nodes] + self._sum[2 * nodes + 1]
            self._min[nodes] = np.minimum(self._min[2 * nodes], self._min[2 * nodes + 1])

    def push(self, obs, action, reward, next_obs, done):
        index = self.ptr
        super().push(obs, action, reward, next_obs, done)
        self._priorities[index] = self.max_priority
        self._set_leaves([index])

    def push_batch(self, obs, actions, rewards, next_obs, dones=False):
        previous = self.ptr
        count = len(obs)
        super().push_batch(obs, actions, rewards, next_obs, dones)
        written = min(count, self.capacity)
        indices = (previous + count - written + np.arange(written)) % self.capacity
        self._priorities[indices] = self.max_priority
        self._set_leaves(indices)

    def sample_prioritized(self, batch_size):
        if self.size == 0 or batch_size <= 0:
            raise ValueError("批量大小必须为正数，且经验池不能为空")
        total = self._sum[1]
        values = self._rng.uniform(0., total, batch_size)
        nodes = np.ones(batch_size, dtype=np.int64)
        while nodes[0] < self._leaf_count:
            left = 2 * nodes
            right = values >= self._sum[left]
            values -= np.where(right, self._sum[left], 0.)
            nodes = left + right.astype(np.int64)
        indices = nodes - self._leaf_count
        uniform = self._rng.uniform(size=batch_size) < self.uniform_mix
        uniform_indices = self._rng.randint(0, self.size, batch_size)
        indices[uniform] = uniform_indices[uniform]
        probabilities = (self.uniform_mix / self.size
                         + (1 - self.uniform_mix) * self._sum[self._leaf_count + indices] / total)
        minimum = self.uniform_mix / self.size + (1 - self.uniform_mix) * self._min[1] / total
        beta = min(1., self.beta_start + (1 - self.beta_start) * self.sample_calls / self.beta_steps)
        weights = (probabilities / minimum) ** (-beta)
        self.sample_calls += 1
        self.sampled_transitions += batch_size
        self.sampled_deaths += int(np.sum((self.rewards[indices, 0] < 0) & (self.dones[indices, 0] == 1)))
        batch = tuple(getattr(self, name)[indices] for name in
                      ('obs', 'actions', 'rewards', 'next_obs', 'dones'))
        return batch, indices, weights.astype(np.float32)[:, None]

    def update_priorities(self, indices, td_errors):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if (indices.shape != errors.shape or np.any(indices < 0) or np.any(indices >= self.size)
                or not np.isfinite(errors).all()):
            raise ValueError("优先级索引和误差必须有效且有限")
        if not len(indices):
            return
        order = np.argsort(indices)
        unique, starts = np.unique(indices[order], return_index=True)
        values = np.maximum.reduceat(np.abs(errors[order]), starts) + self.epsilon
        self._priorities[unique] = values
        self.max_priority = max(self.max_priority, float(values.max()))
        self._set_leaves(unique)

    def state_dict(self):
        state = super().state_dict()
        state['prioritization'] = dict(version=1, alpha=self.alpha, beta=self.beta_start,
            beta_steps=self.beta_steps, uniform_mix=self.uniform_mix, epsilon=self.epsilon,
            max_priority=self.max_priority, sample_calls=self.sample_calls,
            sampled_transitions=self.sampled_transitions, sampled_deaths=self.sampled_deaths,
            priorities=torch.from_numpy(self._priorities[:self.size]))
        return state

    def load_state_dict(self, state):
        if 'prioritization' not in state:
            raise ValueError("均匀经验池无法恢复优先级历史，请使用 --reset_replay")
        meta = state['prioritization']
        for key, value in dict(version=1, alpha=self.alpha, beta=self.beta_start,
                beta_steps=self.beta_steps, uniform_mix=self.uniform_mix, epsilon=self.epsilon).items():
            if meta[key] != value:
                raise ValueError("优先回放配置不匹配：" + key)
        super().load_state_dict(state)
        priorities = meta['priorities'].cpu().numpy()
        if priorities.shape != (self.size,) or not np.isfinite(priorities).all() or np.any(priorities <= 0):
            raise ValueError("已保存的优先级无效")
        self._priorities[:] = 0
        self._priorities[:self.size] = priorities
        self._sum[:] = 0; self._min[:] = np.inf
        self._set_leaves(np.arange(self.size))
        for key in ('max_priority', 'sample_calls', 'sampled_transitions', 'sampled_deaths'):
            setattr(self, key, meta[key])
