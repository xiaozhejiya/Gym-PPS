import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, obs, actions, rewards, next_obs, dones=False):
        """按输入顺序批量写入，保持逐条 push 的环形覆盖结果。

        奖励和终止标记支持一维或单列数组，终止标记也可为标量。
        超容量批次保留末尾 capacity 条数据，所有维度在写入前校验。
        """
        observations = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        following = np.asarray(next_obs, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1:] != self.obs.shape[1:]:
            raise ValueError('obs must have shape (batch, obs_dim)')
        count = observations.shape[0]
        if actions.shape != (count, self.actions.shape[1]):
            raise ValueError('actions must have shape (batch, act_dim)')
        if following.shape != observations.shape:
            raise ValueError('next_obs must have the same shape as obs')
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.shape not in ((count,), (count, 1)):
            raise ValueError('rewards must have shape (batch,) or (batch, 1)')
        rewards = rewards.reshape(count, 1)
        dones = np.asarray(dones, dtype=np.float32)
        if dones.ndim == 0:
            dones = np.broadcast_to(dones, (count, 1))
        elif dones.shape in ((count,), (count, 1)):
            dones = dones.reshape(count, 1)
        else:
            raise ValueError('dones must be scalar, (batch,) or (batch, 1)')
        if not count:
            return
        sources = (observations, actions, rewards, following, dones)
        destinations = (self.obs, self.actions, self.rewards, self.next_obs, self.dones)
        if any(np.may_share_memory(source, destination)
               for source in sources for destination in destinations):
            # 输入与存储共享内存时逐条写入，保留原有的读写顺序。
            for index in range(count):
                self.push(observations[index], actions[index], rewards[index, 0],
                          following[index], dones[index, 0])
            return
        written = min(count, self.capacity)
        start = (self.ptr + count - written) % self.capacity
        first = min(written, self.capacity - start)
        for destination, source in zip(destinations, sources):
            source = source[count - written:]
            destination[start:start + first] = source[:first]
            if first < written:
                destination[:written - first] = source[first:]
        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def state_dict(self):
        """保存有效样本及环形写入位置，空余容量无需写入检查点。

        张量与原 NumPy 数组共享内存，调用方应在下一次写入前完成序列化。
        使用张量还可使检查点兼容 torch.load(weights_only=True)。
        """
        return {
            "capacity": self.capacity,
            "ptr": self.ptr,
            "size": self.size,
            "arrays": {
                name: torch.from_numpy(getattr(self, name)[:self.size])
                for name in ("obs", "actions", "rewards", "next_obs", "dones")
            },
        }

    def load_state_dict(self, state_dict):
        # 容量决定环形覆盖顺序；强制保持一致，才能还原后续写入和采样的行为。
        if self.capacity != state_dict["capacity"]:
            raise ValueError("buffer_size must match the saved replay buffer capacity")
        size = state_dict["size"]
        for name in ("obs", "actions", "rewards", "next_obs", "dones"):
            destination = getattr(self, name)
            source = state_dict["arrays"][name].cpu().numpy()
            if source.shape != destination[:size].shape:
                raise ValueError("Saved replay buffer dimensions do not match the environment")
            destination[:size] = source
        self.ptr = state_dict["ptr"]
        self.size = size
