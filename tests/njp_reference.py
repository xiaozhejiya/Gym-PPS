"""保留优化前的 DDPG 算式，作为独立基准验证参数更新结果。"""
import torch
import torch.nn.functional as F


def soft_update(self, target, source):
    """保留原逐参数更新顺序，检查批量算式没有改变舍入结果。"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - self.tau)
        target_param.data.add_(self.tau * source_param.data)


def update_tensors(self, batch):
    """独立保留原训练过程，避免用待测优化实现验证自身。"""
    obs, action, reward, next_obs, done = batch
    with torch.no_grad():
        next_action = self.actor_target(next_obs)
        target_q = self.critic_target(next_obs, next_action)
        y = reward + self.gamma * (1.0 - done) * target_q
    critic_loss = F.mse_loss(self.critic(obs, action), y)
    self.critic_opt.zero_grad()
    critic_loss.backward()
    self.critic_opt.step()
    actor_loss = -self.critic(obs, self.actor(obs)).mean()
    self.actor_opt.zero_grad()
    actor_loss.backward()
    self.actor_opt.step()
    self._soft_update(self.actor_target, self.actor)
    self._soft_update(self.critic_target, self.critic)
    return actor_loss, critic_loss


def assert_exact(actual, expected, path=()):
    """逐项精确比较训练状态，仅忽略有意切换的优化器后端标志。"""
    if torch.is_tensor(expected):
        assert actual.dtype == expected.dtype, path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            if key not in ('fused', 'foreach', 'capturable'):
                assert_exact(actual[key], expected[key], path + (key,))
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_exact(left, right, path + (index,))
    else:
        assert actual == expected
