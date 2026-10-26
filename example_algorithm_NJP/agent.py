import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class DDPGAgent:
    """同一物种共享一套 DDPG 网络，支持批量推理和完整训练状态恢复。"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=128,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.98,
        tau=0.01,
        device="cpu",
        fused_adam=False,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        # fused Adam 仅在 CUDA 上启用；CPU 保留默认后端以兼容无 GPU 的运行环境。
        self.fused_adam = bool(fused_adam and self.device.type == "cuda")

        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        backend = True if self.fused_adam else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, fused=backend)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, fused=backend)

    @torch.no_grad()
    def act(self, obs, noise_std=0.0):
        return self.act_batch(np.asarray(obs)[None, :], noise_std=noise_std)[0]

    @torch.no_grad()
    def act_batch(self, observations, noise_std=0.0):
        """将 (个体数, 观测维度) 批量映射为同顺序的 (个体数, 动作维度)。"""
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        # 每个物种只做一次前向计算和结果回传，避免逐个体触发 GPU/CPU 同步。
        action = self.actor(obs_t).cpu().numpy()
        if noise_std > 0:
            action += np.random.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def load_policy(self, state_dict):
        """加载策略并同步目标 Actor，不改动 Critic 和优化器。"""
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def state_dict(self):
        """同时保存在线网络、目标网络和优化器，保留继续更新所需的训练状态。"""
        state = {
            name: getattr(self, name).state_dict()
            for name in ("actor", "actor_target", "critic", "critic_target", "actor_opt", "critic_opt")
        }
        state.update(gamma=self.gamma, tau=self.tau)
        return state

    def load_state_dict(self, state_dict):
        for name in ("actor", "actor_target", "critic", "critic_target"):
            getattr(self, name).load_state_dict(state_dict[name])
        for name in ("actor_opt", "critic_opt"):
            optimizer_state = dict(state_dict[name])
            # 先按当前配置选择后端，再加载优化器；这样切换后端时，Adam 步数张量
            # 也会放在正确的设备上。复制参数组，避免改写传入的检查点配置。
            optimizer_state["param_groups"] = [
                dict(group, fused=True if self.fused_adam else None, foreach=None)
                for group in optimizer_state["param_groups"]
            ]
            getattr(self, name).load_state_dict(optimizer_state)
        self.gamma = state_dict["gamma"]
        self.tau = state_dict["tau"]

    def update(self, replay_buffer, batch_size, return_losses=True):
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

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
        # .item() 会等待 CUDA 结果回传；不记录损失时可跳过，梯度更新不受影响。
        if return_losses:
            return actor_loss.item(), critic_loss.item()

    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * source_param.data)
