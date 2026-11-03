"""可选的 TD3 捕食者，保留现有可迁移的 Actor 布局。

目标噪声使用独立且随检查点保存的随机状态，并作为计算图输入。
分别捕获仅更新 Critic、预热目标网络和更新 Actor 的计算图；
捕获预热不会推进实际更新计数，也不会消耗目标噪声。
"""
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from agent import DDPGAgent
from networks import Critic


class TD3Agent(DDPGAgent):
    algorithm = "td3"

    def __init__(self, *args, policy_delay=2, target_noise=.2, target_noise_clip=.5,
                 actor_warmup_updates=2000, target_noise_seed=0, **kwargs):
        if not isinstance(policy_delay, int) or policy_delay < 1:
            raise ValueError("policy_delay 必须为正整数")
        if not isinstance(actor_warmup_updates, int) or actor_warmup_updates < 0:
            raise ValueError("actor_warmup_updates 必须为非负整数")
        if any(not math.isfinite(x) or x < 0 for x in (target_noise, target_noise_clip)):
            raise ValueError("目标噪声设置必须是有限的非负数")
        super().__init__(*args, **kwargs)
        self.policy_delay = policy_delay
        self.target_noise = float(target_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.actor_warmup_updates = actor_warmup_updates
        self.total_updates = 0
        self._mode = (False, False)
        self._graphs = {}
        self._noise_rng = np.random.RandomState(target_noise_seed)
        first = self.critic.fc1
        obs_dim = self.actor.net[0].in_features
        act_dim = self.actor.net[-2].out_features
        self.critic2 = Critic(obs_dim, act_dim, first.out_features,
                              2 + len(self.critic.extra_layers), self.hidden_activation).to(self.device)
        self.critic2_target = Critic(obs_dim, act_dim, first.out_features,
                                     2 + len(self.critic.extra_layers), self.hidden_activation).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.critic_opt.param_groups[0]['lr'],
                                            **self._optimizer_kwargs())

    def _graph_networks(self):
        return super()._graph_networks() + (self.critic2, self.critic2_target)

    def _graph_optimizers(self):
        return super()._graph_optimizers() + (self.critic2_opt,)

    def _invalidate_graph(self):
        if getattr(self, '_graphs', {}):
            torch.cuda.synchronize(self.device)
        self._graphs = {}
        super()._invalidate_graph()

    def algorithm_config(self):
        return dict(policy_delay=self.policy_delay, target_noise=self.target_noise,
                    target_noise_clip=self.target_noise_clip, actor_warmup_updates=self.actor_warmup_updates)

    def state_dict(self):
        state = super().state_dict()
        state.update({name: getattr(self, name).state_dict() for name in ('critic2', 'critic2_target', 'critic2_opt')})
        rng = self._noise_rng.get_state()
        state.update(algorithm='td3', algorithm_config=self.algorithm_config(), total_updates=self.total_updates,
                     target_noise_rng=(rng[0], rng[1].tolist(), *rng[2:]))
        return state

    def load_state_dict(self, state):
        if state.get('algorithm') != 'td3' or state.get('algorithm_config') != self.algorithm_config():
            raise ValueError("TD3 算法或设置不匹配，请使用 policy_only 显式迁移 Actor")
        super().load_state_dict(state)
        self.critic2.load_state_dict(state['critic2'])
        self.critic2_target.load_state_dict(state['critic2_target'])
        optimizer = dict(state['critic2_opt'])
        optimizer['param_groups'] = [dict(group, fused=True if self.fused_adam else None,
                                          foreach=None, capturable=self.cuda_graph) for group in optimizer['param_groups']]
        self.critic2_opt.load_state_dict(optimizer)
        self.total_updates = state['total_updates']
        rng = state['target_noise_rng']
        self._noise_rng.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))

    def update(self, replay_buffer, batch_size, return_losses=True):
        batch = replay_buffer.sample(batch_size)
        noise = np.clip(self._noise_rng.normal(0., self.target_noise, batch[1].shape),
                        -self.target_noise_clip, self.target_noise_clip).astype(np.float32)
        batch = (*batch, noise)
        step = self.total_updates + 1
        self._mode = (step > self.actor_warmup_updates and step % self.policy_delay == 0,
                      step % self.policy_delay == 0)
        if self.cuda_graph and not self._graph_failed:
            from cuda_graph import CudaGraphPreparationError, CudaGraphUpdate, PackedCudaGraphUpdate
            signature = tuple((value.shape, value.dtype.str) for value in batch)
            runner = self._graphs.get(self._mode)
            if runner is not None and runner.signature != signature:
                self._invalidate_graph()
                runner = None
            if runner is None:
                try:
                    cls = PackedCudaGraphUpdate if self.packed_graph_inputs else CudaGraphUpdate
                    runner = cls(self, batch, signature)
                except CudaGraphPreparationError as error:
                    self._graph_failed = True
                    for optimizer in self._graph_optimizers():
                        for group in optimizer.param_groups:
                            group['capturable'] = False
                    warnings.warn("TD3 计算图准备失败，已恢复训练状态：" + str(error), RuntimeWarning)
                else:
                    self._graphs[self._mode] = runner
                    self.graph_capture_seconds += runner.capture_seconds
            if runner is not None:
                self._graph_runner = runner
                losses = runner.replay(batch)
                self.total_updates = step
                return tuple(value.item() for value in losses) if return_losses else None
        tensors = tuple(torch.as_tensor(value, dtype=torch.float32, device=self.device) for value in batch)
        losses = self._update_tensors(tensors)
        self.total_updates = step
        return tuple(value.item() for value in losses) if return_losses else None

    def _update_tensors(self, batch):
        obs, action, reward, next_obs, done, noise = batch
        with torch.no_grad():
            target_action = (self.actor_target(next_obs) + noise).clamp(-1., 1.)
            target_q = torch.minimum(self.critic_target(next_obs, target_action),
                                      self.critic2_target(next_obs, target_action))
            y = reward + self.gamma * (1. - done) * target_q
        critic_loss = F.mse_loss(self.critic(obs, action), y) + F.mse_loss(self.critic2(obs, action), y)
        self.critic_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self.critic2_opt.step()
        actor_loss = torch.zeros((), device=self.device)
        actor_due, target_due = self._mode
        if actor_due:
            parameters = tuple(self.critic.parameters())
            flags = tuple(parameter.requires_grad for parameter in parameters)
            try:
                for parameter in parameters:
                    parameter.requires_grad_(False)
                logits = self.actor.forward_logits(obs)
                actor_loss = -self.critic(obs, torch.tanh(logits)).mean()
                if self.actor_logit_penalty:
                    actor_loss = actor_loss + self.actor_logit_penalty * F.relu(logits.abs() - self.actor_logit_threshold).square().mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
            finally:
                for parameter, flag in zip(parameters, flags):
                    parameter.requires_grad_(flag)
        if target_due:
            if actor_due:
                self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
            self._soft_update(self.critic2_target, self.critic2)
        return actor_loss, critic_loss
