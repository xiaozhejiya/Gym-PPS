import inspect
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class DDPGAgent:
    """同一物种共享一套 DDPG 网络，支持批量推理。"""

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
        cuda_graph=False,
        hidden_layers=2,
        packed_graph_inputs=False,
        hidden_activation="relu",
        actor_logit_penalty=0.0,
        actor_logit_threshold=3.0,
        critic_layer_norm=False,
        critic_grad_clip=0.0,
        critic_geometry=False,
        critic_terminal_weight=0.0,
        critic_terminal_batch_size=32,
        critic_risk_weight=0.0,
        critic_risk_batch_size=256,
    ):
        if not math.isfinite(actor_logit_penalty) or actor_logit_penalty < 0:
            raise ValueError("actor_logit_penalty 必须是有限的非负数")
        if not math.isfinite(actor_logit_threshold) or actor_logit_threshold <= 0:
            raise ValueError("actor_logit_threshold 必须是有限的正数")
        if not math.isfinite(critic_grad_clip) or critic_grad_clip < 0:
            raise ValueError("critic_grad_clip 必须是有限的非负数")
        if not math.isfinite(critic_terminal_weight) or critic_terminal_weight < 0:
            raise ValueError("critic_terminal_weight 必须是有限的非负数")
        if not math.isfinite(critic_risk_weight) or critic_risk_weight < 0:
            raise ValueError("critic_risk_weight 必须是有限的非负数")
        if (not isinstance(critic_risk_batch_size, int) or critic_risk_batch_size < 1
                or (critic_risk_weight and (obs_dim != 178 or act_dim != 4))):
            raise ValueError("近威胁回放要求批量大小为正数，且猎物使用 178 维观测和 4 维动作")
        if (not isinstance(critic_terminal_batch_size, int) or critic_terminal_batch_size < 1
                or (critic_terminal_weight and (obs_dim != 178 or act_dim != 4))):
            raise ValueError("终止值校准要求批量大小为正数，且猎物使用 178 维观测和 4 维动作")
        self.hidden_activation = hidden_activation
        self.actor_logit_penalty = float(actor_logit_penalty)
        self.actor_logit_threshold = float(actor_logit_threshold)
        self.critic_layer_norm = bool(critic_layer_norm)
        self.critic_grad_clip = float(critic_grad_clip)
        self.critic_geometry = bool(critic_geometry)
        self.critic_terminal_weight = float(critic_terminal_weight)
        self.critic_terminal_batch_size = critic_terminal_batch_size
        self.terminal_calibration = None
        self.critic_risk_weight = float(critic_risk_weight)
        self.critic_risk_batch_size = critic_risk_batch_size
        self.risk_replay = None
        self.actor_updates_enabled = True
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        # fused Adam 仅在 CUDA 上启用；CPU 保留默认后端以兼容无 GPU 的运行环境。
        self.fused_adam = bool(fused_adam and self.device.type == "cuda")
        # 图执行需要 CUDA 和 fused Adam；不满足条件时提前报错，默认仍走普通更新。
        if cuda_graph and (not self.fused_adam or not torch.cuda.is_available()):
            raise ValueError("cuda_graph 需要 CUDA 和 fused_adam")
        self.cuda_graph = bool(cuda_graph)
        self.packed_graph_inputs = bool(packed_graph_inputs)
        self._graph_runner = None
        self._graph_failed = False
        self.graph_capture_seconds = 0.0

        self.actor = Actor(obs_dim, act_dim, hidden_dim, hidden_layers, hidden_activation).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim, hidden_layers, hidden_activation).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden_dim, hidden_layers, hidden_activation, critic_layer_norm, critic_geometry).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dim, hidden_layers, hidden_activation, critic_layer_norm, critic_geometry).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        optimizer_kwargs = self._optimizer_kwargs()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, **optimizer_kwargs)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, **optimizer_kwargs)

    def _optimizer_kwargs(self):
        # 默认路径不传 fused 参数，兼容尚未提供该选项的 PyTorch 版本。
        optimizer_kwargs = {}
        if self.fused_adam:
            if "fused" not in inspect.signature(torch.optim.Adam).parameters:
                raise ValueError("当前 PyTorch 版本不支持 --fused_adam，请使用 --no_fused_adam。")
            optimizer_kwargs["fused"] = True
        if self.cuda_graph:
            optimizer_kwargs["capturable"] = True
        return optimizer_kwargs

    @torch.no_grad()
    def act(self, obs, noise_std=0.0, epsilon=0.0):
        return self.act_batch(np.asarray(obs)[None, :], noise_std=noise_std, epsilon=epsilon)[0]

    @torch.no_grad()
    def act_batch(self, observations, noise_std=0.0, epsilon=0.0):
        """将 (个体数, 观测维度) 批量映射为同顺序的 (个体数, 动作维度)。"""
        if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon 必须是 0 到 1 之间的有限数值")
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        # 每个物种只做一次前向计算和结果回传，避免逐个体触发 GPU/CPU 同步。
        action = self.actor(obs_t).cpu().numpy()
        if noise_std > 0:
            action += np.random.normal(0.0, noise_std, size=action.shape)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if epsilon > 0:
            # 为每个个体独立决定是否探索，并同时替换其全部动作分量。
            # 这里采用探索对照实验中的随机动作定义；
            # 原论文未明确给出随机动作的分布。
            explore = np.random.uniform(size=action.shape[0]) < epsilon
            random_action = np.random.uniform(-1.0, 1.0, size=action.shape).astype(np.float32)
            action[explore] = random_action[explore]
        return action

    def load_policy(self, state_dict):
        """加载策略并同步目标 Actor，不改动 Critic 和优化器。"""
        self._invalidate_graph()
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def state_dict(self):
        """同时保存在线网络、目标网络和优化器，保留继续更新所需的训练状态。"""
        state = {
            name: getattr(self, name).state_dict()
            for name in ("actor", "actor_target", "critic", "critic_target", "actor_opt", "critic_opt")
        }
        state.update(gamma=self.gamma, tau=self.tau)
        if self.terminal_calibration is not None:
            state['terminal_calibration'] = self.terminal_calibration.state_dict()
        if self.risk_replay is not None:
            state['risk_replay'] = self.risk_replay.state_dict()
        config = self.network_config()
        # 旧版 ReLU 实验继续使用原有状态字典格式。
        if config != self.legacy_network_config():
            state["network_config"] = config
        return state

    @staticmethod
    def legacy_network_config():
        return dict(hidden_activation="relu", actor_logit_penalty=0.0, actor_logit_threshold=3.0)

    def network_config(self):
        config = dict(hidden_activation=self.hidden_activation,
                    actor_logit_penalty=self.actor_logit_penalty,
                    actor_logit_threshold=self.actor_logit_threshold)
        if self.critic_layer_norm:
            config['critic_layer_norm'] = True
        if self.critic_grad_clip:
            config['critic_grad_clip'] = self.critic_grad_clip
        if self.critic_geometry:
            config['critic_geometry'] = True
        if self.critic_terminal_weight:
            config['critic_terminal_weight'] = self.critic_terminal_weight
            config['critic_terminal_batch_size'] = self.critic_terminal_batch_size
        if self.critic_risk_weight:
            config['critic_risk_weight'] = self.critic_risk_weight
            config['critic_risk_batch_size'] = self.critic_risk_batch_size
        return config

    def set_actor_updates(self, enabled):
        """仅在两次更新之间切换更新模式，并重新捕获相应的计算图。"""
        enabled = bool(enabled)
        if enabled != self.actor_updates_enabled:
            self._invalidate_graph()
            self.actor_updates_enabled = enabled

    def reset_critic(self, seed, learning_rate=1e-4, layer_norm=True, grad_clip=1.0, geometry=False):
        """显式恢复 Critic，同时保留双方 Actor、Actor 优化器及所有全局随机状态。"""
        if getattr(self, 'algorithm', 'ddpg') != 'ddpg':
            raise ValueError("reset_critic 目前仅支持 DDPG")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate 必须是有限的正数")
        if not math.isfinite(grad_clip) or grad_clip < 0:
            raise ValueError("grad_clip 必须是有限的非负数")
        from copy import deepcopy
        self._invalidate_graph()
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            # 在 CPU 上初始化，不重设其他 CUDA 设备的随机种子。
            torch.random.default_generator.manual_seed(seed)
            critic = Critic(self.actor.net[0].in_features, self.actor.net[-2].out_features,
                            self.critic.fc1.out_features, 2 + len(self.critic.extra_layers),
                            self.hidden_activation, layer_norm, geometry).to(self.device)
        self.critic = critic
        self.critic_target = deepcopy(critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate,
            **self._optimizer_kwargs())
        self.critic_layer_norm = bool(layer_norm)
        self.critic_grad_clip = float(grad_clip)
        self.critic_geometry = bool(geometry)
        self.terminal_calibration = None
        self.critic_terminal_weight = 0.
        self.risk_replay = None
        self.critic_risk_weight = 0.

    def install_terminal_calibration(self, observations, arena_L, weight=1., batch_size=32, seed=700001):
        if not math.isfinite(weight) or weight <= 0 or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("终止值校准要求权重和 batch_size 为正数")
        from critic_recovery import TerminalCalibration
        calibration = TerminalCalibration(observations, arena_L, seed)
        self._invalidate_graph()
        self.terminal_calibration = calibration
        self.critic_terminal_weight = float(weight)
        self.critic_terminal_batch_size = batch_size

    def install_risk_replay(self, replay_state, weight=1., batch_size=256, radius=.5):
        if not math.isfinite(weight) or weight <= 0 or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("近威胁回放要求权重和 batch_size 为正数")
        from critic_recovery import NearThreatReplay
        replay = NearThreatReplay(radius)
        arrays = replay_state['arrays']; size = replay_state['size']
        replay.push_batch(*(arrays[n][:size].cpu().numpy() for n in
                            ('obs', 'actions', 'rewards', 'next_obs', 'dones')))
        if not len(replay.buffer): raise ValueError("源经验池中没有近威胁转移样本")
        self._invalidate_graph()
        self.risk_replay = replay
        self.critic_risk_weight = float(weight)
        self.critic_risk_batch_size = batch_size

    def observe_recovery(self, obs, actions, rewards, next_obs, dones):
        if self.risk_replay is not None:
            self.risk_replay.push_batch(obs, actions, rewards, next_obs, dones)

    def load_state_dict(self, state_dict):
        if state_dict.get("algorithm", "ddpg") != getattr(self, "algorithm", "ddpg"):
            raise ValueError("检查点算法不匹配，请选择对应的 predator_algorithm")
        saved_config = state_dict.get("network_config", self.legacy_network_config())
        if saved_config != self.network_config():
            raise ValueError("检查点网络或 Critic 配置不匹配：已保存={!r}，当前请求={!r}".format(
                saved_config, self.network_config()))
        self._invalidate_graph()
        for name in ("actor", "actor_target", "critic", "critic_target"):
            getattr(self, name).load_state_dict(state_dict[name])
        for name in ("actor_opt", "critic_opt"):
            optimizer_state = dict(state_dict[name])
            # 先按当前配置选择后端，再加载优化器；这样切换后端时，Adam 步数张量
            # 也会放在正确的设备上。复制参数组，避免改写传入的检查点配置。
            optimizer_state["param_groups"] = [
                dict(group, fused=True if self.fused_adam else None, foreach=None,
                     capturable=self.cuda_graph)
                for group in optimizer_state["param_groups"]
            ]
            getattr(self, name).load_state_dict(optimizer_state)
        self.gamma = state_dict["gamma"]
        self.tau = state_dict["tau"]
        if 'terminal_calibration' in state_dict:
            from critic_recovery import TerminalCalibration
            self.terminal_calibration = TerminalCalibration.from_state_dict(state_dict['terminal_calibration'])
        else:
            self.terminal_calibration = None
        if self.critic_terminal_weight and self.terminal_calibration is None:
            raise ValueError("检查点缺少终止值校准观测或随机状态")
        if 'risk_replay' in state_dict:
            from critic_recovery import NearThreatReplay
            self.risk_replay = NearThreatReplay.from_state_dict(state_dict['risk_replay'])
        else:
            self.risk_replay = None
        if self.critic_risk_weight and self.risk_replay is None:
            raise ValueError("检查点缺少近威胁回放或随机状态")

    def update(self, replay_buffer, batch_size, return_losses=True):
        prioritized = getattr(replay_buffer, 'prioritized', False)
        if prioritized:
            batch, indices, weights = replay_buffer.sample_prioritized(batch_size)
            batch = (*batch, weights)
        else:
            batch = replay_buffer.sample(batch_size)
        if self.critic_terminal_weight:
            if self.terminal_calibration is None:
                raise ValueError("请先安装终止值校准数据，再执行更新")
            batch = (*batch, *self.terminal_calibration.sample(self.critic_terminal_batch_size))
        if self.critic_risk_weight:
            if self.risk_replay is None:
                raise ValueError("请先安装近威胁回放数据，再执行更新")
            batch = (*batch, *self.risk_replay.buffer.sample(self.critic_risk_batch_size))
        if self.cuda_graph and not self._graph_failed:
            from cuda_graph import CudaGraphPreparationError, CudaGraphUpdate, PackedCudaGraphUpdate
            signature = tuple((array.shape, array.dtype.str) for array in batch)
            if self._graph_runner is None or self._graph_runner.signature != signature:
                self._invalidate_graph()
                try:
                    runner_class = PackedCudaGraphUpdate if self.packed_graph_inputs else CudaGraphUpdate
                    runner = runner_class(self, batch, signature)
                except CudaGraphPreparationError as error:
                    # 只有成功回滚的准备失败才能回退到普通更新。
                    # 回滚失败或回放失败必须向外抛出，避免重复更新参数。
                    self._graph_failed = True
                    for optimizer in (self.actor_opt, self.critic_opt):
                        for group in optimizer.param_groups:
                            group['capturable'] = False
                    warnings.warn("CUDA Graph 准备失败，已恢复训练状态，"
                                  "改用普通更新：{}".format(error), RuntimeWarning)
                else:
                    self._graph_runner = runner
                    self.graph_capture_seconds += runner.capture_seconds
            if self._graph_runner is not None:
                losses = self._graph_runner.replay(batch)
                if prioritized:
                    replay_buffer.update_priorities(indices, self._last_td_errors.cpu().numpy())
                if return_losses:
                    return tuple(loss.item() for loss in losses)
                return None
        tensors = tuple(torch.as_tensor(array, dtype=torch.float32, device=self.device) for array in batch)
        losses = self._update_tensors(tensors)
        if prioritized:
            replay_buffer.update_priorities(indices, self._last_td_errors.cpu().numpy())
        # .item() 会等待 CUDA 结果回传；不记录损失时可跳过，梯度更新不受影响。
        if return_losses:
            return tuple(loss.item() for loss in losses)

    def _invalidate_graph(self):
        """等待已有回放完成后丢弃图；加载权重或训练状态后应重新捕获。"""
        if self._graph_runner is not None:
            torch.cuda.synchronize(self.device)
        self._graph_runner = None
        self._graph_failed = False

    def _graph_networks(self):
        return self.actor, self.actor_target, self.critic, self.critic_target

    def _graph_optimizers(self):
        return self.actor_opt, self.critic_opt

    @property
    def cuda_graph_status(self):
        """区分关闭、待捕获、运行和回退状态，供日志及检查点元数据使用。"""
        if not self.cuda_graph:
            return 'disabled'
        if self._graph_failed:
            return 'fallback'
        return 'active' if self._graph_runner is not None else 'pending'

    def _update_tensors(self, batch):
        """共用同一套更新算式，供普通执行和 CUDA Graph 捕获调用。"""
        base_count = len(batch) - (3 if self.critic_terminal_weight else 0) - (5 if self.critic_risk_weight else 0)
        weighted = base_count == 6
        obs, action, reward, next_obs, done = batch[:5]
        weights = batch[5] if weighted else None

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_action)
            y = reward + self.gamma * (1.0 - done) * target_q

        if weighted:
            error = self.critic(obs, action) - y
            self._last_td_errors = error.detach().abs()
            critic_loss = (weights * error.square()).mean()
        else:
            critic_loss = F.mse_loss(self.critic(obs, action), y)
        if self.critic_terminal_weight:
            terminal_obs, terminal_action, terminal_target = batch[base_count:base_count + 3]
            critic_loss = critic_loss + self.critic_terminal_weight * F.mse_loss(
                self.critic(terminal_obs, terminal_action), terminal_target)
        if self.critic_risk_weight:
            risk_obs, risk_action, risk_reward, risk_next, risk_done = batch[-5:]
            with torch.no_grad():
                risk_y = risk_reward + self.gamma * (1 - risk_done) * self.critic_target(
                    risk_next, self.actor_target(risk_next))
            critic_loss = critic_loss + self.critic_risk_weight * F.mse_loss(self.critic(risk_obs, risk_action), risk_y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.critic_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip, foreach=False)
        self.critic_opt.step()

        if not self.actor_updates_enabled:
            self._soft_update(self.critic_target, self.critic)
            return critic_loss.detach().new_zeros(()), critic_loss

        # Actor 更新仍需经过 Critic 对动作求导，只跳过不使用的 Critic 参数梯度。
        # 临时冻结的标志在异常路径上也必须恢复，保证后续 Critic 仍可训练。
        critic_parameters = tuple(self.critic.parameters())
        requires_grad = tuple(parameter.requires_grad for parameter in critic_parameters)
        try:
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            if weighted:
                logits = self.actor.forward_logits(obs)
                policy_losses = -self.critic(obs, torch.tanh(logits))
                if self.actor_logit_penalty:
                    excess = F.relu(logits.abs() - self.actor_logit_threshold)
                    policy_losses = policy_losses + self.actor_logit_penalty * excess.square().mean(dim=1, keepdim=True)
                actor_loss = (weights * policy_losses).mean()
            elif self.actor_logit_penalty:
                logits = self.actor.forward_logits(obs)
                actions = torch.tanh(logits)
                excess = F.relu(logits.abs() - self.actor_logit_threshold)
                actor_loss = (-self.critic(obs, actions).mean()
                              + self.actor_logit_penalty * excess.square().mean())
            else:
                actor_loss = -self.critic(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        finally:
            for parameter, required in zip(critic_parameters, requires_grad):
                parameter.requires_grad_(required)

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        return actor_loss, critic_loss

    def _soft_update(self, target, source):
        if self.device.type == 'cuda':
            # 批量执行原有的乘、乘、加三步，保留运算顺序和逐步舍入结果。
            targets = [parameter.data for parameter in target.parameters()]
            sources = [parameter.data for parameter in source.parameters()]
            torch._foreach_mul_(targets, 1.0 - self.tau)
            scaled_sources = torch._foreach_mul(sources, self.tau)
            torch._foreach_add_(targets, scaled_sources)
            return
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * source_param.data)
