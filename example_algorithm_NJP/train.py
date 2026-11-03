import os
import sys
import time
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gym

from agent import DDPGAgent
from arguments import get_args
from custom_env import MyAction, MyObs, MyReward, predator_prey_distances
from replay_buffer import ReplayBuffer
from checkpoint_compat import checkpoint_lock, structure, validate_checkpoint

THREE_D_EXPERIMENTS = ("njp_3d", "njp_3d_no_energy", "njp_3d_silu_no_energy", "njp_3d_silu_rule_prey", "njp_3d_silu_wide_obs", "njp_3d_silu_capture")


def make_env(experiment=None, evaluation=False, arena_L=None, n_p=None, n_e=None, **spawn):
    if experiment in THREE_D_EXPERIMENTS:
        from njp_3d import make_env as make_3d_env
        return make_3d_env(evaluation=evaluation, arena_L=arena_L, n_p=n_p, n_e=n_e,
                           no_energy=experiment == "njp_3d_no_energy",
                           silu=experiment == "njp_3d_silu_no_energy",
                           fixed_prey=experiment == "njp_3d_silu_rule_prey",
                           wide_obs=experiment == "njp_3d_silu_wide_obs",
                           capture=experiment == "njp_3d_silu_capture", **spawn)
    if spawn:
        raise ValueError("spawn overrides require a 3D experiment")
    if n_p is not None or n_e is not None:
        raise ValueError("population overrides require --experiment njp_3d")
    if experiment == "paper_swirl":
        from paper_swirl import make_env as make_paper_env
        return make_paper_env(evaluation=evaluation, arena_L=arena_L)
    if arena_L is not None:
        raise ValueError("arena_L requires --experiment paper_swirl")
    if experiment is not None:
        raise ValueError("Unknown experiment: " + experiment)
    # 地图半宽和双方种群规模统一从配置读取，训练与评估共用同一环境定义。
    custom_param = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_param.json")
    env = gym.make("PredatorPreySwarm-v0")
    env = MyReward(env, custom_param)
    env = MyAction(env, custom_param)
    env = MyObs(env, custom_param)
    assert env.n_p > 0 and env.n_e > 0, "This shared DDPG script expects at least 1 predator and 1 prey."
    return env


def exploration_rates(args, local_episode, cumulative_episode):
    if args.exploration_mode == "paper":
        # Match the screened candidate and keep decay continuous on full resume.
        decrement = args.exploration_decay * cumulative_episode
        return (max(args.noise_min, args.noise - decrement),
                max(args.epsilon_min, args.epsilon - decrement))
    return max(args.noise_min, args.noise * (1.0 - local_episode / args.episodes)), 0.0


def train(args):
    started_at = time.perf_counter()
    profile_path = os.environ.get("NJP_PROFILE")
    phases = dict(inference=0., environment=0., metrics_and_replay=0., updates=0., checkpoints=0.)
    if args.policy_only and not args.init_model:
        raise ValueError("policy_only requires init_model")
    reset_replay = getattr(args, "reset_replay", False)
    freeze_predator = getattr(args, "freeze_predator", False)
    prioritized_prey = getattr(args, "prey_replay", "uniform") == "prioritized"
    allow_capture_change = getattr(args, "allow_capture_rule_change", False)
    if freeze_predator and not args.init_model:
        raise ValueError("freeze_predator requires init_model")
    if allow_capture_change and not (args.init_model and reset_replay and not args.policy_only):
        raise ValueError("allow_capture_rule_change requires a full init_model and reset_replay")
    if reset_replay and (not args.init_model or args.policy_only):
        raise ValueError("reset_replay requires a full init_model without policy_only")
    # 在创建环境前校验图执行依赖，避免错误配置启动训练后才失败。
    if args.cuda_graph and (not args.fused_adam or not torch.cuda.is_available()):
        raise ValueError("cuda_graph requires CUDA and --fused_adam")
    if args.episodes <= 0 or args.episode_length <= 0:
        raise ValueError("episodes and episode_length must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be nonnegative")
    if args.render_interval < 0:
        raise ValueError("render_interval must be nonnegative")
    if args.experiment and (not args.init_model or args.experiment in THREE_D_EXPERIMENTS):
        target = os.path.abspath(os.path.join(os.path.dirname(__file__), args.model_path))
        if os.path.exists(target):
            raise FileExistsError("Experiment output already exists; choose another --model_path: " + target)
    if args.exploration_mode == "paper":
        if (not all(np.isfinite(value) for value in (args.epsilon, args.epsilon_min))
                or not 0 <= args.epsilon_min <= args.epsilon <= 1):
            raise ValueError("paper exploration requires 0 <= epsilon_min <= epsilon <= 1")
        if (not all(np.isfinite(value) for value in (args.noise, args.noise_min))
                or not 0 <= args.noise_min <= args.noise):
            raise ValueError("paper exploration requires 0 <= noise_min <= noise")
        if not np.isfinite(args.exploration_decay) or args.exploration_decay < 0:
            raise ValueError("exploration_decay must be finite and nonnegative")
    # total_steps 可缩短最后一回合，使本次新增训练步数精确落在预算内。
    planned_steps = args.episodes * args.episode_length
    if args.total_steps is not None:
        if not 0 < args.total_steps <= planned_steps:
            raise ValueError("total_steps must be positive and no larger than episodes * episode_length")
        planned_steps = args.total_steps
    if args.experiment in THREE_D_EXPERIMENTS and planned_steps % args.episode_length:
        raise ValueError("3D training must finish at a complete episode boundary")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    arena_L = getattr(args, "arena_L", None)
    counts = {name: getattr(args, name, None) for name in ("n_p", "n_e")}
    counts = {name: value for name, value in counts.items() if value is not None}
    from njp_3d import spawn_options
    spawn = spawn_options(args)
    if arena_L is not None or counts or spawn:
        env = make_env(args.experiment, arena_L=arena_L, **counts, **spawn)
    else:
        env = make_env(args.experiment) if args.experiment else make_env()
    learn_predator = env.pursuer_strategy == "input" and not freeze_predator
    if freeze_predator and env.pursuer_strategy != "input":
        raise ValueError("freeze_predator requires an input-controlled predator")
    learn_prey = env.escaper_strategy == "input"
    prey_death = getattr(env.unwrapped, "prey_death_enabled", False)
    obs = env.reset()
    obs_dim = env.observation_space.spaces["predator"].shape[1]
    act_dim = env.action_space.spaces["predator"].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_activation = getattr(args, "hidden_activation", "relu")
    network_options = dict(hidden_activation=hidden_activation,
                           actor_logit_penalty=getattr(args, "actor_logit_penalty", 0.0),
                           actor_logit_threshold=getattr(args, "actor_logit_threshold", 3.0))
    expected_structure = structure(env, args.hidden_dim, args.hidden_layers, hidden_activation)

    predator_class = DDPGAgent
    predator_options = {}
    if getattr(args, "predator_algorithm", "ddpg") == "td3":
        from td3_agent import TD3Agent
        predator_class = TD3Agent
        predator_options = dict(policy_delay=args.policy_delay, target_noise=args.target_noise,
                                target_noise_clip=args.target_noise_clip,
                                actor_warmup_updates=args.actor_warmup_updates, target_noise_seed=args.seed)
    predator_agent = predator_class(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        fused_adam=args.fused_adam,
        cuda_graph=args.cuda_graph,
        hidden_layers=args.hidden_layers,
        packed_graph_inputs=getattr(args, "packed_graph_inputs", False),
        **dict(network_options, **predator_options),
    )
    prey_agent = DDPGAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=getattr(args, 'prey_critic_lr', None) or args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        fused_adam=args.fused_adam,
        cuda_graph=args.cuda_graph,
        hidden_layers=args.hidden_layers,
        packed_graph_inputs=getattr(args, "packed_graph_inputs", False),
        critic_layer_norm=getattr(args, 'prey_critic_layer_norm', False),
        critic_grad_clip=getattr(args, 'prey_critic_grad_clip', 0.0),
        critic_geometry=getattr(args, 'prey_critic_geometry', False),
        critic_terminal_weight=getattr(args, 'prey_critic_terminal_weight', 0.),
        critic_terminal_batch_size=getattr(args, 'prey_critic_terminal_batch_size', 32),
        critic_risk_weight=getattr(args, 'prey_critic_risk_weight', 0.),
        critic_risk_batch_size=getattr(args, 'prey_critic_risk_batch_size', 256),
        **network_options,
    )
    init_model = None
    previous_steps = 0
    previous_episodes = 0
    restored_training_state = False
    warmup_steps = args.start_steps
    random_warmup = args.init_model is None
    checkpoint = None
    critic_repair_info = None
    output_model = os.path.abspath(os.path.join(os.path.dirname(__file__), args.model_path))
    if args.init_model:
        init_model = args.init_model
        if not os.path.isabs(init_model):
            init_model = os.path.join(os.path.dirname(__file__), init_model)
        init_model = os.path.abspath(init_model)
        if init_model == output_model:
            raise ValueError("model_path must differ from init_model to preserve the source checkpoint")
        # 完整检查点还包含大容量回放缓冲，先加载到 CPU，避免占用额外显存。
        try:
            checkpoint = torch.load(init_model, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(init_model, map_location="cpu")
        critic_repair_info = checkpoint.get('training', {}).get('critic_repair')
        if (critic_repair_info and critic_repair_info.get('status') != 'accepted'
                and not getattr(args, 'freeze_prey_actor', False) and not args.policy_only):
            raise ValueError('Critic repair has not passed policy-update validation; keep --freeze_prey_actor')
        if checkpoint.get("kind") == "policy_transfer_only" and not args.policy_only:
            raise ValueError("Actor-only transfer checkpoint requires --policy_only; it has no joint optimizer/replay state")
        validate_checkpoint(checkpoint, expected_structure, policy_only=args.policy_only,
                            allow_capture_change=allow_capture_change)
        if reset_replay and "training_state" not in checkpoint:
            raise ValueError("reset_replay requires a checkpoint with full training_state")
        if args.experiment in THREE_D_EXPERIMENTS:
            previous = checkpoint.get("training", {})
            if (previous.get("last_episode_steps", args.episode_length)
                    != previous.get("episode_length", args.episode_length)
                    and not (prey_death and (previous.get("episode_boundary", False)
                                            or args.policy_only))):
                raise ValueError("3D continuation requires a checkpoint at a complete episode boundary")
        predator_agent.load_policy(checkpoint["predator_actor"])
        prey_agent.load_policy(checkpoint["prey_actor"])
        # 兼容旧检查点：优先取累计计数，缺失时再使用单次运行计数或从零开始。
        previous_training = checkpoint.get("training", {})
        previous_steps = previous_training.get("cumulative_environment_steps", previous_training.get("environment_steps", 0))
        previous_episodes = previous_training.get("cumulative_episodes", previous_training.get("episodes", 0))
    replay_seed = getattr(args, "replay_seed", None)
    replay_options = {} if replay_seed is None else dict(seed=replay_seed)
    predator_replay = ReplayBuffer(args.buffer_size, obs_dim, act_dim, **replay_options)
    if prioritized_prey:
        from prioritized_replay import PrioritizedReplayBuffer
        prey_replay = PrioritizedReplayBuffer(args.buffer_size, obs_dim, act_dim,
            seed=args.seed if replay_seed is None else replay_seed,
            alpha=args.per_alpha, beta=args.per_beta, beta_steps=args.per_beta_steps,
            uniform_mix=args.per_uniform_mix)
    else:
        prey_replay = ReplayBuffer(args.buffer_size, obs_dim, act_dim, **replay_options)
    if checkpoint is not None:
        if "training_state" in checkpoint and not args.policy_only:
            state = checkpoint["training_state"]
            for name, agent, replay in (("predator", predator_agent, predator_replay),
                                        ("prey", prey_agent, prey_replay)):
                agent.load_state_dict(state[name + "_agent"])
                if not reset_replay:
                    replay.load_state_dict(state[name + "_replay"])
            # 网络和缓冲恢复完成后再还原随机状态，避免初始化影响后续采样序列。
            rng = state["numpy_rng"]
            np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
            torch.set_rng_state(state["torch_rng"])
            if device == "cuda" and state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            # 完整续训恢复剩余预热预算；不能将已有缓冲等同于预热已经结束。
            warmup = state.get("warmup")
            if reset_replay:
                # 新缓冲先收集本次指定的预热步数，沿用已加载策略。
                warmup_steps = args.start_steps
                random_warmup = False
            elif warmup is not None:
                warmup_steps = warmup["remaining_steps"]
                random_warmup = warmup["random_actions"]
            elif previous_training.get("experiment") in THREE_D_EXPERIMENTS:
                # 旧 3D 检查点使用累计步数计算剩余预热。
                saved_args = previous_training.get("arguments", {})
                warmup_steps = max(0, saved_args.get("start_steps", args.start_steps) - previous_steps)
                random_warmup = True
            else:
                # 兼容旧完整检查点：其计数是本次运行步数，不能用含策略历史的累计步数。
                saved_args = previous_training.get("arguments", {})
                warmup_steps = max(0, saved_args.get("start_steps", args.start_steps)
                                   - previous_training.get("environment_steps", 0))
                if previous_training.get("restored_training_state", False):
                    warmup_steps = 0  # 旧实现的完整续训已经开始学习。
                random_warmup = saved_args.get("init_model") is None
            restored_training_state = True
            if reset_replay:
                print(f"Restored policies, critics, optimizers and RNG from {init_model}; replay buffers start empty.")
            else:
                print(f"Restored policies, critics, optimizers and replay buffers from {init_model}.")
            del state
        else:
            print(f"Loaded both policies from {init_model}; critics, optimizers and replay buffers start fresh.")
        del checkpoint

    if prey_agent.terminal_calibration is not None:
        if (not prey_death or not env.is_periodic or getattr(env.unwrapped, 'capture_cooldown_steps', 0)
                or env.L != prey_agent.terminal_calibration.arena_L
                or not np.isclose(env.size_p + env.size_e, .095, atol=1e-15, rtol=0)):
            raise ValueError('Terminal calibration requires its original periodic L and 0.095 first-contact capture rule without cooldown')
    prey_agent.set_actor_updates(not getattr(args, 'freeze_prey_actor', False))

    print(
        f"NvsN PPS: n_p={env.n_p}, n_e={env.n_e}, obs_dim={obs_dim}, "
        f"shared_action_dim={act_dim}, L={env.L}, device={device}, "
        f"adam_backend={'fused' if predator_agent.fused_adam else 'default'}, "
        f"previous_env_steps={previous_steps}, planned_env_steps={planned_steps}, "
        f"target_cumulative_env_steps={previous_steps + planned_steps}, render_interval={args.render_interval}, "
        f"cuda_graph_requested={args.cuda_graph}, exploration_mode={args.exploration_mode}"
    )
    total_steps = 0
    best_min_distance = float("inf")
    rendered_episodes = []
    completed_episodes = 0
    episode_steps = 0

    def save_checkpoint():
        save_started = time.perf_counter()
        # 顶层 Actor 权重兼容原评估入口，training_state 用于恢复完整学习过程。
        # 保存发生在回合结束；续训会 reset 环境，不恢复回合中间的物理状态。
        numpy_rng = np.random.get_state()
        checkpoint = {
            "predator_actor": predator_agent.actor.state_dict(),
            "prey_actor": prey_agent.actor.state_dict(),
            "training": {
                "experiment": args.experiment,
                "structure": expected_structure,
                "controllers": {"predator": env.pursuer_strategy, "prey": env.escaper_strategy},
                "environment_steps": total_steps,
                "previous_environment_steps": previous_steps,
                "cumulative_environment_steps": previous_steps + total_steps,
                "episodes": completed_episodes,
                "previous_episodes": previous_episodes,
                "cumulative_episodes": previous_episodes + completed_episodes,
                "episode_length": args.episode_length,
                "last_episode_steps": episode_steps,
                "rendered_episodes": rendered_episodes,
                "render_interval": args.render_interval,
                "restored_training_state": restored_training_state,
                "init_model": init_model,
                "arguments": vars(args),
                "exploration": {"mode": args.exploration_mode,
                                "schedule_episode": previous_episodes + completed_episodes,
                                "noise_std": noise, "epsilon": epsilon},
                "cuda_graph_status": {"predator": predator_agent.cuda_graph_status if learn_predator else "fixed:" + env.pursuer_strategy,
                                      "prey": prey_agent.cuda_graph_status if learn_prey else "fixed:" + env.escaper_strategy},
                "environment": {
                    name: env.get_param(name)
                    for name in ("L", "n_p", "n_e", "dynamics_mode", "pursuer_strategy", "escaper_strategy",
                                 "angle_p_max", "angle_e_max", "dt")
                },
            },
            "training_state": {
                "warmup": {
                    "remaining_steps": max(0, warmup_steps - total_steps),
                    "random_actions": random_warmup,
                },
                "predator_agent": predator_agent.state_dict(),
                "prey_agent": prey_agent.state_dict(),
                "predator_replay": predator_replay.state_dict(),
                "prey_replay": prey_replay.state_dict(),
                # 随机状态中的 NumPy 数组转为列表，使 weights_only 加载无需额外类型。
                "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            },
        }
        if args.experiment == "paper_swirl":
            from paper_swirl import settings
            checkpoint["training"]["experiment_settings"] = settings(arena_L=arena_L)
        elif args.experiment in THREE_D_EXPERIMENTS:
            from njp_3d import settings
            checkpoint["training"]["experiment_settings"] = settings(
                arena_L=arena_L, no_energy=args.experiment == "njp_3d_no_energy",
                silu=args.experiment == "njp_3d_silu_no_energy",
                fixed_prey=args.experiment == "njp_3d_silu_rule_prey",
                wide_obs=args.experiment == "njp_3d_silu_wide_obs",
                capture=args.experiment == "njp_3d_silu_capture", **counts, **spawn)
        if prey_death:
            checkpoint["training"]["episode_boundary"] = (
                episode_terminal or episode_steps == args.episode_length)
            checkpoint["training"]["episode_termination"] = (
                "all_prey_caught" if episode_terminal else
                "time_limit" if episode_steps == args.episode_length else "step_budget")
        if critic_repair_info is not None:
            checkpoint['training']['critic_repair'] = critic_repair_info
        if freeze_predator or prioritized_prey:
            checkpoint["training"]["learning"] = dict(predator_frozen=freeze_predator,
                predator_exploration=False if freeze_predator else True,
                prey_replay="prioritized" if prioritized_prey else "uniform")
        if getattr(args, 'freeze_prey_actor', False) or prey_agent.critic_layer_norm or prey_agent.critic_grad_clip:
            checkpoint['training'].setdefault('learning', {}).update(
                prey_actor_frozen=not prey_agent.actor_updates_enabled,
                prey_critic_layer_norm=prey_agent.critic_layer_norm,
                prey_critic_geometry=prey_agent.critic_geometry,
                prey_critic_terminal_weight=prey_agent.critic_terminal_weight,
                prey_critic_risk_weight=prey_agent.critic_risk_weight,
                prey_critic_grad_clip=prey_agent.critic_grad_clip,
                prey_critic_lr=prey_agent.critic_opt.param_groups[0]['lr'])
        if getattr(args, "predator_algorithm", "ddpg") != "ddpg":
            checkpoint["training"]["algorithms"] = {"predator": args.predator_algorithm, "prey": "ddpg"}
        # 临时文件与目标文件位于同一目录，写完后原子替换，避免中断留下半个检查点。
        temporary_path = output_model + ".tmp"
        os.makedirs(os.path.dirname(output_model), exist_ok=True)
        with checkpoint_lock():
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, output_model)
        phases["checkpoints"] += time.perf_counter() - save_started
        print(f"Saved checkpoint: cumulative_env_steps={previous_steps + total_steps}, path={output_model}")

    loop_started = time.perf_counter()
    local_episode = 0
    while local_episode < args.episodes or (prey_death and total_steps < planned_steps):
        local_episode += 1
        if total_steps >= planned_steps:
            break
        # local_episode 控制本次噪声衰减，累计 episode 控制日志、渲染和保存周期。
        episode_steps = min(args.episode_length, planned_steps - total_steps)
        episode = previous_episodes + local_episode
        obs = env.reset()
        predator_states = obs["predator"]
        prey_states = obs["prey"]
        pred_distances = predator_prey_distances(env)
        episode_reward = 0.0
        episode_prey_reward = 0.0
        min_distance = float(np.min(pred_distances))
        caught_once = False
        noise, epsilon = exploration_rates(args, local_episode, episode)
        render_episode = args.render or (args.render_interval > 0 and episode % args.render_interval == 0)
        if render_episode:
            rendered_episodes.append(episode)
            print(f"Rendering episode {episode}")

        episode_terminal = False
        for step_index in range(episode_steps):
            if profile_path:
                phase_started = time.perf_counter()
            if render_episode:
                env.render(mode="human")

            # 从头训练的预热使用随机动作；仅加载策略的微调使用策略加噪声。
            # 完整续训沿用保存的模式，避免在预热中途改变经验分布。
            if total_steps < warmup_steps and random_warmup:
                predator_actions = (np.random.uniform(-1.0, 1.0, size=(env.n_p, act_dim)).astype(np.float32).T
                                    if learn_predator else np.zeros((act_dim, env.n_p), dtype=np.float32))
                prey_actions = (np.random.uniform(-1.0, 1.0, size=(env.n_e, act_dim)).astype(np.float32).T
                                if learn_prey else np.zeros((act_dim, env.n_e), dtype=np.float32))
            else:
                if args.exploration_mode == "paper":
                    predator_actions = (predator_agent.act_batch(predator_states, noise_std=noise, epsilon=epsilon).T
                                        if learn_predator else np.zeros((act_dim, env.n_p), dtype=np.float32))
                    prey_actions = (prey_agent.act_batch(prey_states, noise_std=noise, epsilon=epsilon).T
                                    if learn_prey else np.zeros((act_dim, env.n_e), dtype=np.float32))
                else:
                    predator_actions = (predator_agent.act_batch(predator_states, noise_std=noise).T
                                        if learn_predator else np.zeros((act_dim, env.n_p), dtype=np.float32))
                    prey_actions = (prey_agent.act_batch(prey_states, noise_std=noise).T
                                    if learn_prey else np.zeros((act_dim, env.n_e), dtype=np.float32))

            if freeze_predator:
                predator_actions = predator_agent.act_batch(predator_states).T
            action = {"predator": predator_actions, "prey": prey_actions}
            if profile_path:
                phases["inference"] += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
            alive_before = env.unwrapped.alive_mask if prey_death else None
            next_obs, rewards, dones, _ = env.step(action)
            if profile_path:
                phases["environment"] += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
            pred_rewards = rewards["predator"]
            prey_rewards = rewards["prey"]
            pred_distances = predator_prey_distances(env)
            caught = (bool(env.unwrapped.capture_contacts.any()) if prey_death else
                      bool(env.unwrapped._is_collide_b2b[env.n_p:env.n_pe, :env.n_p].any())
                      if args.experiment in THREE_D_EXPERIMENTS
                      else bool(np.min(pred_distances) <= env.size_p + env.size_e))
            next_predator_states = next_obs["predator"]
            next_prey_states = next_obs["prey"]
            caught_once = caught_once or caught
            if prey_death:
                terminal = np.asarray(dones, dtype=bool).reshape(-1)
                if learn_predator:
                    predator_replay.push_batch(predator_states, predator_actions.T,
                                               pred_rewards, next_predator_states,
                                               terminal[:env.n_p])
                if learn_prey:
                    living = alive_before[env.n_p:env.n_pe]
                    prey_replay.push_batch(prey_states[living], prey_actions.T[living],
                                           prey_rewards[living], next_prey_states[living],
                                           terminal[env.n_p:env.n_pe][living])
                    prey_agent.observe_recovery(prey_states[living], prey_actions.T[living],
                        prey_rewards[living], next_prey_states[living], terminal[env.n_p:env.n_pe][living])
                episode_terminal = bool(terminal[:env.n_p].all())
            elif learn_predator:
                predator_replay.push_batch(predator_states, predator_actions.T,
                                           pred_rewards, next_predator_states)
            if learn_prey and not prey_death:
                prey_replay.push_batch(prey_states, prey_actions.T,
                                      prey_rewards, next_prey_states)
            # 按原个体顺序整批写入，减少逐个体调用，抽样和终止标记保持原样。

            predator_states = next_predator_states
            prey_states = next_prey_states
            episode_reward += float(np.mean(pred_rewards))
            episode_prey_reward += float(np.mean(prey_rewards))
            min_distance = min(min_distance, float(np.min(pred_distances)))
            total_steps += 1
            if profile_path:
                phases["metrics_and_replay"] += time.perf_counter() - phase_started
                phase_started = time.perf_counter()

            # 保存和恢复剩余预算，使预热前、边界上和预热后的更新时机一致。
            if total_steps >= warmup_steps:
                for _ in range(args.updates_per_step):
                    # 主循环不消费损失值，关闭标量回读以减少不必要的 CUDA 同步。
                    if learn_predator and len(predator_replay) >= args.batch_size:
                        predator_agent.update(predator_replay, args.batch_size, return_losses=False)
                    if learn_prey and len(prey_replay) >= args.batch_size:
                        prey_agent.update(prey_replay, args.batch_size, return_losses=False)
            if profile_path:
                phases["updates"] += time.perf_counter() - phase_started
            if episode_terminal:
                episode_steps = step_index + 1
                break

        best_min_distance = min(best_min_distance, min_distance)
        if episode % 10 == 0 or caught_once or args.experiment in THREE_D_EXPERIMENTS:
            print(
                f"episode={episode:04d} pred_reward={episode_reward:8.3f} "
                f"prey_reward={episode_prey_reward:8.3f} "
                f"min_dist={min_distance:.4f} best={best_min_distance:.4f} "
                f"caught={caught_once} noise={noise:.3f} epsilon={epsilon:.3f} "
                f"cumulative_steps={previous_steps + total_steps}"
            )
        completed_episodes = local_episode
        # 最后一回合交给循环后的统一保存，避免同一步数重复写入大检查点。
        if args.checkpoint_interval and episode % args.checkpoint_interval == 0 and total_steps < planned_steps:
            save_checkpoint()

    if profile_path and device == "cuda":
        torch.cuda.synchronize()
    loop_seconds = time.perf_counter() - loop_started
    save_checkpoint()
    env.close()
    if profile_path:
        import json
        import resource
        if device == "cuda":
            torch.cuda.synchronize()
        profile = {"environment_steps": total_steps, "elapsed_seconds": time.perf_counter() - started_at,
                   "training_loop_seconds": loop_seconds,
                   "phase_host_seconds": phases, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else 0,
                   "graph_status": [predator_agent.cuda_graph_status, prey_agent.cuda_graph_status]}
        os.makedirs(os.path.dirname(os.path.abspath(profile_path)), exist_ok=True)
        with open(profile_path, "w") as handle:
            json.dump(profile, handle, indent=2)
    print(f"Saved {args.model_path}; completed_env_steps={total_steps}, "
          f"cumulative_env_steps={previous_steps + total_steps}, rendered_episodes={rendered_episodes}")
    if args.cuda_graph:
        print(f"CUDA Graph: predator={predator_agent.cuda_graph_status if learn_predator else 'fixed:' + env.pursuer_strategy}, "
              f"prey={prey_agent.cuda_graph_status if learn_prey else 'fixed:' + env.escaper_strategy}, "
              f"capture_seconds={predator_agent.graph_capture_seconds + prey_agent.graph_capture_seconds:.3f}")


if __name__ == "__main__":
    train(get_args())
