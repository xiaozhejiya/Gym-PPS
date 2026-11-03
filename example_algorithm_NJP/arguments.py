import argparse


def get_args():
    parser = argparse.ArgumentParser("Gym-PPS 多对多场景的双方共享 DDPG 策略")
    parser.add_argument("--experiment", choices=("paper_swirl", "njp_3d", "njp_3d_no_energy", "njp_3d_silu_no_energy", "njp_3d_silu_rule_prey", "njp_3d_silu_wide_obs", "njp_3d_silu_capture"), default=None,
                        help="使用独立实验预设，替代 custom_param.json。")
    parser.add_argument("--arena_L", type=float, default=None,
                        help="覆盖所选实验的场地半宽。")
    parser.add_argument("--n_p", type=int, default=None, help="覆盖三维场景的捕食者数量。")
    parser.add_argument("--n_e", type=int, default=None, help="覆盖三维场景的猎物数量。")
    from njp_3d import add_spawn_arguments
    add_spawn_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--buffer_size", type=int, default=5e5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--freeze_predator", action="store_true", help="冻结已加载的捕食者，使用确定性策略动作。")
    parser.add_argument("--freeze_prey_actor", action="store_true", help="仅更新猎物 Critic 和目标 Critic，保留 Actor、目标 Actor 及 Actor 优化器。")
    parser.add_argument("--prey_critic_layer_norm", action="store_true", help="在猎物 Critic 隐藏层中启用 LayerNorm，需要配置匹配的检查点。")
    parser.add_argument("--prey_critic_geometry", action="store_true", help="为 178 维猎物 Critic 添加最近捕食者的派生几何特征，Actor 观测保持不变。")
    parser.add_argument("--prey_critic_grad_clip", type=float, default=0.0, help="猎物 Critic 的梯度范数上限；0 表示沿用原有更新方式。")
    parser.add_argument("--prey_critic_lr", type=float, default=None, help="初始化时的猎物 Critic 学习率；完整续训会恢复已保存的优化器学习率。")
    parser.add_argument("--prey_critic_terminal_weight", type=float, default=0., help="终止值校准权重，需要已保存的校准观测和随机状态。")
    parser.add_argument("--prey_critic_terminal_batch_size", type=int, default=32)
    parser.add_argument("--prey_critic_risk_weight", type=float, default=0.)
    parser.add_argument("--prey_critic_risk_batch_size", type=int, default=256)
    parser.add_argument("--prey_replay", choices=("uniform", "prioritized"), default="uniform")
    parser.add_argument("--replay_seed", type=int, default=None, help="为经验回放使用独立随机状态，保留全局探索随机序列。")
    parser.add_argument("--per_alpha", type=float, default=.6)
    parser.add_argument("--per_beta", type=float, default=.4)
    parser.add_argument("--per_beta_steps", type=int, default=200000)
    parser.add_argument("--per_uniform_mix", type=float, default=.1)
    parser.add_argument("--allow_capture_rule_change", action="store_true",
                        help="在改变捕获规则后显式恢复完整状态进行微调，需要同时指定 --reset_replay。")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, choices=(2, 3), default=2)
    parser.add_argument("--hidden_activation", choices=("relu", "silu"), default="relu")
    parser.add_argument("--predator_algorithm", choices=("ddpg", "td3"), default="ddpg")
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--target_noise", type=float, default=.2)
    parser.add_argument("--target_noise_clip", type=float, default=.5)
    parser.add_argument("--actor_warmup_updates", type=int, default=2000,
                        help="TD3 仅更新 Critic 的预热步数；目标 Critic 仍按延迟更新周期更新。")
    parser.add_argument("--actor_logit_penalty", type=float, default=0.0,
                        help="对 Tanh 之前超过 actor_logit_threshold 的输出施加平方软惩罚。")
    parser.add_argument("--actor_logit_threshold", type=float, default=3.0)
    # 两个开关共用同一配置项，后出现者生效，方便覆盖启动脚本预设的 fused 后端。
    parser.add_argument("--fused_adam", action="store_true", help="在 CUDA 上使用融合 Adam；CPU 保持默认后端。")
    parser.add_argument("--no_fused_adam", dest="fused_adam", action="store_false", help="使用默认 Adam 后端，覆盖 --fused_adam。")
    parser.set_defaults(fused_adam=False)
    # 图执行必须显式开启；两个开关按出现顺序覆盖，默认仍使用普通更新。
    parser.add_argument("--cuda_graph", action="store_true",
                        help="捕获 DDPG 更新计算图，需要 CUDA 和 --fused_adam。")
    parser.add_argument("--no_cuda_graph", dest="cuda_graph", action="store_false")
    parser.set_defaults(cuda_graph=False)
    parser.add_argument("--packed_graph_inputs", action="store_true",
                        help="实验性固定内存打包上传；默认仍按每个物种五次传输执行 CUDA Graph 输入复制。")
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--noise_min", type=float, default=0.05)
    parser.add_argument(
        "--exploration_mode", choices=("gaussian", "paper"), default="gaussian",
        help="gaussian 沿用原有噪声调度；paper 为每个个体增加 epsilon 随机动作，"
             "并按累计回合数逐回合减去 exploration_decay。",
    )
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="paper 探索模式的初始随机动作概率。")
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--exploration_decay", type=float, default=5e-5,
                        help="paper 探索模式中噪声和 epsilon 每回合的递减量。")
    parser.add_argument("--render", action="store_true")
    # 周期渲染与定期保存均按累计回合数计数；间隔为 0 时关闭对应功能。
    parser.add_argument(
        "--render_interval", type=int, default=20,
        help="每 N 个累计回合渲染一次；0 表示关闭周期渲染。",
    )
    parser.add_argument(
        "--total_steps", type=int, default=None,
        # 这是本次运行新增的环境步数，不包含检查点中已经完成的训练步数。
        help="本次新增的准确环境步数，可能缩短最后一个回合。",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=20,
        help="每 N 个回合保存一次输出检查点；0 表示关闭定期保存。",
    )
    parser.add_argument(
        "--init_model",
        type=str,
        default=None,
        help="优先恢复完整训练状态；若检查点仅含策略，则仅初始化双方策略。",
    )
    parser.add_argument(
        "--policy_only", action="store_true",
        help="配合 --init_model 仅加载 Actor；重置 Critic、优化器、经验池和随机状态，以适应变化后的动力学。",
    )
    parser.add_argument(
        "--reset_replay", action="store_true",
        help="配合完整 --init_model 恢复智能体及随机状态，但清空经验池以适应变化后的动力学。",
    )
    parser.add_argument("--model_path", type=str, default="ddpg_both_NvsN.pt")
    selected, _ = parser.parse_known_args()
    if selected.experiment == "paper_swirl":
        from paper_swirl import defaults
        parser.set_defaults(**defaults())
    elif selected.experiment in ("njp_3d", "njp_3d_no_energy", "njp_3d_silu_no_energy", "njp_3d_silu_rule_prey", "njp_3d_silu_wide_obs", "njp_3d_silu_capture"):
        from njp_3d import defaults
        parser.set_defaults(**defaults(no_energy=selected.experiment == "njp_3d_no_energy",
                                       silu=selected.experiment == "njp_3d_silu_no_energy",
                                       fixed_prey=selected.experiment == "njp_3d_silu_rule_prey",
                                       wide_obs=selected.experiment == "njp_3d_silu_wide_obs",
                                       capture=selected.experiment == "njp_3d_silu_capture"))
    return parser.parse_args()
