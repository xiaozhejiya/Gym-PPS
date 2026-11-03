import argparse


def get_args():
    parser = argparse.ArgumentParser("Two shared DDPG policies for N-vs-N Gym-PPS")
    parser.add_argument("--experiment", choices=("paper_swirl", "njp_3d", "njp_3d_no_energy", "njp_3d_silu_no_energy", "njp_3d_silu_rule_prey", "njp_3d_silu_wide_obs", "njp_3d_silu_capture"), default=None,
                        help="Use an isolated experiment preset instead of custom_param.json.")
    parser.add_argument("--arena_L", type=float, default=None,
                        help="Override the selected experiment's arena half-width.")
    parser.add_argument("--n_p", type=int, default=None, help="Override the 3D predator population.")
    parser.add_argument("--n_e", type=int, default=None, help="Override the 3D prey population.")
    from njp_3d import add_spawn_arguments
    add_spawn_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--buffer_size", type=int, default=5e5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--freeze_predator", action="store_true", help="Freeze the loaded predator, using deterministic policy actions.")
    parser.add_argument("--freeze_prey_actor", action="store_true", help="Update the prey Critic and target Critic only; preserve Actor, target Actor and Actor optimizer.")
    parser.add_argument("--prey_critic_layer_norm", action="store_true", help="Opt-in LayerNorm in prey Critic hidden layers; requires a matching checkpoint.")
    parser.add_argument("--prey_critic_geometry", action="store_true", help="Derived nearest-predator geometry in the 178D prey Critic; Actor observations stay unchanged.")
    parser.add_argument("--prey_critic_grad_clip", type=float, default=0.0, help="Prey Critic gradient norm limit; 0 preserves legacy updates.")
    parser.add_argument("--prey_critic_lr", type=float, default=None, help="Prey Critic learning rate for initialization; full resume restores saved optimizer LR.")
    parser.add_argument("--prey_critic_terminal_weight", type=float, default=0., help="Terminal calibration weight; requires saved calibration observations/RNG.")
    parser.add_argument("--prey_critic_terminal_batch_size", type=int, default=32)
    parser.add_argument("--prey_critic_risk_weight", type=float, default=0.)
    parser.add_argument("--prey_critic_risk_batch_size", type=int, default=256)
    parser.add_argument("--prey_replay", choices=("uniform", "prioritized"), default="uniform")
    parser.add_argument("--replay_seed", type=int, default=None, help="Independent replay RNG; preserves global exploration RNG.")
    parser.add_argument("--per_alpha", type=float, default=.6)
    parser.add_argument("--per_beta", type=float, default=.4)
    parser.add_argument("--per_beta_steps", type=int, default=200000)
    parser.add_argument("--per_uniform_mix", type=float, default=.1)
    parser.add_argument("--allow_capture_rule_change", action="store_true",
                        help="Explicit full-state fine-tuning under a changed capture rule; requires --reset_replay.")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, choices=(2, 3), default=2)
    parser.add_argument("--hidden_activation", choices=("relu", "silu"), default="relu")
    parser.add_argument("--predator_algorithm", choices=("ddpg", "td3"), default="ddpg")
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--target_noise", type=float, default=.2)
    parser.add_argument("--target_noise_clip", type=float, default=.5)
    parser.add_argument("--actor_warmup_updates", type=int, default=2000,
                        help="TD3 critic-only warmup; target critics still update on delayed steps.")
    parser.add_argument("--actor_logit_penalty", type=float, default=0.0,
                        help="Soft squared penalty on pre-Tanh outputs beyond actor_logit_threshold.")
    parser.add_argument("--actor_logit_threshold", type=float, default=3.0)
    # 两个开关共用同一配置项，后出现者生效，方便覆盖启动脚本预设的 fused 后端。
    parser.add_argument("--fused_adam", action="store_true", help="Use fused Adam on CUDA; CPU keeps the default backend.")
    parser.add_argument("--no_fused_adam", dest="fused_adam", action="store_false", help="Use the default Adam backend, overriding --fused_adam.")
    parser.set_defaults(fused_adam=False)
    # 图执行必须显式开启；两个开关按出现顺序覆盖，默认仍使用普通更新。
    parser.add_argument("--cuda_graph", action="store_true",
                        help="Capture DDPG updates; requires CUDA and --fused_adam.")
    parser.add_argument("--no_cuda_graph", dest="cuda_graph", action="store_false")
    parser.set_defaults(cuda_graph=False)
    parser.add_argument("--packed_graph_inputs", action="store_true",
                        help="Experimental pinned, packed CUDA Graph uploads; default remains five copies per species.")
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--noise_min", type=float, default=0.05)
    parser.add_argument(
        "--exploration_mode", choices=("gaussian", "paper"), default="gaussian",
        help="gaussian preserves the existing noise schedule; paper adds per-agent epsilon "
             "random actions and subtracts exploration_decay per cumulative episode.",
    )
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Initial random-action probability in paper exploration mode.")
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--exploration_decay", type=float, default=5e-5,
                        help="Per-episode decrement for noise and epsilon in paper exploration mode.")
    parser.add_argument("--render", action="store_true")
    # 周期渲染与定期保存均按累计回合数计数；间隔为 0 时关闭对应功能。
    parser.add_argument(
        "--render_interval", type=int, default=20,
        help="Render every N cumulative episodes; 0 disables periodic rendering.",
    )
    parser.add_argument(
        "--total_steps", type=int, default=None,
        # 这是本次运行新增的环境步数，不包含检查点中已经完成的训练步数。
        help="Exact number of additional environment steps; may shorten the final episode.",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=20,
        help="Save the output checkpoint every N episodes; 0 disables periodic saves.",
    )
    parser.add_argument(
        "--init_model",
        type=str,
        default=None,
        help="Restore full training state when available; otherwise initialize both policies only.",
    )
    parser.add_argument(
        "--policy_only", action="store_true",
        help="With --init_model, load actors only; reset critics, optimizers, replay buffers and RNG for changed dynamics.",
    )
    parser.add_argument(
        "--reset_replay", action="store_true",
        help="With a full --init_model, restore agents and RNG but start empty replay buffers for changed dynamics.",
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
