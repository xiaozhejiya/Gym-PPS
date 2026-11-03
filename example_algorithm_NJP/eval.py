import argparse
import os
import sys
import json

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent import DDPGAgent
from custom_env import predator_prey_distances
from train import (
    make_env,
    THREE_D_EXPERIMENTS,
)
from checkpoint_compat import structure, validate_checkpoint


def get_args():
    parser = argparse.ArgumentParser("评估 Gym-PPS 多对多场景的双方共享 DDPG 策略")
    parser.add_argument("--experiment", choices=("paper_swirl",) + THREE_D_EXPERIMENTS, default=None)
    parser.add_argument("--arena_L", type=float, default=None,
                        help="覆盖所选实验的场地半宽。")
    parser.add_argument("--n_p", type=int, default=None)
    parser.add_argument("--n_e", type=int, default=None)
    from njp_3d import add_spawn_arguments
    add_spawn_arguments(parser)
    parser.add_argument("--headless", action="store_true", help="创建离屏 OpenGL 上下文，用于录制 GIF。")
    parser.add_argument("--camera_view", choices=("oblique", "xy", "xz", "yz"), default="oblique")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--predator_controller", choices=("input", "nearest", "static", "random"), default=None)
    parser.add_argument("--prey_controller", choices=("input", "nearest", "static", "random"), default=None)
    parser.add_argument("--report_path", type=str, default=None, help="将逐步行为诊断结果写入 JSON，不修改仿真状态。")
    parser.add_argument("--allow_capture_transfer", action="store_true",
                        help="显式在变化后的捕获规则下评估 Actor，其他布局检查仍保持严格。")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, choices=(2, 3), default=2)
    parser.add_argument("--hidden_activation", choices=("relu", "silu"), default="relu")
    parser.add_argument("--model_path", type=str, default="ddpg_both_NvsN.pt")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gif_path", type=str, default=None,
                        help="GIF 保存路径；多个回合会追加 _ep001、_ep002 等后缀。")
    parser.add_argument("--gif_fps", type=int, default=25, help="GIF 播放帧率，范围为 1–100。")
    parser.add_argument("--gif_stride", type=int, default=1,
                        help="每 N 个环境步录制一帧；数值越大，GIF 文件越小。")
    selected, _ = parser.parse_known_args()
    if selected.experiment == "paper_swirl":
        from paper_swirl import defaults
        parser.set_defaults(**defaults(evaluation=True))
    elif selected.experiment in THREE_D_EXPERIMENTS:
        from njp_3d import defaults
        parser.set_defaults(**defaults(evaluation=True, no_energy=selected.experiment == "njp_3d_no_energy",
                                       silu=selected.experiment == "njp_3d_silu_no_energy",
                                       fixed_prey=selected.experiment == "njp_3d_silu_rule_prey",
                                       wide_obs=selected.experiment == "njp_3d_silu_wide_obs",
                                       capture=selected.experiment == "njp_3d_silu_capture"))
    return parser.parse_args()


def evaluate(args):
    if args.episodes <= 0 or args.episode_length <= 0:
        raise ValueError("episodes 和 episode_length 必须为正数")
    if not 1 <= args.gif_fps <= 100 or args.gif_stride <= 0:
        raise ValueError("gif_fps 必须在 1 到 100 之间，gif_stride 必须为正数")
    if getattr(args, "headless", False):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        import pyglet
        pyglet.options["headless"] = True
    image_module = None
    if args.gif_path:
        if not args.gif_path.lower().endswith(".gif"):
            raise ValueError("gif_path 必须以 .gif 结尾")
        from PIL import Image
        image_module = Image
        args.gif_path = os.path.abspath(args.gif_path)
        os.makedirs(os.path.dirname(args.gif_path), exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    arena_L = getattr(args, "arena_L", None)
    counts = {name: getattr(args, name, None) for name in ("n_p", "n_e")}
    counts = {name: value for name, value in counts.items() if value is not None}
    from njp_3d import spawn_options
    spawn = spawn_options(args)
    if arena_L is not None or counts or spawn:
        env = make_env(args.experiment, evaluation=True, arena_L=arena_L, **counts, **spawn)
    else:
        env = make_env(args.experiment, evaluation=True) if args.experiment else make_env()
    try:
        for option, parameter in (("predator_controller", "pursuer_strategy"), ("prey_controller", "escaper_strategy")):
            controller = getattr(args, option, None)
            if controller is not None:
                setattr(env.unwrapped, parameter, controller)
        if getattr(env.unwrapped, "spatial_dim", 2) == 3:
            env.unwrapped.camera_view = getattr(args, "camera_view", "oblique")
        _evaluate(args, env, image_module)
    finally:
        env.close()


def _save_gif(frames, path, fps, preserve_frame_count=False):
    # GIF 帧延时以 10 毫秒为单位；交替使用取整后的延时，
    # 使 30 fps 等帧率保持正确平均时长，避免变成约 33 fps。
    durations = [(round((i + 1) * 100 / fps) - round(i * 100 / fps)) * 10
                 for i in range(len(frames))]
    if preserve_frame_count:
        # 即使设置 optimize=False，Pillow 的 save_all 仍会合并连续相同帧。
        # 三维录制逐帧独立编码完整画面，
        # 使静止场景保留每次采样，同时保持像素不变。
        from PIL import GifImagePlugin
        header, _ = GifImagePlugin.getheader(frames[0].copy(), info={"loop": 0, "optimize": False})
        with open(path, "wb") as output:
            for block in header:
                output.write(block)
            for frame, duration in zip(frames, durations):
                # 每个自适应调色板帧保留自己的颜色表。
                for block in GifImagePlugin.getdata(frame, duration=duration, disposal=2,
                                                    include_color_table=True):
                    output.write(block)
            output.write(b";")
    else:
        frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, disposal=2, optimize=False)
    print(f"已保存 GIF：{path}；帧数={len(frames)}，帧率={fps}")


def _caught_in_step(env, distance, contact_distance):
    if getattr(env.unwrapped, "prey_death_enabled", False):
        return bool(env.unwrapped.capture_contacts.any())
    if getattr(env.unwrapped, "spatial_dim", 2) == 3:
        return bool(env.unwrapped._is_collide_b2b[env.n_p:env.n_pe, :env.n_p].any())
    return distance <= contact_distance


def _evaluate(args, env, image_module):
    # 定制包装器保留参数副本；控制器覆盖直接作用于仿真器，
    # 因此应读取仿真器，避免使用包装器中的旧副本。
    physical_env = env.unwrapped
    obs = env.reset()
    obs_dim = env.observation_space.spaces["predator"].shape[1]
    act_dim = env.action_space.spaces["predator"].shape[0]
    requested_device = getattr(args, "device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("已请求 CUDA 评估，但当前 CUDA 不可用")

    hidden_activation = getattr(args, "hidden_activation", "relu")
    predator_agent = DDPGAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
                              hidden_activation=hidden_activation,
                              hidden_layers=args.hidden_layers, device=device)
    prey_agent = DDPGAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
                          hidden_activation=hidden_activation,
                          hidden_layers=args.hidden_layers, device=device)
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
    # 评估只需要两个 Actor；完整检查点先放在 CPU，避免回放缓冲占用显存。
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    validate_checkpoint(checkpoint, structure(env, args.hidden_dim, args.hidden_layers, hidden_activation),
                        policy_only=getattr(args, "allow_capture_transfer", False))
    predator_agent.actor.load_state_dict(checkpoint["predator_actor"])
    prey_agent.actor.load_state_dict(checkpoint["prey_actor"])
    # 网络权重已复制到模型，立即释放不参与评估的训练状态。
    del checkpoint
    predator_agent.actor.eval()
    prey_agent.actor.eval()

    contact_distance = env.size_p + env.size_e
    successes = 0
    min_distances = []
    catch_steps = []
    reports = []
    diagnostics_enabled = args.experiment in THREE_D_EXPERIMENTS or bool(getattr(args, "report_path", None))

    print(
        f"正在评估 {os.path.basename(model_path)}： "
        f"捕食者数={env.n_p}，猎物数={env.n_e}，观测维度={obs_dim}， "
        f"共享动作维度={act_dim}，接触距离={contact_distance:.4f}"
    )

    for episode in range(1, args.episodes + 1):
        if diagnostics_enabled:
            from evaluation_metrics import Diagnostics
            diagnostics = Diagnostics(env)
        obs = env.reset()
        predator_states = obs["predator"]
        prey_states = obs["prey"]
        distances = predator_prey_distances(env)
        min_distance = float(np.min(distances))
        caught_once = False
        first_catch_step = args.episode_length
        frames = []

        for step in range(1, args.episode_length + 1):
            if args.gif_path and (step - 1) % args.gif_stride == 0:
                frame = image_module.fromarray(env.render(mode="rgb_array"))
                # 保存索引色图像以降低录制内存，
                # 无需保留 RGB 数组或生成中间 PNG 文件。
                frames.append(frame.convert("P", palette=image_module.Palette.ADAPTIVE))
            elif args.render:
                env.render(mode="human")

            # 每个物种批量推理一次，再转成环境要求的 (动作维度, 个体数)。
            predator_actions = (predator_agent.act_batch(predator_states).T if physical_env.pursuer_strategy == "input"
                                else np.zeros((act_dim, env.n_p), dtype=np.float32))
            prey_actions = (prey_agent.act_batch(prey_states).T if physical_env.escaper_strategy == "input"
                            else np.zeros((act_dim, env.n_e), dtype=np.float32))

            action = {"predator": predator_actions, "prey": prey_actions}
            next_obs, rewards, dones, _ = env.step(action)
            if diagnostics_enabled:
                diagnostics.add(action, rewards)
            predator_states = next_obs["predator"]
            prey_states = next_obs["prey"]
            distances = predator_prey_distances(env)
            distance = float(np.min(distances))
            min_distance = min(min_distance, distance)

            if _caught_in_step(env, distance, contact_distance) and not caught_once:
                caught_once = True
                first_catch_step = step
            if (getattr(physical_env, "prey_death_enabled", False)
                    and np.asarray(dones, dtype=bool).reshape(-1)[:env.n_p].all()):
                break

        if args.gif_path:
            stem, extension = os.path.splitext(args.gif_path)
            path = (args.gif_path if args.episodes == 1 else
                    f"{stem}_ep{episode:03d}{extension}")
            if getattr(env.unwrapped, "spatial_dim", 2) == 3:
                _save_gif(frames, path, args.gif_fps, preserve_frame_count=True)
            else:
                _save_gif(frames, path, args.gif_fps)

        successes += int(caught_once)
        if diagnostics_enabled:
            reports.append({"episode": episode, "summary": diagnostics.summary(), "trace": diagnostics.rows})
        min_distances.append(min_distance)
        catch_steps.append(first_catch_step)
        print(
            f"回合={episode:03d} 已捕获={caught_once} "
            f"首次捕获步数={first_catch_step:03d} 最小距离={min_distance:.4f}"
        )

    success_rate = successes / args.episodes
    print(
        f"汇总：成功率={success_rate:.2%}， "
        f"平均最小距离={np.mean(min_distances):.4f}， "
        f"平均捕获步数={np.mean(catch_steps):.1f}"
    )
    if getattr(args, "report_path", None):
        report = {"model": os.path.abspath(model_path), "seed": args.seed,
                  "device": device, "episode_length": args.episode_length,
                  "noise_std": 0., "epsilon": 0.,
                  "structure": structure(env, args.hidden_dim, args.hidden_layers, hidden_activation),
                  "environment": {"L": physical_env.L, "n_p": physical_env.n_p, "n_e": physical_env.n_e,
                                  "is_periodic": physical_env.is_periodic,
                                  "pursuer_strategy": physical_env.pursuer_strategy,
                                  "escaper_strategy": physical_env.escaper_strategy,
                                  "energy_enabled": getattr(physical_env, "energy_enabled", True)},
                  "episodes": reports,
                  "success_rate": success_rate, "average_min_distance": float(np.mean(min_distances))}
        if getattr(physical_env, "prey_death_enabled", False):
            report["environment"]["prey_death_enabled"] = True
            report["capture_rule_transfer_allowed"] = getattr(args, "allow_capture_transfer", False)
        os.makedirs(os.path.dirname(os.path.abspath(args.report_path)), exist_ok=True)
        with open(args.report_path, "w") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print("已保存评估诊断：" + os.path.abspath(args.report_path))


if __name__ == "__main__":
    evaluate(get_args())
