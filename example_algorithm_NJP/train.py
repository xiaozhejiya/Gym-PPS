import os
import sys
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


def make_env():
    custom_param = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_param.json")
    env = gym.make("PredatorPreySwarm-v0")
    env = MyReward(env, custom_param)
    env = MyAction(env, custom_param)
    env = MyObs(env, custom_param)
    assert env.n_p > 0 and env.n_e > 0, "This shared DDPG script expects at least 1 predator and 1 prey."
    return env


def train(args):
    if args.episodes <= 0 or args.episode_length <= 0:
        raise ValueError("episodes and episode_length must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be nonnegative")
    if args.render_interval < 0:
        raise ValueError("render_interval must be nonnegative")
    planned_steps = args.episodes * args.episode_length
    if args.total_steps is not None:
        if not 0 < args.total_steps <= planned_steps:
            raise ValueError("total_steps must be positive and no larger than episodes * episode_length")
        planned_steps = args.total_steps
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    obs = env.reset()
    obs_dim = env.observation_space.spaces["predator"].shape[1]
    act_dim = env.action_space.spaces["predator"].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predator_agent = DDPGAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        fused_adam=args.fused_adam,
    )
    prey_agent = DDPGAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        fused_adam=args.fused_adam,
    )
    init_model = None
    previous_steps = 0
    previous_episodes = 0
    restored_training_state = False
    checkpoint = None
    output_model = os.path.abspath(os.path.join(os.path.dirname(__file__), args.model_path))
    if args.init_model:
        init_model = args.init_model
        if not os.path.isabs(init_model):
            init_model = os.path.join(os.path.dirname(__file__), init_model)
        init_model = os.path.abspath(init_model)
        if init_model == output_model:
            raise ValueError("model_path must differ from init_model to preserve the source checkpoint")
        checkpoint = torch.load(init_model, map_location="cpu", weights_only=True)
        predator_agent.load_policy(checkpoint["predator_actor"])
        prey_agent.load_policy(checkpoint["prey_actor"])
        previous_training = checkpoint.get("training", {})
        previous_steps = previous_training.get("cumulative_environment_steps", previous_training.get("environment_steps", 0))
        previous_episodes = previous_training.get("cumulative_episodes", previous_training.get("episodes", 0))
    predator_replay = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    prey_replay = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    if checkpoint is not None:
        if "training_state" in checkpoint:
            state = checkpoint["training_state"]
            for name, agent, replay in (("predator", predator_agent, predator_replay),
                                        ("prey", prey_agent, prey_replay)):
                agent.load_state_dict(state[name + "_agent"])
                replay.load_state_dict(state[name + "_replay"])
            rng = state["numpy_rng"]
            np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
            torch.set_rng_state(state["torch_rng"])
            if device == "cuda" and state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            restored_training_state = True
            print(f"Restored policies, critics, optimizers and replay buffers from {init_model}.")
            del state
        else:
            print(f"Loaded both policies from {init_model}; critics and replay buffers start fresh.")
        del checkpoint

    print(
        f"NvsN PPS: n_p={env.n_p}, n_e={env.n_e}, obs_dim={obs_dim}, "
        f"shared_action_dim={act_dim}, L={env.L}, device={device}, "
        f"adam_backend={'fused' if predator_agent.fused_adam else 'default'}, "
        f"previous_env_steps={previous_steps}, planned_env_steps={planned_steps}, "
        f"target_cumulative_env_steps={previous_steps + planned_steps}, render_interval={args.render_interval}"
    )
    total_steps = 0
    best_min_distance = float("inf")
    rendered_episodes = []
    completed_episodes = 0
    episode_steps = 0

    def save_checkpoint():
        numpy_rng = np.random.get_state()
        checkpoint = {
            "predator_actor": predator_agent.actor.state_dict(),
            "prey_actor": prey_agent.actor.state_dict(),
            "training": {
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
                "environment": {
                    name: env.get_param(name)
                    for name in ("L", "n_p", "n_e", "dynamics_mode", "pursuer_strategy", "escaper_strategy")
                },
            },
            "training_state": {
                "predator_agent": predator_agent.state_dict(),
                "prey_agent": prey_agent.state_dict(),
                "predator_replay": predator_replay.state_dict(),
                "prey_replay": prey_replay.state_dict(),
                "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            },
        }
        temporary_path = output_model + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, output_model)
        print(f"Saved checkpoint: cumulative_env_steps={previous_steps + total_steps}, path={output_model}")

    for local_episode in range(1, args.episodes + 1):
        if total_steps >= planned_steps:
            break
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
        noise = max(args.noise_min, args.noise * (1.0 - local_episode / args.episodes))
        render_episode = args.render or (args.render_interval > 0 and episode % args.render_interval == 0)
        if render_episode:
            rendered_episodes.append(episode)
            print(f"Rendering episode {episode}")

        for _ in range(episode_steps):
            if render_episode:
                env.render(mode="human")

            if total_steps < args.start_steps and init_model is None:
                predator_actions = np.random.uniform(-1.0, 1.0, size=(env.n_p, act_dim)).astype(np.float32).T
                prey_actions = np.random.uniform(-1.0, 1.0, size=(env.n_e, act_dim)).astype(np.float32).T
            else:
                predator_actions = predator_agent.act_batch(predator_states, noise_std=noise).T
                prey_actions = prey_agent.act_batch(prey_states, noise_std=noise).T

            action = {"predator": predator_actions, "prey": prey_actions}
            next_obs, rewards, _, _ = env.step(action)
            pred_rewards = rewards["predator"]
            prey_rewards = rewards["prey"]
            pred_distances = predator_prey_distances(env)
            caught = bool(np.min(pred_distances) <= env.size_p + env.size_e)
            next_predator_states = next_obs["predator"]
            next_prey_states = next_obs["prey"]
            caught_once = caught_once or caught
            for pred_i in range(env.n_p):
                predator_replay.push(
                    predator_states[pred_i],
                    predator_actions[:, pred_i],
                    pred_rewards[pred_i],
                    next_predator_states[pred_i],
                    False,
                )
            for prey_i in range(env.n_e):
                prey_replay.push(
                    prey_states[prey_i],
                    prey_actions[:, prey_i],
                    prey_rewards[prey_i],
                    next_prey_states[prey_i],
                    False,
                )

            predator_states = next_predator_states
            prey_states = next_prey_states
            episode_reward += float(np.mean(pred_rewards))
            episode_prey_reward += float(np.mean(prey_rewards))
            min_distance = min(min_distance, float(np.min(pred_distances)))
            total_steps += 1

            if restored_training_state or total_steps >= args.start_steps:
                for _ in range(args.updates_per_step):
                    if len(predator_replay) >= args.batch_size:
                        predator_agent.update(predator_replay, args.batch_size, return_losses=False)
                    if len(prey_replay) >= args.batch_size:
                        prey_agent.update(prey_replay, args.batch_size, return_losses=False)

        best_min_distance = min(best_min_distance, min_distance)
        if episode % 10 == 0 or caught_once:
            print(
                f"episode={episode:04d} pred_reward={episode_reward:8.3f} "
                f"prey_reward={episode_prey_reward:8.3f} "
                f"min_dist={min_distance:.4f} best={best_min_distance:.4f} "
                f"caught={caught_once} noise={noise:.3f} cumulative_steps={previous_steps + total_steps}"
            )
        completed_episodes = local_episode
        if args.checkpoint_interval and episode % args.checkpoint_interval == 0 and total_steps < planned_steps:
            save_checkpoint()

    save_checkpoint()
    env.close()
    print(f"Saved {args.model_path}; completed_env_steps={total_steps}, "
          f"cumulative_env_steps={previous_steps + total_steps}, rendered_episodes={rendered_episodes}")


if __name__ == "__main__":
    train(get_args())
