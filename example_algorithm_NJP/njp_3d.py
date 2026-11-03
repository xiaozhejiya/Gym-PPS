"""3D preset, deliberately independent of both existing 2D experiments."""
import json
import math
from pathlib import Path

import gym

from custom_env import MyAction, MyObs, MyReward

CONFIG_PATH = Path(__file__).with_name("experiments") / "njp_3d.json"
NO_ENERGY_CONFIG_PATH = CONFIG_PATH.with_name("njp_3d_no_energy.json")
SILU_NO_ENERGY_CONFIG_PATH = CONFIG_PATH.with_name("njp_3d_silu_no_energy.json")
WIDE_OBS_CONFIG_PATH = CONFIG_PATH.with_name("njp_3d_silu_wide_obs.json")
CAPTURE_CONFIG_PATH = CONFIG_PATH.with_name("njp_3d_silu_capture.json")
RULE_PREY_CONFIG_PATH = CONFIG_PATH.with_name("njp_3d_silu_rule_prey.json")

SPAWN_ARGUMENTS = ("prey_spawn_radius", "predator_spawn_inner", "predator_spawn_outer", "spawn_mode")
ENV_ARGUMENTS = SPAWN_ARGUMENTS + ("capture_cooldown_steps",)


def add_spawn_arguments(parser):
    parser.add_argument("--spawn_mode", choices=("uniform", "cluster_shell"), default=None)
    parser.add_argument("--prey_spawn_radius", type=float, default=None)
    parser.add_argument("--predator_spawn_inner", type=float, default=None)
    parser.add_argument("--predator_spawn_outer", type=float, default=None)
    parser.add_argument("--capture_cooldown_steps", type=int, default=None,
                        help="Capture interval in environment steps; zero preserves immediate multi-capture.")


def spawn_options(args):
    return {name: getattr(args, name) for name in ENV_ARGUMENTS
            if getattr(args, name, None) is not None}


def settings(arena_L=None, n_p=None, n_e=None, evaluation=False, no_energy=False, silu=False,
             fixed_prey=False, wide_obs=False, capture=False, **spawn):
    path = (CAPTURE_CONFIG_PATH if capture else
            WIDE_OBS_CONFIG_PATH if wide_obs else
            RULE_PREY_CONFIG_PATH if fixed_prey else
            SILU_NO_ENERGY_CONFIG_PATH if silu else
            NO_ENERGY_CONFIG_PATH if no_energy else CONFIG_PATH)
    config = json.loads(path.read_text())
    params = config["environment"]
    if arena_L is not None:
        if not math.isfinite(arena_L) or arena_L <= max(params["size_p"], params["size_e"]):
            raise ValueError("arena_L must be finite and exceed the largest body radius")
        params["L"] = float(arena_L)
    for name, value in (("n_p", n_p), ("n_e", n_e)):
        if value is not None:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")
            params[name] = value
    if set(spawn) - set(ENV_ARGUMENTS):
        raise TypeError("Unknown environment options: " + str(sorted(set(spawn) - set(ENV_ARGUMENTS))))
    # Set the mode last: wrapper setters validate after each parameter assignment.
    for name in ENV_ARGUMENTS:
        if spawn.get(name) is not None:
            params[name] = spawn[name]
    if evaluation:
        params["render_traj"] = True
    return config


def defaults(evaluation=False, no_energy=False, silu=False, fixed_prey=False, wide_obs=False,
             capture=False):
    config = settings(no_energy=no_energy, silu=silu, fixed_prey=fixed_prey,
                      wide_obs=wide_obs, capture=capture)
    if not evaluation:
        return config["training"]
    return dict(config["evaluation"], hidden_dim=config["training"]["hidden_dim"],
                hidden_layers=config["training"]["hidden_layers"],
                hidden_activation=config["training"].get("hidden_activation", "relu"),
                model_path=config["training"]["model_path"])


def make_env(evaluation=False, arena_L=None, n_p=None, n_e=None, no_energy=False, silu=False,
             fixed_prey=False, wide_obs=False, capture=False, **spawn):
    params = settings(arena_L, n_p, n_e, evaluation, no_energy=no_energy, silu=silu,
                      fixed_prey=fixed_prey, wide_obs=wide_obs, capture=capture, **spawn)["environment"]
    env = gym.make("PredatorPreySwarm3D-v0")
    return MyObs(MyAction(MyReward(env, params), params), params)
