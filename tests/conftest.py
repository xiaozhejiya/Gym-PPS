import importlib
from pathlib import Path
import sys

import gym
import pytest


@pytest.fixture(scope="module")
def njp():
    # 示例使用 agent/train 等顶层模块名；测试前后隔离导入，避免与其他示例串扰。
    directory = Path(__file__).resolve().parents[1] / "example_algorithm_NJP"
    original_path = list(sys.path)
    names = ("agent", "arguments", "custom_env", "cuda_graph", "networks", "replay_buffer", "train")
    original_modules = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        yield importlib.import_module("train")
    finally:
        sys.path[:] = original_path
        for name, module in original_modules.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


@pytest.fixture
def njp_training_env(njp, monkeypatch):
    # 训练回归测试固定场景参数，不依赖示例配置，也不需要修改用户的训练配置文件。
    parameters = {
        "L": 2, "n_p": 12, "n_e": 40,
        "dynamics_mode": "Polar", "is_periodic": True,
        "pursuer_strategy": "input", "escaper_strategy": "input",
        "penalize_control_effort": False, "render_traj": True,
    }

    def make_env():
        env = gym.make("PredatorPreySwarm-v0")
        for wrapper in (njp.MyReward, njp.MyAction, njp.MyObs):
            env = wrapper(env, parameters)
        return env

    monkeypatch.setattr(njp, "make_env", make_env)
