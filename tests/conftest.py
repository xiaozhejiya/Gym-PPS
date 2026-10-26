import importlib
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def njp():
    # 示例使用 agent/train 等顶层模块名；测试前后隔离导入，避免与其他示例串扰。
    directory = Path(__file__).resolve().parents[1] / "example_algorithm_NJP"
    original_path = list(sys.path)
    names = ("agent", "arguments", "custom_env", "networks", "replay_buffer", "train")
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
