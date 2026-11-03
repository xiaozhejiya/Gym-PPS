"""显式检查维度兼容性，保留旧版状态字典格式。"""
from contextlib import contextmanager
import os


@contextmanager
def checkpoint_lock():
    path = os.environ.get("NJP_CHECKPOINT_LOCK")
    if not path:
        yield
        return
    import fcntl
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def structure(env, hidden_dim, hidden_layers, hidden_activation="relu"):
    dim = getattr(env.unwrapped, "spatial_dim", 2)
    result = {"env_id": "PredatorPreySwarm3D-v0" if dim == 3 else "PredatorPreySwarm-v0",
            "spatial_dim": dim,
            "obs_dim": int(env.observation_space.spaces["predator"].shape[1]),
            "act_dim": int(env.action_space.spaces["predator"].shape[0]),
            "hidden_dim": hidden_dim, "hidden_layers": hidden_layers,
            "hidden_activation": hidden_activation,
            "observation_layout": "njp3d-v1" if dim == 3 else "njp2d-legacy",
            "action_layout": "rotation-vector-thrust-v1" if dim == 3 else "njp2d-legacy"}
    if getattr(env.unwrapped, "prey_death_enabled", False):
        result["capture_rule"] = "first-contact-remove-v1"
        if getattr(env.unwrapped, "capture_cooldown_steps", 0):
            result["capture_rule"] = "first-contact-cooldown-v1"
            result["capture_cooldown_steps"] = env.unwrapped.capture_cooldown_steps
    return result


def validate_checkpoint(checkpoint, expected, policy_only=False, allow_capture_change=False):
    saved = checkpoint.get("training", {}).get("structure")
    if saved is None:
        if expected["spatial_dim"] != 2:
            raise ValueError("旧版或二维检查点无法初始化三维实验，请训练新的三维模型")
    else:
        if (not (policy_only or allow_capture_change) and saved.get("capture_rule", "contact-v1")
                != expected.get("capture_rule", "contact-v1")):
            raise ValueError("检查点 capture_rule 不匹配，请使用 --policy_only 迁移 Actor，"
                             "并为变化后的奖励重新初始化 Critic 和经验池")
        for field, value in expected.items():
            if field in ("capture_rule", "capture_cooldown_steps") and (policy_only or allow_capture_change):
                continue
            actual = saved.get(field, "relu" if field == "hidden_activation" else None)
            if actual != value:
                raise ValueError("检查点 {} 不匹配：已保存={!r}，当前请求={!r}".format(field, actual, value))
    for species in ("predator", "prey"):
        actor = checkpoint[species + "_actor"]
        weights = [value for name, value in actor.items() if name.endswith(".weight")]
        if (not weights or weights[0].shape[1] != expected["obs_dim"]
                or weights[-1].shape[0] != expected["act_dim"]
                or weights[0].shape[0] != expected["hidden_dim"]
                or len(weights) != expected["hidden_layers"] + 1):
            raise ValueError("检查点策略维度或网络深度与请求的环境不匹配")
