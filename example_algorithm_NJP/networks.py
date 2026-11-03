import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, hidden_layers=2,
                 hidden_activation="relu"):
        super().__init__()
        if hidden_layers not in (2, 3):
            raise ValueError("hidden_layers 必须为 2 或 3")
        if hidden_activation not in ("relu", "silu"):
            raise ValueError("hidden_activation 必须为 relu 或 silu")
        activation = nn.ReLU if hidden_activation == "relu" else nn.SiLU
        layers = [nn.Linear(obs_dim, hidden_dim), activation()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), activation()))
        layers.extend((nn.Linear(hidden_dim, act_dim), nn.Tanh()))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    def forward_logits(self, obs):
        """返回 Tanh 之前的无界输出，用于可选的软惩罚。"""
        return self.net[:-1](obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, hidden_layers=2,
                 hidden_activation="relu", layer_norm=False, geometry=False):
        super().__init__()
        if hidden_layers not in (2, 3):
            raise ValueError("hidden_layers 必须为 2 或 3")
        if hidden_activation not in ("relu", "silu"):
            raise ValueError("hidden_activation 必须为 relu 或 silu")
        self.hidden_activation = hidden_activation
        self.layer_norm = bool(layer_norm)
        if geometry and (obs_dim != 178 or act_dim != 4):
            raise ValueError("Critic 几何特征要求 178 维 njp3d-v1 猎物观测布局和 4 维动作")
        self.geometry = bool(geometry)
        self.fc1 = nn.Linear(obs_dim + act_dim + (8 if geometry else 0), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.extra_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                          for _ in range(hidden_layers - 2))
        self.q = nn.Linear(hidden_dim, 1)
        # 保留原有参数名、初始化方式及运算顺序。
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(hidden_layers)) if layer_norm else nn.ModuleList()

    def forward(self, obs, action):
        activation = F.relu if self.hidden_activation == "relu" else F.silu
        x = torch.cat([obs, action], dim=-1)
        if self.geometry:
            # 现有观测排序中的首个捕食者邻居即最近邻；
            # 全部派生特征仅使用同一份公开观测。
            delta = obs[..., 6:9]
            distance = delta.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            direction = delta / distance.clamp_min(.095)
            clearance = (distance / .095 - 1).clamp(-1, 10)
            heading = obs[..., -3:]
            opponent_heading = obs[..., 9:12] + heading
            features = torch.cat((clearance, (1 + clearance.clamp_min(0)).reciprocal(),
                direction, (opponent_heading * direction).sum(dim=-1, keepdim=True),
                (heading * direction).sum(dim=-1, keepdim=True),
                (obs[..., 3:6] * direction).sum(dim=-1, keepdim=True)), dim=-1)
            x = torch.cat((x, features), dim=-1)
        if self.layer_norm:
            x = activation(self.norms[0](self.fc1(x)))
            x = activation(self.norms[1](self.fc2(x)))
            for index, layer in enumerate(self.extra_layers, 2):
                x = activation(self.norms[index](layer(x)))
        else:
            x = activation(self.fc1(x))
            x = activation(self.fc2(x))
            for layer in self.extra_layers:
                x = activation(layer(x))
        return self.q(x)
