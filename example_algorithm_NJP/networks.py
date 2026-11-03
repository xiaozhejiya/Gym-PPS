import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, hidden_layers=2,
                 hidden_activation="relu"):
        super().__init__()
        if hidden_layers not in (2, 3):
            raise ValueError("hidden_layers must be 2 or 3")
        if hidden_activation not in ("relu", "silu"):
            raise ValueError("hidden_activation must be relu or silu")
        activation = nn.ReLU if hidden_activation == "relu" else nn.SiLU
        layers = [nn.Linear(obs_dim, hidden_dim), activation()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), activation()))
        layers.extend((nn.Linear(hidden_dim, act_dim), nn.Tanh()))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    def forward_logits(self, obs):
        """Unbounded outputs for an optional soft penalty before Tanh."""
        return self.net[:-1](obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, hidden_layers=2,
                 hidden_activation="relu", layer_norm=False, geometry=False):
        super().__init__()
        if hidden_layers not in (2, 3):
            raise ValueError("hidden_layers must be 2 or 3")
        if hidden_activation not in ("relu", "silu"):
            raise ValueError("hidden_activation must be relu or silu")
        self.hidden_activation = hidden_activation
        self.layer_norm = bool(layer_norm)
        if geometry and (obs_dim != 178 or act_dim != 4):
            raise ValueError('Critic geometry requires the 178D njp3d-v1 prey layout and 4D actions')
        self.geometry = bool(geometry)
        self.fc1 = nn.Linear(obs_dim + act_dim + (8 if geometry else 0), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.extra_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                          for _ in range(hidden_layers - 2))
        self.q = nn.Linear(hidden_dim, 1)
        # Keep legacy parameter names, initialization and arithmetic unchanged.
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(hidden_layers)) if layer_norm else nn.ModuleList()

    def forward(self, obs, action):
        activation = F.relu if self.hidden_activation == "relu" else F.silu
        x = torch.cat([obs, action], dim=-1)
        if self.geometry:
            # First predator neighbor is nearest under the existing observation
            # sort. Derive all features from the same public observation only.
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
