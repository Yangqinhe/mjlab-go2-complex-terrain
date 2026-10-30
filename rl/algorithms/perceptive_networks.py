"""176维教师/学生策略与独立244维critic，GRU状态由调用方持有。"""

import math

import torch
from torch import nn

from .ActorCritic import initialize
from rl.config import TrainConfig


def mlp(dims, final_tanh=False):
    layers = []
    for i, (before, after) in enumerate(zip(dims[:-1], dims[1:])):
        layer = nn.Linear(before, after)
        initialize(layer, 0.01 if i == len(dims)-2 and after == 12 else math.sqrt(2))
        layers.append(layer)
        if i < len(dims)-2:
            layers.append(nn.ELU())
        elif final_tanh:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 5, 2, 2), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ELU(), nn.Flatten(),
            nn.Linear(1536, 128), nn.ELU(), nn.Linear(128, 64), nn.Tanh(),
        )
        self.gru = nn.GRUCell(109, 128)
        self.projection = nn.Sequential(nn.Linear(128, 64), nn.Tanh())

    def forward(self, depth, proprio, hidden):
        feature = self.cnn(depth.float())
        hidden = self.gru(torch.cat([feature, proprio], -1), hidden)
        return self.projection(hidden), hidden


class PerceptiveActor(nn.Module):
    def __init__(self, mode="teacher", config=None):
        super().__init__()
        config = config or TrainConfig()
        self.mode = mode
        self.history = mlp(config.history_dims, final_tanh=True)
        self.velocity_head = mlp(config.velocity_dims)
        self.actor = mlp(config.actor_dims, final_tanh=True)
        self.log_std = nn.Parameter(torch.full((12,), math.log(config.initial_std)))
        self.terrain_encoder = mlp(config.height_dims, final_tanh=True) if mode == "teacher" else None
        self.visual_encoder = DepthEncoder() if mode == "student" else None

    def forward(self, states):
        # states=[本体45, 历史450, 教师高度187或冻结视觉64]；不含age/valid。
        p, history, terrain = states[:, :45], states[:, 45:495], states[:, 495:]
        z_p = self.history(history)
        velocity = (2*self.velocity_head(z_p)).clamp(-4, 4)
        z_v = self.terrain_encoder(terrain) if self.mode == "teacher" else terrain
        fused = torch.cat([p, z_p, z_v, velocity], -1)
        mu = self.actor(fused)
        return mu, self.log_std.clamp(-3, 0).exp().expand_as(mu)

    def velocity(self, states):
        return self.velocity_head(self.history(states[:, 45:495]))

    def copy_common(self, teacher):
        self.history.load_state_dict(teacher.history.state_dict())
        self.velocity_head.load_state_dict(teacher.velocity_head.state_dict())
        self.actor.load_state_dict(teacher.actor.state_dict())
        self.log_std.data.copy_(teacher.log_std.data)


class PrivilegedCritic(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.net = mlp((config or TrainConfig()).critic_dims)

    def forward(self, state):
        return self.net(state)
