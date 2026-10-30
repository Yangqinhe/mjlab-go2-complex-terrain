"""两层 MLP：固定观测缩放、Gaussian actor 和标量 critic。"""
import math

import torch
from torch import nn


def observation_scale(state_dim):
    """不改变环境观测；网络内部缩放单位，训练与播放使用同一份 buffer。"""
    scale = torch.ones(state_dim)
    if state_dim == 58:
        scale[:3] = 2.0
        scale[3:6] = 0.25
        scale[9:12] = torch.tensor([2.0, 2.0, 0.25])
        scale[24:36] = 0.05  # 关节速度 rad/s
        scale[57] = 0.5  # 步频 Hz
    return scale


def initialize(layer, gain=math.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)


class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.register_buffer('obs_scale', observation_scale(state_dim))
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(0.3)))
        initialize(self.fc1)
        initialize(self.fc_hidden)
        initialize(self.fc_mu, 0.01)  # 初始均值接近零动作的站立 PD

    def forward(self, state):
        hidden = torch.tanh(self.fc1(state * self.obs_scale))
        hidden = torch.tanh(self.fc_hidden(hidden))
        mu = torch.tanh(self.fc_mu(hidden))
        std = self.log_std.clamp(-3.0, 0.0).exp().expand_as(mu)
        return mu, std


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.register_buffer('obs_scale', observation_scale(state_dim))
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        initialize(self.fc1)
        initialize(self.fc_hidden)
        initialize(self.fc2, 1.0)

    def forward(self, state):
        hidden = torch.tanh(self.fc1(state * self.obs_scale))
        hidden = torch.tanh(self.fc_hidden(hidden))
        return self.fc2(hidden)
