"""显式维护每世界双帧、GRU状态与10Hz调度，不向actor传递帧状态特征。"""

import torch

from .depth_sensor import DepthSensor


class VisualRuntime:
    def __init__(self, task, encoder):
        self.task, self.encoder = task, encoder
        self.sensor = DepthSensor(task)
        self.steps, self.normal_captures, self.reset_captures = 0, 0, 0
        n, device = task.config.num_envs, task.device
        self.pair = torch.zeros(n, 2, 48, 64, device=device, dtype=torch.float16)
        self.hidden = torch.zeros(n, 128, device=device)
        self.hidden_before = torch.zeros_like(self.hidden)
        self.latent = torch.zeros(n, 64, device=device)
        self.reset(torch.arange(n, device=device))

    @torch.no_grad()
    def reset(self, ids):
        depth = self.sensor.capture()
        self.pair[ids, 0] = depth[ids]
        self.pair[ids, 1] = depth[ids]
        self.hidden[ids] = 0
        self.hidden_before[ids] = 0
        feature, hidden = self.encoder(self.pair[ids], self.task.proprio()[ids], self.hidden[ids])
        self.latent[ids], self.hidden[ids] = feature, hidden
        self.reset_captures += 1

    @torch.no_grad()
    def update(self):
        self.steps += 1
        if self.steps % self.sensor.config.update_steps:
            return False
        depth = self.sensor.capture()
        self.pair[:, 0].copy_(self.pair[:, 1])
        self.pair[:, 1].copy_(depth)
        self.hidden_before.copy_(self.hidden)
        self.latent, self.hidden = self.encoder(self.pair, self.task.proprio(), self.hidden)
        self.normal_captures += 1
        return True
