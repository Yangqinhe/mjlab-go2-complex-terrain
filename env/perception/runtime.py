"""显式维护每世界双帧、GRU状态与10Hz调度，不向actor传递帧状态特征。"""

import torch

from .depth_sensor import DepthSensor


class VisualRuntime:
    def __init__(self, task, encoder, sensor=None):
        self.task, self.encoder = task, encoder
        self.sensor = sensor or DepthSensor(task)
        self.steps, self.normal_captures, self.reset_captures = 0, 0, 0
        n, device = task.config.num_envs, task.device
        self.pair = torch.zeros(n, 2, 48, 64, device=device, dtype=torch.float16)
        self.hidden = torch.zeros(n, 128, device=device)
        self.hidden_before = torch.zeros_like(self.hidden)
        self.latent = torch.zeros(n, 64, device=device)
        self.dr = self.sensor.dr
        if self.dr.enabled:
            self.generator = torch.Generator(device=device).manual_seed(self.dr.seed+3)
            self.pending = torch.zeros(n, 48, 64, device=device, dtype=torch.float16)
            self.pending_proprio = torch.zeros(n, 45, device=device)
            self.ready = torch.full((n,), -1, device=device, dtype=torch.long)
            self.delayed_updates, self.dropped_frames = 0, 0
        self.reset(torch.arange(n, device=device))

    @torch.no_grad()
    def reset(self, ids):
        self.sensor.reset_mount(ids)
        depth = self.sensor.capture()
        if self.dr.enabled:
            self.ready[ids] = -1
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
        if self.dr.enabled:
            return self.update_randomized()
        if self.steps % self.sensor.config.update_steps:
            return False
        depth = self.sensor.capture()
        self.pair[:, 0].copy_(self.pair[:, 1])
        self.pair[:, 1].copy_(depth)
        self.hidden_before.copy_(self.hidden)
        self.latent, self.hidden = self.encoder(self.pair, self.task.proprio(), self.hidden)
        self.normal_captures += 1
        return True

    @torch.no_grad()
    def update_randomized(self):
        # 延迟按控制周期交付，深度与采图本体作为同一个包保存。
        c = self.dr
        if self.steps % self.sensor.config.update_steps == 0:
            depth = self.sensor.capture()
            self.normal_captures += 1
            keep = torch.rand(len(depth), device=self.task.device, generator=self.generator) >= c.frame_dropout
            clean = self.task.env.dr_nominal
            keep |= clean
            self.dropped_frames += int((~keep).sum())
            delay = torch.randint(c.frame_delay_steps+1, (len(depth),), device=self.task.device, generator=self.generator)
            delay[clean] = 0
            self.pending[keep] = depth[keep]
            self.pending_proprio[keep] = self.task.proprio()[keep]
            self.ready[keep] = self.steps+delay[keep]
        ids = ((self.ready >= 0) & (self.ready <= self.steps)).nonzero(as_tuple=False).flatten()
        if not len(ids):
            return False
        self.pair[ids, 0] = self.pair[ids, 1]
        self.pair[ids, 1] = self.pending[ids]
        self.hidden_before[ids] = self.hidden[ids]
        feature, hidden = self.encoder(self.pair[ids], self.pending_proprio[ids], self.hidden[ids])
        self.latent[ids], self.hidden[ids] = feature, hidden
        self.delayed_updates += len(ids)
        self.ready[ids] = -1
        return True
