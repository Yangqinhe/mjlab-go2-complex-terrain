"""每世界维护带噪本体缓存和整控制周期动作延迟，不向策略泄露扰动参数。"""

import torch
from .dynamics import uniform


class RandomizationRuntime:
    def __init__(self, task):
        self.task, self.config = task, task.config.randomization
        n, device = task.config.num_envs, task.device
        self.generator = torch.Generator(device=device).manual_seed(self.config.seed+1)
        self.bias = torch.zeros(n, 12, device=device)
        self.gyro_bias = torch.zeros(n, 3, device=device)
        self.push_clock = torch.zeros(n, device=device)
        self.push_remaining = torch.zeros(n, device=device)
        self.push = torch.zeros(n, 1, 3, device=device)
        self.base_ids, _ = task.robot.find_bodies(['base'])
        self.rotation_error = torch.zeros(n, 3, device=device)
        self.delay = torch.zeros(n, dtype=torch.long, device=device)
        self.queue = torch.zeros(self.config.action_delay_steps+1, n, 12, device=device)
        self.observation = torch.zeros(n, 45, device=device)
        self.cursor = 0
        self.ids = torch.arange(n, device=device)

    def reset(self, ids):
        c, device = self.config, self.task.device
        self.bias[ids] = uniform((len(ids), 12), -c.joint_bias, c.joint_bias, device, self.generator)
        self.gyro_bias[ids] = uniform((len(ids), 3), -c.gyro_bias, c.gyro_bias, device, self.generator)
        self.push_clock[ids] = uniform((len(ids),), 8., 15., device, self.generator)
        self.push_remaining[ids] = 0
        self.push[ids] = 0
        self.rotation_error[ids] = uniform((len(ids), 3), -c.gravity_angle, c.gravity_angle, device, self.generator)
        self.delay[ids] = torch.randint(c.action_delay_steps+1, (len(ids),), device=device, generator=self.generator)
        clean = self.task.env.dr_nominal[ids]
        self.bias[ids[clean]], self.rotation_error[ids[clean]], self.delay[ids[clean]] = 0., 0., 0
        self.gyro_bias[ids[clean]] = 0
        self.queue[:, ids] = 0

    def action(self, action):
        if self.config.push_force:
            self.push_clock -= self.task.dt
            self.push_remaining -= self.task.dt
            self.push[self.push_remaining <= 0] = 0
            ids = ((self.push_clock <= 0) & ~self.task.env.dr_nominal).nonzero(as_tuple=False).flatten()
            if len(ids):
                force = self.config.push_force
                self.push[ids, 0, :2] = uniform((len(ids), 2), -force, force, self.task.device, self.generator)
                self.push_clock[ids] = uniform((len(ids),), 8., 15., self.task.device, self.generator)
                self.push_remaining[ids] = .1
            self.task.robot.write_external_wrench_to_sim(self.push, torch.zeros_like(self.push), body_ids=self.base_ids)
        self.queue[self.cursor] = action
        result = self.queue[(self.cursor-self.delay) % len(self.queue), self.ids].clone()
        self.cursor = (self.cursor+1) % len(self.queue)
        return result

    def update(self, raw, ids=None):
        ids = self.ids if ids is None else ids
        c, n = self.config, len(ids)
        noisy = raw[ids].clone()
        scale = torch.tensor([c.gyro_noise*.25]*3+[0.]*6+[c.position_noise]*12+
                             [c.velocity_noise*.05]*12+[0.]*12, device=raw.device)
        noise = torch.randn(n, 45, device=raw.device, generator=self.generator).clamp(-3, 3)*scale
        clean = self.task.env.dr_nominal[ids]
        noise[clean] = 0
        noisy += noise
        noisy[:, :3] += self.gyro_bias[ids]*.25
        noisy[:, 9:21] += self.bias[ids]
        # 小角度误差通过旋转近似再归一化，保证重力仍是单位方向。
        gravity = noisy[:, 3:6]
        gravity = gravity+torch.cross(self.rotation_error[ids], gravity, dim=-1)
        noisy[:, 3:6] = gravity/gravity.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.observation[ids] = noisy.clamp(-5, 5)
