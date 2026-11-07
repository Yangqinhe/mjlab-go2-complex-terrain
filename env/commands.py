"""根据路线横向偏差和航向生成转向指令，保留固定vx及原有3维指令接口。"""

import torch


def route_command(command, lateral, yaw, config):
    command = command.clone()
    desired = torch.atan2(-config.lateral_gain*lateral, torch.ones_like(lateral))
    error = desired-yaw
    error = torch.atan2(error.sin(), error.cos())
    command[:, 2] = (config.heading_gain*error).clamp(-config.max_yaw_rate, config.max_yaw_rate)
    return command


class VelocitySchedule:
    """缓存每步真实生效的连续速度指令；采样目标与加速度限制分开。"""

    def __init__(self, task):
        self.task, self.config = task, task.config.command
        n, device = task.config.num_envs, task.device
        self.generator = torch.Generator(device=device).manual_seed(task.config.seed+101)
        self.target = torch.zeros(n, 3, device=device)
        self.applied = torch.zeros_like(self.target)
        self.remaining = torch.zeros(n, dtype=torch.long, device=device)
        self.anchor = torch.zeros(n, dtype=torch.bool, device=device)
        self.stop_script = torch.zeros_like(self.anchor)
        self.elapsed = torch.zeros(n, device=device)

    def sample(self, ids):
        from domain_randomization.dynamics import uniform
        c, device = self.config, self.task.device
        def draw(low, high):
            return uniform((len(ids),), low, high, device, self.generator)
        self.target[ids, 0] = draw(*c.vx)
        self.target[ids, 1] = 0.
        self.target[ids, 2] = draw(*c.wz)
        stops = draw(0., 1.) < c.stop_fraction
        self.target[ids[stops]] = 0.
        self.target[ids[self.anchor[ids]]] = torch.tensor([.3, 0., 0.], device=device)
        self.remaining[ids] = (draw(4., 8.)/self.task.dt).long()
        if c.stop_hold_seconds is not None:
            hold=(draw(*c.stop_hold_seconds)/self.task.dt).long()
            self.remaining[ids[stops & ~self.anchor[ids]]] = hold[stops & ~self.anchor[ids]]

    def reset(self, ids):
        self.anchor[ids] = torch.rand(len(ids), device=self.task.device,
                                      generator=self.generator) < self.config.anchor_fraction
        self.sample(ids)
        self.elapsed[ids] = 0
        if self.config.stop_script_fraction:
            self.stop_script[ids] = ((torch.rand(len(ids),device=self.task.device,generator=self.generator)
                                     < self.config.stop_script_fraction) & ~self.anchor[ids])
            self.target[ids[self.stop_script[ids]]] = torch.tensor([.3,0.,0.],device=self.task.device)
            self.remaining[ids[self.stop_script[ids]]] = 1000000
        # 保持基线的初始向前步态，再平滑进入随机指令；停车目标仍有专门概率质量。
        self.applied[ids] = torch.tensor([.3, 0., 0.], device=self.task.device)

    def step(self):
        self.elapsed += self.task.dt
        self.remaining -= 1
        ids = (self.remaining <= 0).nonzero(as_tuple=False).flatten()
        if len(ids):
            self.sample(ids)
        if self.config.stop_script_fraction:
            self.target[self.stop_script] = 0.
            walking=self.stop_script & (self.elapsed < 6.)
            self.target[walking,0] = .3
        limit = torch.tensor([self.config.acceleration, self.config.acceleration,
                              self.config.yaw_acceleration], device=self.task.device)*self.task.dt
        self.applied += (self.target-self.applied).clamp(-limit, limit)


def nominal_flat_training_config(source, num_envs, seed):
    """复用正常平地审计的场景设置，用独立训练seed保持已有步态。"""
    from copy import deepcopy
    config=deepcopy(source)
    config.num_envs, config.seed=num_envs, seed
    # 独立平地集中学习接触，及时重置以避免长期侧漂撞到装饰物。
    config.episode_seconds=12.5
    config.terrain.seed=seed
    config.randomization.seed=200000+seed
    config.randomization.enabled=False
    config.camera.pitch_down_degrees=25.
    config.terrain.obstacles=False
    config.terrain.levels, config.terrain.layouts=2, 1
    config.terrain.width, config.terrain.flat_length, config.terrain.spawn_x=20., 20., 10.
    config.route.enabled=False
    config.command.mode='legacy'
    config.command.vx=(.3,.3)
    config.command.vy=config.command.wz=(0.,0.)
    return config


def free_training_config(source, d435i_scenes=False):
    """自由速度任务显式设置启停，防止继承路线阶段的stop_fraction=0。"""
    from copy import deepcopy
    config=deepcopy(source)
    config.route.enabled=False
    config.command.mode='free'
    config.command.vx=(-.3,.3)
    config.command.vy=(0.,0.)
    config.command.wz=(-.5,.5)
    config.command.stop_fraction=.2
    config.command.anchor_fraction=.2
    config.episode_seconds=24.
    config.terrain.obstacles=False
    config.terrain.levels=2
    config.terrain.layouts=4 if d435i_scenes else 1
    config.terrain.width=20.
    config.terrain.flat_length=20.
    config.terrain.spawn_x=10.
    config.randomization.enabled=True
    config.randomization.payload_max=.5
    if d435i_scenes:
        config.terrain.corridor_width=(12.,16.)
    return config
