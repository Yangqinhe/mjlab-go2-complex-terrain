"""提供加载阶段的速度跟踪、姿态和动作平滑奖励，不代表最终步态奖励。"""

import torch


def tracking_velocity(env, error_scale):
    command = env.command_manager.get_command("velocity")
    velocity = env.scene["robot"].data.root_link_lin_vel_b
    return torch.exp(-((command[:, :2] - velocity[:, :2]) ** 2).sum(-1) / error_scale)


def orientation_error(env):
    return env.scene["robot"].data.projected_gravity_b[:, :2].square().sum(-1)


def action_rate(env):
    manager = env.action_manager
    return (manager.action - manager.prev_action).square().sum(-1)
