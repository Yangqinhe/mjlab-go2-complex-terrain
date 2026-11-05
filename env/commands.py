"""根据路线横向偏差和航向生成转向指令，保留固定vx及原有3维指令接口。"""

import torch


def route_command(command, lateral, yaw, config):
    command = command.clone()
    desired = torch.atan2(-config.lateral_gain*lateral, torch.ones_like(lateral))
    error = desired-yaw
    error = torch.atan2(error.sin(), error.cos())
    command[:, 2] = (config.heading_gain*error).clamp(-config.max_yaw_rate, config.max_yaw_rate)
    return command
