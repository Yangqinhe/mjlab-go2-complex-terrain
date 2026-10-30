"""构造固定顺序的 48 维观测，保留三维速度指令供后期扩展。"""

import torch


def policy_observation(env):
    robot = env.scene["robot"]
    return torch.cat([
        robot.data.root_link_lin_vel_b,             # 0:3，m/s，机体系
        robot.data.root_link_ang_vel_b,             # 3:6，rad/s，机体系
        robot.data.projected_gravity_b,             # 6:9
        env.command_manager.get_command("velocity"),  # 9:12，vx/vy/wz
        robot.data.joint_pos - robot.data.default_joint_pos,  # 12:24
        robot.data.joint_vel,                       # 24:36
        env.action_manager.action,                 # 36:48，生成新观测时为已执行动作
    ], dim=-1)
