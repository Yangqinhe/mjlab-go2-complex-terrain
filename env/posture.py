"""平地前腿姿态的机体系测量与软约束；默认权重为0，不改变原训练。"""

import torch


def foot_positions_body(task):
    r = task.robot.data
    v = task.env.sim.data.geom_xpos[:, task.foot_ids]-r.root_link_pos_w[:, None]
    # wxyz四元数逆旋转，避免把转向后的世界系足间距当成站宽。
    q = r.root_link_quat_w[:, None].expand(-1, 4, -1)
    cross = torch.cross(q[..., 1:], v, dim=-1)
    return v + 2*(torch.cross(q[..., 1:], cross, dim=-1)-q[..., :1]*cross)


def configure_front_posture(config, scale):
    w = config.training_reward
    w.front_hip_deviation = -.3*scale
    w.front_stance_width = -.2*scale


def front_posture_terms(task, command, flat):
    w = task.weights
    feet = foot_positions_body(task)
    width = feet[:, 0, 1]-feet[:, 1, 1]
    ids = [list(task.robot.joint_names).index(name) for name in ('FL_hip_joint', 'FR_hip_joint')]
    hips = task.robot.data.joint_pos[:, ids]
    # 只约束平地直行/站立；上下楼梯与明显转向保留外展余量，不硬裁动作。
    active = flat & (command[:, 2].abs() < .1)
    hip = ((hips.abs()-w.front_hip_deadband).clamp(min=0)/w.front_hip_deadband).square().mean(-1)
    spread = ((width-w.front_width_target).abs()-w.front_width_tolerance).clamp(min=0)
    return dict(front_hip_posture=w.front_hip_deviation*hip*active,
                front_stance_width=w.front_stance_width*(spread/w.front_width_tolerance).square()*active)
