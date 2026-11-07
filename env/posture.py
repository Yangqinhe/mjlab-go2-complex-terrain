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


def configure_normal_gait(config, scale):
    configure_front_posture(config, scale)
    config.training_reward.all_hip_deviation = -.2*scale
    config.training_reward.foot_landing = -.4*scale
    config.training_reward.slip = -2.*scale


def normal_foot_terms(task, command, flat):
    w = task.weights
    feet = foot_positions_body(task)
    home = feet.new_tensor(list(zip(w.foot_home_x, w.foot_home_y)))
    tolerance = feet.new_tensor([w.foot_landing_x_tolerance, w.foot_landing_y_tolerance])
    error = ((feet[..., :2]-home).abs()-tolerance).clamp(min=0)/tolerance
    contact = task.env.scene['feet'].data.force[:, task.foot_order, 2].abs()>task.config.gait.contact_force
    touchdown = contact & ~task.gait.previous_contact & (task.gait.air_time>=task.config.gait.minimum_swing)
    # 落点惩罚只作用在落地瞬间；支撑期额外约束横向位置，不固定摆动相轨迹。
    landing = (error.square().sum(-1)*touchdown).sum(-1)
    lateral = (error[..., 1].square()*contact).mean(-1)
    ids = [i for i, name in enumerate(task.robot.joint_names) if name.endswith('_hip_joint')]
    hip = ((task.robot.data.joint_pos[:, ids].abs()-w.front_hip_deadband).clamp(min=0)
           /w.front_hip_deadband).square().mean(-1)
    active = flat & (command[:, 2].abs()<.1)
    return dict(all_hip_posture=w.all_hip_deviation*hip*active,
                normal_foot_landing=w.foot_landing*(landing+lateral)*active)


def foot_velocities_world(task, flat_contact_point=False):
    r = task.robot.data
    feet = task.env.sim.data.geom_xpos[:, task.foot_ids]
    if flat_contact_point:
        # 平地球形足底接触点近似，避免把球形足滚动的中心速度全部算成滑移。
        feet=feet.clone();feet[...,2]-=task.foot_radius
    lever = feet-r.body_link_pos_w[:, task.calf_ids]
    return r.body_link_lin_vel_w[:, task.calf_ids]+torch.cross(
        r.body_link_ang_vel_w[:, task.calf_ids], lever, dim=-1)
