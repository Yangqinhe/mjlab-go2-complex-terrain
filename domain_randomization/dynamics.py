"""按世界重置刚性负载、组合惯性、接触摩擦和原生PD参数，保留标称快照。"""

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix


def uniform(shape, low, high, device, generator):
    return torch.rand(shape, device=device, generator=generator)*(high-low)+low


def combine_payload(mass, com, tensor, payload, position):
    """两刚体平行轴合成；输入惯量在机身坐标系中表达，输出主惯量/四元数。"""
    n = len(payload)
    eye = torch.eye(3, device=payload.device).expand(n, 3, 3)
    total = mass+payload
    center = (mass[:, None]*com+payload[:, None]*position)/total[:, None]
    def shift(delta):
        return delta.square().sum(-1)[:, None, None]*eye-delta[:, :, None]*delta[:, None, :]
    size = torch.tensor([.20, .12, .08], device=payload.device)
    box = payload[:, None]*(size.square().sum()-size.square())/12
    tensor = tensor+mass[:, None, None]*shift(com-center)
    tensor += torch.diag_embed(box)+payload[:, None, None]*shift(position-center)
    values, vectors = torch.linalg.eigh(tensor)
    # 主轴矩阵必须是右手系，才可表示为旋转四元数。
    vectors[:, :, 2] *= torch.linalg.det(vectors)[:, None]
    return total, center, values, quat_from_matrix(vectors)


@requires_model_fields('body_mass', 'body_ipos', 'body_inertia', 'body_iquat',
                       'geom_friction', 'actuator_gainprm',
                       'actuator_biasprm', 'actuator_forcerange', 'dof_damping', 'dof_frictionloss',
                       recompute=RecomputeLevel.set_const)
def reset_dynamics(env, env_ids, config):
    ids = env_ids.long()
    device, n = env.device, len(ids)
    if not hasattr(env, 'dr_generator'):
        env.dr_generator = torch.Generator(device=device).manual_seed(config.seed)
        env.dr_samples = torch.zeros(env.num_envs, 8, device=device)
        env.dr_nominal = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    def draw(shape, low, high):
        return uniform(shape, low, high, device, env.dr_generator)
    nominal = draw((n,), 0., 1.) < config.nominal_fraction
    env.dr_nominal[ids] = nominal
    base = env.sim.mj_model.body('robot/base').id
    default = env.sim.get_default_field
    mass, com = default('body_mass')[base], default('body_ipos')[base]
    inertia = default('body_inertia')[base]
    rotation = matrix_from_quat(default('body_iquat')[base])
    scale = draw((n,), *config.mass_scale)
    scale[nominal] = 1.
    payload = draw((n,), 0., config.payload_max)
    payload[nominal] = 0.
    position = draw((n, 3), -1., 1.)*torch.tensor([.03, .01, .02], device=device)
    position[:, 2] += .12
    # 密度倍率先同时作用本体mass/inertia，再叠加未缩放的实际负载。
    center0 = com+draw((n, 3), -config.com_error, config.com_error)
    center0[nominal] = com
    base_tensor = scale[:, None, None]*(rotation@torch.diag(inertia)@rotation.T)
    total, center, values, quaternion = combine_payload(mass*scale, center0, base_tensor, payload, position)
    model = env.sim.model
    model.body_mass[ids, base] = total
    model.body_ipos[ids, base] = center
    model.body_inertia[ids, base] = values
    model.body_iquat[ids, base] = quaternion
    # 无扰动环境精确恢复标称主轴，避免特征分解导致无意义的表示差异。
    clean = ids[nominal]
    for name in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat'):
        getattr(model, name)[clean, base] = default(name)[base]
    feet = torch.tensor([env.sim.mj_model.geom('robot/'+leg).id
                         for leg in ('FL', 'FR', 'RL', 'RR')], device=device)
    mu = draw((n,), *config.friction)
    mu[nominal] = default('geom_friction')[feet[0], 0]
    model.geom_friction[ids[:, None], feet] = default('geom_friction')[feet]
    model.geom_friction[ids[:, None], feet, 0] = mu[:, None]
    kp = draw((n, 12), *config.gain_scale)
    kd = draw((n, 12), *config.gain_scale)
    strength = draw((n, 12), *config.effort_scale)
    kp[nominal], kd[nominal], strength[nominal] = 1., 1., 1.
    controls = torch.cat([a.global_ctrl_ids for a in env.scene['robot'].actuators])
    model.actuator_gainprm[ids[:, None], controls, 0] = default('actuator_gainprm')[controls, 0]*kp
    model.actuator_biasprm[ids[:, None], controls, 1] = default('actuator_biasprm')[controls, 1]*kp
    model.actuator_biasprm[ids[:, None], controls, 2] = default('actuator_biasprm')[controls, 2]*kd
    model.actuator_forcerange[ids[:, None], controls] = default('actuator_forcerange')[controls]*strength[:, :, None]
    damping = draw((n, 1), *config.damping_scale)
    damping[nominal] = 1.
    model.dof_damping[ids] = default('dof_damping')*damping
    model.dof_frictionloss[ids] = default('dof_frictionloss')*damping
    env.dr_samples[ids] = torch.cat([payload[:, None], position, scale[:, None], mu[:, None],
                                    kp.mean(-1, keepdim=True), kd.mean(-1, keepdim=True)], -1)
