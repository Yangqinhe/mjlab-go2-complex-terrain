"""集中保存配置、checkpoint与CSV；训练代码不写入PPO内部。"""

from dataclasses import asdict
from datetime import datetime
import csv
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch


ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def deployment_config(task):
    return dict(joint_order=list(task.robot.joint_names),
                default_joint_pos=task.robot.data.default_joint_pos[0].cpu().tolist(),
                camera=asdict(task.config.camera), proprio_dim=45, history_steps=10,
                fusion_dim=176, depth_geometry_groups=([0,4] if task.config.terrain.surroundings else [0]),
                observation_version='perceptive_176_v1',
                route=asdict(task.config.route), command=asdict(task.config.command),
                randomization=asdict(task.config.randomization),
                gait=asdict(task.config.gait),
                control_dt=task.dt, physics_dt=task.config.physics_dt,
                action_scale=task.config.action_scale, kp=task.config.kp, kd=task.config.kd,
                joint_target_limits_by_pattern=task.env.cfg.actions['joint_pos'].clip,
                proprio_fields=[dict(name='angular_velocity_body', dims=3, scale=.25),
                    dict(name='projected_gravity', dims=3, scale=1.),
                    dict(name='velocity_command', dims=3, scale=[2., 2., .25]),
                    dict(name='joint_position_minus_home', dims=12, scale=1.),
                    dict(name='joint_velocity', dims=12, scale=.05),
                    dict(name='previous_executed_action', dims=12, scale=1.)],
                proprio_clip=5., raw_action_clip=1., velocity_feature_scale=2.,
                depth_normalization='(clip(depth_m, near, far)-near)/(far-near)-0.5')


def new_run(stage, env_config, train_config):
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    path = ROOT/'outputs'/f'{stamp}_{stage}_seed{env_config.seed}'
    (path/'checkpoints').mkdir(parents=True)
    (path/'evaluations').mkdir()
    save_json(path/'config.json', dict(env=asdict(env_config), train=asdict(train_config), stage=stage))
    save_json(path/'randomization.json', asdict(env_config.randomization))
    terrain = env_config.terrain
    save_json(path/'terrain_manifest.json', dict(seed=terrain.seed,
        obstacles=terrain.obstacles, length=terrain.length, width=terrain.width,
        regions=terrain.regions(), layouts=terrain.layouts,
        surroundings=terrain.surroundings, corridor_width=terrain.corridor_width,
        stair_heights=[terrain.step_height(i/max(terrain.levels-1, 1))
                       for i in range(terrain.levels)]))
    return path


def restore_curriculum(task, saved):
    if not saved or 'curriculum_levels' not in saved:
        return
    old = saved['env_config']['terrain']
    if not old.get('obstacles', True) or not task.config.terrain.obstacles:
        return  # 平地的虚拟等级不对应实际台阶高度。
    levels = saved['curriculum_levels'].to(task.device).float()
    height = old['step_height_min']+levels/(old['levels']-1)*(old['step_height_max']-old['step_height_min'])
    indices = torch.linspace(0, len(height)-1, task.config.num_envs, device=task.device).long()
    course = task.config.terrain
    new_levels = ((height[indices]-course.step_height_min)/(course.step_height_max-course.step_height_min)
                  *(course.levels-1)).round().long().clamp(0, course.levels-1)
    terrain = task.env.scene.terrain
    terrain.terrain_levels[:] = new_levels
    terrain.env_origins[:] = terrain.terrain_origins[new_levels, terrain.terrain_types]
    task.reset()


def save_checkpoint(path, agent, iteration, env_config, train_config, stage, task=None, **extra):
    payload = dict(format_version=3, observation_version='perceptive_176_v1',
                   network_version='depth48x64_gru128_fusion176_v1', stage=stage,
                   iteration=iteration, env_config=asdict(env_config),
                   train_config=asdict(train_config), actor=agent.actor.state_dict(),
                   critic=agent.critic.state_dict(), actor_optimizer=agent.actor_optimizer.state_dict(),
                   critic_optimizer=agent.critic_optimizer.state_dict(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), **extra)
    if task is not None:
        payload['curriculum_levels'] = task.env.scene.terrain.terrain_levels.cpu()
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def env_from_dict(data):
    from env.config import EnvConfig, VelocityCommandConfig, TrainingRewardConfig, RouteCommandConfig, GaitConfig
    from terrains.config import TerrainConfig
    data = dict(data)
    data['command'] = VelocityCommandConfig(**data['command'])
    data['terrain'] = TerrainConfig(**data['terrain'])
    data['training_reward'] = TrainingRewardConfig(**data.get('training_reward', {}))
    data['route'] = RouteCommandConfig(**data.get('route', {}))
    data['gait'] = GaitConfig(**data.get('gait', {}))
    from domain_randomization.config import RandomizationConfig
    data['randomization'] = RandomizationConfig(**data.get('randomization', {}))
    from env.perception.config import DepthConfig
    data['camera'] = DepthConfig(**data.get('camera', {}))
    return EnvConfig(**data)


def load_policy(checkpoint, device='cuda:0'):
    from rl.config import TrainConfig
    from rl.algorithms.perceptive_networks import PerceptiveActor
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get('observation_version') != 'perceptive_176_v1':
        raise ValueError('Checkpoint observation version is not perceptive_176_v1')
    mode = 'teacher' if saved['stage'] == 'teacher' else 'student'
    actor = PerceptiveActor(mode, TrainConfig(**saved['train_config'])).to(device)
    actor.load_state_dict(saved['actor'])
    actor.eval()
    return actor, env_from_dict(saved['env_config']), saved
