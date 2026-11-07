"""冻结感知和本体编码，监督拟合平地/复杂地形参考动作，再交回PPO续训。"""

import argparse
import json
import shutil
import time

import torch
import torch.nn.functional as F

from env.gait import flat_terrain_mask
from env.perception.config import DepthConfig
from env.perception.depth_sensor import DepthSensor
from env.perception.runtime import VisualRuntime
from env.training_task import Go2Task
from rl.algorithms.perceptive_networks import PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import TrainConfig
from rl.run_io import append_csv, deployment_config, load_policy, new_run, save_checkpoint, save_json


def reference_visual(task, policy, saved):
    camera=DepthConfig(**saved['env_config'].get('camera', {}))
    groups=[0,4] if saved['env_config']['terrain'].get('surroundings') else [0]
    return VisualRuntime(task, policy.visual_encoder,
                         DepthSensor(task, camera, geometry_groups=groups, randomize=False))


def fit_actions(actor, optimizer, states, targets, args):
    total, count=0., 0
    for _ in range(args.epochs):
        for ids in torch.randperm(len(states), device=states.device).split(args.batch_size):
            loss=F.mse_loss(actor(states[ids])[0], targets[ids])
            if not torch.isfinite(loss):
                raise RuntimeError('参考动作监督损失出现非有限值')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), 1.)
            optimizer.step()
            total+=loss.item()
            count+=1
    return total/count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--flat-reference', required=True)
    parser.add_argument('--terrain-reference', required=True)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--teacher-mix', type=float, default=.5)
    parser.add_argument('--training-levels', type=int, nargs='+', default=[0,6,9,12])
    parser.add_argument('--seed', type=int, default=89991)
    args=parser.parse_args()
    if not 0 <= args.teacher_mix <= 1:
        parser.error('teacher-mix应在0到1之间')
    torch.set_num_threads(1)
    actor, config, saved=load_policy(args.resume)
    flat, _, flat_saved=load_policy(args.flat_reference)
    rough, _, rough_saved=load_policy(args.terrain_reference)
    assert actor.mode==flat.mode==rough.mode=='student'
    actor.requires_grad_(False)
    actor.actor.requires_grad_(True)
    flat.requires_grad_(False)
    rough.requires_grad_(False)
    training=TrainConfig(**saved['train_config'])
    critic=PrivilegedCritic(training).to('cuda:0')
    critic.load_state_dict(saved['critic'])
    critic.requires_grad_(False)
    # 复用检查点格式；本阶段仅MSE更新动作MLP，不采集PPO优势或更新critic。
    agent=PPO(actor=actor, critic=critic, config=training, device='cuda:0')
    agent.actor_optimizer=torch.optim.Adam(actor.actor.parameters(), lr=args.learning_rate)
    config.num_envs, config.seed=args.num_envs, args.seed
    config.randomization.enabled=False
    config.terrain.seed=56000
    ppo_iteration=saved.get('source_ppo_iteration', saved['iteration']) if saved['stage']=='distill' else saved['iteration']
    ppo_steps=saved.get('source_ppo_total_steps', saved.get('total_steps',0)) if saved['stage']=='distill' else saved.get('total_steps',0)
    folder=new_run('reference_control_distill', config, training)
    metadata=dict(status='running', algorithm='reference_action_supervision', parent=args.resume,
                  source_ppo_iteration=ppo_iteration, source_visual_iteration=saved.get('source_visual_iteration'),
                  arguments=vars(args), numerical_failures=0, randomization_enabled=False)
    save_json(folder/'metadata.json', metadata)
    print('RUN_DIR='+str(folder), flush=True)
    task=Go2Task(config, curriculum=False)
    try:
        terrain=task.env.scene.terrain
        levels=torch.tensor(args.training_levels, device=task.device)
        terrain.terrain_levels[:]=levels[torch.arange(args.num_envs, device=task.device)%len(levels)]
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        task.reset()
        visual=VisualRuntime(task, actor.visual_encoder)
        flat_visual=reference_visual(task, flat, flat_saved)
        rough_visual=reference_visual(task, rough, rough_saved)
        hips=[i for i,name in enumerate(task.robot.joint_names) if name.endswith('_hip_joint')]
        hip_limit=config.training_reward.front_hip_deadband/config.action_scale
        deployment=deployment_config(task)
        save_json(folder/'deployment.json', deployment)
        torch.manual_seed(args.seed)
        began=time.monotonic()
        for iteration in range(1, args.iterations+1):
            states, targets=[], []
            episodes=successes=highest_episodes=highest_successes=0
            flat_fraction=0.
            for _ in range(args.horizon):
                with torch.no_grad():
                    state,_=task.inputs(visual.latent)
                    flat_state,_=task.inputs(flat_visual.latent)
                    rough_state,_=task.inputs(rough_visual.latent)
                    local_x=task.robot.data.root_link_pos_w[:,0]-task.env.scene.env_origins[:,0]+config.terrain.spawn_x
                    is_flat=flat_terrain_mask(task, local_x)
                    target=torch.where(is_flat[:,None], flat(flat_state)[0], rough(rough_state)[0])
                    # 与标称PPO相同的平地髋参考限制；执行动作仍由任务作统一限幅。
                    straight=is_flat & (task.command()[:,2].abs()<.1)
                    target[:,hips]=torch.where(straight[:,None], target[:,hips].clamp(-hip_limit,hip_limit), target[:,hips])
                    action=args.teacher_mix*target+(1-args.teacher_mix)*actor(state)[0]
                    states.append(state.clone())
                    targets.append(target.clone())
                    flat_fraction+=is_flat.float().mean().item()
                    _, terminated, truncated, info=task.step(action)
                    for runtime in (visual, flat_visual, rough_visual):
                        runtime.update()
                    ids=(terminated|truncated).nonzero(as_tuple=False).flatten()
                    if len(ids):
                        highest=ids[terrain.terrain_levels[ids]==max(args.training_levels)]
                        episodes+=len(ids)
                        successes+=int(info['success'][ids].sum())
                        highest_episodes+=len(highest)
                        highest_successes+=int(info['success'][highest].sum())
                        task.reset(ids)
                        for runtime in (visual, flat_visual, rough_visual):
                            runtime.reset(ids)
            loss=fit_actions(actor, agent.actor_optimizer, torch.cat(states), torch.cat(targets), args)
            row=dict(iteration=iteration, action_loss=loss, teacher_mix=args.teacher_mix,
                     flat_reference_fraction=flat_fraction/args.horizon, episodes=episodes, successes=successes,
                     highest_level=max(args.training_levels), highest_episodes=highest_episodes,
                     highest_successes=highest_successes, numerical_failures=task.numerical_failures,
                     elapsed_seconds=time.monotonic()-began)
            append_csv(folder/'metrics.csv', row)
            with (folder/'train.log').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
            if iteration%25==0 or iteration==args.iterations:
                save_checkpoint(folder/'checkpoints/latest.pt', agent, iteration, config, training, 'distill',
                    task=task, deployment=deployment, source_ppo_iteration=ppo_iteration,
                    source_ppo_total_steps=ppo_steps, source_visual_iteration=saved.get('source_visual_iteration'),
                    source_reference_control_updates=iteration, supervised_total_steps=iteration*args.horizon*args.num_envs,
                    flat_reference=args.flat_reference, terrain_reference=args.terrain_reference,
                    algorithm='reference_action_supervision')
                shutil.copyfile(folder/'checkpoints/latest.pt', folder/f'checkpoints/iteration_{iteration}.pt')
        shutil.copyfile(folder/'checkpoints/latest.pt', folder/'checkpoints/final.pt')
        metadata.update(status='finished', iteration=iteration, numerical_failures=task.numerical_failures)
        save_json(folder/'metadata.json', metadata)
        print('CONTROL_DISTILLATION_FINISHED='+str(folder), flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=str(error), numerical_failures=task.numerical_failures)
        save_json(folder/'metadata.json', metadata)
        raise
    finally:
        task.close()


if __name__=='__main__':
    main()
