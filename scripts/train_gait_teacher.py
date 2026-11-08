"""统一特权教师PPO：共同采集复杂路线和独立平地，学习一致的接触与落点。"""
import argparse
from dataclasses import asdict
import json
import time

import torch

from env.commands import nominal_flat_training_config
from env.training_task import Go2Task
from rl.algorithms.perceptive_networks import PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import TrainConfig
from rl.run_io import append_csv,deployment_config,load_policy,new_run,save_checkpoint,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',required=True)
    parser.add_argument('--route-envs',type=int,default=512)
    parser.add_argument('--free-envs',type=int,default=256)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--actor-lr',type=float,default=5e-6)
    parser.add_argument('--stance-width-scale',type=float,help='仅调整平地站宽软成本；省略则沿用权重配置')
    parser.add_argument('--training-levels',type=int,nargs='+',default=[0,6,9,12])
    parser.add_argument('--seed',type=int,default=90015)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    actor,config,saved=load_policy(args.resume)
    assert actor.mode=='teacher'
    actor.train();actor.requires_grad_(True)
    training=TrainConfig(**saved['train_config'])
    training.horizon=64;training.batch_size=8192;training.actor_lr=args.actor_lr
    training.critic_lr=3e-4;training.anchor_coef=0.;training.freeze_proprio=False
    training.training_levels=tuple(args.training_levels)
    critic=PrivilegedCritic(training).to('cuda:0')
    critic.load_state_dict(saved['critic'])
    agent=PPO(actor=actor,critic=critic,config=training,device='cuda:0',
              actor_lr=training.actor_lr,critic_lr=training.critic_lr)
    agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
    agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
    for group in agent.actor_optimizer.param_groups:group['lr']=args.actor_lr
    for group in agent.critic_optimizer.param_groups:group['lr']=training.critic_lr
    config.num_envs,config.seed=args.route_envs,args.seed
    config.randomization.enabled=False
    config.terrain.seed=56000
    if args.stance_width_scale is not None:
        config.training_reward.stance_width_scale=args.stance_width_scale
    free_config=nominal_flat_training_config(config,args.free_envs,args.seed+1)
    folder=new_run('gait_teacher_multitask',config,training)
    metadata=dict(status='running',parent=args.resume,arguments=vars(args),
                  privileged_input=True,student_goal_complete=False,randomization_enabled=False)
    save_json(folder/'metadata.json',metadata)
    save_json(folder/'task_configs.json',dict(route=asdict(config),free=asdict(free_config)))
    print('RUN_DIR='+str(folder),flush=True)
    tasks=[]
    try:
        route=Go2Task(config,curriculum=False);tasks.append(route)
        free=Go2Task(free_config,curriculum=False);tasks.append(free)
        terrain=route.env.scene.terrain
        levels=torch.tensor(args.training_levels,device='cuda:0')
        terrain.terrain_levels[:]=levels[torch.arange(args.route_envs,device='cuda:0')%len(levels)]
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
        route.reset()
        deployment=deployment_config(route)
        start=saved['iteration'];total=saved.get('total_steps',0);began=time.monotonic()
        std_scale=torch.cat([torch.ones(args.route_envs,1,device='cuda:0'),
                            torch.full((args.free_envs,1),.5,device='cuda:0')])
        for iteration in range(start+1,start+args.iterations+1):
            stats=[dict(episodes=0,successes=0,falls=0,timeouts=0,tracking=0.) for _ in tasks]
            highest=dict(episodes=0,successes=0,falls=0)
            for _ in range(training.horizon):
                with torch.no_grad():
                    states=[];critics=[]
                    for index,task in enumerate(tasks):
                        state,value_state=task.inputs()
                        if index==1:value_state[:,-1]=-1.  # critic区分平地任务，Actor仍是原682维。
                        states.append(state);critics.append(value_state)
                    action,sample=agent.sample_batch(torch.cat(states),torch.cat(critics),std_scale=std_scale)
                    rewards=[];dones=[];terms=[];next_critics=[];results=[]
                    for index,(task,command) in enumerate(zip(tasks,action.split([args.route_envs,args.free_envs]))):
                        # NanGuard会恢复异常世界，提前保留实际输入状态，供碰撞/动作异常定位。
                        before_qpos=task.env.sim.data.qpos.clone()
                        before_qvel=task.env.sim.data.qvel.clone()
                        reward,terminated,truncated,info=task.step(command)
                        failed=info['numerical_failure'].nonzero().flatten()
                        if len(failed):
                            terrain_state=task.env.scene.terrain
                            step=task.env.common_step_counter
                            save_json(folder/f'numerical_failure_{iteration}_{index}_{step}.json',dict(
                                iteration=iteration,task='route' if index==0 else 'free',
                                global_step=step,episode_steps=task.env.episode_length_buf[failed].cpu().tolist(),
                                env_ids=failed.cpu().tolist(),
                                levels=terrain_state.terrain_levels[failed].cpu().tolist(),
                                layouts=terrain_state.terrain_types[failed].cpu().tolist(),
                                qpos_before=before_qpos[failed].cpu().tolist(),
                                qvel_before=before_qvel[failed].cpu().tolist(),
                                action=command[failed].cpu().tolist(),
                                physics_dt=task.config.physics_dt,control_dt=task.dt))
                        _,next_value=task.inputs()
                        if index==1:next_value[:,-1]=-1.
                        rewards.append(reward);dones.append(terminated|truncated);terms.append(terminated)
                        next_critics.append(next_value);results.append((truncated,info))
                    # 先保存真实末状态用于截断bootstrap，再重置各任务及其高度缓存。
                    agent.store_batch(sample,torch.cat(rewards),torch.cat(dones),torch.cat(terms),torch.cat(next_critics))
                    for index,(task,done,(truncated,info)) in enumerate(zip(tasks,dones,results)):
                        stats[index]['tracking']+=info['terms']['tracking_xy'].mean().item()
                        ids=done.nonzero().flatten()
                        if len(ids):
                            stats[index]['episodes']+=len(ids)
                            stats[index]['successes']+=int(info['success'][ids].sum())
                            stats[index]['falls']+=int(info['fall'][ids].sum())
                            stats[index]['timeouts']+=int(truncated[ids].sum())
                            if index==0:
                                top=ids[terrain.terrain_levels[ids]==levels.max()]
                                highest['episodes']+=len(top)
                                highest['successes']+=int(info['success'][top].sum())
                                highest['falls']+=int(info['fall'][top].sum())
                            task.reset(ids)
            if any(t.numerical_failures for t in tasks):raise RuntimeError('教师仿真数值异常，本批保留拒用')
            metrics=agent.update_batch()
            total+=training.horizon*(args.route_envs+args.free_envs)
            row=dict(iteration=iteration,total_steps=total,elapsed_seconds=time.monotonic()-began,
                     numerical_failures=0,route_max_level=int(levels.max()),
                     **{f'route_max_level_{key}':value for key,value in highest.items()},**metrics)
            for index,name in enumerate(('route','free')):
                row.update({name+'_'+key:(value/training.horizon if key=='tracking' else value)
                            for key,value in stats[index].items()})
            append_csv(folder/'metrics.csv',row)
            with (folder/'train.log').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if iteration==start+1 or (iteration-start)%10==0:print(json.dumps(row),flush=True)
            if (iteration-start)%25==0 or iteration==start+args.iterations:
                extra=dict(task=route,deployment=deployment,total_steps=total,
                           task_configs=dict(route=asdict(config),free=asdict(free_config)),
                           privileged_input=True,student_acceptance=False,critic_free_task_marker=True)
                save_checkpoint(folder/f'checkpoints/iteration_{iteration:06d}.pt',agent,iteration,config,training,'teacher',**extra)
                save_checkpoint(folder/'checkpoints/latest.pt',agent,iteration,config,training,'teacher',**extra)
                if iteration==start+args.iterations:
                    save_checkpoint(folder/'checkpoints/final.pt',agent,iteration,config,training,'teacher',**extra)
        metadata.update(status='finished',iteration=iteration,numerical_failures=0)
        save_json(folder/'metadata.json',metadata)
        print('GAIT_TEACHER_MULTITASK_FINISHED='+str(folder),flush=True)
    except BaseException as error:
        metadata.update(status='failed',error=str(error),numerical_failures=sum(t.numerical_failures for t in tasks))
        save_json(folder/'metadata.json',metadata);raise
    finally:
        for task in tasks:task.close()


if __name__=='__main__':main()
