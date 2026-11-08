"""合并已保存的学生控制网络和特权高度编码，准备统一步态教师的独立PPO起点。"""
import argparse
import hashlib
import math
from pathlib import Path

import torch

from env.posture import configure_normal_gait
from rl.algorithms.perceptive_networks import PerceptiveActor,PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import TrainConfig
from rl.run_io import env_from_dict,new_run,save_checkpoint,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-source',required=True)
    parser.add_argument('--height-source',required=True)
    parser.add_argument('--seed',type=int,default=90011)
    parser.add_argument('--num-envs',type=int,default=512)
    args=parser.parse_args()
    torch.set_num_threads(1)
    control=torch.load(args.control_source,map_location='cpu',weights_only=False)
    height=torch.load(args.height_source,map_location='cpu',weights_only=False)
    assert height['stage']=='teacher'
    config=env_from_dict(control['env_config'])
    config.num_envs,config.seed=args.num_envs,args.seed
    config.randomization.enabled=False
    config.terrain.seed=56000
    config.height_scan_update_steps=5
    configure_normal_gait(config,.5)
    training=TrainConfig(**control['train_config'])
    training.horizon=64;training.batch_size=8192;training.save_interval=25
    training.actor_lr=2e-5;training.critic_lr=3e-4
    training.freeze_proprio=False;training.anchor_coef=0.;training.target_action_std=.03
    training.training_levels=(0,6,9,12)
    actor=PerceptiveActor('teacher',training).to('cuda:0')
    state=dict(height['actor'])
    # 控制、本体和速度延续原基线；高度编码保留原教师，重新建立独立优化器。
    state.update({k:v for k,v in control['actor'].items() if not k.startswith('visual_encoder.')})
    actor.load_state_dict(state)
    with torch.no_grad():actor.log_std.fill_(math.log(training.target_action_std))
    critic=PrivilegedCritic(training).to('cuda:0')
    critic.load_state_dict(control['critic'])
    agent=PPO(actor=actor,critic=critic,config=training,device='cuda:0',
              actor_lr=training.actor_lr,critic_lr=training.critic_lr)
    folder=new_run('gait_teacher_initial',config,training)
    save_checkpoint(folder/'checkpoints/initial.pt',agent,0,config,training,'teacher',total_steps=0,
        control_source=args.control_source,height_source=args.height_source,
        source_control_ppo_iteration=control.get('source_ppo_iteration',control['iteration']),
        source_control_supervision_updates=control.get('source_reference_control_updates',0),
        privileged_input=True,student_acceptance=False)
    save_json(folder/'initialization.json',dict(control_source=args.control_source,height_source=args.height_source,
        control_sha256=hashlib.sha256(Path(args.control_source).read_bytes()).hexdigest(),
        height_sha256=hashlib.sha256(Path(args.height_source).read_bytes()).hexdigest(),
        policy_role='统一特权教师，最终仍需视觉学生验收',teacher_iteration=0,
        height_scan_update_steps=5,randomization_enabled=False))
    print('TEACHER_INITIAL='+str(folder/'checkpoints/initial.pt'),flush=True)


if __name__=='__main__':main()
