"""核对停车脚本、历史配置兼容，以及参考约束与PPO共享同一KL早停预算。"""

from types import SimpleNamespace
import torch

from env.config import EnvConfig
from env.commands import VelocitySchedule
from rl.config import TrainConfig
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.algorithms.ppo import PPO


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    cfg=EnvConfig(num_envs=64)
    cfg.command.stop_script_fraction=1.;cfg.command.anchor_fraction=0.
    cfg.command.vx=(-.3,.3);cfg.command.wz=(-.5,.5)
    schedule=VelocitySchedule(SimpleNamespace(config=cfg,device='cpu',dt=.02))
    schedule.reset(torch.arange(64));assert schedule.stop_script.all()
    for _ in range(290):schedule.step()
    assert (schedule.applied[:,0]-.3).abs().max()<1e-6
    for _ in range(160):schedule.step()
    assert schedule.applied.abs().max()<1e-5
    config=TrainConfig(epochs=2,batch_size=8,anchor_coef=10.,target_kl=0.,
                       velocity_coef=0.,entropy_coef=0.)
    actor=PerceptiveActor('student',config);critic=PrivilegedCritic(config)
    agent=PPO(actor=actor,critic=critic,config=config,device='cpu',actor_lr=1e-3,epochs=2)
    states=torch.randn(4,559)*.1;critics=torch.zeros(4,244)
    with torch.no_grad():target=agent.actor(states)[0]+.1
    scale=torch.tensor([[.5],[.5],[2.],[2.]])
    for _ in range(2):
        action,sample=agent.sample_batch(states,critics,std_scale=scale)
        assert torch.allclose(sample['stds'],actor(states)[1]*scale)
        sample['anchor_mu']=target.clone();sample['anchor_mu'][2:]=100.
        sample['anchor_weight']=torch.tensor([1.,1.,0.,0.])
        agent.store_batch(sample,torch.zeros(4),torch.zeros(4,dtype=torch.bool),
                          torch.zeros(4,dtype=torch.bool),critics)
    before=(actor(states)[0][:2]-target[:2]).square().mean().item()
    result=agent.update_batch()
    after=(actor(states)[0][:2]-target[:2]).square().mean().item()
    assert result['actor_steps']==1, result
    assert abs(result['anchor_loss']-.01)<1e-5,result
    assert after<before,(before,after)
    print('MULTITASK_CHECKS_PASSED',result,flush=True)


if __name__=='__main__':main()
