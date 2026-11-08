"""逐脚检查平地接触与落点，保存接触时序、落脚CSV和足印图；只评估学生。"""

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from env.gait import gait_passed, summarize_gait, MAX_STANCE_POINT_SPEED
from env.perception.runtime import VisualRuntime
from env.posture import foot_positions_body, foot_velocities_world
from env.training_task import Go2Task
from rl.run_io import load_policy, save_json


LEGS = ('FL', 'FR', 'RL', 'RR')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[56200])
    parser.add_argument('--nominal',action='store_true')
    parser.add_argument('--steps',type=int,default=600)
    args=parser.parse_args()
    torch.set_num_threads(1)
    actor,original,saved=load_policy(args.checkpoint)
    positions=[];hips=[];speeds=[];contacts=[];velocity=[];landed=[[] for _ in LEGS]
    gait_records=[];traces=[];landings=[];failed_total=0
    for seed in args.seeds:
        cfg=deepcopy(original);cfg.num_envs=4;cfg.seed=seed;cfg.terrain.seed=seed
        cfg.randomization.seed=200000+seed
        if args.nominal:
            cfg.randomization.enabled=False;cfg.camera.pitch_down_degrees=25.
        cfg.terrain.obstacles=False;cfg.terrain.levels=2;cfg.terrain.layouts=1
        cfg.terrain.width=20.;cfg.terrain.flat_length=20.;cfg.terrain.spawn_x=10.
        cfg.route.enabled=False;cfg.command.mode='legacy'
        cfg.command.vx=(.3,.3);cfg.command.vy=(0.,0.);cfg.command.wz=(0.,0.)
        task=Go2Task(cfg,curriculum=False)
        previous=torch.zeros(4,4,dtype=torch.bool,device='cuda:0')
        air=torch.zeros(4,4,device='cuda:0');failed=torch.zeros(4,dtype=torch.bool,device='cuda:0')
        ids=[list(task.robot.joint_names).index(leg+'_hip_joint') for leg in LEGS]
        try:
            visual=VisualRuntime(task,actor.visual_encoder) if actor.mode=='student' else None
            for step in range(args.steps):
                with torch.no_grad():
                    state,_=task.inputs(visual.latent if visual is not None else None)
                    _,terminated,truncated,_=task.step(actor(state)[0])
                    if visual is not None:visual.update()
                    failed|=terminated|truncated
                    contact=task.env.scene['feet'].data.force[:,task.foot_order,2].abs()>cfg.gait.contact_force
                    touchdown=contact & ~previous & (air>=cfg.gait.minimum_swing)
                    air=torch.where(contact,0.,air+task.dt);previous=contact.clone()
                    if step<50:continue
                    valid=~failed
                    feet=foot_positions_body(task)
                    slip=foot_velocities_world(task,flat_contact_point=True)[...,:2].norm(dim=-1)
                    if valid.any():
                        positions.append(feet[valid].cpu());hips.append(task.robot.data.joint_pos[valid][:,ids].cpu())
                        contacts.append(contact[valid].cpu());speeds.append(slip[valid].cpu())
                        velocity.append(task.robot.data.root_link_lin_vel_b[valid,0].cpu())
                    for leg in range(4):
                        mask=touchdown[:,leg]&valid
                        if mask.any():
                            landed[leg].append(feet[mask,leg,:2].cpu())
                            for env in mask.nonzero().flatten().tolist():
                                xyz=feet[env,leg].cpu().tolist()
                                landings.append(dict(seed=seed,env=env,seconds=(step+1)*task.dt,
                                                     foot=LEGS[leg],x=xyz[0],y=xyz[1],z=xyz[2]))
                    if seed==args.seeds[0] and not failed[0]:
                        traces.append(dict(seconds=(step+1)*task.dt,**dict(zip(LEGS,contact[0].int().cpu().tolist()))))
            failed_total+=int(failed.sum())
            gait_records.extend(dict(gait=task.gait.snapshot(i)) for i in range(4))
        finally:task.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not positions:
        save_json(args.output,dict(passed=False,error=100.,reason='没有有效行走样本',
            policy_mode=actor.mode,privileged_input=actor.mode=='teacher',failed_environments=failed_total));return
    p=torch.cat(positions);h=torch.cat(hips).abs();c=torch.cat(contacts);v=torch.cat(speeds)
    home_x=torch.tensor(original.training_reward.foot_home_x)
    home_y=torch.tensor(original.training_reward.foot_home_y)
    widths=torch.stack([p[:,0,1]-p[:,1,1],p[:,2,1]-p[:,3,1]],-1)
    crossing=(p[...,1]*home_y.sign()<=0).float().mean().item()
    duty=c.float().mean(0)
    slip_p95=[v[:,i][c[:,i]].quantile(.95).item() if c[:,i].any() else 100. for i in range(4)]
    foot_reports=[]
    for i,leg in enumerate(LEGS):
        xy=torch.cat(landed[i]) if landed[i] else torch.empty(0,2)
        inside=((xy[:,0]-home_x[i]).abs()<=.16)&((xy[:,1]-home_y[i]).abs()<=.04)
        foot_reports.append(dict(foot=leg,touchdowns=len(xy),inside_fraction=inside.float().mean().item() if len(xy) else 0.,
                                 duty_factor=duty[i].item(),stance_slip_p95_m_s=slip_p95[i],hip_abs_p95_rad=h[:,i].quantile(.95).item()))
    gait=summarize_gait(gait_records)
    rmse=(torch.cat(velocity)-.3).square().mean().sqrt().item()
    hip_p95=h.quantile(.95,dim=0)
    mean_width=widths.mean(0);p95_width=widths.quantile(.95,dim=0)
    pose_pass=bool(((mean_width>=.244)&(mean_width<=.324)&(p95_width<=.344)).all() and (hip_p95<=.25).all())
    feet_pass=all(r['touchdowns']>=4*len(args.seeds) and r['inside_fraction']>=.95
                  and .3<=r['duty_factor']<=.8 and r['stance_slip_p95_m_s']<=MAX_STANCE_POINT_SPEED for r in foot_reports)
    contact_pass=gait_passed(dict(gait=gait))
    error=(float(((mean_width-.324).clamp(min=0)+(.244-mean_width).clamp(min=0)).sum()/.04)
           +float((p95_width-.344).clamp(min=0).sum()/.04)+float((hip_p95-.25).clamp(min=0).sum()/.15)
           +sum(max(0.,.95-r['inside_fraction'])*10+max(0.,r['stance_slip_p95_m_s']-MAX_STANCE_POINT_SPEED)*10
                 +int(r['touchdowns']<4*len(args.seeds)) for r in foot_reports)+10*crossing)
    # 排名也必须计入生存、接触、速度和支撑占比，不能让减速或跌倒的候选凭低滑移占据最佳。
    gate_penalty=(100.*failed_total+100.*int(not contact_pass)+10*max(0.,rmse-.1)/.1
        +10*sum(max(0.,.3-r['duty_factor'])+max(0.,r['duty_factor']-.8) for r in foot_reports))
    error+=gate_penalty
    report=dict(selection_gate_penalty=gate_penalty,
        policy_mode=actor.mode,privileged_input=actor.mode=='teacher',
        checkpoint=str(Path(args.checkpoint).resolve()),iteration=saved['iteration'],seeds=args.seeds,
        slip_measurement='平地球形足底接触点速度近似；不是接触求解器直接测量值',
        nominal=args.nominal,randomization_enabled=False if args.nominal else original.randomization.enabled,
        passed=failed_total==0 and pose_pass and feet_pass and contact_pass and crossing<=.001 and rmse<=.1,
        error=error,failed_environments=failed_total,samples=len(p),vx_rmse=rmse,pose_passed=pose_pass,
        contact_passed=contact_pass,foot_placement_passed=feet_pass,crossing_fraction=crossing,
        mean_width_m=mean_width.tolist(),p95_width_m=p95_width.tolist(),feet=foot_reports,gait=gait)
    for suffix,rows in (('_contacts.csv',traces),('_landings.csv',landings)):
        if rows:
            with args.output.with_name(args.output.stem+suffix).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,leg in enumerate(LEGS):
        if traces:axes[0].step([r['seconds'] for r in traces],[r[leg]*.7+i for r in traces],where='post',label=leg)
        xy=torch.cat(landed[i]) if landed[i] else torch.empty(0,2)
        if len(xy):axes[1].scatter(xy[:,0],xy[:,1],s=8,label=leg)
        axes[1].add_patch(Rectangle((home_x[i]-.16,home_y[i]-.04),.32,.08,fill=False,linestyle='--'))
    axes[0].set(xlabel='Seconds',ylabel='Contact (offset by foot)',title='Measured contacts, env 0')
    axes[1].set(xlabel='Body x (m)',ylabel='Body y (m)',title='Touchdown positions');axes[1].set_aspect('equal')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();plot=args.output.with_suffix('.png');fig.savefig(plot,dpi=140);plt.close(fig)
    report['plot']=str(plot);save_json(args.output,report)
    print('NORMAL_GAIT_AUDIT='+str(args.output),flush=True)
    print(json.dumps({k:report[k] for k in ('passed','error','mean_width_m','feet')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
