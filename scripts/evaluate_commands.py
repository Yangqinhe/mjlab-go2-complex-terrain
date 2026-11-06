"""独立评估正负速度/转向和0.3到0停车；报告逐案例误差、漂移及真实接触。"""

import argparse
import copy
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import hashlib

import torch

from env.training_task import Go2Task
from env.perception.runtime import VisualRuntime
from rl.run_io import env_from_dict, load_policy, save_json


def evaluate(checkpoint, env_config, seeds=(50100, 50101, 50102), seconds=20., device='cuda:0', output=None):
    actor, original, saved = load_policy(checkpoint, device)
    source = env_from_dict(json.loads(Path(env_config).read_text())) if env_config else original
    if source.command.mode != 'free' or source.route.enabled or source.terrain.obstacles:
        raise ValueError('变速专项使用宽阔平地free配置；复杂路线需单独验收')
    scenarios = [(vx, wz, 'constant') for vx in (-.3, 0., .3) for wz in (-.5, 0., .5)]
    scenarios += [(.3, 0., 'stop'), (.3, 0., 'reverse')]
    records = []
    evaluated_configs = []
    for seed in seeds:
        config = copy.deepcopy(source)
        config.num_envs = len(scenarios)
        config.seed, config.randomization.seed = seed, seed+10000
        config.episode_seconds = seconds+1
        evaluated_configs.append(asdict(config))
        task = Go2Task(config, device, curriculum=False)
        try:
            visual = VisualRuntime(task, actor.visual_encoder)
            task.schedule.remaining[:] = 1000000
            task.schedule.anchor[:] = False
            target = torch.tensor([[vx, 0., wz] for vx, wz, _ in scenarios], device=device)
            alive = torch.ones(config.num_envs, dtype=torch.bool, device=device)
            errors, references, velocities, positions, contacts, active = [], [], [], [], [], []
            flight = torch.zeros(config.num_envs, device=device)
            max_flight = torch.zeros_like(flight)
            failed = torch.zeros_like(alive)
            samples = task.env.dr_samples.clone() if config.randomization.enabled else None
            for step in range(round(seconds/task.dt)):
                time = step*task.dt
                target[-2, 0] = .3 if time < 6 else 0.
                target[-1, 0] = .3 if time < 6 else -.3
                task.schedule.target.copy_(target)
                with torch.no_grad():
                    state, _ = task.inputs(visual.latent)
                    action = actor(state)[0]
                    action[~alive] = 0
                    _, terminated, truncated, info = task.step(action)
                    visual.update()
                velocity = torch.stack([task.robot.data.root_link_lin_vel_b[:,0],
                                        task.robot.data.root_link_lin_vel_b[:,1],
                                        task.robot.data.root_link_ang_vel_b[:,2]], -1)
                contact = task.gait.previous_contact.clone()
                count = contact.sum(-1)
                if time >= .5:
                    flight = torch.where((count == 0) & alive, flight+task.dt, 0.)
                    max_flight = torch.maximum(max_flight, flight)
                active.append(alive.clone())
                errors.append((velocity-info['command']).square())
                references.append(info['command'].clone())
                velocities.append(velocity)
                positions.append(task.robot.data.root_link_pos_w.clone())
                contacts.append(contact)
                failed |= (terminated & alive)
                alive &= ~(terminated | truncated)
            error = torch.stack(errors);reference = torch.stack(references)
            velocity = torch.stack(velocities);pos = torch.stack(positions)
            contact = torch.stack(contacts);active = torch.stack(active)
            for i, (vx,wz,kind) in enumerate(scenarios):
                begin = 10. if kind == 'reverse' else 9. if kind == 'stop' else 6.
                steady = active[:,i].clone();steady[:round(begin/task.dt)] = False
                if steady.any():
                    rmse = error[steady,i].mean(0).sqrt().cpu().tolist()
                else:
                    rmse = [None]*3
                mask = active[:,i].clone();mask[:round(.5/task.dt)] = False
                flight_fraction = (contact[mask,i].sum(-1) == 0).float().mean().item() if mask.any() else 1.
                stopped = None
                if kind == 'stop':
                    index = round(9./task.dt)
                    drift = (pos[index:,i,:2]-pos[index,i,:2]).norm(dim=-1).max().item()
                    stopped = dict(mean_planar_speed=velocity[index:,i,:2].norm(dim=-1).mean().item(),
                                   maximum_drift_m=drift,
                                   passed=bool(alive[i]) and drift <= .10 and velocity[index:,i,:2].norm(dim=-1).mean().item() <= .03)
                passed = (bool(alive[i]) and rmse[0] is not None and rmse[0] <= .1
                          and rmse[1] <= .1 and rmse[2] <= .15
                          and max_flight[i].item() <= .060001 and flight_fraction <= .01
                          and (stopped is None or stopped['passed']))
                records.append(dict(seed=seed,scenario=kind,vx=vx,wz=wz,passed=passed,
                    terrain_layout=int(task.env.scene.terrain.terrain_types[i]),
                    camera_pitch=(visual.sensor.pitch_degrees[i].item() if config.randomization.enabled else config.camera.pitch_down_degrees),
                    survived=bool(alive[i]),failure=bool(failed[i]),steady_start_seconds=begin,
                    steady_rmse=rmse,full_rmse=error[active[:,i],i].mean(0).sqrt().cpu().tolist(),
                    max_flight_seconds=max_flight[i].item(),flight_fraction=flight_fraction,
                    stop=stopped,randomization=(samples[i].cpu().tolist() if samples is not None else None)))
        finally:
            task.close()
    report = dict(checkpoint=str(Path(checkpoint).resolve()),iteration=saved['iteration'],
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  evaluated_configs=evaluated_configs,
                  seeds=list(seeds),seconds=seconds,command_ranges=dict(vx=[-.3,.3],vy=[0,0],wz=[-.5,.5]),
                  cases=len(records),pass_rate=sum(r['passed'] for r in records)/len(records),
                  scope='边界速度、转向、停止和反向专项；不代替复杂地形路线与完整步态验收',
                  config=env_config,records=records)
    out = Path(output) if output else Path(checkpoint).parent.parent/'evaluations'/('commands_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    save_json(out,report)
    print(f'COMMAND_EVALUATION={out} PASS_RATE={report["pass_rate"]:.3f}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--env-config')
    parser.add_argument('--seeds',type=int,nargs='+',default=[50100,50101,50102])
    parser.add_argument('--output')
    args=parser.parse_args()
    torch.set_num_threads(1)
    evaluate(args.checkpoint,args.env_config,args.seeds,output=args.output)


if __name__ == '__main__':main()
