"""持续联合训练学生：按真实路线/指令共同选模，完整随机化验收通过前不按轮数结束。"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time

import torch

from env.gait import gait_error, gait_passed
from rl.config import TrainConfig
from rl.run_io import ROOT, env_from_dict, new_run, save_json
from scripts.train_student_multitask import REFERENCE, VISUAL, task_configs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates-per-round',type=int,default=100)
    parser.add_argument('--route-envs',type=int,default=384)
    parser.add_argument('--free-envs',type=int,default=128)
    parser.add_argument('--visual-source',default=str(VISUAL))
    parser.add_argument('--initial-checkpoint')
    parser.add_argument('--reference-checkpoint',default=str(REFERENCE))
    parser.add_argument('--flat-gait-first',action='store_true')
    parser.add_argument('--nominal-max-level',type=int,choices=(0,6,9,12,15,17),default=17)
    parser.add_argument('--nominal-screen-seeds',type=int,nargs='+',default=[54000])
    parser.add_argument('--flat-anchor-coef',type=float,default=20.)
    parser.add_argument('--nominal-actor-lr',type=float,default=5e-6)
    parser.add_argument('--nominal-route-std-scale',type=float,default=.5)
    parser.add_argument('--nominal-free-std-scale',type=float,default=2.)
    parser.add_argument('--nominal-anchor-coef',type=float,default=2.)
    parser.add_argument('--nominal-terrain-reference')
    parser.add_argument('--nominal-reference-transition',type=float,default=0.)
    parser.add_argument('--nominal-continuous-updates',action='store_true',help='标称训练分支累计更新，最佳验收权重单独保留')
    parser.add_argument('--nominal-gait-first',action='store_true',help='先关闭随机化纠正四足步态，数值和视频复核通过才开放随机化')
    parser.add_argument('--require-normal-gait',action='store_true')
    parser.add_argument('--resume-controller',type=Path,help='恢复控制目录，保留选模与课程，并优先评估已完成的训练批次')
    parser.add_argument('--front-posture-finetune',action='store_true',help='从指定学生权重微调前腿站姿；可显式选择当前随机化强度')
    parser.add_argument('--initial-strength',type=float,help='显式沿用来源随机化强度；省略时前腿阶段默认1、常规训练默认0.25')
    parser.add_argument('--visual-updates-per-refresh',type=int,default=100)
    parser.add_argument('--visual-refresh-every',type=int,default=3)
    parser.add_argument('--screen-only',action='store_true',help='只用于候选初始化对比，不是持续训练默认行为')
    args=parser.parse_args()
    if args.flat_gait_first:args.nominal_gait_first=True
    if args.nominal_gait_first:args.front_posture_finetune=True
    if args.front_posture_finetune and not (args.initial_checkpoint or args.resume_controller):
        parser.error('前腿微调必须指定作为起点的完整学生检查点')
    if args.initial_strength is not None and not (0<args.initial_strength<=1 or (args.initial_strength==0 and args.nominal_gait_first)):
        parser.error('initial-strength必须在(0,1]')
    torch.set_num_threads(1)
    recovered=None;pending_run=None
    if args.resume_controller:
        folder=args.resume_controller.resolve()
        recovered=json.loads((folder/'status.json').read_text())
        history=folder/'resume_history';history.mkdir(exist_ok=True)
        save_json(history/f'{datetime.now():%Y%m%d_%H%M%S_%f}_before_resume.json',recovered)
        args.front_posture_finetune=recovered.get('front_posture_finetune',False)
        args.nominal_gait_first=recovered.get('nominal_gait_first',False)
        args.flat_gait_first=recovered.get('nominal_substage')=='flat'
        args.require_normal_gait=recovered.get('require_normal_gait',False)
        for key,value in recovered.get('controller_config',{}).items():
            if key in ('updates_per_round','nominal_actor_lr','nominal_anchor_coef','flat_anchor_coef','route_envs','free_envs','nominal_max_level','nominal_screen_seeds','nominal_terrain_reference','nominal_continuous_updates','nominal_route_std_scale','nominal_free_std_scale','nominal_reference_transition'):
                setattr(args,key,value)
        args.reference_checkpoint=recovered['reference']
        args.visual_source=recovered['visual_source']
        pending_run=recovered.get('pending_run')
        if not pending_run and str(recovered.get('stage','')).endswith('_train'):
            log=Path(recovered['log'])
            if log.exists():
                finished=[line.split('=',1)[1] for line in log.read_text().splitlines()
                          if line.startswith('MULTITASK_FINISHED=')]
                if finished:pending_run=finished[-1]
    else:
        folder=ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_continual_student'
        folder.mkdir()
    reference_path=Path(args.reference_checkpoint)
    ref=torch.load(reference_path,map_location='cpu',weights_only=False)
    phi=torch.load(args.visual_source,map_location='cpu',weights_only=False)
    source=env_from_dict(ref['env_config'])
    initial_strength=recovered['strength'] if recovered else (args.initial_strength if args.initial_strength is not None
        else (0. if args.nominal_gait_first else 1. if args.front_posture_finetune else .25))
    def configured_tasks(strength,route_envs,free_envs,seed):
        route,free=task_configs(source,strength,route_envs,free_envs,seed,flat_gait=args.flat_gait_first)
        if args.front_posture_finetune:
            from env.posture import configure_front_posture
            for cfg in (route,free):configure_front_posture(cfg,1.)
        if args.nominal_gait_first or args.require_normal_gait:
            from env.posture import configure_normal_gait
            for cfg in (route,free):configure_normal_gait(cfg,1.)
        return route,free
    route,free=configured_tasks(initial_strength,args.route_envs,args.free_envs,85000)
    if recovered:
        state=recovered
        checkpoint=Path(state['selected_checkpoint'])
        state['round']=max((r['round'] for r in state['records']),default=0)
        state.update(status='running',resumed=True,pending_run=pending_run,folder=str(folder))
        state.pop('error',None)
    else:
        training=TrainConfig(**ref['train_config'])
        initial=new_run('student_ppo',route,training)
        seed=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False) if args.initial_checkpoint else dict(ref)
        seed['actor']=dict(seed['actor'])
        if not args.initial_checkpoint:
            for k,v in phi['actor'].items():
                if k.startswith('visual_encoder.'):seed['actor'][k]=v
        seed['env_config']=asdict(route)
        seed['initialization']=dict(common_source=args.initial_checkpoint or str(reference_path),
                                    visual_source='included in checkpoint' if args.initial_checkpoint else args.visual_source,ppo_updates=0)
        checkpoint=initial/'checkpoints/final.pt';torch.save(seed,checkpoint)
        save_json(initial/'metadata.json',dict(status='initialization_only',**seed['initialization']))
        deployment=deepcopy(ref['deployment']);deployment.update(camera=asdict(route.camera),
            randomization=asdict(route.randomization),depth_geometry_groups=[0,4])
        save_json(initial/'deployment.json',deployment)
        save_json(initial/'task_configs.json',dict(route=asdict(route),free=asdict(free)))
        state=dict(status='running',round=0,strength=initial_strength,front_posture_finetune=args.front_posture_finetune,selected_checkpoint=str(checkpoint),
                   reference=str(reference_path),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                   nominal_gait_first=args.nominal_gait_first,require_normal_gait=args.require_normal_gait,
                   phase='nominal_gait' if args.nominal_gait_first else 'training',
                   nominal_substage='flat' if args.flat_gait_first else 'complex',
                   controller_config={key:getattr(args,key) for key in ('updates_per_round','nominal_actor_lr',
                       'nominal_anchor_coef','flat_anchor_coef','route_envs','free_envs','nominal_max_level','nominal_screen_seeds','nominal_terrain_reference','nominal_continuous_updates','nominal_route_std_scale','nominal_free_std_scale','nominal_reference_transition')},
                   visual_source=args.visual_source,folder=str(folder),records=[],
                   goal=('先在标称环境纠正四足接触与落脚；数值和视频复核均通过后再训练域随机化' if args.nominal_gait_first
                         else '完整域随机化下六档固定120+独立120路线全完成、速度/60ms步态通过，33项变速停车及20/35/50度边界通过'))
    save_json(folder/'status.json',state);print(f'CONTINUAL_DIR={folder}',flush=True)

    def run(module,arguments,label):
        log=folder/(label+'.log')
        state.update(stage=label,log=str(log));save_json(folder/'status.json',state)
        if log.exists():
            history=folder/'resume_history';history.mkdir(exist_ok=True)
            shutil.copy2(log,history/f'{datetime.now():%Y%m%d_%H%M%S_%f}_{log.name}')
        with log.open('w') as f:
            subprocess.run([sys.executable,'-m',module,*map(str,arguments)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        return log.read_text()

    def evaluate_route(ckpt,level,seeds,label,pitch=None,route_config=None,video=None):
        arguments=['--checkpoint',ckpt,'--num-envs',4,'--seeds',*seeds,'--level',level]
        if pitch is not None:arguments+=['--camera-pitch',pitch]
        if video is not None:arguments+=['--video',video]
        if route_config is not None:
            config_path=folder/(label+'_environment.json');save_json(config_path,asdict(route_config))
            arguments+=['--env-config',config_path]
        text=run('scripts.evaluate',arguments,label)
        path=Path(next(x.split('=',1)[1] for x in text.splitlines() if x.startswith('EVALUATION=')))
        return path,json.loads(path.read_text())

    def evaluate_commands(ckpt,cfg,label,seeds=(54500,54501,54502)):
        config=folder/(label+'_config.json');save_json(config,asdict(cfg))
        output=folder/(label+'.json')
        run('scripts.evaluate_commands',['--checkpoint',ckpt,'--env-config',config,'--seeds',*seeds,'--output',output],label)
        return output,json.loads(output.read_text())

    def route_pass(report):
        return (report['success_rate']==1. and report['vx_rmse']<=.1 and gait_passed(report)
                and not report['partial_evaluation'] and report['numerical_failure_rate']==0.)

    def stance_check(ckpt,label,seed=56200,seeds=None):
        if args.nominal_gait_first or args.require_normal_gait:
            output=folder/(label+'_normal_gait.json')
            arguments=['--checkpoint',ckpt,'--output',output,'--seeds',*(seeds or [seed])]
            if args.nominal_gait_first:arguments+=['--nominal']
            run('scripts.audit_normal_gait',arguments,label+'_normal_gait')
            report=json.loads(output.read_text())
            return dict(passed=report['passed'],error=report['error'],report=str(output))
        output=folder/(label+'_front_stance.json')
        run('scripts.audit_front_stance',['--checkpoint',ckpt,'--output',output,'--seed',seed],label+'_front_stance')
        report=json.loads(output.read_text())
        if not report['samples']:return dict(passed=False,error=100.,report=str(output))
        passed=(report['failed_environments']==0 and report['samples']>=1000
            and .244<=report['front_width_mean_m']<=.324 and report['front_width_p95_m']<=.344
            and max(report['front_hip_abs_p95_rad'])<=.25)
        error=(max(0.,.244-report['front_width_mean_m'])/.04
            +max(0.,report['front_width_mean_m']-.324)/.04
            +max(0.,report['front_width_p95_m']-.344)/.04
            +max(0.,max(report['front_hip_abs_p95_rad'])-.25)/.15)
        return dict(passed=passed,error=error,report=str(output))

    def screen(ckpt,cfg,label,route_config=None):
        if args.flat_gait_first:
            stance=stance_check(ckpt,label)
            return dict(score=-30*stance['error'],front_stance=stance,route_pass=False,route_min_success=None,
                command_pass_rate=None,stop_passes=0,reports=[stance['report']],passed=stance['passed'],
                note='标称平地中间课程，复杂地形尚未复核')
        reports=[];paths=[]
        screen_levels=sorted({0,args.nominal_max_level}) if args.nominal_gait_first else (0,17)
        for level in screen_levels:
            selection_seeds=args.nominal_screen_seeds if args.nominal_gait_first else [54000]
            path,report=evaluate_route(ckpt,level,selection_seeds,label+f'_route{level}',route_config=route_config)
            reports.append(report);paths.append(str(path))
        if args.nominal_gait_first:
            command=dict(pass_rate=None,records=[])
        else:
            path,command=evaluate_commands(ckpt,cfg,label+'_commands');paths.append(str(path))
        stop=[r['stop'] for r in command['records'] if r['scenario']=='stop']
        gait_safe=all(gait_passed(r) for r in reports)
        score=(100*min(r['success_rate'] for r in reports)+30*sum(r['success_rate'] for r in reports)/len(reports)
               -10*sum(r['vx_rmse'] for r in reports)-10*sum(gait_error(r) for r in reports)
               +20*(command['pass_rate'] or 0.)-10*sum(r['maximum_drift_m'] for r in stop)/max(len(stop),1))
        if not gait_safe:score-=1000.
        stance=stance_check(ckpt,label) if (args.front_posture_finetune or args.require_normal_gait) else dict(passed=True,error=0.)
        score-=30*stance['error']
        return dict(score=score,selection_seeds=selection_seeds,front_stance=stance,gait_safe=gait_safe,route_min_success=min(r['success_rate'] for r in reports),
                    route_pass=all(route_pass(r) for r in reports),command_pass_rate=command['pass_rate'],
                    stop_passes=sum(r['passed'] for r in stop),reports=paths,
                    passed=all(route_pass(r) for r in reports) and (args.nominal_gait_first or command['pass_rate']==1.) and stance['passed'])

    def full_validation(ckpt,cfg,label,final=False,route_config=None):
        if args.flat_gait_first:
            normal=stance_check(ckpt,label+'_flat_independent',seeds=[56300,56301,56302])
            return dict(passed=normal['passed'],normal_gait=normal,scope='flat_curriculum_only',randomization_enabled=False)
        paths=[]
        for level in (0,6,9,12,15,17):
            path,report=evaluate_route(ckpt,level,list(range(54000,54005)),label+f'_selection{level}',route_config=route_config)
            paths.append(str(path))
            if not route_pass(report):return dict(passed=False,reports=paths,reason='selection_route')
        if args.nominal_gait_first:
            fresh_start=54800+state.get('independent_attempts',0)*10
            state['independent_attempts']=state.get('independent_attempts',0)+1
            for level in (0,6,9,12,15,17):
                path,report=evaluate_route(ckpt,level,list(range(fresh_start,fresh_start+5)),
                    label+f'_nominal_fresh{level}',route_config=route_config)
                paths.append(str(path))
                if not route_pass(report):return dict(passed=False,reports=paths,reason='nominal_fresh_route')
            normal=stance_check(ckpt,label+'_independent',seeds=[56300,56301,56302])
            if not normal['passed']:return dict(passed=False,reports=paths,normal_gait=normal,reason='contact_or_placement')
            videos=[]
            for level in (0,17):
                video=folder/(label+f'_review_{level}.mp4')
                path,report=evaluate_route(ckpt,level,[fresh_start+5],label+f'_review{level}',
                    route_config=route_config,video=video)
                paths.append(str(path));videos.append(str(video))
                if not route_pass(report):return dict(passed=False,reports=paths,reason='review_video_route')
            return dict(passed=True,reports=paths,normal_gait=normal,videos=videos,randomization_enabled=False)
        path,report=evaluate_commands(ckpt,cfg,label+'_commands_full')
        if report['pass_rate']!=1.:return dict(passed=False,reports=paths+[str(path)],reason='commands')
        if not final:return dict(passed=True,reports=paths,commands=str(path))
        # 独立seed仅在候选通过固定选择后使用；失败数据保留，下次候选用新的独立组。
        fresh_start=54800+state.get('independent_attempts',0)*10
        state['independent_attempts']=state.get('independent_attempts',0)+1
        save_json(folder/'status.json',state)
        fresh=[]
        for level in (0,6,9,12,15,17):
            path,report=evaluate_route(ckpt,level,list(range(fresh_start,fresh_start+5)),label+f'_fresh{level}',route_config=route_config)
            fresh.append(str(path))
            if not route_pass(report):return dict(passed=False,reports=paths+fresh,reason='fresh_route')
        for pitch in (20,35,50):
            for level in (0,17):
                path,report=evaluate_route(ckpt,level,[54900],label+f'_pitch{pitch}_level{level}',pitch,route_config=route_config)
                if not route_pass(report):return dict(passed=False,reports=paths+fresh+[str(path)],reason='camera_boundary')
        path,report=evaluate_commands(ckpt,cfg,label+'_commands_independent',seeds=(54700,54701,54702))
        if report['pass_rate']!=1.:return dict(passed=False,reports=paths+fresh+[str(path)],reason='fresh_commands')
        stance=stance_check(ckpt,label+'_independent',seed=56300) if (args.front_posture_finetune or args.require_normal_gait) else None
        if stance and not stance['passed']:
            return dict(passed=False,reports=paths+fresh,front_stance=stance,reason='front_stance')
        return dict(passed=True,reports=paths,height_evaluations=fresh,commands=str(path),front_stance=stance)

    strength=state['strength'];best=str(checkpoint);best_screen=state.get('best_screen') if recovered else None
    training_checkpoint=state.get('training_checkpoint',best)
    anchor=.2 if args.front_posture_finetune else 20.
    lr=2e-6 if args.front_posture_finetune else 5e-6
    stale=0;stop_fraction=.25;freeze=True
    route_envs=args.route_envs;free_envs=args.free_envs
    errors=0
    if recovered:
        anchor=state.get('next_anchor',anchor);lr=state.get('next_lr',lr)
        for record in reversed(state['records']):
            if record.get('invalid_environment'):continue
            if record['selected']:break
            stale+=1
        stop_fraction=state.get('next_stop_fraction',stop_fraction)
        freeze=state.get('next_freeze_proprio',args.front_posture_finetune or (best_screen['route_min_success'] is not None and best_screen['route_min_success']<1.))
        route_envs=state.get('next_route_envs',route_envs);free_envs=state.get('next_free_envs',free_envs)
        if pending_run:
            previous=json.loads((Path(pending_run)/'metadata.json').read_text())['arguments']
            anchor=previous['anchor_coef'];lr=previous['actor_lr'];stop_fraction=previous['stop_script_fraction']
            freeze=previous['freeze_proprio'];route_envs=previous['route_envs'];free_envs=previous['free_envs']
    if args.nominal_gait_first:
        anchor=args.flat_anchor_coef if args.flat_gait_first else args.nominal_anchor_coef;lr=args.nominal_actor_lr
    try:
        if best_screen is None:
            best_screen=screen(best,free,'initial',route_config=route)
            state.update(initial_screen=best_screen,selected_score=best_screen['score']);save_json(folder/'status.json',state)
        if args.screen_only:
            state.update(status='screen_only_finished');save_json(folder/'status.json',state);return
        while True:
            if (args.nominal_gait_first and not args.flat_gait_first
                and args.nominal_max_level<17 and best_screen['passed']):
                # 已评估最佳权重也能通过当前档位时，直接测下一档，不等待新候选重新通过。
                previous_level=args.nominal_max_level
                args.nominal_max_level=next(level for level in (0,6,9,12,15,17) if level>previous_level)
                state['controller_config']['nominal_max_level']=args.nominal_max_level
                training_checkpoint=best;state['training_checkpoint']=best
                state.setdefault('terrain_baseline_advances',[]).append(dict(checkpoint=best,
                    passed_level=previous_level,next_level=args.nominal_max_level,screen=best_screen))
                next_route,next_free=configured_tasks(strength,route_envs,free_envs,85000+state['round'])
                best_screen=screen(best,next_free,f'baseline_round_{state["round"]:05d}_level{args.nominal_max_level}',
                    route_config=next_route)
                stale=0
                state.update(best_screen=best_screen,selected_score=best_screen['score'])
                save_json(folder/'status.json',state)
                continue
            state['round']+=1;round_id=state['round'];label=f'round_{round_id:05d}'
            route_cfg,free_cfg=configured_tasks(strength,route_envs,free_envs,85000+round_id)
            if not pending_run and not args.front_posture_finetune and not args.require_normal_gait and stale and stale%args.visual_refresh_every==0:
                # PPO固定视觉编码器以控制显存；停滞时单独顺序BPTT训练视觉，教师始终冻结。
                aligned=deepcopy(route_cfg)
                aligned.randomization.frame_delay_steps=0;aligned.randomization.frame_dropout=0.
                aligned.num_envs=64
                config_path=folder/(label+'_visual_environment.json');save_json(config_path,asdict(aligned))
                teacher=ROOT/'outputs/20261105_212457_360463_teacher_seed62001/checkpoints/final.pt'
                text=run('scripts.distill_student',['--teacher',teacher,'--resume',best,
                    '--env-config',config_path,'--num-envs',64,'--iterations',args.visual_updates_per_refresh,
                    '--seed',87000+round_id,'--terrain-seed',88000+round_id,
                    '--visual-steps',20,'--burn-in',5,'--sequence-batch',16,'--visual-lr',2e-5,
                    '--common-lr',0,'--freeze-common','--latent-coef',.02,'--teacher-mix',.5,
                    '--training-levels',0,6,9,12,15,17,17,17],label+'_visual_train')
                visual_dir=Path(next(x.split('=',1)[1] for x in text.splitlines() if x.startswith('DISTILLATION_FINISHED=')))
                visual_ckpt=str(visual_dir/'checkpoints/final.pt')
                visual_result=screen(visual_ckpt,free_cfg,label+'_visual',route_config=route_cfg)
                visual_accepted=visual_result['score']>best_screen['score']
                state.setdefault('visual_records',[]).append(dict(round=round_id,checkpoint=visual_ckpt,
                    screen=visual_result,selected=visual_accepted,teacher=str(teacher),teacher_frozen=True))
                if visual_accepted:best=visual_ckpt;best_screen=visual_result
                save_json(folder/'status.json',state)
            resume_checkpoint=training_checkpoint if (args.nominal_gait_first and args.nominal_continuous_updates) else best
            arguments=['--resume',resume_checkpoint,'--reference',reference_path,'--visual',args.visual_source,
                       '--iterations',args.updates_per_round,'--route-envs',route_envs,'--free-envs',free_envs,
                       '--strength',strength,'--anchor-coef',anchor,'--actor-lr',lr,
                       '--std',.02 if args.front_posture_finetune else .04,'--stop-script-fraction',stop_fraction,'--seed',85000+round_id]
            if args.front_posture_finetune:
                arguments+=['--front-posture-scale',1. if args.flat_gait_first else min(1.,.1*round_id),'--free-anchor-weight',0.]
            if args.nominal_gait_first:
                arguments+=['--nominal-gait','--training-max-level',args.nominal_max_level,
                            '--route-std-scale',args.nominal_route_std_scale,'--free-std-scale',args.nominal_free_std_scale]
            if args.nominal_gait_first and args.nominal_terrain_reference:
                arguments+=['--terrain-reference',args.nominal_terrain_reference,
                            '--terrain-reference-transition',args.nominal_reference_transition]
            if args.flat_gait_first:arguments+=['--flat-gait-only','--route-std-scale',2.]
            if args.require_normal_gait:
                arguments+=['--normal-foot-scale',1.,'--free-anchor-weight',0.]
            if freeze:arguments+=['--freeze-proprio']
            try:
                if pending_run:
                    run_dir=Path(pending_run);pending_run=None
                    print(f'RECOVER_COMPLETED_TRAINING={run_dir}',flush=True)
                else:
                    text=run('scripts.train_student_multitask',arguments,label+'_train')
                    run_dir=Path(next(x.split('=',1)[1] for x in text.splitlines() if x.startswith('MULTITASK_FINISHED=')))
                state['pending_run']=str(run_dir);save_json(folder/'status.json',state)
                candidate=str(run_dir/'checkpoints/final.pt')
                metadata=json.loads((run_dir/'metadata.json').read_text())
                if metadata['numerical_failures']:
                    # 单批异常必须拒绝，但不应中止整个续训；从上一已评估权重重新采样。
                    training_checkpoint=best;state['training_checkpoint']=best
                    rejection=dict(round=round_id,strength=strength,checkpoint=candidate,selected=False,
                        screen=dict(passed=False,reason='numerical_failure',reports=[],
                                    numerical_failures=metadata['numerical_failures']))
                    state['records'].append(rejection);state.pop('pending_run',None)
                    failures=state.get('consecutive_numerical_rejections',0)+1
                    state.update(status='running',consecutive_numerical_rejections=failures,
                                 next_anchor=anchor,next_lr=lr)
                    save_json(folder/'status.json',state)
                    print(f'REJECT_NUMERICAL_CANDIDATE={candidate}',flush=True)
                    if failures>=3:raise RuntimeError('连续三批发生数值异常，需要检查仿真，未采用异常权重')
                    continue
                state['consecutive_numerical_rejections']=0
                cfg=env_from_dict(json.loads((run_dir/'task_configs.json').read_text())['free'])
                # 同时检查保存的中途权重，避免仅凭最后一次更新丢掉更好的正常步态。
                candidates=[Path(candidate)]
                if args.nominal_gait_first:
                    candidates+=sorted((run_dir/'checkpoints').glob('iteration_*.pt'))
                seen=set();candidate_screens=[]
                for path in candidates:
                    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
                    if checksum in seen:continue
                    seen.add(checksum)
                    trial_label=label if str(path)==candidate else label+'_'+path.stem
                    trial=screen(str(path),cfg,trial_label,route_config=route_cfg)
                    candidate_screens.append(dict(checkpoint=str(path),sha256=checksum,screen=trial))
                selected=max(candidate_screens,key=lambda item:item['screen']['score'])
                candidate=selected['checkpoint'];result=selected['screen']
                errors=0
            except subprocess.CalledProcessError as error:
                errors+=1;lr=max(1e-6,lr/2)
                state.setdefault('errors',[]).append(dict(round=round_id,error=str(error),log=state['log']))
                save_json(folder/'status.json',state)
                if errors>=3:raise
                continue
            accepted=result['score']>best_screen['score']
            record=dict(round=round_id,strength=strength,checkpoint=candidate,screen=result,
                        selected=accepted,anchor=metadata['arguments']['anchor_coef'],
                        actor_lr=metadata['arguments']['actor_lr'],freeze_proprio=metadata['arguments']['freeze_proprio'],
                        terrain_reference=metadata.get('terrain_reference'),
                        reference_target_revision=metadata.get('reference_target_revision','legacy_age_mask_v1'),
                        reference_transition=metadata['arguments'].get('terrain_reference_transition',0.),
                        candidate_screens=candidate_screens)
            if args.nominal_gait_first and args.nominal_continuous_updates:
                # 保留学习分支的累计更新；验收最佳不受影响。失败分支不直接发布。
                eligible=[trial for trial in candidate_screens if trial['screen']['front_stance']['passed']
                    and (trial['screen']['route_min_success'] is None or trial['screen']['route_min_success']>0)]
                if eligible:training_checkpoint=eligible[0]['checkpoint']
                state['training_checkpoint']=training_checkpoint
                record['next_training_checkpoint']=training_checkpoint
            state['records'].append(record)
            state.pop('pending_run',None)
            if accepted:
                best=candidate;best_screen=result;stale=0
            else:stale+=1
            if result['passed'] and args.nominal_gait_first and not args.flat_gait_first and args.nominal_max_level<17:
                # 当前高度必须同时保住正常平地步态，才增加训练难度；最终仍验收全部六档。
                best=candidate
                previous_level=args.nominal_max_level
                args.nominal_max_level=next(level for level in (0,6,9,12,15,17) if level>previous_level)
                state['controller_config']['nominal_max_level']=args.nominal_max_level
                training_checkpoint=best;state['training_checkpoint']=best
                record['terrain_curriculum']=dict(passed_level=previous_level,next_level=args.nominal_max_level)
                best_screen=screen(best,free_cfg,label+'_next_terrain_baseline',route_config=route_cfg)
                stale=0
                state.update(selected_checkpoint=best,best_screen=best_screen,selected_score=best_screen['score'])
                save_json(folder/'status.json',state)
                continue
            if result['passed']:
                validation=full_validation(candidate,cfg,label,final=strength==1.,route_config=route_cfg)
                record['validation']=validation
                if validation['passed']:
                    best=candidate;best_screen=result
                    if args.flat_gait_first:
                        # 平地只是中间课程。仍关闭随机化，回到完整标称复杂地形并重新测基线。
                        state['flat_gait_validation']=dict(checkpoint=best,validation=validation)
                        training_checkpoint=best;state['training_checkpoint']=best
                        args.flat_gait_first=False
                        reference_path=Path(best);ref=torch.load(best,map_location='cpu',weights_only=False)
                        source=env_from_dict(ref['env_config']);anchor=args.nominal_anchor_coef;stale=0
                        next_route,next_free=configured_tasks(0.,route_envs,free_envs,85000+round_id)
                        best_screen=screen(best,next_free,label+'_nominal_complex_baseline',route_config=next_route)
                        state.update(status='running',nominal_substage='complex',selected_checkpoint=best,
                            reference=str(reference_path),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                            best_screen=best_screen,selected_score=best_screen['score'],next_anchor=anchor,next_lr=lr)
                        save_json(folder/'status.json',state)
                        continue
                    if args.nominal_gait_first:
                        checksum=hashlib.sha256(Path(best).read_bytes()).hexdigest()
                        state.update(status='nominal_gait_passed_needs_visual_review',checkpoint=best,
                                     selected_checkpoint=best,best_screen=result,selected_score=result['score'],validation=validation,stage='gait_video_review')
                        save_json(folder/'status.json',state)
                        save_json(folder/'gait_review.template.json',dict(approved=False,checkpoint_sha256=checksum,
                            contact_timing_reviewed=False,foot_placement_reviewed=False,videos=validation['videos']))
                        if shutil.which('notify-send'):
                            try:
                                subprocess.run(['notify-send','Go2标称步态数值通过',
                                    '接触、落点和路线数值门槛通过，等待视频复核；域随机化仍关闭。'],timeout=10,check=False)
                            except (OSError,subprocess.TimeoutExpired) as error:
                                print(f'GAIT_REVIEW_NOTIFICATION_FAILED={error}',flush=True)
                        while True:
                            review_path=folder/'gait_review.json'
                            try:review=json.loads(review_path.read_text()) if review_path.exists() else {}
                            except json.JSONDecodeError:review={}
                            if (review.get('approved') is True and review.get('checkpoint_sha256')==checksum
                                and review.get('contact_timing_reviewed') is True and review.get('foot_placement_reviewed') is True
                                and review.get('videos')==validation['videos']
                                and all(Path(v).is_file() for v in validation['videos'])):break
                            time.sleep(30)
                        # 只有同一权重的实际视频复核通过，才开放随机化；使用正常步态参考保留落脚方式。
                        args.nominal_gait_first=False;args.front_posture_finetune=False;args.require_normal_gait=True
                        reference_path=Path(best);ref=torch.load(best,map_location='cpu',weights_only=False)
                        source=env_from_dict(ref['env_config']);strength=.25;anchor=20.;lr=5e-6
                        freeze=True;stale=0;route_envs=args.route_envs;free_envs=args.free_envs;stop_fraction=.25
                        state.update(status='running',phase='domain_randomization',strength=strength,
                            nominal_gait_first=False,front_posture_finetune=False,require_normal_gait=True,
                            nominal_gait_validation=validation,gait_review=str(review_path),reference=str(reference_path),
                            reference_sha256=checksum,selected_checkpoint=best)
                        next_route,next_free=configured_tasks(strength,route_envs,free_envs,85000+round_id)
                        best_screen=screen(best,next_free,label+'_start_randomization',route_config=next_route)
                        state.update(best_screen=best_screen,selected_score=best_screen['score'])
                        save_json(folder/'status.json',state)
                        continue
                    if strength==1.:
                        state.update(status='performance_passed_needs_delivery_review',stage='review',
                                     checkpoint=best,validation=validation)
                        save_json(folder/'status.json',state);print(f'CONTINUAL_PERFORMANCE_PASSED={folder}',flush=True);return
                    strength=min(1.,strength+.25)
                    next_route,next_free=configured_tasks(strength,route_envs,free_envs,85000+round_id)
                    # 新强度必须重新测基线，不能拿弱随机化的高分压住后续选模。
                    best_screen=screen(best,next_free,label+'_next_strength',route_config=next_route)
                    stale=0
            if args.front_posture_finetune:
                anchor=(args.flat_anchor_coef if args.flat_gait_first else args.nominal_anchor_coef) if args.nominal_gait_first else .2;freeze=True
                route_envs=args.route_envs;free_envs=args.free_envs
            elif best_screen['route_min_success']<1.:
                anchor=min(80.,anchor*1.1);freeze=True
                route_envs=args.route_envs+args.free_envs//2;free_envs=args.free_envs//2
            else:
                anchor=max(5.,anchor*.98);freeze=False
                route_envs=args.route_envs;free_envs=args.free_envs
            if not args.nominal_gait_first and best_screen['stop_passes']<3 and stale>=2:stop_fraction=min(.6,stop_fraction+.05)
            if stale>=5:lr=max(1e-6,lr*.8);stale=0
            state.update(status='running',strength=strength,selected_checkpoint=best,
                         selected_score=best_screen['score'],best_screen=best_screen,
                         next_anchor=anchor,next_lr=lr,next_stop_fraction=stop_fraction,
                         next_freeze_proprio=freeze,next_route_envs=route_envs,next_free_envs=free_envs)
            save_json(folder/'status.json',state)
            print(json.dumps(dict(round=round_id,strength=strength,selected=best,score=best_screen['score'])),flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed_needs_fix',error=str(error))
        save_json(folder/'status.json',state);raise


if __name__=='__main__':main()
