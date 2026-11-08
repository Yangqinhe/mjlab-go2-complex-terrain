"""先查正常平地再查完整路线，明确区分视觉学生与中间特权教师。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from env.gait import gait_passed
from rl.run_io import ROOT, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--iterations',type=int,nargs='+',required=True)
    parser.add_argument('--route-config')
    parser.add_argument('--policy-mode',choices=('student','teacher'),default='student')
    parser.add_argument('--level',type=int,default=9)
    parser.add_argument('--max-vx-rmse',type=float,default=.1)
    parser.add_argument('--seeds',type=int,nargs='+',default=[54000,54001,54002,54003,54004])
    args=parser.parse_args()
    privileged=args.policy_mode=='teacher'
    if not privileged and not args.route_config:
        parser.error('学生候选需要显式route-config；教师默认使用自身高度采样配置')
    folder=args.run.resolve();status_path=folder/('teacher_acceptance.json' if privileged else 'visual_acceptance.json')
    state=dict(status='running',runner_pid=os.getpid(),run=str(folder),
        arguments={**vars(args),'run':str(folder)},privileged_input=privileged,
        deployment_validation=not privileged,records=[])

    def save():save_json(status_path,state)

    def run(module,arguments,label):
        log=folder/(label+'.log')
        state.update(stage=label,log=str(log));save()
        with log.open('w') as stream:
            child=subprocess.Popen([sys.executable,'-m',module,*map(str,arguments)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            code=child.wait()
        if code:raise RuntimeError(f'{label}失败，见{log}')
        return log.read_text()

    prefix='TEACHER' if privileged else 'VISUAL'
    save();print(prefix+'_ACCEPTANCE_STATUS='+str(status_path),flush=True)
    try:
        for iteration in args.iterations:
            name=f'iteration_{iteration:06d}.pt' if privileged else f'iteration_{iteration}.pt'
            checkpoint=folder/'checkpoints'/name
            state.update(stage=f'wait_{iteration}',child_pid=None);save()
            while not checkpoint.exists():
                metadata=json.loads((folder/'metadata.json').read_text())
                if metadata['status']!='running':raise RuntimeError(f'缺少{checkpoint}')
                time.sleep(2)
            time.sleep(.5)  # 不在不可变检查点复制尚未完成时加载。
            normal_path=folder/f'candidate_{iteration}_normal_gait.json'
            run('scripts.audit_normal_gait',['--checkpoint',checkpoint,'--nominal',
                '--seeds',56200,'--output',normal_path],f'candidate_{iteration}_normal_gait')
            normal=json.loads(normal_path.read_text())
            record=dict(iteration=iteration,checkpoint=str(checkpoint),normal_report=str(normal_path),
                normal_passed=normal['passed'],normal_error=normal['error'],
                privileged_input=privileged,deployment_validation=not privileged,adopted=False,passed=False)
            if normal['policy_mode']!=args.policy_mode:
                raise RuntimeError('候选角色与指定policy-mode不一致')
            if normal['passed']:
                route_args=['--checkpoint',checkpoint,'--num-envs',4,
                    '--seeds',*args.seeds,'--level',args.level]
                if args.route_config:route_args+=['--env-config',args.route_config]
                text=run('scripts.evaluate',route_args,
                    f'candidate_{iteration}_route{args.level}')
                path=Path(next(line.split('=',1)[1] for line in text.splitlines() if line.startswith('EVALUATION=')))
                report=json.loads(path.read_text())
                record.update(route_report=str(path),success_rate=report['success_rate'],
                    vx_rmse=report['vx_rmse'],max_flight=report['gait']['all_max_flight'],
                    episodes=report['episodes'],unique_layouts=report['unique_layouts'],
                    passed=(report['success_rate']==1. and report['vx_rmse']<=args.max_vx_rmse
                        and gait_passed(report) and not report['partial_evaluation']
                        and report['numerical_failure_rate']==0.
                        and report['episodes']==4*len(args.seeds)
                        and report['unique_layouts']==4*len(args.seeds)
                        and report['visual_ablation']=='none' and not report['proprio_only']
                        and report['privileged_input']==privileged))
            state['records'].append(record);save();print(json.dumps(record),flush=True)
        state.update(status='finished',child_pid=None);save()
        print(prefix+'_ACCEPTANCE_FINISHED='+str(status_path),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
