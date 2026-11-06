"""从已验证195权重分批续训，保存独立指令评估与负载/混合地形课程记录。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from rl.run_io import ROOT, env_from_dict, save_json


BASELINE = ROOT/'outputs/20261106_135343_676539_student_ppo_seed92037/checkpoints/final.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(BASELINE))
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--d435i-scenes', action='store_true', help='先适应20-50度及侧墙视觉，再续训控制')
    parser.add_argument('--skip-visual-adaptation', action='store_true', help='从已经适应的相机检查点继续，保留视觉权重')
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    original = env_from_dict(saved['env_config'])
    if args.d435i_scenes:
        from domain_randomization.config import d435i_scene_config
        original = d435i_scene_config(original)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = ROOT/'outputs'/f'{stamp}_domain_randomization_pipeline'
    folder.mkdir()
    source_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    status = dict(status='running', baseline=args.checkpoint, baseline_sha256=source_hash,
                  requested=dict(vx=[-.3,.3],vy=[0.,0.],wz=[-.5,.5],stop_test='0.3 -> 0'),
                  folder=str(folder), runs=[], evaluations=[], current_checkpoint=args.checkpoint,
                  d435i_scenes=args.d435i_scenes,skip_visual_adaptation=args.skip_visual_adaptation)
    save_json(folder/'status.json',status)
    print(f'DOMAIN_PIPELINE={folder}',flush=True)

    def run(command, log):
        with (folder/log).open('w') as stream:
            subprocess.run([sys.executable, '-m', *command], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        return (folder/log).read_text()

    def train(config, checkpoint, count, seed, label, lr=2e-5, std=.08):
        path=folder/(label+'_config.json')
        save_json(path,asdict(config))
        status.update(stage=label,current_checkpoint=checkpoint,log=str(folder/(label+'.log')))
        save_json(folder/'status.json',status)
        text=run(['scripts.train_go2_ppo','--stage','student_ppo','--resume',checkpoint,
                  '--env-config',str(path),'--num-envs',str(args.num_envs),
                  '--iterations',str(count),'--horizon','128','--batch-size','32768',
                  '--training-levels',*(['0'] if not config.terrain.obstacles else ['0','6','9','12','15','17']),
                  '--actor-lr',str(lr),'--no-freeze-proprio','--target-action-std',str(std),
                  '--save-interval','25','--seed',str(seed),'--reset-optimizer'],label+'.log')
        directory=Path(next(line.split('=',1)[1] for line in text.splitlines() if line.startswith('TRAINING_FINISHED=')))
        status['runs'].append(str(directory));status['current_checkpoint']=str(directory/'checkpoints/final.pt')
        save_json(folder/'status.json',status)
        return status['current_checkpoint']

    def commands(checkpoint,config,label):
        path=folder/(label+'_evaluation_config.json');save_json(path,asdict(config))
        report=folder/(label+'_commands.json')
        status.update(stage=label+'_commands',log=str(folder/(label+'_commands.log')))
        save_json(folder/'status.json',status)
        run(['scripts.evaluate_commands','--checkpoint',checkpoint,'--env-config',str(path),
             '--seeds','50100','50101','50102','--output',str(report)],label+'_commands.log')
        result=json.loads(report.read_text());status['evaluations'].append(str(report));save_json(folder/'status.json',status)
        return result

    from env.commands import free_training_config
    config=free_training_config(original,args.d435i_scenes)
    save_json(folder/'initial_free_config.json',asdict(config))
    checkpoint=args.checkpoint
    best=None;best_score=-1.
    try:
        if args.d435i_scenes and not args.skip_visual_adaptation:
            status.update(stage='d435i_visual_adaptation',log=str(folder/'d435i_visual_adaptation.log'))
            save_json(folder/'status.json',status)
            text=run(['scripts.adapt_camera_scene','--checkpoint',checkpoint,
                      '--num-envs','128','--iterations','250','--seed','82200'],
                     'd435i_visual_adaptation.log')
            adaptation=Path(next(line.split('=',1)[1] for line in text.splitlines()
                                 if line.startswith('VISUAL_ADAPTATION_FINISHED=')))
            checkpoint=str(adaptation/'checkpoints/final.pt')
            status.update(visual_adaptation=str(adaptation),current_checkpoint=checkpoint)
            save_json(folder/'status.json',status)
        for batch in range(args.batches):
            config.randomization.seed=81110+batch;config.seed=81110+batch
            checkpoint=train(config,checkpoint,args.iterations,81110+batch,f'free_batch{batch+1}')
            result=commands(checkpoint,config,f'free_batch{batch+1}')
            score=result['pass_rate']
            if score>best_score:
                best_score=score;best=checkpoint
            status.update(best_commands_checkpoint=best,best_commands_pass_rate=best_score)
            save_json(folder/'status.json',status)
            # 达到全部33个指令案例后，才能解锁负载1kg；不靠训练reward升级。
            if score >= .999:
                if config.randomization.payload_max < 1.:
                    config.randomization.payload_max=1.
                    config.randomization.friction=(.55,1.1)
                    config.randomization.gain_scale=(.9,1.1)
                else:
                    break
        # 正向复杂地形与自由倒退/转向是不同任务；各自保留权重并评估，禁止只凭最后权重发布。
        route=env_from_dict(asdict(original));route.randomization.enabled=True
        route.randomization.seed=81200;route.command.mode='route';route.command.vx=(.15,.3)
        route.command.vy=(0.,0.);route.command.wz=(0.,0.);route.command.stop_fraction=0.
        route.route.enabled=True;route.episode_seconds=200.
        route_checkpoint=train(route,best or checkpoint,100,81200,'complex_route',lr=1e-5,std=.05)
        for level in (0,17):
            status.update(stage=f'complex_route_evaluate_level{level}')
            save_json(folder/'status.json',status)
            text=run(['scripts.evaluate','--checkpoint',route_checkpoint,'--num-envs','4',
                      '--seeds','50200','50201','--level',str(level)],f'complex_evaluate_{level}.log')
            marker=next(line.split('=',1)[1] for line in text.splitlines() if line.startswith('EVALUATION='))
            status['evaluations'].append(marker)
        commands(route_checkpoint,config,'after_complex')
        # 只完成预算不代表新任务通过；完整矩阵、独立seed和步态仍需后续验收。
        status.update(status='batch_budget_finished_needs_review',stage='review',
                      complex_checkpoint=route_checkpoint)
        assert hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()==source_hash
        save_json(folder/'status.json',status)
        print(f'DOMAIN_BATCHES_FINISHED={folder}',flush=True)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        save_json(folder/'status.json',status)
        raise


if __name__ == '__main__':main()
