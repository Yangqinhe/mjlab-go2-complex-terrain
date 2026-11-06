"""平均同一训练分支的两个actor快照；不增加推理模块，结果仍须完整独立验收。"""

import argparse
import hashlib
from pathlib import Path
import shutil

import torch

from rl.config import TrainConfig
from rl.run_io import env_from_dict, new_run, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--second-weight', type=float, default=.5)
    args = parser.parse_args()
    if not 0 <= args.second_weight <= 1:
        parser.error('--second-weight须在[0,1]')
    paths = [args.first.resolve(), args.second.resolve()]
    first, second = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
    if (first['stage'] != 'student_ppo' or second['stage'] != 'student_ppo'
            or first['env_config'] != second['env_config']
            or first['train_config'] != second['train_config']):
        raise ValueError('只平均同一配置下的视觉PPO学生快照')
    for name, value in second['actor'].items():
        second['actor'][name] = torch.lerp(first['actor'][name], value, args.second_weight)
    # actor已改变，旧Adam动量不再对应此权重；critic仍完整保留第二份快照。
    second['actor_optimizer']['state'].clear()
    provenance = dict(method='actor_parameter_average', second_weight=args.second_weight,
        sources=[dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in paths], additional_ppo_updates=0,
        critic_source=str(paths[1]), actor_optimizer_state='cleared',
        acceptance='Independent full-route and gait evaluation required')
    second['actor_average'] = provenance
    folder = new_run('student_average', env_from_dict(second['env_config']),
                     TrainConfig(**second['train_config']))
    for name in ['deployment.json', 'metrics.csv']:
        shutil.copyfile(paths[1].parent.parent/name, folder/name)
    torch.save(second, folder/'checkpoints/final.pt')
    save_json(folder/'metadata.json', provenance)
    print(f'AVERAGED_CHECKPOINT={folder / "checkpoints/final.pt"}', flush=True)


if __name__ == '__main__':
    main()
