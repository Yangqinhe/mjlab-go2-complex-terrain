"""将已验证的学生控制网络与教师地形编码组合成仅供采样的特权引导策略。"""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path

import torch

from rl.algorithms.perceptive_networks import PerceptiveActor
from rl.config import TrainConfig
from rl.run_io import ROOT, load_policy, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student',required=True)
    parser.add_argument('--terrain-teacher',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    student,_,saved=load_policy(args.student,'cpu')
    teacher,_,terrain_saved=load_policy(args.terrain_teacher,'cpu')
    assert student.mode=='student' and teacher.mode=='teacher'
    guide=PerceptiveActor('teacher',TrainConfig(**saved['train_config']))
    guide.copy_common(student)
    guide.terrain_encoder.load_state_dict(teacher.terrain_encoder.state_dict())
    guide.eval();guide.requires_grad_(False)
    # 等价于诊断中的学生动作网络接收同一个真实地形latent，不重置已学会的步态。
    states=torch.randn(16,682,generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        z=teacher.terrain_encoder(states[:,495:])
        expected=student(torch.cat([states[:,:495],z],-1))[0]
        actual=guide(states)[0]
    assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-6)
    folder=ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_privileged_gait_guide'
    folder.mkdir()
    provenance=dict(role='privileged_sampling_guide',student=args.student,
        student_stage=saved['stage'],student_iteration=saved['iteration'],
        student_sha256=hashlib.sha256(Path(args.student).read_bytes()).hexdigest(),
        terrain_teacher=args.terrain_teacher,
        terrain_teacher_sha256=hashlib.sha256(Path(args.terrain_teacher).read_bytes()).hexdigest(),
        constructed_not_trained=True,sampling_only=True,deployment_validation=False,
        sampling_height_updates="每策略周期50Hz；oracle对照为10Hz保持，需单独检验轨迹",
        common_network_max_error=(actual-expected).abs().max().item())
    # teacher表示加载时使用187维地形输入；iteration=0表示该组合没有另行训练。
    checkpoint=dict(format_version=saved['format_version'],
        observation_version=saved['observation_version'],network_version=saved['network_version'],
        actor=guide.state_dict(),critic=saved['critic'],
        env_config=saved['env_config'],train_config=saved['train_config'],
        stage='teacher',iteration=0,construction=provenance)
    path=folder/'guide.pt';torch.save(checkpoint,path)
    restored,_,_=load_policy(path,'cpu')
    with torch.no_grad():assert torch.allclose(restored(states)[0],actual,atol=1e-7,rtol=1e-6)
    save_json(folder/'construction.json',provenance)
    print('PRIVILEGED_GUIDE='+str(path),flush=True)


if __name__=='__main__':main()
