"""复核参考不依赖回合年龄，原统计宽限保留，障碍前标签平滑建立。"""

from types import SimpleNamespace

import torch

from env.config import EnvConfig
from env.gait import flat_mask,flat_terrain_mask
from rl.reference_targets import terrain_reference_weight


def main():
    config=EnvConfig();config.terrain.obstacles=True
    task=SimpleNamespace(config=config,dt=.02,
        env=SimpleNamespace(episode_length_buf=torch.zeros(4)))
    x=torch.tensor([1.,3.5,5.,9.])
    assert flat_terrain_mask(task,x).tolist()==[True,False,True,False]
    assert not flat_mask(task,x).any()
    initial=terrain_reference_weight(task,x,.8)
    assert terrain_reference_weight(task,x).tolist()==[0.,1.,0.,1.]
    task.env.episode_length_buf.fill_(50)
    assert torch.equal(initial,terrain_reference_weight(task,x,.8))
    assert torch.equal(flat_mask(task,x),flat_terrain_mask(task,x))
    approaching=terrain_reference_weight(task,torch.tensor([1.,2.3,2.5,2.7,3.]),.8)
    assert approaching[0]==0 and approaching[-1]==1
    assert (approaching[1:]>=approaching[:-1]).all()
    assert ((initial>=0)&(initial<=1)).all()
    adjacent=terrain_reference_weight(task,torch.tensor([2.64,2.66]),.8)
    assert (adjacent[1]-adjacent[0]).abs()<.1
    print('REFERENCE_TARGETS_PASSED')


if __name__=='__main__':main()
