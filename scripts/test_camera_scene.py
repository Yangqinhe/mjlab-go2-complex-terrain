"""核对20～50°射线、墙壁可见性/碰撞、高度扫描隔离并导出真实仿真预览。"""

from dataclasses import asdict
from pathlib import Path
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from domain_randomization.config import d435i_scene_config
from env.perception.depth_sensor import DepthSensor
from env.training_task import Go2Task
from rl.run_io import ROOT, load_policy, save_json




def main():
    torch.set_num_threads(1)
    folder=ROOT/'outputs/20261106_d435i_scene_setup';folder.mkdir(exist_ok=True)
    _, original, _=load_policy(ROOT/'outputs/20261106_200325_915130_student_ppo_seed81112/checkpoints/final.pt')
    config=d435i_scene_config(original)
    task=Go2Task(config,curriculum=False)
    try:
        terrain=task.env.scene.terrain
        terrain.terrain_levels[:]=0
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
        task.reset()
        # 只为相机检查放在首级前0.7m，不作为路线成功样本。
        root=task.robot.data.default_root_state.clone()
        root[:,:3]+=task.env.scene.env_origins
        root[:,0]+=1.3
        task.robot.write_root_state_to_sim(root)
        task.env.sim.forward()
        sensor=DepthSensor(task);sensor.reset_mount(torch.arange(config.num_envs,device=task.device))
        sensor.capture()
        # 同相机分别包含/排除scene组，直接证明画面变化来自墙体/物体几何。
        clean=DepthSensor(task,config.camera,geometry_groups=[0],randomize=False)
        clean.capture()
        model=task.env.sim.mj_model
        assert any('_up_' in model.geom(i).name for i in range(model.ngeom))
        vec=clean.vec[0]
        distance=-clean.pnt[0,:,2]/vec[:,2].clamp(max=-1e-6)
        flat_depth=(distance*(-clean.directions[:,2])).clamp(.15,3.)
        visible_steps=int(((clean.raw_depth[0].flatten().clamp(.15,3.)-flat_depth).abs() > .005).sum())
        assert visible_steps>3, '楼梯必须实际出现在深度图中'
        groups=torch.tensor(model.geom_group,device=task.device)
        hit=sensor.geom.clamp(min=0)
        scene_pixels=((groups[hit] == 4) & (sensor.geom>=0)).sum(-1)
        assert scene_pixels.max()>0
        ids=torch.arange(config.num_envs,device=task.device)
        # 强制两个已知俯角，检查真正的中心光轴，而非只检查配置数字。
        for angle in (20.,50.):
            config.randomization.camera_pitch_degrees=(angle,angle)
            sensor.reset_mount(ids)
            direction=-sensor.rotations[:,:,2]
            measured=torch.rad2deg(torch.atan2(-direction[:,2],direction[:,0]))
            active=~task.env.dr_nominal
            assert (measured[active]-angle).abs().max()<1.1
        config.randomization.camera_pitch_degrees=(20.,50.)
        sensor.reset_mount(ids);sensor.capture()
        # HeightQuery表仅包含可行走组0，不将数米高墙顶当作足下高度。
        assert task.ground.boxes.shape[1] == len([i for i in range(model.ngeom)
            if model.geom(i).name.startswith('course_0000_v0_') and model.geom_group[i]==0])
        mid=task.env.scene.env_origins[:,:2].clone()
        assert task.ground(mid[:,None]).abs().max()<1e-5
        # CPU射线与GPU光轴深度对照，包含真实scene碰撞体。
        cpu=mujoco.MjData(model)
        for index in range(4):
            cpu.qpos[:]=task.env.sim.data.qpos[index].cpu().numpy()
            mujoco.mj_forward(model,cpu)
            group_mask=np.array([1,0,0,0,1,0],dtype=np.uint8)
            pixel=24*64+32
            origin=sensor.pnt[index,pixel].cpu().numpy();vec=sensor.vec[index,pixel].cpu().numpy()
            distance=mujoco.mj_ray(model,cpu,origin,vec,group_mask,1,-1,np.zeros(1,dtype=np.int32))
            expected=distance*(-sensor.directions[pixel,2].item()) if distance>=0 else config.camera.far
            assert abs(expected-sensor.raw_depth[index,24,32].item())<1e-4
        fig,axes=plt.subplots(2,4,figsize=(16,7))
        render_model=copy.copy(model);render_model.vis.map.znear=.02/render_model.stat.extent
        renderer=mujoco.Renderer(render_model,height=360,width=480)
        for column,mode in enumerate(('Open','Left wall','Right wall','Both walls')):
            index=int((terrain.terrain_types == column).nonzero()[0])
            tag=f'course_0000_v{column}_'
            for i in range(render_model.ngeom):
                if render_model.geom(i).name.startswith('course_'):
                    render_model.geom_group[i]=model.geom_group[i] if model.geom(i).name.startswith(tag) else 5
            data=mujoco.MjData(render_model);data.qpos[:]=task.env.sim.data.qpos[index].cpu().numpy();mujoco.mj_forward(render_model,data)
            camera=mujoco.MjvCamera();camera.distance=4.;camera.azimuth=110;camera.elevation=-75;camera.lookat[:]=data.qpos[:3]+np.array([1.,0.,0.])
            option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,1,0]
            renderer.update_scene(data,camera=camera,scene_option=option)
            axes[0,column].imshow(renderer.render());axes[0,column].set_title(mode)
            image=sensor.raw_depth[index].clamp(.15,3.).cpu().numpy()
            axes[1,column].imshow(image,vmin=.15,vmax=3.,cmap='viridis')
            axes[1,column].set_title(f'Depth: pitch {sensor.pitch_degrees[index].item():.1f} deg')
            for row in range(2):axes[row,column].axis('off')
        renderer.close();fig.tight_layout();fig.savefig(folder/'scene_preview.png',dpi=140);plt.close(fig)
        save_json(folder/'config.json',asdict(config))
        save_json(folder/'verification.json',dict(status='passed',
            checks=['camera_20_and_50_degree_actual_rays','walls_visible_to_depth','walls_excluded_from_walkable_height',
                    'cpu_gpu_optical_depth_matches','four_scene_layouts'],
            pitch_degrees=sensor.pitch_degrees.cpu().tolist(),scene_hit_pixels=scene_pixels.cpu().tolist(),
            camera=asdict(config.camera),visible_step_pixels=visible_steps,position_note='沿用机身前0.27m/上0.10m，未获得真实安装点标定'))
        print('D435I_SCENE_CHECKS_PASSED',folder,flush=True)
    finally:task.close()


if __name__=='__main__':main()
