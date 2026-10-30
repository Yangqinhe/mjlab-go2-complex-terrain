"""从项目内 Go2 MJCF 创建 12 关节 PD 控制模型，保持原关节顺序。"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/unitree_go2/go2.xml"


def get_spec():
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    # XML 原 motor 接收力矩，不能与新位置执行器同时保留，否则重复驱动关节。
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for keyframe in list(spec.keys):
        spec.delete(keyframe)
    # MJWarp 的 MULTICCD 不支持此模型非零 margin 的碰撞组合。
    # 仅在运行时清零接触 margin，保留磁盘 XML 和实际碰撞体尺寸。
    for i, geom in enumerate(spec.geoms):
        geom.margin = 0.0
        if geom.group == 3 and not geom.name:
            geom.name = f"collision_{i}"
    # 场景挂接不会继承实体的全局 option；统一由 env 的 MujocoCfg 管理。
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    return spec


def make_robot_config(config):
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, config.base_height),
            joint_pos={".*_hip_joint": 0.0, ".*_thigh_joint": 0.9,
                       ".*_calf_joint": -1.8},
            joint_vel={".*": 0.0},
        ),
        sort_actuators=True,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                    stiffness=config.kp, damping=config.kd, effort_limit=23.7,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_calf_joint",), stiffness=config.kp,
                    damping=config.kd, effort_limit=45.43,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
