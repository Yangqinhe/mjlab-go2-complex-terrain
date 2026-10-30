# Go2 模型来源

迁自工作区 `mujoco_project_go2_stairs/models/unitree_go2/`，保留原始 XML、素材和 LICENSE。
原说明中的来源为 Unitree Go2 URDF 转换的 MJCF。

磁盘 XML 未改写；加载时移除原 torque motor 与 keyframe，在 mjlab 中创建 12 个 PD 位置执行器。
为适配 MuJoCo Warp MULTICCD，在运行时将 geom margin 从 XML 的 0.001 m 改为 0，碰撞体尺寸不变。
实体全局 cone/impratio 运行时恢复默认，实际场景值由 env 中 MujocoCfg 统一设置（elliptic、10）。
保留 XML 的被动关节阻尼、armature 和 frictionloss，不能视为与旧显式 PD 控制器完全相同。
XML SHA-256：`50adb09a4365293e2acdaf2010ae35a82b0f09fea18ae51806fef91e310ed04a`。
