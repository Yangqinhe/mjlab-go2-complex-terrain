# Go2 复杂地形 GPU 并行强化学习

2026-11-05根据平地跳跃步态反馈重新训练：旧代码已独立备份，全部历史`outputs/`保留。新教师已逐档通过平地、3/9/12/15/18/20cm路线及步态验收；第2000轮20cm为20/20完成、最长腾空0.02s，各档代表录像和接触曲线已检查，最终教师中间高度的新seed复查也全部通过。视觉蒸馏已完成，学生PPO第200轮通过3～18cm；20cm虽17/20完成，但最长腾空0.26s，正在继续修正。随后还需新seed全高度最终验收，整体训练尚未完成。见[步态纠正与重新训练](docs/步态纠正与重新训练.md)。此前旧策略的路线通过结果不代表步态合格。

此前路线纠偏和旧策略的训练命令、运行目录与历史数据见[持续训练与路线纠偏记录](docs/持续训练与路线纠偏记录.md)。

2026-11-06：经数据覆盖修正和固定教师共享网络，蒸馏1000轮纯视觉学生六档路线完成率95%～100%，20cm为20/20；但9/12/18/20cm仍有腾空超限。已保留所有候选和六档比较基线，转入固定1e-6学习率的小幅PPO微调；最终步态及新seed全高度验收尚未完成。

PPO50轮六档选模集已全部通过；新seed复核虽120/120走完，但20cm有1例0.08s腾空超限，仍未发布，继续微调至100轮。

后续微调出现高档退化，已保留全部候选。根据同策略探索噪声诊断，当前改用std目标0.02、每批25次、固定1e-6学习率，并冻结本体特征继续动作头微调；最终独立验收尚未完成。

目标仍为**单级台阶最高20cm**；旧教师虽通过路线评估，步态验收不足，现从平地重训。[旧高台阶课程记录](docs/高台阶20cm课程与训练记录.md)保留逐档参数和实验历史。

2026-11-02已从第2400轮继续训练，采用较大minibatch、较长rollout及KL自适应学习率；当前运行目录和结果见[PPO续训与对照记录](docs/PPO续训与对照记录.md)。

目标：基于 mjlab / MuJoCo Warp 构建 Go2 复杂地形环境，沿用已有自写 PPO，支持无窗口训练、训练后评估和 PPT 视频导出。每个正式任务环境包含平地、上楼梯、下楼梯、碎石的连续混合路线，楼梯高度按配置和难度变化。

`terrains/` 管理同一场景中的连续混合地形，`domain_randomization/` 预留后期域随机化，当前关闭。固定速度指令为机体系 `[vx, vy, wz] = [0.3, 0, 0]`。

## 当前可运行内容

已接入自写 PPO 的批量训练、176维教师/视觉学生网络、48×64双帧深度、顺序蒸馏、评估和离屏录像。教师以2048环境训练，视觉阶段默认256环境；具体行为是否达标以独立评估JSON为准，训练reward不能代替通过率。当前运行记录见[训练实施与运行记录](docs/训练实施与运行记录.md)。

训练输入为45维本体及历史，不使用旧加载测试的48维观测。学生融合：当前本体45 + 本体历史特征64 + 深度CNN/GRU特征64 + 估计速度3 = 176；输出12维关节位置偏移。没有age/valid特征。

```bash
cd /home/user/Documents/YQH/工作/强化学习实验/mujoco_go2/mjlab_project_go2_complex_terrain

# 独立环境；当前机器已安装，换机器时按锁文件安装
uv sync --frozen

# 固定 vx=0.3，单环境 GPU 加载与步进，离屏导出两张场景图
./run.sh python -m scripts.test_env --num-envs 1 --steps 200 --render

# 16 个环境，允许初始分配到三档难度，默认无窗口
./run.sh python -m scripts.test_env --num-envs 16 --level 2 --steps 200

# 2048 个独立 Go2 世界，复用三档静态地形几何，机器人状态彼此独立
./run.sh python -m scripts.test_env --num-envs 2048 --level 2 --steps 200

# 有图形桌面时可打开窗口（本次仅验证了离屏渲染）
./run.sh python -m scripts.test_env --viewer
```

`run.sh` 默认按 UUID 选择本机 RTX 3070；换机器时设置对应 `CUDA_VISIBLE_DEVICES`。核心环境只接收 `device`，不依赖个人路径或原项目目录。无需自动降级到 CPU；如需 CPU 调试，可显式传 `--device cpu`，本次未测试此路径。

测试输出位于 `outputs/<时间>_load_env_seed42/`，包括配置、地形清单、检查报告及可选 PNG。当前混合路线为平地 → 上楼梯 → 平台 → 下楼梯 → 过渡 → 静态碎石 → 终点；三档单级台阶高度为 3/6/9 cm。`--level` 为初始最高难度，不是强制所有环境使用该等级。

路线尺寸 13.7 m × 3 m 是每台机器人可走的单条路线，不是将 2048 台机器人放入同一个物理世界。早期2048环境零动作测试各200步耗时约3.27秒；该吞吐不代表PPO训练。当前已接入2048环境教师PPO、8档难度×4种碎石布局，RTX3070占用约3.3GB，训练统计见各运行CSV。视觉阶段默认256，2048视觉容量仍待单独测量。

## 方案与约定

- [代码详解：网络、环境、PPO与迁移部署](docs/代码详解_网络环境训练与部署.md)：按当前源码解释入口、教师/学生网络、观测和奖励、GPU并行、蒸馏、评估，以及Sim2Sim/Sim2Real的实现边界与后续接口。
- [文件格式与代码风格要求](docs/文件格式.md)
- [方案设计](docs/方案设计.md)：独立地形目录、单环境混合路线、可变台阶高度、域随机化预留、参数管理、PPO 迁移、录像、代码风格与 GitHub 交付。
- [固定速度行走：环境与地形加载方案](docs/固定速度行走_环境与地形加载方案.md)：本阶段的实际文件、接口、运行命令与后续训练边界。
- [视觉与本体感知融合：176维实施方案（当前版）](docs/视觉与本体感知融合_176维实施方案.md)：已获批准实施，四路176维融合与12关节输出。
- [旧178维技术方案（历史记录）](docs/视觉与本体感知融合_技术设计与实施方案.md)：已被176维方案取代，不再作为实施依据。
- [旧178维网络图说明（历史记录）](docs/视觉本体融合网络图说明.md)：PNG尚未重绘，其中帧状态和178维融合不适用于当前方案。
- [变更记录](CHANGELOG.MD)

## 训练与评估

```bash
# 最小检查：GAE/超时bootstrap/重置隔离；深度标定/调度/网络维度
./run.sh python -m scripts.test_training
./run.sh python -m scripts.test_perception

# 全流程：平地教师 → 混合地形教师 → 蒸馏 → 冻结视觉PPO → 评估与录像
# 教师未通过20个独立布局的低台阶评估时，不自动启动蒸馏
./run.sh python -m scripts.train_pipeline

# 从已有复杂地形教师继续，跳过已完成的平地/低障碍阶段，逐档训练到20cm目标
./run.sh python -m scripts.train_pipeline --teacher-start <teacher.pt> --skip-pretraining --max-step-height 0.20

# 等待指定流水线；成功后打包完整学生pt/配置/评估/视频并上传本项目GitHub Release
# 未达标或流水线失败时停止，不上传“成功模型”
./run.sh python -m scripts.publish_result --pipeline <outputs/pipeline目录> --wait

# 单独启动教师，或从已有教师继续到复杂地形；resume会新建实验目录
./run.sh python -m scripts.train_go2_ppo --flat --iterations 300 --actor-lr 0.0001 --velocity-coef 0.01 --kl-interval 1
./run.sh python -m scripts.train_go2_ppo --resume <teacher.pt> --complex --iterations 500

# 视觉蒸馏前50轮冻结公共网络，150轮后完全由学生控制；建议至少200轮
./run.sh python -m scripts.distill_student --teacher <qualified_teacher.pt> --iterations 200
./run.sh python -m scripts.train_go2_ppo --stage student_ppo --resume <student.pt> --num-envs 256 --iterations 300

# 确定性评估；--max-steps仅作片段诊断，会标记partial_evaluation
./run.sh python -m scripts.evaluate --checkpoint <checkpoint.pt> --seeds 10000 10001 10002 10003 10004
./run.sh python -m scripts.record_video --checkpoint <checkpoint.pt> --seeds 10005 --video <output.mp4>
./run.sh python -m scripts.plot_training <outputs/实验目录/metrics.csv>
```

`outputs/<时间>_<阶段>_seed42/`包含配置、地形清单、CSV、checkpoint和评估。流水线另保存各阶段stdout和`status.json`，所有训练均不启动窗口。`latest.pt`用于继续训练，`final.pt`只表示本轮结束，是否通过验收仍须查看评估。resume恢复网络/优化器/随机状态，但从新回合开始，不声称逐物理步复现。

实际课程为平地→5mm–3cm过渡→上限9/12/15/18/20cm五档。每档同时通过3cm和该档最高台阶评估才升档；教师完成目标档后再蒸馏，学生按3/9/12/15/18/20cm逐档验收。各档动作尺度、关节限位和训练目标分别保存。代码仓库：[Yangqinhe/mjlab-go2-complex-terrain（私有）](https://github.com/Yangqinhe/mjlab-go2-complex-terrain)。

## 已知限制

当前深度为静态地形射线，只渲染地形碰撞组，不包含机器人自遮挡、真实相机噪声和延迟；域随机化关闭。布局变化主要是碎石位置/尺寸。基础地形8档3–9cm；目标档18档3–20cm的场景已通过加载检查，行走能力须另行验收。48/58/历史178维旧权重均不兼容当前`perceptive_176_v1`。

加载时为 MJWarp 适配替换执行器、清零 margin，并用场景配置统一接触参数；磁盘原 XML 保留。后端对部分 cylinder 碰撞组合会提示最多生成单接触点，这属于当前后端支持范围，后续训练前应结合行为评估，不能据此声称接触与旧 MuJoCo 完全一致。
