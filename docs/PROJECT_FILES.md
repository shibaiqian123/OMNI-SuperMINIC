# 项目文件说明

本文面向准备阅读和扩展本项目的开发者，说明当前文件的职责、数据流和修改入口。所有路径相对于仓库根目录，文档中的链接可直接打开对应文件。

## 1. 核查范围与当前状态

- 仓库：229 服务器 `/home/cqu023/SuperMimic-Repro-Sim2Real`。
- 源码核查基准：`2f46eaa2f6a8dc46ebd81da6c117052a6f439150`，2026-11-06。
- 依据：实际目录、Python 模块说明与顶层定义、关键训练/策略/部署代码，以及现有 README 和文档。本说明不等于完整代码审计。
- 当前实现是 G1 参考动作跟踪：仿真采样、学习 World Model、通过可微 imagination/BPTT 更新策略，再导出 ONNX 进行 Sim2Sim 和部署。
- OmniMimic 的 FSQ、多模态 Encoder、实时 VR Teleop、VLA Token/Residual、分离式 ONNX 尚属于 [开发清单](to_do_list.md)，不能把清单中的建议文件当作已有实现。
- 本次仅补充文档，没有运行训练、测试、导出、仿真或机器人控制。仓库中上游报告的实验结果与本次验证状态分开理解。

## 2. 目录总览

```text
SuperMimic-Repro-Sim2Real/
├── README.md / README_EN.md     项目入口和使用说明
├── pyproject.toml / uv.lock     依赖、命令入口和环境锁定
├── LICENSE / LICENSES/         项目及第三方许可
├── docs/                       方法、数据、部署和开发计划
├── motions/                    仓库附带的动作 CSV、拼接信息和重置点
├── src/supermimic/              可导入的核心实现
│   ├── config.py               训练配置树
│   ├── scripts/                安装后的训练命令入口
│   ├── training/               训练总循环、分布式和学习率
│   ├── data/                   转换、采样记录和 ReplayBuffer
│   ├── envs/                   仿真环境及 MDP 各项
│   ├── policy/                 Goal Encoder、Decoder、Critic
│   ├── world_model/            状态表示、动力学与模型拟合
│   ├── imagination/            可微 rollout 和策略优化
│   ├── kinematics/             G1 可微 FK 和旋转数学
│   ├── eval/                   统一评测实现
│   └── deploy/                 模型导出及共用部署核心
├── scripts/                    训练辅助、评估、数据和部署入口
│   ├── data/
│   ├── train/
│   └── deploy/bridge/          C++ DDS bridge 与 Python 适配
├── tools/video2motion/         视频→人体估计→G1 动作工具
└── tests/                      当前回归测试
```

`data/`、`logs/`、`deploy_artifacts/`、`deploy_logs/`、`work/video2motion/` 是工具使用的输入或产物位置，不表示这些目录、数据或权重已经随当前源码提供。实际路径由配置和脚本参数决定。

## 3. 先理解四条主链路

### 3.1 训练链路

`sm-train → scripts/train.py → training/loop.py:Trainer` 负责构造环境、策略、World Model、Critic 和优化器。环境执行策略动作后，`data/recorder.py` 将状态、动作及上下文送入 `data/replay_buffer.py`；`world_model/wm_trainer.py` 从回放窗口拟合预测模型；`imagination/rollout.py` 使用模型进行可微展开，`imagination/policy_trainer.py` 更新 Actor 与 Critic。

这里的 `scripts/train.py` 指包内 `src/supermimic/scripts/train.py`，不是仓库根目录 `scripts/train/` 中的特定实验配方。

### 3.2 动作数据链路

G1 retarget CSV → `scripts/data/csv_to_npz_batch.py` → `data/motion_convert.py` 重采样和 FK → NPZ。大数据集可经筛选和 `build_library.py` 打包；`envs/mdp/motion_library.py` 提供参考状态与未来 goal，`commands.py` 负责片段推进、采样和 reset。

视频路线在前面多一段：`tools/video2motion/run_pipeline.sh` 调用外部 GVHMR/GMR 环境，输出 G1 CSV，然后进入上述转换流程。视频人体估计输出不等于训练系统已经支持 SMPL 模态输入。

### 3.3 策略与 imagination 链路

默认 G1 future goal 680D → `policy/nets.py:MotionEncoder` → 32D 连续 latent；latent 与 930D proprio history 一起进入 `Decoder`，输出 29D raw action。imagination 将该动作传给 World Model，再通过解析 FK、reward 和 observation 构造下一步输入，形成多步梯度链。

Actor 更新时冻结 WM/Critic 的参数，不代表可以对 WM 输入或输出随意 `detach()`。新增 FSQ 必须保留所需的动作到目标函数梯度。

### 3.4 部署链路

训练 checkpoint → `deploy/export.py` → 单体 `policy.onnx`、配置、参考动作等产物 → `deploy/core.py:DeployCore` → Sim2Sim 或 G1 runner。bridge 版本再由 `g1_bridge_io.py`、C++ bridge 和 Unitree SDK 负责 DDS 通信。DeployCore 是仿真与真机共享的策略输入、动作转换和部分安全检查入口；runner/bridge 还承担各自的状态机、通信监控及保护，不能只替换网络而绕过这些层。

## 4. 核心接口约定

- 默认 proprio 单帧为 93D：关节相对位置29、关节速度29、机体角速度3、投影重力3、上一 raw action29。10帧历史为930D，使用 term-major、旧→新排列，不是直接逐帧拼接。
- 默认 goal 每帧68D：关节位置29、速度29、投影重力3、机体角速度3、heading坐标线速度3、高度1。10个未来参考帧为680D；部署按当前 k 的 k+1…k+10 取值并处理片尾。
- `anchor_obs` 可选追加9D：相对位置3+相对旋转6；默认 goal 为689D。当前没有独立的 Actor orientation-only 6D 模式。
- WM canonical feature 中已有6D旋转表示；Critic 的 `e_root` 是位置误差3+yaw误差1。这两个概念与 Actor 的9D anchor输入不同。
- 默认 latent 为32D连续量，当前没有2×32 FSQ。未来 FSQ 的64D编码不能直接加载到原32D Decoder，也不能默认复用 SONIC token 语义。
- raw action 为29D，部署目标关节角为 `q_target = default_joint_pos + action_scale * action`，不能把 raw action 直接发送成电机角度。
- 训练/部署控制约定为50Hz。CSV输入四元数为xyzw，转换后的内部约定为wxyz；坐标、关节顺序和参考时刻必须连同维度一起核对。

## 5. 根目录与现有文档

- [README.md](../README.md)：中文总入口：来源、安装、数据准备、训练、评测和部署路线。
- [README_EN.md](../README_EN.md)：英文项目说明。
- [pyproject.toml](../pyproject.toml)：依赖和开发工具配置；注册 `sm-train = supermimic.scripts.train:main`。训练 Python 约束为 >=3.12,<3.13，mjlab 固定到指定提交。
- [uv.lock](../uv.lock)：依赖解析锁文件；环境复现时应与 pyproject.toml 配套。
- [.python-version](../.python-version)：Python 版本选择文件。
- [.gitignore](../.gitignore)：Git 忽略规则；不代表被忽略的运行产物不存在。
- [LICENSE](../LICENSE)：仓库自身 MIT 许可。
- [LICENSES/THIRD_PARTY_NOTICES.md](../LICENSES/THIRD_PARTY_NOTICES.md)：第三方代码及相关资源声明。
- [LICENSES/unitree_sdk2-BSD-3-Clause.txt](../LICENSES/unitree_sdk2-BSD-3-Clause.txt)：Unitree SDK2 的许可文本。
- [docs/METHOD.md](../docs/METHOD.md)：方法细节、指标、超参数及与论文不同的实现选项。
- [docs/CUSTOM_MOTIONS.md](../docs/CUSTOM_MOTIONS.md)：自定义动作的数据准备、训练与部署流程。
- [docs/VIDEO_TO_MOTION.md](../docs/VIDEO_TO_MOTION.md)：视频转动作的外部依赖、环境和处理流程。
- [docs/SIM2SIM.md](../docs/SIM2SIM.md)：部署导出、仿真门禁及历史评估说明。
- [docs/DEPLOY_G1.md](../docs/DEPLOY_G1.md)：真机部署操作手册：模式、按键、状态机、保护及未验证事项。
- [docs/REAL_ROBOT_LEAN35KFS.md](../docs/REAL_ROBOT_LEAN35KFS.md)：特定转体/落地动作与 e24 模型的经验记录；不能当作通用模型验收结论。
- [docs/REFERENCES.md](../docs/REFERENCES.md)：引用与 BibTeX。
- [docs/to_do_list.md](../docs/to_do_list.md)：OmniMimic P0–P6 未来开发清单；当前已放在 docs 内。
- [docs/assets/shanshenbu_demo.gif](../docs/assets/shanshenbu_demo.gif)：README 使用的演示资源。
- [docs/assets/shanshenbu_pro_demo.gif](../docs/assets/shanshenbu_pro_demo.gif)：README 使用的 Pro 演示资源。

## 6. 训练入口与调度

- [src/supermimic/config.py](../src/supermimic/config.py)：`DataCfg/EnvCfg/ScheduleCfg/TrainConfig` 组成配置树；新增训练选项首先检查这里及下游透传。
- [src/supermimic/scripts/train.py](../src/supermimic/scripts/train.py)：`main()` 解析训练参数、处理启动并进入训练逻辑。
- [src/supermimic/training/loop.py](../src/supermimic/training/loop.py)：`Trainer` 组装环境与各网络，组织采样→WM拟合→策略更新，管理日志、checkpoint和恢复；goal维度也在这里设置。
- [src/supermimic/training/distributed.py](../src/supermimic/training/distributed.py)：多进程多卡初始化、参数广播和显式梯度 all-reduce；当前不是 DDP wrapper，新增分支需核对参与同步的梯度。
- [src/supermimic/training/schedule.py](../src/supermimic/training/schedule.py)：`cosine_lr()` 计算学习率调度。

## 7. 数据与仿真环境

- [src/supermimic/data/motion_convert.py](../src/supermimic/data/motion_convert.py)：`convert_csv_to_npz()` 处理 G1 CSV、四元数顺序、重采样、速度与CPU MuJoCo FK；`G1_JOINT_NAMES` 是重要顺序契约。
- [src/supermimic/data/recorder.py](../src/supermimic/data/recorder.py)：`ReplayRecorder` 在环境 step/reset 钩子中记录转移，连接仿真与回放。
- [src/supermimic/data/replay_buffer.py](../src/supermimic/data/replay_buffer.py)：`ReplayBuffer/WindowBatch/StartBatch` 管理按环境与时间排列的GPU环形回放；区分clean action和executed action，处理terminal行与连续窗口。
- [src/supermimic/envs/tracking_env_cfg.py](../src/supermimic/envs/tracking_env_cfg.py)：`make_env_cfg()/make_env()` 组装机器人、观测、动作、command、reward、终止和可选模型扰动。
- [src/supermimic/envs/actions.py](../src/supermimic/envs/actions.py)：`NoisyJointPositionAction` 保留clean action供历史观测使用，执行时添加噪声；包含动作延迟相关逻辑。
- [src/supermimic/envs/state_io.py](../src/supermimic/envs/state_io.py)：`StateIO` 读写71D完整状态；锁腰模型也映射到统一29关节接口。
- [src/supermimic/envs/sim_context.py](../src/supermimic/envs/sim_context.py)：`SimContext` 管理33D上下文：摩擦1、编码器偏差29、躯干COM偏移3。
- [src/supermimic/envs/lock_waist.py](../src/supermimic/envs/lock_waist.py)：锁腰G1模型处理、缺失关节补位和常量映射；29D软件接口不等于29个物理可动关节。
- [src/supermimic/envs/imu_model.py](../src/supermimic/envs/imu_model.py)：可选训练/imagined IMU饱和与重力估计模型；包含 `ImuImagCfg` 和可微积分辅助函数。
- [src/supermimic/envs/mdp/motion_library.py](../src/supermimic/envs/mdp/motion_library.py)：`MotionLibrary` 以统一帧轴管理多片段，支持full/compact存储、参考查询与 `goal_window()`；compact利用FK恢复相关量。
- [src/supermimic/envs/mdp/commands.py](../src/supermimic/envs/mdp/commands.py)：`MultiClipMotionCommand` 处理参考推进、return-based reset采样、失败点和真实状态reset；向环境暴露参考字段。
- [src/supermimic/envs/mdp/observations.py](../src/supermimic/envs/mdp/observations.py)：goal和完整关节观测适配，包括锁腰关节补位。
- [src/supermimic/envs/mdp/rewards.py](../src/supermimic/envs/mdp/rewards.py)：`tracking_reward_terms()` 等可微跟踪reward、重锚定及附加惩罚；环境与imagination应维持一致。
- [src/supermimic/envs/mdp/terminations.py](../src/supermimic/envs/mdp/terminations.py)：根节点高度、倾斜和片段结束判定。
- [src/supermimic/envs/mdp/events.py](../src/supermimic/envs/mdp/events.py)：可选随机化事件，例如膝力矩限制缩放、指定窗口的骨盆推力。

## 8. 策略、世界模型与可微优化

- [src/supermimic/policy/nets.py](../src/supermimic/policy/nets.py)：`PolicyCfg/MotionEncoder/Decoder/Policy/Critic`；当前只注册motion encoder。扩展FSQ、多模态和拆分encode/decode的首要入口。
- [src/supermimic/world_model/state_spec.py](../src/supermimic/world_model/state_spec.py)：`StateSpec` 定义原始状态、heading归一化特征、预测量及状态重建；不能只按71D总数判断语义相同。
- [src/supermimic/world_model/dynamics.py](../src/supermimic/world_model/dynamics.py)：通用 `MLP` 和动力学 `DynamicsMLP` 网络；Policy也复用MLP。
- [src/supermimic/world_model/world_model.py](../src/supermimic/world_model/world_model.py)：`WorldModel` 结合学习到的动力学/终止head与解析obs/reward，并组织多步模型损失。
- [src/supermimic/world_model/wm_trainer.py](../src/supermimic/world_model/wm_trainer.py)：`WorldModelTrainer` 从回放窗口拟合WM，提供参考帧查询和优化步骤。
- [src/supermimic/world_model/heads.py](../src/supermimic/world_model/heads.py)：`AnalyticHeads/ReferenceFrame` 以及 `analytic_observation()`、`anchor_relative_obs()`、`flatten_history_term_major()`；是训练/想象/部署一致性的关键参照。
- [src/supermimic/imagination/rollout.py](../src/supermimic/imagination/rollout.py)：`imagine()` 构造多步可微轨迹；读取参考、形成goal/history、执行策略、预测下一状态并累积reward和终止量。
- [src/supermimic/imagination/compiled.py](../src/supermimic/imagination/compiled.py)：`ImagStep` 将单步动力学、FK和解析head集中到可编译调用，减少小算子启动开销。
- [src/supermimic/imagination/objective.py](../src/supermimic/imagination/objective.py)：策略目标与 `gae_targets()`，定义Actor优化及Critic训练目标。
- [src/supermimic/imagination/policy_trainer.py](../src/supermimic/imagination/policy_trainer.py)：`PolicyTrainer` 通过冻结WM进行短时域Actor优化，再更新Critic；扩展时重点检查梯度与冻结边界。
- [src/supermimic/kinematics/g1_fk.py](../src/supermimic/kinematics/g1_fk.py)：`G1Kinematics` 从MuJoCo模型构建可微批量FK，计算body位置、姿态与速度。
- [src/supermimic/kinematics/rotations.py](../src/supermimic/kinematics/rotations.py)：wxyz四元数、rotvec、6D旋转、heading/yaw与投影重力工具；修改前核对所有生产/消费方。
- [src/supermimic/eval/benchmark.py](../src/supermimic/eval/benchmark.py)：统一的确定性整片段评测、checkpoint加载、结果汇总与保存；入口脚本应复用这里的协议。

## 9. 导出与共用部署核心

- [src/supermimic/deploy/export.py](../src/supermimic/deploy/export.py)：`freeze/config/clips/onnx/mjcf` 等导出流程；依赖训练环境，当前导出单体policy.onnx并做ORT对照，不是独立Encoder/Decoder导出。
- [src/supermimic/deploy/core.py](../src/supermimic/deploy/core.py)：`DeployConfig/Clip/SafetyCfg/OnnxPolicy/DeployCore`：构造历史与goal、运行策略、映射动作及执行核心安全检查；只依赖轻量部署栈。
- [src/supermimic/deploy/imu_model.py](../src/supermimic/deploy/imu_model.py)：Sim2Sim侧可选NumPy IMU模型；应与训练侧 `envs/imu_model.py` 的假设对应。
- [src/supermimic/deploy/waist_coupling.py](../src/supermimic/deploy/waist_coupling.py)：为MuJoCo模型加入可选腰部roll/pitch可达域耦合约束，并解析对应配置。
- [src/supermimic/deploy/waist_roll.py](../src/supermimic/deploy/waist_roll.py)：可选腰部roll范围、阻尼和摩擦模型；同时涉及训练和Sim2Sim参数一致性。

## 10. 评估、数据与训练辅助脚本

- [scripts/bench_imagination.py](../scripts/bench_imagination.py)：比较imagination策略更新的eager/torch.compile前向和反向耗时。
- [scripts/bench_sim_speed.py](../scripts/bench_sim_speed.py)：测量真实环境配置下的仿真吞吐，包括记录器等开销。
- [scripts/eval_benchmark.py](../scripts/eval_benchmark.py)：对checkpoint与动作集运行统一benchmark的命令行入口。
- [scripts/eval_push.py](../scripts/eval_push.py)：推扰恢复实验；成功/稳定判据是本仓库定义，不能自动视作论文同协议。
- [scripts/eval_scope_stats.py](../scripts/eval_scope_stats.py)：按BONES-SEED不同筛选范围重新汇总已有benchmark结果。
- [scripts/diag_drift.py](../scripts/diag_drift.py)：诊断全局root位置/yaw漂移及其对速度误差的影响。
- [scripts/play.py](../scripts/play.py)：交互回放策略与参考机器人。
- [scripts/render_video.py](../scripts/render_video.py)：无窗口渲染策略/参考对照视频及逐帧信息。
- [scripts/tb_tail.py](../scripts/tb_tail.py)：读取TensorBoard最新标量，用于快速查看训练状态。
- [scripts/pull_and_eval.sh](../scripts/pull_and_eval.sh)：循环从远程同步checkpoint并本地评测；会写入logs，不是单次只读查看脚本。
- [scripts/data/csv_to_npz_batch.py](../scripts/data/csv_to_npz_batch.py)：批量转换CSV并输出manifest，支持输入帧率与逐片段覆盖。
- [scripts/data/filter_bones_seed.py](../scripts/data/filter_bones_seed.py)：按名字与物理可行性筛选BONES-SEED，输出保留/排除原因。
- [scripts/data/build_library.py](../scripts/data/build_library.py)：将筛选后的NPZ打包成compact参考状态库。
- [scripts/data/build_real_reset_states.py](../scripts/data/build_real_reset_states.py)：将真机日志的指定时刻转换为训练reset状态，处理朝向、接触和状态估计约定。
- [scripts/train/train_lean35kfs_e24.sh](../scripts/train/train_lean35kfs_e24.sh)：特定动作的e24微调配方；依赖父checkpoint，不能当作从零复现实验脚本。
- [scripts/train/gate_lean35kfs.sh](../scripts/train/gate_lean35kfs.sh)：特定e24候选模型的导出与部署门禁编排；部分已有输出会跳过，复测时需核对产物版本。

## 11. 部署、Sim2Sim与动作处理脚本

- [scripts/deploy/g1_common.py](../scripts/deploy/g1_common.py)：G1状态提取、遥控器按键、DDS初始化、模式检查、锁腰处理等公共工具。
- [scripts/deploy/g1_runner.py](../scripts/deploy/g1_runner.py)：Python DDS版真机runner，负责策略循环、状态机、保护与日志；会发送低层控制命令。
- [scripts/deploy/check_robot.py](../scripts/deploy/check_robot.py)：上机前检查入口；具体读取/交互行为以参数与部署手册为准，不能把全部deploy脚本都视为只读。
- [scripts/deploy/sim2sim.py](../scripts/deploy/sim2sim.py)：CPU MuJoCo+ORT部署门禁主程序；通过DeployCore检验跟踪、扰动、切换和时序等部署行为。
- [scripts/deploy/sim2sim_ref.py](../scripts/deploy/sim2sim_ref.py)：在训练环境中产生确定性参考rollout，供Sim2Sim对齐obs/goal/action。
- [scripts/deploy/sim_g1_mujoco.py](../scripts/deploy/sim_g1_mujoco.py)：无窗口DDS仿真机器人，用于runner端到端测试，支持保护带和相关仿真控制。
- [scripts/deploy/run_sim_e2e.sh](../scripts/deploy/run_sim_e2e.sh)：编排Python runner与DDS仿真的端到端测试。
- [scripts/deploy/realstate_gate.py](../scripts/deploy/realstate_gate.py)：从真实记录状态起步的反事实Sim2Sim门禁，诊断落地等局部故障；不是直接在真机重放。
- [scripts/deploy/cap_joint_velocity.py](../scripts/deploy/cap_joint_velocity.py)：局部平滑超速关节参考；后续链路需重算相应body状态，不能把部分更新当作完整一致NPZ。
- [scripts/deploy/knee_feasibility.py](../scripts/deploy/knee_feasibility.py)：针对膝速度/机械限位的参考动作修正，尽量保持原时序。
- [scripts/deploy/spin_feasibility.py](../scripts/deploy/spin_feasibility.py)：转体/落地参考动作可行性修正与检查，包含轨迹优化辅助函数。
- [scripts/deploy/make_combo_clip.py](../scripts/deploy/make_combo_clip.py)：将多动作组合成连续参考，处理过渡、朝向、时序及相关状态。
- [scripts/deploy/make_thermal_stop_clips.py](../scripts/deploy/make_thermal_stop_clips.py)：生成热保护停止过程的仿真测试参考；标明SIM ONLY，不能当作真机急停验证。
- [scripts/deploy/make_source_bg.py](../scripts/deploy/make_source_bg.py)：生成与动作参考同步的原视频背景素材。
- [scripts/deploy/bg_sync_server.py](../scripts/deploy/bg_sync_server.py)：接收runner同步信息并播放背景视频；属于展示同步，不是策略或运动控制器。

## 12. C++ DDS bridge

- [scripts/deploy/bridge/build.sh](../scripts/deploy/bridge/build.sh)：编译g1_bridge与libg1bridge_client.so，需要Unitree SDK2及目标架构依赖。
- [scripts/deploy/bridge/g1_bridge.cpp](../scripts/deploy/bridge/g1_bridge.cpp)：承担低状态/低命令DDS I/O、CRC、共享内存更新、发送节拍与watchdog；runner失联时按实现进入锁定阻尼。
- [scripts/deploy/bridge/g1_bridge_shm.h](../scripts/deploy/bridge/g1_bridge_shm.h)：C++/Python之间共享内存、ring、状态与socket命令协议；字段/版本修改需要两端同时核对。
- [scripts/deploy/bridge/g1_bridge_client.cpp](../scripts/deploy/bridge/g1_bridge_client.cpp)：向Python提供C ABI的共享内存读取器，处理seqlock/ring等读一致性。
- [scripts/deploy/bridge/g1_bridge_io.py](../scripts/deploy/bridge/g1_bridge_io.py)：通过ctypes客户端读取bridge状态并通过Unix socket发送指令；桥接Python runner与C++进程。
- [scripts/deploy/bridge/g1_runner_bridge.py](../scripts/deploy/bridge/g1_runner_bridge.py)：bridge版runner；保留策略、状态机、安全和日志逻辑，DDS I/O交给C++。
- [scripts/deploy/bridge/ro_bridge_check.py](../scripts/deploy/bridge/ro_bridge_check.py)：使用listen-only bridge进行只读检查，报告lowstate频率、CRC、间隔、温度及其他发布者。
- [scripts/deploy/bridge/test_bridge.py](../scripts/deploy/bridge/test_bridge.py)：在隔离DDS domain 2/lo验证CRC、字段、命令、ring、watchdog和退出等；不是在真实机器人上运行的测试。
- [scripts/deploy/bridge/run_sim_e2e_bridge.sh](../scripts/deploy/bridge/run_sim_e2e_bridge.sh)：bridge版本的DDS仿真端到端编排。

## 13. 视频转动作工具

- [tools/video2motion/run_pipeline.sh](../tools/video2motion/run_pipeline.sh)：编排视频转码、人体框、外部GVHMR、GMR重定向与预览；GVHMR/GMR/SMPL-X资源需用户另行配置。
- [tools/video2motion/shot_bbx.py](../tools/video2motion/shot_bbx.py)：处理镜头切换场景的人体框，避免跨镜头跟踪/平滑导致错位。
- [tools/video2motion/gvhmr_to_supermimic.py](../tools/video2motion/gvhmr_to_supermimic.py)：GVHMR人体结果→GMR G1重定向→CSV，含IK初始化、重启和限位处理。
- [tools/video2motion/g1_model.py](../tools/video2motion/g1_model.py)：视频工具环境使用的G1模型、关节顺序、足部及CSV工具；这里的重复约定必须与核心转换器一致。
- [tools/video2motion/render_preview.py](../tools/video2motion/render_preview.py)：源视频与G1运动学参考并排预览；不代表策略闭环已能跟踪。
- [tools/video2motion/extend_full.py](../tools/video2motion/extend_full.py)：针对特定视频近景/全景切换的实验性动作补全，不是通用人体遮挡恢复方案。
- [tools/video2motion/render_full_preview.py](../tools/video2motion/render_full_preview.py)：检查扩展动作与原视频逐帧对应、镜头切换及接续位置。

## 14. 测试文件与覆盖目标

- [tests/test_env_smoke.py](../tests/test_env_smoke.py)：环境构建、step、goal、replay、状态读写与片段边界。
- [tests/test_motion_convert.py](../tests/test_motion_convert.py)：CSV重采样、FK、速度与关节顺序约定。
- [tests/test_motion_library_compact.py](../tests/test_motion_library_compact.py)：full/compact库的goal和body状态一致性。
- [tests/test_g1_fk.py](../tests/test_g1_fk.py)：Torch FK与MuJoCo对照、CPU/CUDA及梯度。
- [tests/test_world_model.py](../tests/test_world_model.py)：状态表示、预测模型梯度及小样本拟合。
- [tests/test_analytic_heads.py](../tests/test_analytic_heads.py)：解析reward/obs/termination及anchor特征与环境一致性。
- [tests/test_imagination.py](../tests/test_imagination.py)：GAE手算对照及短时端到端训练smoke。
- [tests/test_deploy_parity.py](../tests/test_deploy_parity.py)：NumPy DeployCore/ORT与训练环境的策略输入输出对齐。
- [tests/test_sampler.py](../tests/test_sampler.py)：return-based采样与微调恢复选项。
- [tests/test_action_delay.py](../tests/test_action_delay.py)：执行指令延迟随机化与默认关闭行为。
- [tests/test_imu_model.py](../tests/test_imu_model.py)：训练和imagined IMU模型、reset及梯度。
- [tests/test_deploy_imu_model.py](../tests/test_deploy_imu_model.py)：部署IMU模型的限幅、积分与配置。
- [tests/test_lock_waist.py](../tests/test_lock_waist.py)：锁腰模型、状态补位、参考与解析head一致性。
- [tests/test_joint_torque.py](../tests/test_joint_torque.py)：关节力矩惩罚的手算、默认行为及梯度。
- [tests/test_joint_vel_hinge.py](../tests/test_joint_vel_hinge.py)：关节速度hinge惩罚及梯度。
- [tests/test_tilt_hinge.py](../tests/test_tilt_hinge.py)：倾斜误差、部署判据对应及梯度。
- [tests/test_yaw_rate_hinge.py](../tests/test_yaw_rate_hinge.py)：yaw角速度惩罚和环境/解析实现一致性。
- [tests/test_waist_coupling.py](../tests/test_waist_coupling.py)：腰部耦合模型、配置及仿真约束。
- [tests/test_waist_roll.py](../tests/test_waist_roll.py)：腰部roll范围、阻尼/摩擦及训练/仿真配置。
- [tests/test_real_reset.py](../tests/test_real_reset.py)：真机日志reset及可选推扰/膝力矩随机化。
- [tests/test_realstate_gate.py](../tests/test_realstate_gate.py)：真实状态起步门禁的输入处理、网格与判定。
- [tests/test_sim2sim_peaks.py](../tests/test_sim2sim_peaks.py)：高频关节速度峰值统计与超阈计数。
- [tests/test_spin_feasibility.py](../tests/test_spin_feasibility.py)：转体可行性工具的QP、差分及角速度辅助计算。

测试文件存在不代表测试已通过。部分测试依赖GPU、mjlab、转换后的动作或部署环境；应分别记录 PASS、FAIL、SKIPPED 和 NOT_RUN。bridge协议测试位于 `scripts/deploy/bridge/test_bridge.py`，不在上述tests目录内。

## 15. 动作资源与包初始化文件

- `motions/custom_dance/` 中的 `gouxiong_duosuomao.csv`、`langzi_tiqiu.csv`、`shanshenbu.csv`、`shanshenbu_combo4.csv`、`yaotou_huangnao.csv` 是随仓库提供的动作输入；`shanshenbu_combo4.json` 是关联的组合描述。它们不是训练checkpoint或部署policy。
- `motions/shanshen_duosuomao/reset_points_lean35kfs.csv` 是特定动作的重置点资源，与完整动作库、真实状态文件用途不同。
- `src/supermimic/` 及各子包中的 `__init__.py` 声明包并可能暴露公共入口；其中 `envs/__init__.py` 暴露环境构造接口。阅读实现时继续跳转到实际定义文件，不能把所有初始化文件都当作空文件。

## 16. 按开发目标寻找修改入口

### G1 FSQ与Actor朝向

优先阅读 `policy/nets.py`，再检查 `config.py`、`training/loop.py`、`world_model/heads.py` 和 `imagination/rollout.py`。同时追踪环境goal生产、缩放、checkpoint加载、export和DeployCore消费。仅改网络输入维度不能完成orientation或FSQ接入。

### 多模态数据与跨模态BPTT

从 `data/motion_convert.py`、`envs/mdp/motion_library.py`、`commands.py` 和ReplayBuffer开始，再到 `policy/nets.py`、`imagination/rollout.py` 和 `policy_trainer.py`。目前数据主链仍是G1参考动作，多模态配对、mask、因果窗口和新Encoder需按开发清单补充。

### VR Teleop与VLA

当前没有专用Teleop/VLA目录。可复用策略Decoder和DeployCore，但需要新增设备协议、标定、因果输入、异步调度、失联处理及版本化token契约。参考视频工具只负责离线数据准备，不能直接充当实时遥操接收端。

### ONNX与真机

先看 `deploy/export.py → deploy/core.py → scripts/deploy/sim2sim.py`，再读 `g1_runner_bridge.py` 与bridge协议。模型导出一致性、Sim2Sim、DDS仿真和真机阶段分别验收；保留既有控制权、超时、倾斜、速度、力矩和热保护逻辑。

## 17. 建议阅读顺序

1. README、METHOD、`config.py`：理解目标、假设和开关。
2. `policy/nets.py → motion_library.py → world_model/heads.py`：核对实际输入输出。
3. `state_spec.py → world_model.py → wm_trainer.py`：理解预测模型学什么。
4. `imagination/rollout.py → policy_trainer.py → training/loop.py`：理解梯度与训练编排。
5. `deploy/core.py → export.py → test_deploy_parity.py`：理解训练到部署的契约。
6. SIM2SIM、DEPLOY_G1及runner/bridge：理解验证边界和控制链。
7. [to_do_list.md](to_do_list.md)：基于当前实现推进P0–P6，新增文件落地后同步更新本说明。
