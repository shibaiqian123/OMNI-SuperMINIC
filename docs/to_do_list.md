# OmniMimic 开发待办清单（NJU SuperMimic 基线）

## 范围、依据与状态

- 目标仓库：229 服务器 `10.242.3.229:/home/cqu023/SuperMimic-Repro-Sim2Real`。
- 核查日期：2026-11-06；当前提交：`65a16ae5b5b17a9648e98ac6436f7a159e44cc80`。创建前 Git 工作区干净，根目录没有 `to_do_list.md`；仓库及核查的上级目录未发现适用的 AGENTS.md。
- 本次只创建本文件。下文所有代码/测试/文档修改均为未来任务，不代表已经实现；不运行训练、导出、仿真或真机控制。
- 原 ChatGPT 会话可读取的问题涉及 NJU、Actor orientation 和完整开发计划，但助手正文仅返回内容引用占位符。因此以下 P0–P6 是根据用户本次明确范围、历史架构记录及本次实际源码核查重建的可执行计划，不声称是原计划逐字恢复。
- “已存在”表示本次核对过的真实路径，不代表其中已有待办所述能力；“建议新增”是拟议路径，写入时逐项检查未存在。同一拟议文件可在后续阶段复用，不重复创建。
- 所有 checkbox 默认未完成；验收后才勾选，并附日志/报告位置。数值门槛未由基线确定的项目必须在实验前冻结，不能看结果后调整。
- 开发依赖：P0→P1→P2→P3；P4 依赖因果 Teleop 数据与训练；P5 依赖稳定且版本固定的 FSQ/BFM；P6 的导出/parity 可从 P1 开始，真机准入必须等待相应模态的完整仿真与安全门禁通过。

## 本次源码已确认的关键边界

1. `src/supermimic/policy/nets.py` 默认 MotionEncoder 接收 680D goal，输出 32D 连续 latent；Decoder 拼接 930D proprio history 后输出 29D raw action。当前仅注册 motion encoder，没有 FSQ。
2. G1 goal 每帧 68D、默认 10 个未来帧；proprio 单帧 93D、10 帧按 term-major 展平。实时 Teleop 不能默认获得这 10 个未来帧。
3. `anchor_obs` 默认关闭，打开时追加 9D（3D 相对位置+6D 相对旋转）；World Model canonical state 已有 rot6d；Critic 的 `e_root` 为另一个 4D 特权量。三者不是同一个特征。
4. `imagination/rollout.py` 已构建可微模型 rollout，`policy_trainer.py` 在 Actor 更新中冻结 WM/Critic 参数但保留输入梯度。FSQ 必须保持该梯度链。
5. 当前 `deploy/export.py` 导出单体 policy.onnx，DeployCore 管理 50Hz 控制约定、历史、参考时序和安全检查，动作映射为 `q_target = q0 + scale * action`。
6. G1 CSV→NPZ、compact motion library、视频转动作、Sim2Sim 和 G1 runner 已有实际入口；这些并不等于已经具备 SMPL/Teleop 配对库、多模态训练、实时 VR 或 VLA 接口。

## P0 — 固定 NJU 基线与接口契约
- [ ] **P0.01 基线来源与可复现记录**
  - 已存在：`README.md`、`README_EN.md`、`docs/METHOD.md`、`pyproject.toml`、`uv.lock`。
  - 建议新增：`docs/OMNIMIMIC_BASELINE.md`。
  - 具体动作：记录 remote、commit、依赖、机器人变体、训练命令、数据划分与权重摘要；区分论文、NJU 实现和本项目选项，不以仓库名称推定实现一致。
  - 验收标准：基线报告可追溯到同一 commit/配置/数据/权重；未执行项目标为 NOT_RUN，不能引用上游结果充当本机复现。

- [ ] **P0.02 原始训练与评估回归**
  - 已存在：`src/supermimic/scripts/train.py`、`src/supermimic/training/loop.py`、`scripts/eval_benchmark.py`、`scripts/eval_push.py`、`tests/test_env_smoke.py`、`tests/test_world_model.py`、`tests/test_imagination.py`。
  - 具体动作：在不启用 Omni 功能时运行环境、World Model、imagination 检查及小规模训练；按同一固定评估集保存基线回报、成功率、误差和耗时。
  - 验收标准：环境/梯度检查通过；训练无 NaN；记录随机种子、样本量及失败项，建立后续比较基准。

- [ ] **P0.03 观测、动作、时间契约**
  - 已存在：`src/supermimic/world_model/heads.py`、`src/supermimic/envs/mdp/motion_library.py`、`src/supermimic/deploy/core.py`。
  - 建议新增：`docs/OMNIMIMIC_CONTRACT.md`。
  - 具体动作：逐字段定义 93D 单帧观测、10 帧 term-major 历史、68D goal 帧、未来 k+1…k+10、29D action、关节顺序、四元数顺序、坐标系、单位、50Hz 和 reset/backfill。
  - 验收标准：训练、imagination、部署三路样例逐字段一致；action 明确经 q0+scale*a 转为关节目标。

- [ ] **P0.04 分支配置与兼容加载**
  - 已存在：`src/supermimic/config.py`、`src/supermimic/policy/nets.py`、`src/supermimic/training/loop.py`。
  - 建议新增：`tests/test_policy_compat.py`。
  - 具体动作：增加显式 policy variant、latent/goal schema 版本与 checkpoint 元数据；原版默认行为保持可复现；不允许用 strict=False 静默掩盖 32D→64D 的不匹配。
  - 验收标准：原始 checkpoint 正常加载；Omni 不兼容权重明确报错或使用有日志的迁移；配置可恢复。

- [ ] **P0.05 基线部署回归与开发分工**
  - 已存在：`src/supermimic/deploy/export.py`、`tests/test_deploy_parity.py`、`docs/SIM2SIM.md`、`docs/DEPLOY_G1.md`。
  - 具体动作：先建立原版 PyTorch/ORT/DeployCore 对照；229 开发，32 由用户同步代码后训练；数据、权重及运行产物使用配置路径。
  - 验收标准：记录实测 parity 与现有部署门槛；没有训练或硬件记录时保持未验证；不在本任务启动训练或控制机器人。

## P1 — G1 2×32 FSQ Actor 与显式 6D orientation
- [ ] **P1.01 FSQ 数值与编码定义**
  - 已存在：`src/supermimic/policy/nets.py`。
  - 建议新增：`src/supermimic/policy/fsq.py`、`tests/test_fsq.py`。
  - 具体动作：实现 64D pre-quant latent→[B,2,32] 分组 FSQ→64D decoder latent；将每分量 levels 作为独立配置（本计划初值为 32），固定偶数量化级的边界、归一化、round/tie 和 STE 公式。
  - 验收标准：2×32 是两个 32 维分组，不是两个 32 类 token；每维恰有配置数量的可达级别；训练前向与部署一致，STE 梯度有限且非零；不把整个组打包为会溢出的 int64。

- [ ] **P1.02 拆分策略接口**
  - 已存在：`src/supermimic/policy/nets.py`、`src/supermimic/policy/__init__.py`。
  - 建议新增：`tests/test_omni_policy.py`。
  - 具体动作：增加 encode/quantize/decode 接口及结构化输出（prequant、quantized、indices）；G1 encoder 输出 64D，共享 Decoder 接收 latent 与原有 proprio history，仍输出 29D raw action。
  - 验收标准：batch=1/多 batch 形状一致；forward 等于组合接口；基线模式仍为原 32D 连续 latent；非法 interface/shape 明确失败。

- [ ] **P1.03 区分现有 4D、6D 和 9D**
  - 已存在：`src/supermimic/policy/nets.py`、`src/supermimic/world_model/state_spec.py`、`src/supermimic/world_model/heads.py`、`src/supermimic/config.py`。
  - 具体动作：保留 Critic e_root 的 3D 位置误差+1D yaw；注明 WM canonical rot6d 已存在；现有 anchor_obs 是 3D pos+6D ori，不能宣称 Actor 已有独立 ori6d 开关。
  - 验收标准：契约分别列出 critic-only e_root、WM rot6d、Actor none/ori6d/pose9d；不把 e_root 特权量直接加入部署 Actor。

- [ ] **P1.04 Actor orientation-only 特征**
  - 已存在：`src/supermimic/world_model/heads.py`、`src/supermimic/kinematics/rotations.py`、`src/supermimic/envs/mdp/observations.py`、`src/supermimic/envs/tracking_env_cfg.py`、`src/supermimic/envs/mdp/commands.py`。
  - 建议新增：`tests/test_orientation_contract.py`。
  - 具体动作：新增显式 orientation 模式，使用 ref anchor 相对 robot anchor 的旋转；复用现有矩阵前两列及其实际展平顺序，固定坐标和 quaternion 顺序；默认 K=10 时 none/ori6d/pose9d 分别为 680/686/689D。
  - 验收标准：单位旋转、90°旋转、q/-q、批量样例通过；与现有 9D 特征的 orientation 部分一致；误用行/列或排列能被测试检出。

- [ ] **P1.05 6D 三路生产与消费一致**
  - 已存在：`src/supermimic/policy/nets.py`、`src/supermimic/training/loop.py`、`src/supermimic/imagination/rollout.py`、`src/supermimic/deploy/core.py`、`src/supermimic/deploy/export.py`、`tests/test_analytic_heads.py`、`tests/test_deploy_parity.py`。
  - 具体动作：统一维度、scale、噪声、当前参考时刻和状态相关特征计算；取消仅凭 goal_dim divmod(68) 猜 schema 的扩展方式；真机 orientation 来源必须可观测并标定。
  - 验收标准：同状态/参考下环境、imagination、NumPy 部署特征及 action 在预设容差内一致；未知朝向源拒绝启动对应模式；不从仿真真值泄露部署输入。

- [ ] **P1.06 G1 单模态 BPTT 接入**
  - 已存在：`src/supermimic/imagination/rollout.py`、`src/supermimic/imagination/policy_trainer.py`、`src/supermimic/imagination/compiled.py`、`src/supermimic/training/loop.py`、`tests/test_imagination.py`。
  - 具体动作：先仅接 G1 FSQ，保留 WM/Reward/ReplayBuffer 主链；量化保留 STE，不 detach Actor→WM→reward 路径；检查 compiled/eager 分支。
  - 验收标准：小批量 backward 中 encoder/decoder 有有限梯度，Actor 更新阶段 WM 参数不更新但输入梯度存在；compiled/eager 结果一致。

- [ ] **P1.07 G1 消融与 token 健康度**
  - 已存在：`scripts/eval_benchmark.py`、`src/supermimic/training/loop.py`。
  - 建议新增：`scripts/eval_omni.py`。
  - 具体动作：比较原始连续32、连续64、FSQ64，分别测试 none/ori6d/pose9d；记录每分量级别占用、饱和率、量化误差和动作/跟踪误差。
  - 验收标准：固定数据/预算/种子并报告波动；先冻结容许退化门槛再比较；不能把未经训练的随机网络结果或旧 SONIC token 当作本模型能力。

## P2 — G1 / SMPL / Teleop 数据与多模态 Encoder
- [ ] **P2.01 统一配对数据 schema**
  - 已存在：`src/supermimic/data/motion_convert.py`、`src/supermimic/data/recorder.py`。
  - 建议新增：`src/supermimic/data/multimodal_dataset.py`、`docs/OMNIMIMIC_DATA.md`、`tests/test_multimodal_dataset.py`。
  - 具体动作：定义 clip/person/source ID、时间戳、fps、G1 状态/参考、SMPL 参数、稀疏头手 pose、mask、坐标/单位、质量标签；有配对与无配对数据分开。
  - 验收标准：可 roundtrip；同一动作跨模态按时间索引取样；缺失不默认为有效零；训练/验证按人物/源动作隔离并检测重复。

- [ ] **P2.02 G1 现有数据复用**
  - 已存在：`src/supermimic/data/motion_convert.py`、`scripts/data/csv_to_npz_batch.py`、`scripts/data/build_library.py`、`src/supermimic/envs/mdp/motion_library.py`、`tests/test_motion_convert.py`、`tests/test_motion_library_compact.py`。
  - 具体动作：保留 CSV xyzw→内部 wxyz、FK、重采样、关节顺序和 compact library；扩展关联 ID/manifest，保持现有 G1 数据可独立使用。
  - 验收标准：full/compact 同片段参考一致；无错序、NaN、跨片段窗口或单位误差；清楚统计过滤前后数量。

- [ ] **P2.03 SMPL 转换及 G1 对齐**
  - 已存在：`tools/video2motion/gvhmr_to_supermimic.py`、`tools/video2motion/g1_model.py`、`tools/video2motion/run_pipeline.sh`。
  - 建议新增：`scripts/data/build_smpl_pairs.py`、`src/supermimic/data/smpl_features.py`。
  - 具体动作：核对视频流程真实输出，新增 SMPL/SMPL-X 参数到选定人体特征及 G1 retarget 配对；记录模型类型、shape、root heading、尺度、足接触与拟合质量；依赖和人体资产另行配置。
  - 验收标准：可视化检查选定样本且误差统计达预设阈值；坐标/时间对齐可复现；不声称已有工具天然提供完整 SMPL 配对。

- [ ] **P2.04 离线稀疏 Teleop 数据**
  - 已存在：`src/supermimic/kinematics/g1_fk.py`。
  - 建议新增：`scripts/data/build_teleop_pairs.py`、`src/supermimic/data/teleop_features.py`。
  - 具体动作：从完整动作生成头/双手稀疏信号与配对 G1 目标；加入跟踪缺失、延迟、抖动和标定扰动；与真实 VR 数据保留来源区分。
  - 验收标准：离线抽稀与实时输入同 schema；mask/延迟可回放；不以合成 Teleop 验证替代真实 VR 验证。

- [ ] **P2.05 多模态 Encoder 与共享空间**
  - 已存在：`src/supermimic/policy/nets.py`、`src/supermimic/config.py`。
  - 建议新增：`src/supermimic/policy/encoders.py`、`tests/test_multimodal_policy.py`。
  - 具体动作：注册 g1/smpl/teleop encoder（保留 motion 兼容映射），输出同一 FSQ schema 并复用 Decoder；各模态独立输入缩放和 mask，明确接口维度。
  - 验收标准：每个 encoder 均能独立前向/反向；配对样本共享语义通过解码动作和轨迹衡量，不能仅凭 latent 维度相同宣称对齐。

- [ ] **P2.06 因果窗口与数据质量门禁**
  - 已存在：`src/supermimic/envs/mdp/motion_library.py`。
  - 建议新增：`src/supermimic/data/goal_provider.py`、`scripts/data/validate_multimodal.py`。
  - 具体动作：区分离线未来窗口、实时当前/历史窗口、显式预测窗口；处理 clip 尾部、reset、缺失和重采样；记录延迟补偿方法。
  - 验收标准：实时输入无未来信息泄漏；边界测试不过片；坏数据给出原因；若缓冲未来帧必须显式计入端到端延迟。

## P3 — 跨模态可微 imagination / BPTT 训练
- [ ] **P3.01 接口无关 Goal Provider**
  - 已存在：`src/supermimic/imagination/rollout.py`、`src/supermimic/training/loop.py`、`src/supermimic/data/replay_buffer.py`、`src/supermimic/envs/mdp/commands.py`、`src/supermimic/data/recorder.py`。
  - 建议新增：`tests/test_goal_provider.py`。
  - 具体动作：接入 P2 Goal Provider，Replay/批次保留 clip/frame/interface/mask 关联；各 rollout step 正确推进时间；状态相关相对朝向由 imagined state 重算。
  - 验收标准：多步回放与参考时刻对齐；状态相关特征保留梯度；静态目标数据和 actor-dependent 特征边界清晰。

- [ ] **P3.02 跨模态采样与课程**
  - 已存在：`src/supermimic/config.py`、`src/supermimic/training/loop.py`、`src/supermimic/training/schedule.py`、`src/supermimic/training/distributed.py`。
  - 具体动作：按 G1 先收敛→配对 SMPL/Teleop→联合训练推进；配置模态比例、缺失规则、冻结/解冻阶段，处理 DDP 未使用 encoder 参数。
  - 验收标准：各模态批次计数与配置一致；单/多卡无 hang 或静默漏更新；旧 G1 能力按固定回归集检查。

- [ ] **P3.03 BPTT 与 Critic 条件化**
  - 已存在：`src/supermimic/imagination/policy_trainer.py`、`src/supermimic/imagination/rollout.py`、`src/supermimic/policy/nets.py`、`src/supermimic/imagination/objective.py`、`tests/test_imagination.py`。
  - 具体动作：让各模态可沿共享 decoder/WM 优化；选择并记录按模态 Critic 或统一完整参考 Critic 的实现，统一目标、e_root 和 bootstrap；Actor 不读取 Critic 特权信息。
  - 验收标准：每模态完整 H 步梯度有限；WM 冻结参数而不截断输入梯度；终止/survival/bootstrap 对齐；缺模态批次不会串用旧 goal。

- [ ] **P3.04 配对监督与语义对齐损失**
  - 已存在：`src/supermimic/imagination/policy_trainer.py`、`src/supermimic/config.py`。
  - 建议新增：`src/supermimic/policy/losses.py`、`tests/test_multimodal_losses.py`。
  - 具体动作：将配对动作蒸馏、latent 对齐或任务跟踪辅助项与 BPTT 分开配置；只在可信配对/mask 上监督；固定 teacher 和 stop-gradient 边界。
  - 验收标准：消融可单独关闭辅助损失；无配对样本不强制 token 一致；同时报告任务误差与编码健康度，避免全 token 塌缩的伪对齐。

- [ ] **P3.05 训练恢复和故障记录**
  - 已存在：`src/supermimic/training/loop.py`、`src/supermimic/training/schedule.py`、`src/supermimic/config.py`。
  - 具体动作：checkpoint 保存所有 encoder/FSQ/decoder/critic、optimizer、schema、随机状态、课程和采样状态；记录梯度、各模态回报/崩溃与性能。
  - 验收标准：断点恢复能重现下一小段训练；NaN/不兼容 schema 有明确故障原因；32 上真实训练、验证结果单独归档。

- [ ] **P3.06 跨模态闭环评估**
  - 已存在：`src/supermimic/eval/benchmark.py`、`scripts/eval_benchmark.py`。
  - 建议新增：`tests/test_multimodal_rollout.py`。
  - 具体动作：按同一 G1 目标比较不同模态输入的跟踪、跌倒、手端误差、泛化和模态切换连续性。
  - 验收标准：提供留出集逐模态报告及最差样本；完成实际闭环 rollout 后才标记通过，不把 offline token MSE 当作闭环精度。

## P4 — VR Teleop 实时链路
- [ ] **P4.01 设备接入与输入协议**
  - 建议新增：`src/supermimic/teleop/protocol.py`、`src/supermimic/teleop/receiver.py`、`scripts/teleop/record_vr.py`。
  - 具体动作：接入实际选定的 PICO/OpenXR/XR 数据源；协议携带版本、序号、设备/接收时间、头手 pose、跟踪有效性及操作按钮；记录可回放原始数据。
  - 验收标准：使用真实设备录制并回放同一片段；时钟差/乱序/重复/非法值被识别；未接设备时明确 UNVERIFIED。

- [ ] **P4.02 标定与人体约束**
  - 建议新增：`src/supermimic/teleop/calibration.py`、`src/supermimic/teleop/retarget.py`、`tests/test_teleop_calibration.py`。
  - 具体动作：定义世界→机器人、头手→躯干的变换、yaw reset、身高/臂长缩放、腕坐标偏移与可达范围；持久化标定版本。
  - 验收标准：静止和各轴旋转样例方向正确；重标定无无界突跳；不可达目标有明确约束和状态反馈。

- [ ] **P4.03 因果稀疏编码与下肢补全**
  - 已存在：`src/supermimic/policy/nets.py`、`src/supermimic/config.py`。
  - 建议新增：`src/supermimic/teleop/goal_provider.py`。
  - 具体动作：从当前/历史头手和可选明确 locomotion 命令生成 Teleop goal；训练补全下肢与接触，不能把三点输入伪装成已知全身/未来 G1 goal。
  - 验收标准：站立、伸手、转身等定义范围内闭环稳定；数据与实时预处理一致；超出训练范围可降级，延迟不靠偷看未来降低。

- [ ] **P4.04 实时调度与失联处理**
  - 已存在：`src/supermimic/deploy/core.py`。
  - 建议新增：`src/supermimic/teleop/session.py`、`scripts/teleop/run_teleop.py`、`tests/test_teleop_timeout.py`。
  - 具体动作：分离 VR 接收/编码和 50Hz 控制，使用有界最新帧缓存；定义 deadman、stale、丢帧、重连、模式切换、平滑进入退出；接入现有安全状态机。
  - 验收标准：仿真故障注入覆盖断流/卡顿/乱序/掉追踪；失联在配置时限内触发既定安全状态，重连不自动恢复运动。

- [ ] **P4.05 实时遥操质量验证**
  - 已存在：`scripts/deploy/sim2sim.py`。
  - 建议新增：`scripts/teleop/eval_latency.py`。
  - 具体动作：在目标部署 CPU 测采集→接收→编码→控制全链路 p50/p95/p99/max，记录时钟同步质量；检查手端误差、动作速度和过冲。
  - 验收标准：给出真实设备+真实权重+闭环仿真的延迟报告；控制周期预算满足部署门槛，超预算有确定处理；不以网络 ping 或随机网络前向代替端到端延迟。

## P5 — VLA Token / Residual 基座接口
- [ ] **P5.01 Token 数据契约与提取**
  - 已存在：`src/supermimic/policy/nets.py`。
  - 建议新增：`src/supermimic/vla/token_codec.py`、`scripts/data/export_vla_tokens.py`、`tests/test_token_codec.py`。
  - 具体动作：冻结选定 BFM encoder/FSQ/decoder 版本，导出时序 token indices、quantized latent 与动作、观测时间对齐信息；定义两个32维组的序列化，不把64个标量索引误称两个词表 ID。
  - 验收标准：encode/index/decode 往返准确；每个分量索引范围合法；schema/hash 不匹配拒绝加载；SONIC token 不直接混用。

- [ ] **P5.02 VLA 训练样本与监督**
  - 建议新增：`src/supermimic/vla/dataset.py`、`scripts/data/build_vla_dataset.py`。
  - 具体动作：组织图像、语言、proprio、token chunk、动作/任务成功、时间戳与有效 mask；明确相机标定、图像采样、动作频率和窗口延迟，提供外部 VLA 训练器可消费格式。
  - 验收标准：逐样本可回放验证相机和动作对齐；按任务/场景隔离验证集；低层 BFM 数据与实际操作数据的覆盖差异可见。

- [ ] **P5.03 VLA Token 推理适配**
  - 建议新增：`src/supermimic/vla/adapter.py`、`scripts/vla/run_vla.py`、`tests/test_vla_adapter.py`。
  - 具体动作：定义 VLA token chunk→反量化→共享 Decoder 的接口、chunk 长度/频率、时效、取消和替换；高层异步输出，低层按当前 proprio 解码。
  - 验收标准：模拟慢响应/越界 token/乱序 chunk 有确定拒绝策略；控制不阻塞等待 VLA；同 token 与参考 decoder 输出一致。

- [ ] **P5.04 Residual 模式与界限**
  - 已存在：`src/supermimic/config.py`、`src/supermimic/deploy/core.py`。
  - 建议新增：`src/supermimic/vla/residual.py`、`tests/test_vla_residual.py`。
  - 具体动作：显式区分 token-only 与 token+residual；先固定 residual 作用空间（建议在 raw action 空间），定义物理/数值尺度、幅度/变化率约束与训练监督，最终统一进入 DeployCore。
  - 验收标准：residual=0 与 token-only 一致；极端 residual 不绕过限幅/力矩/速度门禁；latent residual 若另行实验必须独立 schema 和重新验证。

- [ ] **P5.05 VLA 基座训练与消融**
  - 已存在：`src/supermimic/policy/nets.py`。
  - 建议新增：`scripts/vla/eval_vla.py`、`docs/VLA_INTERFACE.md`。
  - 具体动作：先冻结 BFM 训练高层 token，再单独评估 residual；后续解冻需独立版本和全量回归；定义操作任务、夹爪/手接口与成功判定，29D 不包含独立手部命令时另立协议。
  - 验收标准：比较 token-only/residual 的成功率、末端误差、延迟、动作安全性及 BFM 能力保持；真实任务数据缺失时不得宣称 VLA 已训练或可部署。

## P6 — 分离式 ONNX、Sim2Sim 与 G1 安全部署
- [ ] **P6.01 分离式导出**
  - 已存在：`src/supermimic/deploy/export.py`。
  - 建议新增：`tests/test_split_onnx.py`。
  - 具体动作：导出 encoder_g1.onnx、encoder_smpl.onnx、encoder_teleop.onnx 和 decoder.onnx；定义 FSQ 放在 encoder 输出边界及 VLA codec 的统一实现，encoder 输出含量化 latent/必要索引；保留原 monolithic 基线路径。
  - 验收标准：ONNX checker 通过；图中无训练 STE 依赖；batch/dtype/shape/round 语义固定；缺模态权重不产出伪可用模型。

- [ ] **P6.02 部署产物 manifest**
  - 已存在：`src/supermimic/deploy/export.py`、`src/supermimic/deploy/core.py`。
  - 建议新增：`src/supermimic/deploy/manifest.py`、`tests/test_deploy_manifest.py`。
  - 具体动作：保存各图与 checkpoint hash、schema、FSQ levels、orientation 模式、关节/历史顺序、scale、频率和机器人变体；加载前验证 encoder/decoder 配套。
  - 验收标准：篡改图/混用版本/错机器人配置明确拒绝；无外部权重文件或完整记录依赖；产物能离线复现。

- [ ] **P6.03 分离运行时与数值一致性**
  - 已存在：`src/supermimic/deploy/core.py`、`tests/test_deploy_parity.py`。
  - 建议新增：`src/supermimic/deploy/omni_runtime.py`。
  - 具体动作：增加 NumPy/ORT 分离式 runtime 和模态路由；共用现有历史/reset、安全与动作映射；比较 PyTorch、单体 ONNX、分离 ONNX 与 VLA codec。
  - 验收标准：随机、真实录制、量化边界输入均检查；确定性索引应一致，action 最大绝对误差目标 <1e-4；任何边界差异必须解释并阻止未经确认的发布。

- [ ] **P6.04 原有 Sim2Sim 全门禁**
  - 已存在：`scripts/deploy/sim2sim.py`、`scripts/deploy/sim2sim_ref.py`、`docs/SIM2SIM.md`、`tests/test_sim2sim_peaks.py`。
  - 具体动作：接入分离 runtime，在相同 checkpoint/参考下复跑文档现有 G0–G5 门禁及目标机器人变体；测试延迟、质量/摩擦/增益扰动、初始化和结束。
  - 验收标准：逐门禁保存命令/配置/日志/通过条件，不降低既有阈值；汇报跌倒、跟踪误差、速度/力矩峰值和 CPU 周期；原模型历史通过不代表 Omni 通过。

- [ ] **P6.05 Teleop/VLA 专项 Sim2Sim**
  - 已存在：`scripts/deploy/sim2sim.py`。
  - 建议新增：`tests/test_omni_mode_switch.py`、`scripts/deploy/sim2sim_omni.py`。
  - 具体动作：增加跨模态切换、VR 断流、VLA 延迟、token 越界、残差饱和、姿态 reset、结束/重启故障场景；覆盖锁腰/非锁腰实际目标变体。
  - 验收标准：异常可重复触发预期安全状态且不自动复动；切换瞬态满足预设动作/力矩变化门槛；有逐场景报告。

- [ ] **P6.06 DDS 与 bridge 仿真端到端**
  - 已存在：`scripts/deploy/run_sim_e2e.sh`、`scripts/deploy/bridge/run_sim_e2e_bridge.sh`、`scripts/deploy/bridge/test_bridge.py`、`scripts/deploy/sim_g1_mujoco.py`、`scripts/deploy/bridge/g1_runner_bridge.py`。
  - 具体动作：先在隔离仿真 DDS 域接入新 runtime；保留 shared-memory、heartbeat、CRC/消息与状态机契约，测试进程退出/卡死和急停。
  - 验收标准：仿真控制链完整通过；确认不会连接真实机器人域；仅软件测试结果不标为真机验证。

- [ ] **P6.07 G1 上机前只读核查**
  - 已存在：`scripts/deploy/check_robot.py`、`scripts/deploy/bridge/ro_bridge_check.py`、`scripts/deploy/g1_common.py`、`scripts/deploy/realstate_gate.py`、`docs/DEPLOY_G1.md`。
  - 具体动作：核对实际型号/锁腰状态、关节/IMU/控制频率、网络、占用发布者、标定、产物 hash 和起始参考可行性；完成文档 UNVERIFIED 清单。
  - 验收标准：检查报告关联具体机器人与产物；状态陈旧、非有限数、控制权冲突或起始姿态不符时拒绝进入策略。

- [ ] **P6.08 保留低层安全保护**
  - 已存在：`scripts/deploy/g1_runner.py`、`scripts/deploy/bridge/g1_runner_bridge.py`、`scripts/deploy/bridge/g1_bridge.cpp`、`scripts/deploy/bridge/g1_bridge_client.cpp`、`src/supermimic/deploy/core.py`、`tests/test_realstate_gate.py`。
  - 具体动作：将新策略接入既有状态机，保留倾斜、速度、力矩、温度、电机错误、低状态 stale、超时与 heartbeat 门禁；新数据源不得绕过这些保护。
  - 验收标准：故障注入验证拒绝启动/停止路径；独立急停和控制线程 watchdog 有实测记录；不以放宽阈值解决模型不稳定。

- [ ] **P6.09 分阶段真机验证与退回基线**
  - 已存在：`docs/DEPLOY_G1.md`、`scripts/deploy/g1_runner.py`、`scripts/deploy/bridge/g1_runner_bridge.py`。
  - 建议新增：`docs/OMNIMIMIC_DEPLOY.md`。
  - 具体动作：在仿真门禁通过后由现场人员按已有 air/feet/ground 分级流程执行，先低幅静态，再短段 G1，再 Teleop，最后 VLA；支撑/吊架与遥控急停就绪；保留已验证基线产物。
  - 验收标准：每阶段独立记录设备、checkpoint、时长、故障和操作者；失败停止并分析首发日志，不自动重试；无真机证据保持 UNVERIFIED。

- [ ] **P6.10 发布验收与可追溯报告**
  - 已存在：`README.md`、`docs/METHOD.md`、`docs/SIM2SIM.md`、`docs/DEPLOY_G1.md`。
  - 建议新增：`docs/OMNIMIMIC_VALIDATION.md`。
  - 具体动作：汇总 P0–P6 状态、数据/权重/代码版本、所有门禁、已知限制和回退流程；区分 inspected/tested/trained/simulated/hardware-verified。
  - 验收标准：每一项勾选有报告或日志证据；缺数据/设备写 BLOCKED，未运行写 NOT_RUN；不存在仅靠源码审阅宣称完整可部署。

## 交付边界

本文件是开发计划。编写时仅完成路径与相关源码核查；上述实现、训练、ONNX、Sim2Sim、VR 设备和 G1 真机验收均未运行。本次不得据此启动机器人或勾选开发任务。
