# OmniMimic 数据与模型接口契约

核查日期：2026-11-07；源码基准 `09ef1a8b81f42bfbf15d7bfaccb8962fc320d2e0`，代码与导入基线 `65a16ae` 无已提交差异。本文完成C01范围的现有接口说明和拟新增接口边界。完整环境/imagination/ORT parity仍未运行；C01文档验收与原逐文件P0.02的动态golden验收分别记录。

约定：切片均为Python左闭右开；B为batch，H=10为默认历史长度，K=10为默认goal窗口，Nbody为body数。训练浮点通常float32；compact库存储可为float16，查询转换不代表无量化误差。部署预处理可为NumPy float64，OnnxPolicy边界转float32。

## 1. 坐标、单位和关节顺序

世界坐标位置单位m；线速度m/s；关节/旋转角rad；角速度rad/s；周期s。四元数内部采用wxyz，表示body到world姿态；`quat_apply_inverse(q,v_world)`得到body向量。heading坐标仅移除root yaw，不等同于完整body坐标。root为pelvis，anchor为torso_link，不能用pelvis IMU直接冒充torso姿态。

CSV原始四元数为xyzw，`data/motion_convert.py`转换成wxyz；NPZ的body_quat_w也是wxyz。必须以字段来源确认世界角速度还是机体角速度，不能只看shape为3。

29关节顺序（来自 `G1_JOINT_NAMES`；部署模型仍需按导出joint_names核对）：

- 00：`left_hip_pitch_joint`
- 01：`left_hip_roll_joint`
- 02：`left_hip_yaw_joint`
- 03：`left_knee_joint`
- 04：`left_ankle_pitch_joint`
- 05：`left_ankle_roll_joint`
- 06：`right_hip_pitch_joint`
- 07：`right_hip_roll_joint`
- 08：`right_hip_yaw_joint`
- 09：`right_knee_joint`
- 10：`right_ankle_pitch_joint`
- 11：`right_ankle_roll_joint`
- 12：`waist_yaw_joint`
- 13：`waist_roll_joint`
- 14：`waist_pitch_joint`
- 15：`left_shoulder_pitch_joint`
- 16：`left_shoulder_roll_joint`
- 17：`left_shoulder_yaw_joint`
- 18：`left_elbow_joint`
- 19：`left_wrist_roll_joint`
- 20：`left_wrist_pitch_joint`
- 21：`left_wrist_yaw_joint`
- 22：`right_shoulder_pitch_joint`
- 23：`right_shoulder_roll_joint`
- 24：`right_shoulder_yaw_joint`
- 25：`right_elbow_joint`
- 26：`right_wrist_roll_joint`
- 27：`right_wrist_pitch_joint`
- 28：`right_wrist_yaw_joint`

锁腰13/14槽保留在29维软件接口；对应观测补零、动作不执行，不能当作同一个非锁腰物理模型。

## 2. Actor观测与历史

生产：环境 `envs/tracking_env_cfg.py` 的policy_terms；imagination `world_model/heads.py:analytic_observation()`；部署 `deploy/core.py:obs_frame()`。消费：`policy/nets.py:Decoder.forward()`。

单帧 `[B,93]`：

- `[0:29]`：`q_measured-q0`，rad；仿真解析路径为 `q+bias-q0`，部署传入的关节读数已代表测量值，不再重复加仿真bias。
- `[29:58]`：dq，rad/s。
- `[58:61]`：pelvis角速度，body frame，rad/s。
- `[61:64]`：`R(root_quat)^T*[0,0,-1]`，无量纲单位重力方向，不乘9.81。
- `[64:93]`：上一clean/raw action，29D，无量纲；不是上一q_target，不是加噪执行动作。

历史张量语义 `[H,B,93]`，旧→新。term-major输出 `[B,930]`：q历史`[0:290]`、dq历史`[290:580]`、角速度历史`[580:610]`、重力历史`[610:640]`、action历史`[640:930]`。每个term内部按时间再按分量展开。不可直接reshape完整 `[H,B,93]`。

Decoder内部除数按上述term分别为1、5、2、1、1；预处理层不要再次除一遍。DeployCore第一次调用observation回填10份首帧；reset将last_action清零并清hist；switch_clip保留hist和last_action。训练环境首帧行为仍需真实环境parity确认，不能仅以部署实现推定。

## 3. Goal、frame与orientation

### 3.1 基础goal

`envs/mdp/motion_library.py:goal_features()`每帧 `[B,68]`：

- `[0:29]` reference joint_pos-q0，rad。
- `[29:58]` reference joint_vel，rad/s。
- `[58:61]` reference projected gravity，body frame，无量纲。
- `[61:64]` reference root_ang_vel_world转reference body frame，rad/s。
- `[64:67]` reference root_lin_vel_world转reference heading frame，m/s。
- `[67:68]` reference root height，m。

goal窗口按帧拼接 `[B,K*68]`，这与proprio的term-major不同。MotionEncoder内部每帧除数为1、5、1、2、1、1；先scale再MLP和tanh，输出连续32D。

### 3.2 必须分开的frame变量

- MotionLibrary使用整库global frame；clip含start和length。`goal_window(f,K)`精确取 `min(f+i,clip_last)`，i=0…K-1，**函数本身没有额外+1**。
- MultiClipMotionCommand.goal直接传入cmd.frame；`_update_command()`推进cmd.frame并refresh reference。
- ReplayRecorder保存当步command frame；`imagine()`从StartBatch.frame开始，调用同一goal_window，每步frame+1并clip。
- DeployCore.frame是部署的当前clip-local进度k；`goal()`精确取 `min(k+1+i,L-1)`。两边对齐要求是 `cmd.frame-clip_start == min(core.frame+1,L-1)`，不是令两个frame整数相等。
- 现有 `tests/test_deploy_parity.py:record_env/replay_core`分别记录上述首goal帧做对照。实际mjlab reset/step钩子顺序不能仅凭本仓库函数定义宣称已验证；完整parity本轮NOT_RUN。
- 例：clip有12帧，部署k=0取1…10；等效库调用f=clip_start+1。k=10取10份第11帧；短clip不能跨入下一clip。
- reward使用转移目标ref评价s_next；不要把goal窗口第一帧、观测对应状态、reward参考时刻混为一谈。

### 3.3 已有9D与拟新增6D

当前配置只有anchor_obs bool：False为680D，True在窗口后追加3D相对位置和6D相对旋转，共689D。相对位置 `R_robot_anchor^T*(p_ref_anchor-p_robot_anchor)`；相对旋转 `R_robot_anchor^T*R_ref_anchor`。

6D的精确展平为 `[R00,R01,R10,R11,R20,R21]`，即先取矩阵前两列再按行展开；**不是** `[R00,R10,R20,R01,R11,R21]`。单位旋转为 `[1,0,0,1,0,0]`，绕z正90度为 `[0,-1,1,0,0,0]`。q和-q应表示同一结果。

未来拟增加orientation_mode：none/ori6d/pose9d，K=10分别680/686/689D。ori6d接口尚不存在；pose9d需要合法位置来源，ori6d也需要指定anchor的可观测姿态和标定；不能用仿真真值冒充部署可用信号。当前DeployCore/Clip只实现基础680D路径，不能因训练有anchor_obs就宣称现有部署支持689D。

## 4. Actor / Critic / WM 的信息边界

### Actor（现有）

`Policy.forward(obs_hist,goal,interface="motion") -> [B,29]`。只注册motion encoder，输入goal与proprio。不能直接读取xi、完整71D仿真状态、root世界线速度、e_root或训练标签。可选anchor特征也必须来自对应模式允许的输入来源。

### Critic（现有）

`Critic.forward(feat,goal,xi_norm,e_root) -> [B]`：canonical state71 + goal680 + xi33 + e_root4 = 788D默认输入；anchor模式为797D。goal在Critic内部缩放，feat已经由StateSpec.canonical归一化，xi已由normalize_xi归一化。

e_root=`[R_ref_heading^T*(p_robot-p_ref), wrap(yaw_robot-yaw_ref)]`；前三维m，最后一维rad。它是Critic-only特权量，不是6D朝向。当前实际只有一个Critic实例，未来按模态head是规划，不是已实现。

### World Model（现有）

原始状态 `[B,71]`：root_pos`[0:3]`、root_quat_wxyz`[3:7]`、qj`[7:36]`、root_lin_vel_world`[36:39]`、root_ang_vel_body`[39:42]`、dqj`[42:71]`。位置移除各仿真env origin，不能忽略这一转换。

canonical feature顺序/切片：root_z`[0:1]`、heading移除后的rot6d`[1:7]`、heading线速度`[7:10]`、body角速度`[10:13]`、绝对qj`[13:42]`、dqj`[42:71]`。除数为1、1、1、2、1、5；这里qj不是Actor的q-q0。

xi `[B,33]`：摩擦`[0:1]`（无量纲）、encoder bias`[1:30]`（rad）、torso COM offset`[30:33]`（m）。归一化offset为0.75/0/0、scale为0.45/0.01/0.05。

Dynamics输入 `[canonical71,action29,xi_norm33]`，共133D；不直接拼接goal。输出delta70：dv_h`[0:3]`、dw_b`[3:6]`、dqd`[6:35]`、dp_h`[35:38]`、dtheta_b`[38:41]`、dq`[41:70]`，乘out_scale=(0.3,0.5,2,0.01,0.02,0.02)后积分。dt=0.02；零delta仍按速度外推，不是状态恒等。

WM训练window_loss消费executed action；Actor/imagination优化使用策略action并保留图。WM的解析reward和termination可读取完整参考与状态；TerminationHead接高度误差/tilt特征及绝对值，不能把这一“读参考”误述为DynamicsMLP直接读goal。

BPTT时WM/Critic参数冻结而输入梯度保留。当前策略目标包含折扣survival奖励和可导终端V_H；Critic训练的GAE targets再detach。FSQ必须使用STE才能保留量化处梯度。

## 5. 动作与部署时间

Actor输出29D raw action；仿真执行可添加高斯噪声，回放同时保存clean与executed。目标角度为 `q0+action_scale*action`，不是raw action本身；kp/kd/effort/scale来自实际模型导出，本文不以合成测试参数替代机器人配置。

DeployCore使用50Hz/20ms；机器人低层DDS发布频率与策略频率是不同概念。q_target输出可经力矩/软限位处理；last_action存网络raw action而不是限幅后的目标。IMU合法性、速度、倾斜、幅值等检查与runner/bridge控制权、stale、watchdog、热保护共同生效。

未来VR/VLA接入不得阻塞控制循环；使用接收单调时间判断stale，记录源时间与同步误差；重连不能自动复动。终止/取消必须走原runner状态机。

## 6. Omni拟新增schema（尚未实现）

- 建议schema ID：`omni-g1-fsq-v1`；本ID目前仅为文档约定，不存在加载器校验实现。
- Encoder输出prequant `[B,64]`；量化分组 `[B,2,32]`；indices同shape、整型；z_q展平 `[B,64]`；Decoder输入z_q及obs `[B,930]`，输出action `[B,29]`。
- levels与2×32维度分开配置；拟初始每标量L=32。当前尚未实现FSQ，bound/中心偏移/round ties/归一化公式没有源码可核验。**这些字段在P1.fsQ实现前必须冻结并给边界golden向量，不把“32 levels”当作完整数值契约。** 暂不允许产出声称兼容的token/ONNX bundle。
- 必填metadata：schema_version、variant、encoder/decoder/hash、modality、feature_names/order/units/scale、history/goal offsets、quaternion_order、rotation6d_layout、joint_names、robot_variant、orientation/anchor、FSQ levels/bound/round/normalization、control_dt。
- 每次接口不兼容变更必须升级schema；不能strict=False静默吞掉32D→64D不匹配，不假设SONIC与新token语义一致。
- Teleop拟单时刻30D：头/左手/右手各pos3+rot6d6，再附3个valid mask；T帧历史按显式schema堆叠，不能套用proprio的term-major。输入仅当前/过去，未来预测另标记。
- SMPL/SMPL-X关节集映射、人体shape和root处理仍需P2固定，不靠截断张量维度兼容。
- VLA token chunk拟为 `[T,2,32]`，每分量索引独立；Residual首版固定29D raw action空间。手/夹爪通道单独定义，不默认包含于29D body action。

## 7. Golden样例、实际检查与验收边界

### G01：DeployCore离线检查（已运行，范围有限）

运行地点：本地文档工作环境；从229读出当前deploy/core.py源码副本，未修改或执行远程控制。NumPy 2.5.0，无Torch/mjlab/ORT。本样例为故意简化的合成输入，不是机器人可执行姿态，也不是策略行为验证。

以下脚本可在含NumPy的环境中重放：保存为仓库外的临时脚本，传入当前 `src/supermimic/deploy/core.py` 路径。它只调用observation/goal/targets及重力函数，不调用robot runner、不发送命令。

```python
import runpy, numpy as np, json, sys
m = runpy.run_path(sys.argv[1])
q0=np.arange(29,dtype=float)*0.01
cfg=m['DeployConfig'](joint_names=[f'fixture_joint_{i}' for i in range(29)],default_joint_pos=q0,action_scale=np.full(29,0.25),kp=np.ones(29),kd=np.zeros(29),effort_limit=np.full(29,1000.),joint_lower=np.full(29,-10.),joint_upper=np.full(29,10.),velocity_limit=np.full(29,100.))
table=np.arange(12*68,dtype=np.float32).reshape(12,68)
clip=m['Clip']('golden-v1',table,np.zeros((12,29)),np.tile([0.,0.,-1.],(12,1)))
core=m['DeployCore'](cfg,lambda obs,goal:np.zeros(29),clip,m['SafetyCfg'](clamp_torque=False,clamp_soft_limits=False))
frames=[]
for t in range(10):
 q=q0+np.arange(29)*0.001+t*0.01;dq=np.arange(29)*0.02+t*0.03;gyro=np.array([t*.01,t*.02,t*.03]);prev=np.arange(29)*.001+t*.005
 core.last_action=prev.copy()
 frame=np.concatenate([q-q0,dq,gyro,[0.,0.,-1.],prev]);frames.append(frame)
 obs=core.observation(q,dq,gyro,[1.,0.,0.,0.])
 if t==0:
  ref=np.concatenate([np.tile(frame[a:b],10) for a,b in [(0,29),(29,58),(58,61),(61,64),(64,93)]])
  np.testing.assert_allclose(obs,ref,atol=1e-12)
expected=np.concatenate([np.array(frames)[:,a:b].reshape(-1) for a,b in [(0,29),(29,58),(58,61),(61,64),(64,93)]])
np.testing.assert_allclose(obs,expected,atol=1e-12)
core.frame=0;np.testing.assert_array_equal(core.goal(),table[1:11].reshape(-1))
core.frame=10;np.testing.assert_array_equal(core.goal(),np.tile(table[11],10))
a=np.linspace(-1,1,29);target,_=core.targets(a,q0,np.zeros(29));np.testing.assert_allclose(target,q0+.25*a,atol=1e-12)
np.testing.assert_array_equal(m['projected_gravity'](np.array([1.,0.,0.,0.])),[0,0,-1])
print(json.dumps({'scope':'actual DeployCore snapshot; synthetic inputs; no Torch/mjlab/ORT/robot','checks_passed':['initial_history_backfill','term_major_930','goal_frame_1_to_10','goal_tail_repeat','raw_action_mapping','identity_gravity'],'max_history_error':float(np.max(np.abs(obs-expected))),'goal_shape':[680],'obs_shape':list(obs.shape)},ensure_ascii=False))
```

实际输出摘录（不是完整环境测试日志）：

```json
{"scope": "actual DeployCore snapshot; synthetic inputs; no Torch/mjlab/ORT/robot", "checks_passed": ["initial_history_backfill", "term_major_930", "goal_frame_1_to_10", "goal_tail_repeat", "raw_action_mapping", "identity_gravity"], "max_history_error": 0.0, "goal_shape": [680], "obs_shape": [930]}
```

源码副本SHA-256：`62a508bf13aa6c6c64abfafe0010623230cf28a23cee0303844970e124b66435`。该记录与本文脚本可以复查；没有在仓库另新增测试程序或日志文件。

### G02：环境与imagination消费同一golden样例（定义完成；执行NOT_RUN）

使用G01同一组q0、q/dq/gyro/prev_action和单位root姿态，构造71D state，xi的encoder bias设0；按时间顺序分别调用 `analytic_observation()` 和 `flatten_history_term_major()`。应得到G01 expected的 `[1,930]`，容差float32下atol=1e-6；此步骤需要Torch及模块依赖。

goal库构造与G01相同12×68索引表：环境command frame设clip_start+1，imagination start.frame同值，部署core.frame=0，三者应取完全相同1…10帧。再测试库frame=clip_start+11和部署k=10的片尾重复。索引表仅用来检查窗口排列；物理goal_features另用单位root姿态、零参考速度、已知qj/q0与height构造预期68D向量。

环境完整测试还必须核对reset后首goal帧与history回填，录制同一真实checkpoint rollout并送入 `tests/test_deploy_parity.py`；不能把人为设置frame的离线用例等同于真实reset钩子测试。现有parity测试要求CUDA、指定NPZ、checkpoint、ONNX及ORT，本轮不满足。

### G03：朝向golden（书面数值；执行NOT_RUN）

robot anchor单位姿态，ref同姿态：rot6d=[1,0,0,1,0,0]；ref绕z+90度：rot6d=[0,-1,1,0,0,0]。相对位置ref-robot=[1,2,3]且robot旋转为单位时，pose9d前3维应为[1,2,3]。分别检查q/-q同结果、torso/root不混用。未来ori6d只能在其实际实现后验收。

### P0.02状态

- 文档与字段源码核查：完成。
- 当前DeployCore合成预处理检查：PASS，6项，history最大误差0。
- 环境/imagination完整共同golden、真实权重parity：NOT_RUN / 依赖与产物缺失，32认证受阻。
- FSQ、ori6d、Teleop/VLA schema：设计，尚未实现，不能报PASS。

C01文档验收通过：数据契约已按当前源码固定，设计未定项与NOT_RUN边界明确。完整动态验收仍未通过；须确认实际训练环境、动作数据和选定checkpoint，再按G02运行，记录日志/模型hash/误差后更新其状态。不能据C01完成将原P0.02完整三路golden勾为PASS。

## 8. 源码定位

- [src/supermimic/deploy/core.py](../src/supermimic/deploy/core.py)
- [src/supermimic/world_model/heads.py](../src/supermimic/world_model/heads.py)
- [src/supermimic/world_model/state_spec.py](../src/supermimic/world_model/state_spec.py)
- [src/supermimic/envs/mdp/motion_library.py](../src/supermimic/envs/mdp/motion_library.py)
- [src/supermimic/envs/mdp/commands.py](../src/supermimic/envs/mdp/commands.py)
- [src/supermimic/imagination/rollout.py](../src/supermimic/imagination/rollout.py)
- [src/supermimic/policy/nets.py](../src/supermimic/policy/nets.py)
- [src/supermimic/data/motion_convert.py](../src/supermimic/data/motion_convert.py)
- [src/supermimic/envs/tracking_env_cfg.py](../src/supermimic/envs/tracking_env_cfg.py)
- [src/supermimic/envs/sim_context.py](../src/supermimic/envs/sim_context.py)
- [tests/test_deploy_parity.py](../tests/test_deploy_parity.py)

## 9. C01源码指纹复核

2026-11-07再次从229读取以下文件并与字段审查副本比对；14项全部相同。此检查确保本文描述对应实际当前源码，不替代执行各模块。

```text
62a508bf13aa6c6c64abfafe0010623230cf28a23cee0303844970e124b66435  src/supermimic/deploy/core.py
5320c27d9831f774e517f08b1a42936fd843c0e8ca76937694eebd56f4cbad94  src/supermimic/world_model/heads.py
cf224f94a8993b04d3d8cc7bc772e79e8385674affb41e5a9ecc974fcbde2484  src/supermimic/world_model/state_spec.py
892bf0ca1353b77bab358b1d01ff0a6baa60dc9ccaf70be0e7683bda6a8be072  src/supermimic/envs/mdp/motion_library.py
c7b0d549a38e36b4f179e9be3f037d49601410abb22b68d98dd1b594bda04181  src/supermimic/envs/mdp/commands.py
1005bcb1c979b437ca92338fafb53e754e899d8ca04add3b72eb78a9028fab9d  src/supermimic/imagination/rollout.py
02e33aeaf0b371129f349223429682819868a831f4dac8dd834fdc848e57e471  src/supermimic/policy/nets.py
032047915b7dde7b348e72e5dca1529cee15d7259800d5d81144a5c9a82f5bc1  src/supermimic/data/motion_convert.py
dd6835fd4e1c805f398298a5589521e323245d5a00976315a9999e7eaf270c57  src/supermimic/envs/tracking_env_cfg.py
3efcf858d60357a66ae7bdb0709ef3c735c2bf544d961f466786078b10ee331f  src/supermimic/envs/sim_context.py
25372f7d2186dd4a53ecc5f859395101645478ca809757c268f01d169f2b5d1f  tests/test_deploy_parity.py
692730ceadaa6b09522840886c72147ceab2882546fd4a328b55e76d89ee96f1  pyproject.toml
0b7f650ed31337967af38d1735380631f2da020f6aaec913bd8d5fb048e7732b  uv.lock
e63fecc96fa5764fc4df3d54e1b31737d602e0194b55c5713bea2aa2e18b9bea  README.md
```

两份文档的相对链接在写入前核对；最终提交范围由Git检查，仅允许本档案与契约文档。

## 10. P1实现补充（2026-11-09）

第6节记录的是C01时的拟议schema；以下为当前已落地的G1 P1契约：

- `policy_variant=continuous`：原始32D连续latent，旧参数名和严格checkpoint加载保持不变。
- `policy_variant=continuous64`：64D连续latent，不经过量化，用作宽度消融。
- `policy_variant=fsq64`：Encoder输出`[B,64]`，FSQ重排为`[B,2,32]`，Decoder展平后与930D observation拼接并输出29D action。
- 每个FSQ分量在`[-1,1]`上使用32个包含端点的等距中心。索引公式为`floor((clip(x,-1,1)+1)*(L-1)/2+0.5)`，即最近中心、精确半格向上；索引范围`[0,31]`。训练前向使用同一量化值，反向使用STE；确定性路径不使用STE。
- checkpoint metadata保存variant、token数量、token维度、levels、范围和round规则。32D/64D、continuous/FSQ或goal schema不匹配时在加载权重前失败，不使用`strict=False`。
- `Policy.encode()`、`quantize()`和`decode()`是拆分接口；组合调用必须与`forward()`数值一致。当前仅注册`motion`接口，SMPL/Teleop仍属于P2之后工作。
- Goal仅接受`K*68`或`K*68+9`。`anchor_obs=False`为默认680D；`anchor_obs=True`复用现有torso相对位置3D和相对旋转6D，得到689D。没有新增ori-only 686D模式。
- PolicyTrainer在Actor BPTT期间保留WM/Critic原始`requires_grad`状态并在异常时恢复；FSQ训练记录全局level coverage、饱和比例和平均绝对量化误差。

32独立验证checkout上的离线结果：目标测试`47 passed`；隐藏CUDA后的全CPU回归为`120 passed, 39 skipped`。跳过项主要需要CUDA、数据或部署产物，不能计作通过。G1 FSQ训练、anchor训练、闭环benchmark、ONNX和Sim2Sim尚未通过；当前GPU资源状态为`BLOCKED_RESOURCE`。
