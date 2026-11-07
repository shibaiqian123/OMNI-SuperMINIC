# OmniMimic / NJU 基线档案

核查日期：2026-11-07。范围：229服务器当前仓库只读核查；32尝试只读连接；不执行训练、安装、模型导出或机器人控制。

本文件完成P0.01的档案建立。已核查事实、仓库声明、计划命令与未运行结果分别标注；这不是“训练成功”报告。

## 1. 来源与代码

- 开发仓库：`https://github.com/shibaiqian123/OMNI-SuperMINIC.git`。
- 本机remote：仅 `origin`，fetch/push均为上述地址；未配置名为upstream的remote，无需为本档案强行添加。
- NJU来源（当前README声明）：`https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real`。
- 算法代码上游（当前README声明）：`https://github.com/CuboiLeo/SuperMimic-Reproduced`。本次未访问网络核对上游最新版本。
- 导入基线提交：`65a16ae5b5b17a9648e98ac6436f7a159e44cc80`，提交消息 `Initial import from original project`。这是自有仓库导入提交，不是已经证明相等的上游原始commit；上游精确原始commit仍为UNVERIFIED。
- 当前分支：`omni`；HEAD：`09ef1a8b81f42bfbf15d7bfaccb8962fc320d2e0`。
- 从导入基线到HEAD的已提交变化：仅 `docs/to_do_list.md` 和 `docs/PROJECT_FILES.md`；`src/`、`pyproject.toml`、`uv.lock` 的该范围diff为空。
- 首次采集时工作区曾有 ` M docs/to_do_list.md`。C01收尾复核时工作区已干净，两份目标文档不存在；本次依据重新校验的源码恢复文档，仅将这两个文件提交到omni，不切分支、不推送。
- 32的代码提交/分支/数据路径：BLOCKED，尚未实测。不能默认两端代码同步。

证据E01（229，已执行，只读）：

```bash
cd /home/cqu023/SuperMimic-Repro-Sim2Real
git remote -v
git branch --show-current
git rev-parse HEAD
git rev-parse 65a16ae
git status --short
git diff 65a16ae HEAD --stat
git diff 65a16ae HEAD -- src pyproject.toml uv.lock
```

输出摘要：origin为自有仓库，分支omni，HEAD和基线如上；已提交diff为2份文档、548行新增；核心源码/依赖diff为空。以上结果为首次采集工具输出摘录；C01收尾时再次核对HEAD与源码指纹，结果见第6节。未另外生成远程运行日志文件。

## 2. 环境与机器人

### 2.1 仓库环境声明（非已安装环境）

- `.python-version`：3.12；pyproject要求 `>=3.12,<3.13`。
- mjlab锁定git：`8bb2bc9e187797689ec82c78c35019afd2fd0376`；uv.lock包版本：1.6.0。
- Linux torch锁定版本：`2.14.1+cu130`；这是锁文件内容，不是32实装版本。pyproject声明下限为torch>=2.14.0。
- pyproject.toml SHA-256：`692730ceadaa6b09522840886c72147ceab2882546fd4a328b55e76d89ee96f1`。
- uv.lock SHA-256：`0b7f650ed31337967af38d1735380631f2da020f6aaec913bd8d5fb048e7732b`。

### 2.2 229本次SSH命令实际所见

- `python3 --version`：Python 3.10.12。
- 该解释器的importlib.metadata中torch/mjlab/numpy均未找到。只说明这个解释器，不代表整台机器没有其他环境。
- 仓库根目录 `.venv`：本次检查不存在。
- GPU：NVIDIA GeForce RTX 4090；driver：580.178.04。
- torch CUDA版本：NOT_RUN，该解释器未发现torch；不能把驱动支持版本当torch构建版本。
- 这不是符合项目Python要求的已确认训练环境；本轮不安装/修复它。

证据E02：`python3 --version`、`importlib.metadata`查询、`nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`、`sha256sum pyproject.toml uv.lock`的实际输出摘录如上。

### 2.3 32训练环境

状态：BLOCKED_AUTH；环境核查、仓库核查及训练均未运行。

证据E03（2026-11-07连接尝试）：

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 2025slx@10.246.1.32 ...
2025slx@10.246.1.32: Permission denied (publickey,password).
```

不能从认证失败推断32缺仓库、缺GPU或缺数据。后续在32真实训练解释器内回填Python/torch/torch CUDA/mjlab来源、驱动、GPU、commit与数据路径；不把229系统Python结果填入32环境栏。

### 2.4 机器人配置

源码默认是G1 29维动作接口；`EnvCfg.lock_waist=False`、`anchor_obs=False`。root为`pelvis`，anchor为`torso_link`；IMU饱和、腰部耦合等额外选项默认关闭。以实际训练保存配置为最终依据，不能用默认值推断已有checkpoint。

目标实机锁腰状态、序列号、IMU标定、控制模式：UNVERIFIED（未连接机器人）。若使用锁腰变体，13/14槽保留但不对应活动关节；必须单独记录配置与数据可行性处理。

## 3. 数据

229仓库根目录的 `data/`、`logs/`、`deploy_artifacts/` 本次均不存在。这里仅描述已检查的位置，不推断服务器其他路径没有数据/权重。

随仓库附带的原始动作资源已计算指纹，属于版本化输入；并非32训练数据集，也不是转换后的NPZ动作库：

```text
71b3471dfc07efb8f1862153aa214e32fc7c47ce3662446e247c8fe5a619cd54  motions/custom_dance/gouxiong_duosuomao.csv
9ae7f612a32c182717a004abaa34e285d115d759b4bce19a23742030dc8f1750  motions/custom_dance/langzi_tiqiu.csv
82e2b408447f980e389b063728ac3c18d653a290e91f4ade14f3060f054e5997  motions/custom_dance/shanshenbu.csv
1ef51910dba048f6392df3817ab7c749c6273a15cae34007ab624e4310b57a18  motions/custom_dance/shanshenbu_combo4.csv
89fbb963efa7d186c212edb562d7566cdb2299b3d879e970d813750dc76ccf00  motions/custom_dance/shanshenbu_combo4.json
3b87675f3bcb046b1b1113a585f907a48ffc365fa32b692d154ba03344fa0403  motions/custom_dance/yaotou_huangnao.csv
8065f1e9d6686f3a42bb90b5b772f05843603b4bf7abe0aa337a7ab43e0ed7e4  motions/shanshen_duosuomao/reset_points_lean35kfs.csv
```

证据E04：对 `find motions -type f` 返回的7个文件逐一执行sha256sum。训练数据路径/数量/划分/manifest/library hash：NOT_RUN，待32访问与数据选择后核对。manifest hash只证明manifest内容，不自动覆盖全部NPZ。

## 4. 实验记录

### 4.1 上游报告

当前README有LAFAN1、BONES-SEED和e24舞蹈真机报告；细节见 [METHOD.md](METHOD.md) 和 [REAL_ROBOT_LEAN35KFS.md](REAL_ROBOT_LEAN35KFS.md)。这些属于上游项目陈述，本次不复填为“本机成绩”。e24父checkpoint未随对应发布提供的限制见原说明，不能宣称已具备逐位复现条件。

### 4.2 本项目实际运行

- 32基线训练：NOT_RUN；日志/命令/输出checkpoint：无实际运行记录，访问受阻。
- 固定数据集完整评测：NOT_RUN；Return/失败率/耗时：NOT_RUN。
- World Model和imagination真实训练检查：NOT_RUN。
- ONNX导出与ORT parity：NOT_RUN。
- Sim2Sim、VR、VLA和G1真机：NOT_RUN。
- 文档契约离线检查：已执行6项NumPy/DeployCore源码副本检查，见 [OMNIMIMIC_CONTRACT.md](OMNIMIMIC_CONTRACT.md) 的G01；仅检查合成输入的预处理/映射，不能填为训练结果。

### 4.3 待执行命令模板（不是运行证据）

以下保留README已有入口；在32确认仓库、训练环境、数据及算力预算后再执行。路径需替换；本次没有启动这些命令。

```bash
# 在32实际仓库、实际训练环境中执行；先检查双方commit及依赖。
uv run python --version
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
uv run python -m pip show mjlab
uv run sm-train --help
# 模板：原始G1连续32、anchor_obs默认关闭；完整记录最终实际命令。
uv run sm-train --run-name nju_baseline --data.npz-dir /实际路径/npz   --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours 3
uv run python scripts/eval_benchmark.py   --ckpt /实际路径/ckpt_XXXXXXX.pt --npz-dir /实际路径/npz
```

资源参数是README参考值，不是对32当前资源的推荐或实测。若uv环境无pip，用 `uv run python -c 'import importlib.metadata as m; print(m.version("mjlab"))'` 查询包版本；安装来源/commit还需核对direct_url或锁文件。实际运行时保存stdout/stderr、最终config.json、seed、数据划分及起止时间；将命令和日志路径回填本节。

## 5. 产物与验收

- 实际选定checkpoint路径/大小/SHA-256：NOT_RUN，尚未发现并选定候选；不下载或伪造权重。
- 实际训练数据产物与hash：NOT_RUN；仅第3节附带动作资源已核实。
- ONNX/config/clip bundle及hash：NOT_RUN。
- 本次文档产物：本文件、[OMNIMIMIC_CONTRACT.md](OMNIMIMIC_CONTRACT.md)。

档案完整性：已将已知内容、来源及缺失原因如实写明。P0.01文档工作完成；32环境与实验结果仍缺证据，不能由“文档完成”推导“基线训练通过”。C01按文档范围验收，见下节。

## 6. C01文档提交验收（2026-11-07）

验收范围：固定代码基线、说明当前数据接口、区分已有实现与拟议接口、明确实验缺证据项。此范围对应交互清单C01，不要求先运行训练或真机。

- PASS：再次确认分支omni、父提交09ef1a8b81f42bfbf15d7bfaccb8962fc320d2e0、写入前工作区干净。
- PASS：14份已审阅源码/依赖/README与原核查副本SHA-256逐一相同，差异列表为空，适用上级及仓库AGENTS.md未发现。
- PASS：两份文档分别覆盖来源/环境/数据/实验/产物和观测/goal/FSQ/动作/时间/schema；真实值、锁文件声明、计划与NOT_RUN分开标注。
- PASS：复跑当前源码副本的6项DeployCore离线golden检查，history最大误差0；输出与脚本完整保存在契约文档中。
- PASS：C01范围仅包含本档案和契约文档；不包含程序修改、依赖安装、数据转换或模型训练。
- 待后续运行验证：32环境确认、真实checkpoint、环境/imagination/ORT完整parity、Sim2Sim及真机。不得把本组PASS当作这些事项通过。

结论：**C01文档改动完成，本组文档验收通过。** 本次提交由Git自身记录commit，避免在提交内容中写入尚不存在的自身提交号。浏览器进度不由仓库文档回写。
