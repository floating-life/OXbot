# OXbot 竞赛版（基于 FableDan）—— 9800X3D + RTX 5080 训练与上线手册

当前 OXbot 部署使用 C++17，训练仍按本页运行。训练完成后按
[C++ 迁移说明](../../docs/fabledan_cpp.md) 导出 FBDN 权重并构建；本页第 3 节保留的是
Python 对照包上传流程。

> 竞赛版仅用于 Botzone 等非商业用途（FableDan 许可：Apache 2.0 + 非商业限制）。商业版不得使用本仓库代码与训练出的权重，除非取得作者书面许可。

## 0. 一次性准备（约 15 分钟）

1. 安装 Python 3.12（勾选 "Add to PATH" 与 "py launcher"），显卡驱动更新到最新。
2. 解压本包到一个**不在 OneDrive 同步目录**里的路径，例如 `D:\coding\OXbot-comp`。
3. 双击或在 cmd 里运行 `scripts\setup_windows.bat`：
   - 建虚拟环境 `.venv`，装 numpy 和 **CUDA 12.8 版 PyTorch**（RTX 50 系必须 cu128 或更新，装成 CPU 版或旧 CUDA 版会报 `no kernel image` 或 `cuda False`）；
   - 自检 GPU、bf16 矩阵乘、跑测试。最后一行出现 `SETUP OK` 即可。
4. 建议：Windows 电源计划设为"高性能/卓越性能"，关闭自动睡眠；在 Windows 安全中心把本目录加入"排除项"（Defender 实时扫描会拖慢检查点写入和多进程启动）。
5. 把官方裁判源码（当前核验副本 SHA-256 为 `910cba94…`）放到 `judge\judge_official.py`。缺少官方裁判时上传检查会直接失败，不会退回修正版。安装脚本还会运行 `tools\check_environment.py`，用默认四层模型做一次真实 CUDA 前向、反向、优化器更新和 NumPy 导出一致性检查；它使用合成数据，只证明训练链路可用。

## 1. 开训

### 真题监督训练（当前 `real-v2`）

如果要复现当前交付的竞赛模型，先确保 `data\processed\fabledan-real-v2`
已准备好，再运行：

```
scripts\train_real_5080.bat
```

它默认使用 `ckpts\real-v2`，训练 8 个 epoch，只读取 train/validation；中断后再次运行会从
`latest.pt` 续训。held-out test 必须单独评估：

```
.venv\Scripts\python.exe train_real.py evaluate --data ..\data\processed\fabledan-real-v2 --checkpoint ckpts\real-v2\best.pt --split test --out reports\real-v2-test.json --device cuda:0
```

### 自博弈训练

下面的 `train_5080.bat` 是持续自博弈路线，默认输出 `ckpts\run1`；它与真题监督训练可以并行保留，不能混用检查点。

```
scripts\train_5080.bat
```

在 WSL2 中使用现有 CUDA 环境时运行 `bash scripts/setup_wsl.sh`，然后运行
`scripts/train_5080_wsl.sh`。两个启动器都支持 `--cycles 1` 这类额外参数，检测到
`latest.pt` 会自动续训；设置 `OXBOT_NO_RESUME=1` 可强制新实验，`OXBOT_OUT`、
`OXBOT_ACTORS`、`OXBOT_RING`、`OXBOT_LADDER_FRAC` 和 `OXBOT_MICRO_BATCH` 可覆盖默认值。

- 中断后（关机、报错、手动 Ctrl+C）**直接再运行一次**，会从 `ckpts\run1\latest.pt` 续训（回放池会重新攒，几分钟就满）。检查点和 `latest.npz` 采用临时文件替换，异常中断不会覆盖上一个完整文件。
- 关键参数（都在 bat 里）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--actors 8` | 8 | 自博弈进程数。9800X3D 是 8 核 16 线程，另有推理进程和训练进程要占 CPU |
| `--ring 32` | 32 | 每个进程同时跑的牌局数；越大 GPU 批次越大，往返越少 |
| `--ladder-frac 0.5` | 0.5 | 一半牌局按 Botzone 默认设定（打 2、不进贡、0 号先出），一半随机级牌与进贡 |
| `--batch 4096` | 4096 | 每个优化器步骤的样本数 |
| `--micro-batch 128` | 128 | 梯度累积分块；单张 16 GiB 5080 与推理进程同卡时的保守显存余量，可用 `OXBOT_MICRO_BATCH` 覆盖 |
| `--eval-games 100` | 100 | 每次评测局数（同牌换座，成对出现） |
| `--export-cycles 50` | 50 | 每 50 个周期导出一次 `latest.npz`（上传用的 numpy 权重） |

### 怎么调到最快
日志每个周期打印 `... samples/s`，推理进程每 30 秒打印 `[infer] N decisions/s`。目标是 **samples/s 最大**：

- 任务管理器 → 性能 → GPU → 把一个小图切到 **Cuda**：
  - GPU 长期接近 100%、CPU 没满 → GPU 是瓶颈，把 `--actors` 降到 6；
  - CPU 100%、GPU 明显没满 → 自博弈进程是瓶颈，先试 `--ring 48`，再把 `--actors` 调到 10。
- 每次只改一个参数，跑 10 分钟比较 samples/s。

（本包已把自博弈进程提速约 2.3 倍；30,772 个局面上的牌型枚举、特征和分词与基线逐位一致，有测试 `tests\test_fast_paths.py` 保证。推理端仍保留 512 个合法动作上限，用于控制 Botzone 内存。）

### 一键：从 real-v2 长时自博弈 + duplicate 评测（推荐）

`scripts/selfplay_eval_5080_wsl.sh` 把“训练 → 冻结候选 → 评测 → 晋级”串成一条命令：

```bash
bash scripts/selfplay_eval_5080_wsl.sh                 # 默认训练 24 小时，然后评测
OXBOT_HOURS=48 bash scripts/selfplay_eval_5080_wsl.sh  # 训练 48 小时
bash scripts/selfplay_eval_5080_wsl.sh eval            # 冻结 latest.pt 并评测对应权重
# 冒烟测试（几分钟）：
OXBOT_HOURS=0.1 OXBOT_DEALS=20 OXBOT_JUDGE_GAMES=8 OXBOT_OUT=ckpts/dmc-smoke \
    bash scripts/selfplay_eval_5080_wsl.sh
```

Windows PowerShell：`.\competition\scripts\selfplay_eval_5080.ps1 -Hours 24`。默认独立后台运行 `all`（训练及完整验收），
启动后入口命令会返回；前台调试需显式加 `-Foreground`。不要在聊天的临时终端里直接启动长时 WSL 训练。
Windows 冒烟：`.\competition\scripts\selfplay_eval_5080.ps1 -Hours 0.1 -Deals 20 -JudgeGames 8 -Out ckpts/dmc-smoke`。

1. **训练**：用 `ckpts/real-v2/best.pt` 做 `--warm-start`，跑 `train_fast` 的 DMC 自博弈，`--ladder-frac 0.5`。
   输出到 `ckpts/dmc-realv2`。再次运行会从 `latest.pt` 续训，并扣除检查点累计训练时长，补足原来的总预算。
2. **评测**：冻结 `latest.pt` 为 `candidate.pt`，从该检查点导出对应的 `candidate.npz`，并冻结基线及已有冠军；
   然后用 `tools/duplicate_eval.py` 多进程跑 1,000 副同牌换座（2,000 局）。
   对手依次是 real-v2 BC、规则机器人，以及已存在的冠军。统计单位是“副牌”，报告配对 bootstrap 95% 置信区间。
   评测 seed 固定，保证不同候选面对同一套牌。
3. **合法性**：用官方裁判跑 200 局随机混合场景，并加 `--require-model`；报告必须完整记录指定局数且零错误。
4. **晋级只看一条规则**：至少 1,000 副同牌换座，对 real-v2 和对现任冠军的置信区间都完全大于 0，且裁判 0 错误，才写入
   `ckpts/dmc-realv2/champion/`。首次没有冠军时以 real-v2 为基线，并在报告中明确记录。
   冒烟或 32 局级别的结果只验证流程，不会晋级。

训练失败会停止后续评测，避免误用陈旧权重；同一输出目录由进程锁保护。候选须包含实际 DMC 优化器更新，
权重及报告身份必须一致。`tools/selfplay_verdict.py` 集中检查晋级门槛，缺失或不完整报告会使流程失败。

### 后台运行、故障记录与通知

Windows 入口通过本机 WMI 服务启动隐藏监督进程，使训练不依赖 Codex 终端的生命周期。监督进程只在本轮运行期间
请求阻止系统自动睡眠，结束后释放；不会更改系统电源方案，也不会在 CUDA 故障后自动重试。

```powershell
.\competition\scripts\selfplay_eval_5080.ps1 -Mode status
```

每次启动都会保存 `launch-<run_id>.json` 和 `logs/<run_id>.*`，分别保留训练标准输出、错误输出和监督日志。
`supervisor_status.json` 记录监督进程及子进程身份，每 15 秒刷新；`last_pipeline.json` 记录训练、各对手评测、
官方裁判和最终验收阶段。训练器另写 `training_progress.json`，区分响应时间与样本/优化器实际推进时间。
周期检查点的 `stop_reason=running` 不代表训练完成；必须核对成功退出、累计预算和完整正式报告。

监督进程发现结束或失败后写 `notification-needed.json` 并尝试显示一次 Windows 桌面提示；是否显示仍取决于系统
通知设置。在本聊天配置的定时检查还会识别监督进程消失、心跳过期和训练长时间无进展，只在异常或全部验收完成时通知。
本地聊天定时检查需要电脑开机且 Codex 应用运行；应用关闭期间，独立监督进程和落盘记录仍保留。

### CUDA 长跑稳定性

在 RTX 5080/WSL2 的长时间自博弈中，Windows NVIDIA 驱动曾记录 `nvlddmkm` 的 `GPUID: 100` 错误，
随后 CUDA 上下文失效；Python 报错位置可能只是异步错误被发现的位置。脚本默认开启保守推理模式：推理进程禁用
BF16 flash/memory-efficient SDPA，改用 math SDPA + FP32；学习器仍保持原来的 AMP 和训练参数。输入在送入 GPU
前会检查 token 范围、特征形状和有限值，训练失败时不会被二次保存异常掩盖。短时诊断可额外设置
`CUDA_LAUNCH_BLOCKING=1`；只有在确认驱动稳定且需要对照时才设置 `OXBOT_SAFE_CUDA=0`，不建议长跑关闭。
为避免 math SDPA 与 learner 争用 16 GiB 显存，安全模式会把推理请求按每次最多 8 条分块；普通路径不改变批量策略。

报告都在 `ckpts/dmc-realv2/eval/<时间戳>/`（`vs_*.json`、`judge.json`、`summary.json`）。
晋级后，按 [C++ 迁移说明](../../docs/fabledan_cpp.md) 导出 FBDN，再运行 `check_submission.py`。
### C++ 上线门槛：本机 5080 对战代替 BotZone 天梯

主推 C++ 版本，强度验收全部在本机完成，不需要在 BotZone 打 500 局。
冠军晋级后，流水线会自动运行 `scripts/cpp_release_5080_wsl.sh`；也可以单独运行：

```bash
bash scripts/cpp_release_5080_wsl.sh ckpts/dmc-realv2/champion/champion.npz
```

1. 构建 `bin/oxbot`（含 C++ 测试），把候选导出为 FBDN 权重，并做 Python/C++ 数值对齐检查。
2. 用 `tools/cpp_duel.py` 在官方裁判下，让候选 C++ bot 对线上 cf8（`OXBOT_CF8_MODEL`）打
   500 副同牌换座。双方都以长时运行进程参加，协议与 BotZone 相同。如果已有上一版 C++ 发布，也要对它打一遍。
3. **发布条件**：对 cf8 和上一版的 95% 区间都完全大于 0，裁判 0 错误，模型全程出牌，并且本机 p99 每步用时低于 0.5 秒。
   满足后生成 `dist/release/oxbot-fabledan-<sha8>-cpp-fp32.cpp` 和同名 `.fbd`，`dist/release/current.json` 记录当前版本。
   手动上传 BotZone 只需要做一次，用来确认平台能编译、能加载模型。

下一轮训练从冠军继续，新候选必须打赢现任冠军：

```bash
OXBOT_OUT=ckpts/dmc-r2 OXBOT_WARM_START=ckpts/dmc-realv2/champion/champion.pt \
    OXBOT_CHAMPION_DIR=ckpts/dmc-realv2/champion bash scripts/selfplay_eval_5080_wsl.sh
```

C++ 推理已优化：权重转置后做向量化计算，每个注意力分数只算一次，候选动作批量计算，长时运行时缓存历史的 key/value。
输出与原实现逐位一致。在模拟 G++ 7.2 -O2 的条件下，512 token、120 个候选的单步用时从 1.7 秒降到约 0.3 秒；
实际对局中有缓存，中位数约 10 毫秒。

## 2. 看哪些数

- `eval vs rule: xx%`：对内置规则机器人的胜率。第一阶段目标 **≥ 95%**（日志会打印"里程碑"），之后会饱和，不再有参考价值。
- `vs snapshot(-500 cyc): xx%`：对 500 个周期前的自己。**持续 > 53% 说明还在变强**；长期 ≤ 50–53% 说明进入平台期，该换招（见第 6 节）。
- 括号里的 `avg` 是每局平均得分差（含 3/2/1），比胜率更接近天梯的计分方式。

## 3. 第一次上传（vs rule ≥ 95% 之后）

1. 跑检查并打包：
   ```
   scripts\check_before_upload.bat
   ```
   无参数时检查当前真题模型 `ckpts\real-v2\best.npz`；也可以显式传入其他兼容的 `.npz`。
   依次做：单元测试 → 训练引擎与官方裁判的进贡规则一致性 → **官方裁判 200 局合法性扫描（用真模型推理）** → 再用"按座位排列"的 history 格式跑 40 局 → 生成代码 zip → 在临时 `data\` 用户存储目录中以真实 zip 进程分别跑传统和长时模式。任何一步失败都不要上传。
2. 产物：
   - `dist\OXbot-real-v2.zip` —— 作为 Bot **源码**上传，编译器选 **Python 3.6.5**；
   - `dist\fabledan_w_XXXXXXXX.npz` —— 上传到**用户存储空间**，**文件名保持不变**（zip 里记着要加载哪个文件，所以多个版本的权重可以并存，旧版本 Bot 不受影响）。
3. Botzone 设置：
   - 建一个新 Bot（例如 `OXbot-F`），游戏 GuanDan；现有 C++ 版 OXbot 先保留，等新版本在天梯上明显更强再替换；
   - **不勾**"使用简单交互"（用 JSON）；"长时运行"勾不勾都可以（本 Bot 每回合输出后自行退出）。
4. 先在游戏桌开一局 4 个座位都是自己的对局，打开回放，看首回合 debug：
   - `model=transformer w=data/fabledan_w_XXXXXXXX.npz` 才说明权重加载成功；
   - 如果是 `model=rule w=missing(...)`，说明存储空间里的文件名不对。
5. 确认无误后加入天梯。之后每 1–2 天：`scripts\duel.bat 新.npz 旧.npz` 比一比，新的明显更强就重复第 3 步，上传新版本。

## 4. 确认天梯的实际设定（重要，决定训练分布）

从任意一局天梯对局的日志里，看我方 Bot 收到的**第一条请求**（deal 阶段）的 `global`：

- 若 `level` 总是 `"2"` 且 `tribute` 总是 `0`：说明天梯只打"打 2、不进贡"。把 `OXBOT_LADDER_FRAC` 改成 **0.85**，然后续训。保留 15% 的随机局面，用来保持泛化。
- 若出现别的级牌或进贡：保持 0.5，或按实际比例调整。

## 5. 时间预期

FableDan 作者的经验是：训练 1–2 天后，对规则机器人的胜率达到 95%，就可以先上传占一个天梯位置，然后持续训练、隔一两天换一次版本。本包的自博弈速度比原版快约 2.3 倍，但单机能跑的自博弈局数仍然是牌力上限。如果需要更快，最直接的办法是租多核 CPU + GPU 的云机器，跑同一个脚本。

## 6. 进入平台期后的升级顺序（一次只改一处）

1. 出牌时搜索：在 Bot 端用模型做前瞻（不影响训练，单独上传做 A/B 对比）；
2. 对手池：自博弈时混入旧版本，防止"只会打自己"；
3. TD-λ 混合目标：降低训练噪声；
4. 加大模型（`--n-blocks 6` 起步），之后视推理耗时决定是否需要蒸馏。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available()` 为 False / `no kernel image` | 装成了 CPU 版或旧 CUDA 版 PyTorch：在 .venv 里 `pip uninstall torch`，然后重新 `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| 训练窗口一闪而过 | 在 cmd 里手动运行 bat，查看报错信息 |
| 多进程启动很慢、samples/s 偏低 | 检查 Defender 排除项；确认不在 OneDrive 目录里 |
| 上传后 debug 显示 `model=rule` | 存储空间里的文件名必须与 `dist\` 下的 npz 文件名完全一致 |
| 首回合超时 | 检查是否误把权重打进 zip（不要用 `--embed-weights`），并以 `tools\check_submission.py` 的 p99 结果为本机参考 |
