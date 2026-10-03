# OXbot 竞赛版（基于 FableDan）—— 9800X3D + RTX 5080 训练与上线手册

> 竞赛版仅用于 Botzone 等非商业用途（FableDan 许可：Apache 2.0 + 非商业限制）。商业版不得使用本仓库代码与训练出的权重，除非取得作者书面许可。

## 0. 一次性准备（约 15 分钟）

1. 安装 Python 3.12（勾选 "Add to PATH" 与 "py launcher"），显卡驱动更新到最新。
2. 解压本包到一个**不在 OneDrive 同步目录**里的路径，例如 `D:\coding\OXbot-comp`。
3. 双击或在 cmd 里运行 `scripts\setup_windows.bat`：
   - 建虚拟环境 `.venv`，装 numpy 和 **CUDA 12.8 版 PyTorch**（RTX 50 系必须 cu128 或更新，装成 CPU 版或旧 CUDA 版会报 `no kernel image` 或 `cuda False`）；
   - 自检 GPU、bf16 矩阵乘、跑测试。最后一行出现 `SETUP OK` 即可。
4. 建议：Windows 电源计划设为"高性能/卓越性能"，关闭自动睡眠；在 Windows 安全中心把本目录加入"排除项"（Defender 实时扫描会拖慢检查点写入和多进程启动）。
5. 把官方裁判源码（你们 SHA 为 `910cba94…` 的那份）放到 `judge\judge_official.py`。没有它时，检查脚本会退回到附件里的修正版裁判 `judge\judge_fixed.py` 并打印警告。

## 1. 开训

```
scripts\train_5080.bat
```

- 中断后（关机、报错、手动 Ctrl+C）**直接再运行一次**，会从 `ckpts\run1\latest.pt` 续训（回放池会重新攒，几分钟就满）。
- 关键参数（都在 bat 里）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--actors 12` | 12 | 自博弈进程数。9800X3D 是 8 核 16 线程，另有推理进程和训练进程要占 CPU |
| `--ring 32` | 32 | 每个进程同时跑的牌局数；越大 GPU 批次越大，往返越少 |
| `--ladder-frac 0.5` | 0.5 | 一半牌局按 Botzone 默认设定（打 2、不进贡、0 号先出），一半随机级牌与进贡 |
| `--eval-games 100` | 100 | 每次评测局数（同牌换座，成对出现） |
| `--export-cycles 50` | 50 | 每 50 个周期导出一次 `latest.npz`（上传用的 numpy 权重） |

### 怎么调到最快
日志每个周期打印 `... samples/s`，推理进程每 30 秒打印 `[infer] N decisions/s`。目标是 **samples/s 最大**：

- 任务管理器 → 性能 → GPU → 把一个小图切到 **Cuda**：
  - GPU 长期接近 100%、CPU 没满 → GPU 是瓶颈，把 `--actors` 降到 10；
  - CPU 100%、GPU 明显没满 → 自博弈进程是瓶颈，试 `--ring 48`，或 `--actors 14`。
- 每次只改一个参数，跑 10 分钟比较 samples/s。

（本包已把自博弈进程提速约 2.3 倍：同一局面下的出牌结果与原版逐位一致，有测试 `tests\test_fast_paths.py` 保证。）

## 2. 看哪些数

- `eval vs rule: xx%`：对内置规则机器人的胜率。第一阶段目标 **≥ 95%**（日志会打印"里程碑"），之后会饱和，不再有参考价值。
- `vs snapshot(-500 cyc): xx%`：对 500 个周期前的自己。**持续 > 53% 说明还在变强**；长期 ≤ 50–53% 说明进入平台期，该换招（见第 6 节）。
- 括号里的 `avg` 是每局平均得分差（含 3/2/1），比胜率更接近天梯的计分方式。

## 3. 第一次上传（vs rule ≥ 95% 之后）

1. 跑检查并打包：
   ```
   scripts\check_before_upload.bat ckpts\run1\latest.npz
   ```
   依次做：单元测试 → 训练引擎与官方裁判的进贡规则一致性 → **官方裁判 200 局合法性扫描（用真模型推理）** → 再用"按座位排列"的 history 格式跑 40 局 → 打包。任何一步失败都不要上传。
2. 产物：
   - `dist\oxbot_fable.zip` —— 作为 Bot **源码**上传，编译器选 **Python 3.6.5**；
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

- 若 `level` 总是 `"2"` 且 `tribute` 总是 `0`：说明天梯只打"打 2、不进贡"。把 `train_5080.bat` 里的 `--ladder-frac` 改成 **0.85**，然后续训。保留 15% 的随机局面，用来保持泛化。
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
| 首回合超时 | 首回合时限翻倍，权重加载通常 <1 秒；若仍超时，检查是否误把权重打进 zip（不要用 `--embed-weights`） |
