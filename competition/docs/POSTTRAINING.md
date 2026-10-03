# OXbot 后训练：团队回报、换牌复盘与对战验收

当前线上 `real-v2` 来自行为模仿（BC）。动作模仿准确率不能证明牌力，模型也不会
自动知道某次过牌、拆牌或压住队友损害了团队结果。新增流程把这些反馈接到真实梯度
更新上。默认后训练已升级为224维花色/出后结构特征（feature version 2），保留四层
网络与48 token；Python、C++和导出同步支持。旧80维权重仍可加载，使用
`--feature-dim 80` 可显式运行旧特征路线。详细布局与验证见 [组牌特征v2](STRUCTURE_FEATURES.md)。

## 使用

在 `D:\coding\OXbot` 的 PowerShell 中运行，使用已配置的 WSL / RTX 5080：

```powershell
# 短流程：真实权重、真实梯度更新，仅验工程链路。
.\competition\scripts\posttrain_5080.ps1 -Smoke -Out ckpts/posttrain-smoke-new

# 正式有限流程：默认10个DMC周期、16局复盘、4轮纠错训练、800局成对评测。
# 另有200局异策略搭档诊断。输出目录必须全新，原BC权重不变。
.\competition\scripts\posttrain_5080.ps1 -Out ckpts/posttrain-v1
```

WSL 中等价入口：`bash competition/scripts/posttrain_5080_wsl.sh --out ckpts/posttrain-v1`。
脚本切换到 `competition/` 后调用 `posttrain.py`；输出路径相对该目录。
`python posttrain.py --help` 可查看规模参数。DMC 阶段默认最多运行一小时；复盘和评测
按有限局数执行，不计入这个一小时时限。停止或报错会记录到 `pipeline.json`。

这些默认规模用于第一轮实验，不是声称足够达到竞赛水平。完整流程结束也不自动
替换、打包或上传线上模型。

## 实际训练了什么

| 环节 | 实现与作用 | 不作出的推断 |
|---|---|---|
| BC 初始化 | `--warm-start` 保留编码器，80→224只为首层新增144个零输入列；重置优化器和计数，BC 最后一个 Q 线性层乘正数0.05 | 迁移保持相同动作输入的旧分数排序；缩放不是胜率或Q值校准 |
| 团队回报自我对弈 | 同队两人共享终局团队分差，除以3归一化；学习争头游、双下及减少失分 | 不按个人出牌数、个人名次奖励，不固定惩罚过牌 |
| 全候选探索 | 15%探索，`--top-k 1` 在当前实现中表示从全部合法候选采样 | 不是仅在原模型偏爱的前10手里循环 |
| 换牌复盘 | 复制真实模拟局面，对实际动作、pass和不同组合候选分别续打至终局 | 在指定续打策略下的条件估计，不等于证明最优动作 |
| 纠错训练 | 回归平均团队分差；只有各续打场景不反转方向、平均差至少1个原始分时才建立偏好；用冻结模型约束过度漂移 | 不是给所有败局动作贴“错误”标签 |
| 组牌采样 | 单独抽取领出局面，覆盖不同组合类型、长度和花色代表；编码出后面库存及组合潜力 | 候选是有预算的代表集合，不是穷举所有物理子集 |
| 配合采样 | 采样队友控牌、敌人剩1–2张等局面；续打包括原BC搭档 | 不硬编码“永不盖队友”或“有大牌必接” |

奖励为 `本队官方积分 - 对方官方积分`，范围 `[-3,3]`，训练目标再除以3。
它与官方“赢家获1/2/3分、输家0分”的原始积分不同；评测同时保留胜率、双方原始
积分和团队分差，不将两个队友的相同奖励加算两次。

DMC阶段仍是当前模型全桌自我对弈。混合策略出现在后续复盘：自身续打、冻结BC对手、
规则对手、冻结BC搭档四种场景；没有把DMC虚称为已实现完整历史对手联盟。

复盘的模拟器可持有完整发牌，但所有出牌策略只接收自己的手牌和公开历史。
保存的训练输入也只有公开token和本人的版本化候选特征，不存其他玩家手牌。
标签使用这个模拟发牌的终局回报，因此有条件估计偏差；同一局面及其所有分支只能
属于同一数据划分，按整局分开训练/验证，最终牌力评测再用独立种子。

## 验收与产物

每个运行目录包含：

- `dmc/latest.pt`：可继续DMC的权重、Adam状态、经验FIFO、学习器随机状态及来源SHA。
- `reviews.json`：换牌分支、各续打场景团队回报、估计损失、候选覆盖数、编码/规则SHA。
- `credit/best.pt` 与 `best.npz`：按复盘验证损失选出的候选，仍需对战检验。
- `credit/training.json`：实际梯度训练与偏好样本计数，不把低损失当作牌力证明。
- `evaluation.json`：冻结BC/规则对手同牌换座结果，以及单模型搭配BC队友的独立诊断。
- `pipeline.json`：完整流程状态、参数、文件SHA和晋级判断。

正式实力门槛至少每个对手200个同牌换座对，两个对手合计800局。置信区间以
“一对发牌”整体重采样；对BC的团队分差95%区间下界必须大于0，对规则对手不能
跌破0分差。小样本不会通过门槛。异策略搭档测试按同牌四座位块统计，独立展示，
不拿自我克隆队友的成绩替代协同证据。

`promotion_gate.passed` 仅表示本地实力门槛。`release_eligible` 始终为false，候选
还需要 [C++导出、数值对齐、官方裁判和真实进程时限验收](../../docs/fabledan_cpp.md)。
重复查看相同种子的评测会过拟合；扩大规模或再迭代时换新的最终验收种子，不反复
挑同一小测试集的偶然赢家。

## 继续训练及独立运行

以下命令从 `competition/` 执行。DMC恢复时须保持原来的buffer、belief配置，必要时
显式传入；actor重新开牌，恢复的是学习器，不承诺跨进程逐位复现。

```bash
# 从BC重新开始DMC，明确不沿用BC优化器。
python -m fabledan.train_fast --warm-start ckpts/real-v2/best.pt \
  --feature-dim 224 --out ckpts/posttrain-next/dmc --cycles 10 --lr 0.00002 \
  --eps 0.15 --top-k 1 --ladder-frac 0.8 --belief-weight 0 --buffer 32768

# 接回上一轮纠错候选：已是团队回报尺度，不再次压缩Q输出。
python -m fabledan.train_fast --warm-start ckpts/posttrain-v1/credit/best.pt \
  --warm-start-q-scale 1 --out ckpts/posttrain-v2/dmc --cycles 10

# 在相同DMC输出目录续训；cycles是累计目标，而非追加周期数。
python -m fabledan.train_fast --resume ckpts/posttrain-v1/dmc/latest.pt \
  --out ckpts/posttrain-v1/dmc --cycles 20 --belief-weight 0 --buffer 32768

python -m fabledan.posttrain_credit collect \
  --candidate ckpts/posttrain-v1/dmc/latest.pt --anchor ckpts/real-v2/best.pt \
  --out ckpts/posttrain-next/reviews.json --games 32 --positions 8 --device cuda:0

python -m fabledan.posttrain_credit fit --checkpoint ckpts/posttrain-v1/dmc/latest.pt \
  --reviews ckpts/posttrain-next/reviews.json --out ckpts/posttrain-next/credit --device cuda:0

python -m fabledan.posttrain_eval --candidate ckpts/posttrain-next/credit/best.pt \
  --anchor ckpts/real-v2/best.pt --pairs 200 --mixed-pairs 50 --seed 93001 \
  --backend torch --device cuda:0 --report ckpts/posttrain-next/evaluation.json
```

## 当前边界

这次增加的是可执行后训练闭环，不是已经获得更强的线上模型。真实权重短流程已经
贯通；短流程产物没有资格上传。长期训练仍需根据正式评测决定保留或淘汰。

224维路线已解决旧80维无法区分花色及余牌结构的表示盲点；同一54面不同副本仍是
等价牌，不强行区分。组合潜力是每种窗口/牌型的独立可行性统计，不等同于全手最少
几次出完的最优分解。完整历史对手池、隐牌重采样反事实和真实BotZone日志自动导入
仍属于后续扩展，当前复盘源为自生成模拟对局。
