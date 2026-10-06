# 实施进度与证据

记录日期：2026-10-02。正式模型尚未上线；优先级只覆盖 BotZone 首发。

| 项目 | 已验证结果 | 证据/限制 |
|---|---|---|
| 本机训练 | RTX 5080、sm_120、PyTorch 2.11.0+cu128；前向/反向/优化器更新通过 | `reports/training_environment.json`；这是合成环境验证 |
| 规则差分 | 9,997 随机 claim、30,386 生成动作、6,104 大小比较；小手牌配子穷举通过 | `reports/rules_subagent.json` 是附件 oracle 的历史证据；当前 C++ core 已用 `RuleVariant::BotzoneCompat` 实现官方还牌边界，平台仍未验证 |
| 状态和完整对局 | 10,000 局、868,093 次决策；13 等级、0/1/2 贡，手牌与四家余牌数全部对账通过 | `reports/state_harness10000.json`；规则策略 |
| JSON 进程 | stdin 保持打开时响应并退出；正常阶段及畸形输入检查通过 | `tests/test_protocol_process.py` |
| 特征编码 | 52 组观测、5,923 动作，Python/C++ 最大误差 0 | `reports/feature_parity.json` |
| 网络前向 | 真实训练权重最大分数误差约 2.9e-6；128 个真实验证观测、21,753 候选全部 argmax 对齐 | `reports/trained_network_parity.json`、`reports/trained-observation-parity.json` |
| 模型接入 | 嵌入模型 26 局、4,165 Play 全由模型执行，零降级并通过附件裁判 | `reports/bc_v1_embedded_oracle26.json`；不是牌力证明 |
| 单文件构建 | 实际嵌入权重后 2,383,202 字节，本地 C++17 编译及 stdin-open 测试通过 | `dist/oxbot-bc-v1.cpp`；目标平台 G++ 7.2.0 尚未测试 |
| 数据清洗 | 138 完整数据文件、1,300 副牌、132,638 个有效 P；历史/累计出牌全量核对通过 | `docs/data_audit.md`；v3 manifest 固定 |
| BC 训练 | 8 轮、10,232 优化器步骤，156 秒，峰值显存约 430 MiB；训练 81,827、验证 24,224、测试 21,662 | `models/bc-v1/training.json`；epoch8 按验证 NLL 选出 |
| 离线示范匹配 | 验证 63.06%，测试 63.63%；测试仅执行一次 | `reports/bc-v1-test.json`；不是对战胜率 |
| 基础牌力 | 首批 32 副同牌互换、64 盘均负于规则基线，候选 v1 不满足首发要求 | 已完成首轮诊断；不得用上述离线匹配率掩盖此结果 |
| 候选组 A/B | 冻结 v1 权重、同一 32 组/64 盘换座：raw 与 group-logmeanexp 均 0/64 胜规则；模型合法率、必需模型降级和本地预算超限均为 0 | `reports/selection_ab_32x2.json`；仅附件裁判和本地 guardrail，不是牌力结论 |

已完成第一轮 v1 行为诊断，并继续做了候选监督目标实验。独立在线诊断 4 副牌、134 次模型决策中有 80 pass、53 single、仅一次 5 张牌；11 次领出全部为 single。134 次决策的线上 Bot、独立 C++ 推理与 Python checkpoint 完全一致，特征最大误差为 0，见 `reports/strength_bc_v1_diagnostic.json`。后续 rank005 结果已记录在本文末尾；未根据 held-out test 调参。

诊断更正：早期 `tools/analyze_bc_behavior.py` 漏载 checkpoint，曾报告的“验证集炸弹 4,148/对子14”等统计来自随机初始化，全部作废；已要求修复加载并核验权重 SHA。上述真实对战、训练、parity 和 held-out test 使用独立正确加载工具，结论不受该统计脚本错误影响。

候选组实验：新增 `group-logmeanexp-v1` 选择器，但 raw 仍是默认且未改动 v1 权重。组键与 Python 参考实现保持一致（牌型/长度/牌点/副牌点/实际牌面秩计数/当前级红桃癞子数；同花顺保留完整花色面计数），先按组内 raw 分数 logmeanexp，再取选中组内 raw 最高成员。`tools/diagnose_strength.py` 的 68 次在线 parity 为 C++/Python 68/68；嵌入单文件 group 策略 2 局也通过附件裁判。策略来源可由模型 header manifest 或 `--strategy raw|raw-pass-bias|group-logmeanexp` 覆盖，debug 带 `selection_strategy=*-v1`。

固定 20261020 seed 的 32 组 duplicate A/B（raw/group 各 64 盘）：raw 配对模型点差 `-186`，group 配对模型点差 `-185`；两者均 0/64 胜规则，差异不足以支持替换 raw。报告只作工程/行为 guardrail，未读取 test、未作强度或置信结论。下一步仍需改进训练目标并重新评测，不能把 group 实验写成牌力提升。

v1 的 10,000 局模型压力测试已停止，实际完成 525 局、88,488 次决策；报告状态为 `interrupted`，不能当作完成验收。新候选需重新完整验证。v1 源码/权重保留用于复现，打包 manifest 的 `release_eligible` 仍为 false。

最新训练目标实验（均只读取 `bc-v2-full` 的 train/validation，未读取 held-out test）：

- RTX 5080 FP32 全候选 `multi_weight=2`：checkpoint `73274b9f…076e`，验证 NLL `1.20968`、choice `45.13%`。size 4/5/6 同大小召回升至 `16.81%/38.23%/40.32%`，但 exact score margin 变差；16 组同牌换座 raw 配对为 `0` 正、`2` 平、`14` 负，点差 `-56`，不首发。
- 同一 multi2 权重改用 `group-logmeanexp`：验证 face accuracy `66.70%`，但 16 组配对为 `0` 正、`1` 平、`15` 负，点差 `-48`，不设默认策略。
- 全候选 FP32 + pass-mass auxiliary `beta=0.1`：checkpoint `df5578b7…1144`，验证 NLL `1.18652`、raw `63.52%`、choice `45.90%`；follow 预测 pass 误差由 `+11.15pp` 降至 `+9.55pp`，但 129–256 候选和 size≥4 召回下降，未通过筛选门槛，未做实战。

当前保留的训练 anchor 是 `models/bc-v2-full-fp32`，线上默认仍不切换；下一步应在不读取 test 的前提下重新设计具体动作排序/牌型价值目标，而不是继续叠加未通过门槛的校准项。

2026-10-02 outcome-weighted BC 筛选：使用开发集 sidecar
`reports/action_utility_targets_v2.json`（仅 train/validation）训练了
alpha=0.25 和 alpha=0.50。alpha=0.25 的验证 NLL 为 1.179704，略差于
anchor 的 1.172835；其 C++ parity 全部通过，但固定 seed=20261020 的
16 副牌同牌换座 32 盘为 0 胜、配对总分差 -96，明确不晋级。alpha=0.50
最佳验证 NLL 为 1.203901，验证集已明显回退，未导出、未做强度评测。完整
指标、SHA 和限制见 `reports/action-utility-alpha-screening.md`；anchor
仍未改变，也没有上传 BotZone。

hard-negative ranking 实验（`rank005`，仅 train/validation）已完成：在 marginal NLL 外，对同牌型/同动作长度的负候选加入确定性 pairwise softplus（每行最多 32 个，权重 `lambda=0.05`）。RTX 5080 训练 8 轮，checkpoint `models/bc-v2-full-fp32-rank005/best.pt`，SHA256 `0a27a6e1d9bdcc16833580738e67cc470a68223f21e5b3c6679cdd656e876139`；导出 `models/oxbot-bc-v2-full-fp32-rank005.bin`，文件 SHA256 `70255109db5d838240a72e35c62a2026c86145cad7946d48ac5d6fd06ca9cc75`，payload SHA256 `f117d62ca56cfc7b0eedea6ec96b4647ef28c3e7237a8187290197e75038c489`。最终验证 NLL `1.1702345167`、choice `47.0372%`，hard-pair accuracy `74.2029%`（83,649 对/9,914 行）；这些指标不代表对局牌力。固定 seed `20261020` 的 16 组互换队伍离线 probe 通过合法性和模型使用检查，但模型 32 盘胜 `1`、配对总点差 `-89`，整组 bootstrap 单盘点差 `-2.78125`（95% CI `[-3.0,-2.5]`），故候选明确不晋级；`rank010` 跳过，线上仍保留 `bc-v2-full-fp32` anchor。报告：`reports/rank005_probe_16_summary.json`、`reports/rank005_probe_team0_16.json`、`reports/rank005_probe_team1_16.json`。
rank005 同时生成了本地候选包 `dist/oxbot-bc-v2-full-fp32-rank005.cpp`，2,400,030 bytes，SHA256 `82cfe55292592b3f257821dc6fba41a1c96d5cef658a5c840a6225081ad72083`；WSL g++13.3 C++17/O2 编译及 6 项 stdin-open 普通 JSON 进程检查通过。该包的 manifest 仍为 `release_eligible=false`，不能据此替代 BotZone G++7.2/私有对局验收。

发布诊断标识已修正：新模型头可携带受限字符集的 `model_version`，旧/冻结二进制缺省仍显示 `oxbot-model-v1`；打包器可用 `--candidate-version` 设置包级标识，避免未来模型误报 v1。新增模型选择/未来版本 fail-safe 测试，并重新构建通过 C++17、本地协议、特征和网络 parity。当前源码一致的 rank005 候选包为 `dist/oxbot-bc-v2-full-fp32-rank005-current.cpp`（2,403,252 bytes，SHA256 `f6978934d60ce0b3ff67b2ae87942e7a8a24486c12896e96cd2828f5f37e740b`）；它仍是未晋级候选，`release_eligible=false`。

不得把“生成模型文件”当成发布，也不得把规则策略合法率当成模型强度。模型应分别报告实际决策次数、降级、非法/超时、运行时间和相对基线得分。

## 线上待完成项

附件 `裁判代码-修正版.py` 保持原样，SHA256 为 `FA63589D3F69CE9127093CEC417F1635D8CC17205D80E70F03E44BC6809FC622`。该附件只作为历史离线 oracle；官方源码已取得并完成初步 diff，当前 C++ core 已按官方 BotZone 还牌语义实现，但线上平台兼容性仍未完成。

当官方差异核验完成后，需做平台私有编译/对局和发布。Windows 整局人机、微信小游戏、iOS/Android 均未启动。

2026-10-02 本机训练条件确认：RTX 5080 可用于本地训练和离线评测；训练数据、checkpoint、目标 sidecar 均不上传 BotZone。新增开发审计 sidecar
`reports/short_horizon_targets_v1.json`（SHA256
`c2c0d256ea2051b90d68357b993836a1a220939fa1c6b28dbd282c2dd93b3854`），由
`train/build_short_horizon_targets.py` 只读取 train/validation，未读取 test、result
或对手私牌。共 110,128 个目标（train 84,993、validation 25,135）；其中可用局部
控牌 credit 的 train/validation 行分别为 39,867/12,462。该 credit 只是下一次公开领出
的局部标签，不是反事实 Q 值；尚未接入模型训练或强度评测。单测
`tests/test_short_horizon_targets.py` 通过（1 OK）。
在已配置依赖的 WSL2/5080 环境中完整回归为 48/48；Windows 系统 Python 未安装
NumPy/PyTorch，不能用它代表训练环境。

本地 BotZone 包审计已完成，见
`reports/botzone_package_audit.md`：`dist/oxbot-bc-v2-full-fp32-rank005-current.cpp`
为 2,403,252 bytes，小于 4 MB；嵌入模型、普通 JSON、stdin 保持打开、模型真实参与
决策和 `/tmp` 启动均已验证。当前 C++ core 已按官方 `BotzoneCompat` 语义构建；既有
包审计是本机 artifact 快照，仍缺官方 G++ 7.2、平台私有对局与 1 秒时限核验，manifest
保持 `release_eligible=false`，未上传。

anchor 单文件包已按官方 C++ 语义重建并审计：
`dist/oxbot-bc-v2-full-fp32-anchor.cpp` 为 2,400,969 bytes，源码 SHA256
`a9ccf9504f653f68ae89a9934aac6fba77240154bcf3a10f9d19237ed66ef827`，嵌入 anchor 模型
`cccb8ecf…f7232`，包级身份 `bc-v2-full-fp32-anchor`。manifest SHA256 为
`eff1768b7449806d12fb07caf73d47f48f7788c76647b2e154d76b8ab72461e9`，本机二进制 SHA256 为
`495bc236543619b0c5d98ba3b3ece5912ec0dcb2d9f7dabfeb8e14b389e1d18c`，manifest 仍为
`release_eligible=false`。WSL g++13.3 的 C++17/Werror、stdin-open 普通 JSON 进程检查和
官方还牌边界 golden（13 个等级、零 mismatch）通过；证据见
`reports/official_return_boundaries.json` 与 `reports/botzone_anchor_package_audit.md`。
本机计时不等价 BotZone CPU 时限，anchor 牌力和官方平台门槛仍未通过。

候选局部 residual 审计 helper 已完成：`train/residual_features.py` 及
`reports/candidate-local-residual-plan.md`。它只从自己的手牌删除候选牌，再调用离线
C++ core 统计“未来轮到自己领出时”的合法选项、炸弹/同花顺/火箭保留等字段；不读取
对手手牌或结果，也不进入 BotZone 包。`tests/test_residual_features.py` 与短视 credit
测试在 WSL 中共 4/4 通过。该 sidecar 尚未接入模型，当前没有候选级因果 Q 标签。

候选 residual 小批审计已完成，见
`reports/candidate-local-residual-audit-v1.md` 与
`reports/candidate-local-residual-audit-v1.json`（SHA256
`fc35ba9d919bf98cbb62edf15d532e8450754414b6332c8ae7ba6dced45277a5`）。固定 seed
在 train/validation 各抽 128 行、每行最多 16 个候选；共检查 1,455 个候选，unknown、
物理删除错误和 C++ validate 错误均为 0。`base_lead_total` 长尾最高 2,893，说明
后续若接入模型必须明确 `log1p`/分位数归一化和新的 feature contract。该审计没有生成
Q 标签，没有接入模型，也没有读取 test。当前 WSL2/5080 完整回归为 49/49。

新增 anchor 的完整本地整局压力：`reports/anchor_model_team0_1000_offline.json`。
单核、固定 seed `20261020`、附件裁判 oracle、1000 局中模型队伍实际使用 anchor，
模型队伍未发生非法动作、模型降级或本地预算超限（规则对手的 33,209 次预期
`rule_fallback` 单独计数）；描述性结果为模型胜 11/1000、总点差 `-2890`。
这不是线上兼容性证明，但确认当前 anchor 仍只能作为研究基线，不能上传。

Candidate continuation sidecar 的最小审计已完成：`reports/candidate-continuation-target-plan.md`
与 `reports/candidate_continuation_targets_audit128.json`。它仅读 train/validation，
`test_used=false`；128 行上限审计得到 train/validation 各 29/16 个有正负 pair 的行，
但输出明确 `audit_only=true`、`trainable=false`，尚未接入训练、未改 C++ 特征契约，也未
生成可发布模型。标签是公开续演兼容性，不是 Q 值或团队收益 credit。

## 2026-10-02 后续工程化训练与筛选

上面的 128 行审计是历史审计记录；随后已完成 candidate continuation sidecar 与
prepared shard 的工程化对齐。`reports/candidate_continuation_targets_v1.json` 的
SHA256 为 `aef35f97fb0de22a6ca82019f4bafc373cfb190354696e9072248fc715d981e0`，只读取
train/validation，`test_used=false`；对齐后 train 13,512 行、validation 4,655 行，
共 18,167 行。`reports/candidate_continuation_alignment.json` 显示两个 split 的候选数
不匹配均为 0；train 有 2 个未进入 prepared shard 的额外 ID，已 fail-closed 过滤。sidecar
来源、ID、候选数量和 provenance 现在由 `train_bc.py` 在启动时校验，训练使用
continuation pairwise loss（lambda=0.05）；`tests/test_continuation_alignment.py` 覆盖
这些对齐检查。sidecar 仍只用于开发训练，不进入 BotZone 发布包。

基于该 sidecar 的 cont005 候选在 RTX 5080 上完成 8 epoch FP32 训练。最佳 epoch=8，
checkpoint 为 `models/bc-v2-full-fp32-cont005/best.pt`，SHA256
`671437b6a272b7549d1c8fc8673a474c6f4e2b7418e14ba42bb9b5aca60050d0`；validation NLL
`1.190878`、raw accuracy `63.416%`、choice accuracy `45.752%`、hard-pair accuracy
`73.812%`。导出文件 `models/oxbot-bc-v2-full-fp32-cont005.bin` 的文件 SHA256 为
`666881c91a149fee097444c6ec9d45bb69c7faa84f04176eb63279e5ccfce007`，payload SHA256 为
`2b03e119b4e2dc12fa08dee850c4724dd073d5e9b98e6e64d328adcf0db67a6a`。网络 parity 最大
误差为 `3.814697e-6`，真实 validation 观测 128/128 argmax 一致；证据见
`reports/cont005_network_parity.json` 与 `reports/cont005_trained_observation_parity.json`。

强度门槛未通过：固定 seed `20261020`、16 组同牌换座的两队配对共 32 局，cont005
模型胜 `0/32`，总点差 `-95`，平均每局 `-2.96875`，bootstrap 95% CI
`[-3.0, -2.90625]`（`reports/cont005_probe_summary.json`）。因此 cont005 明确淘汰，
不替换线上默认模型、不生成发布包，也不上传 BotZone。

当前 anchor 仍为 `models/bc-v2-full-fp32`（SHA256
`cccb8ecf58c88ee51ca4fb231162f90f14521a78d172ed7b74739fbad37f7232`），未作任何替换。
RTX 5080 训练条件已确认可用（PyTorch 2.11.0+cu128、CUDA sm_120、真实梯度更新通过），
官方裁判语义定向测试已在当前 C++ core 和重建单文件包上通过，但官方 G++ 7.2、平台私有对局
和 1 秒时限仍未完成；在这些验收完成前，anchor 也保持未上传状态。

## 2026-10-02 官方裁判取得与差异

已通过 BotZone GuanDan 游戏页的已登录会话取得当前官方裁判源码，游戏 ID 为
`65490c16ec1ab1389702dced`。证据副本为
`reports/official_judge_botzone_2026-10-02.py`；规范化源码 SHA256
`49a7346ddc0a06ebd8922e582d535fbc6fb5418ddea16b023a5f3384fc9b6fd9`。与附件
`裁判代码-修正版.py`（SHA256 `fa63589d3f69ce9127093cec417f1635d8cc17205d80e70f03e44bc6809fc622`）
的逐行报告见 `reports/official_judge_diff_2026-10-02.md`。

差异集中在 `isValidReturn`：官方源码按等级 9/其他等级分别以 8/9 为上界，附件修正版则允许自然
10 并显式排除当前级牌；另有说明注释和文件末尾换行差异。该边界现已落实到 C++
`RuleVariant::BotzoneCompat`（state/protocol/policy 默认走此变体），并由
`tests/rules_test.cpp` 的 level-9、level-10、普通等级边界断言覆盖。此前产生的本地 oracle
回归和模型 probe 仍基于附件，不能改写为官方对局结果；当前 C++ 源码已按官方语义，且
`dist/oxbot-bc-v2-full-fp32-anchor.cpp` 已按该源码重建，并以官方裁判副本完成边界 golden
和 16+16 局同牌换座。重建包、编译和对局证据仍是本机证据，平台 G++ 7.2、1 秒时限和私有
对局仍未验收。

## 2026-10-02 当前 anchor 重建与官方 oracle 回归

- 定向官方还牌 golden：`reports/official_return_boundaries.json`，13 个等级全部通过；C++
  `RuleVariant::BotzoneCompat` 与官方 `isValidReturn` 一致。该检查使用官方源码副本，附件
  修正版仍保持独立，不被覆盖。
- 当前单文件包：`dist/oxbot-bc-v2-full-fp32-anchor.cpp`，2,400,969 bytes，SHA256
  `a9ccf9504f653f68ae89a9934aac6fba77240154bcf3a10f9d19237ed66ef827`；manifest SHA256
  `eff1768b7449806d12fb07caf73d47f48f7788c76647b2e154d76b8ab72461e9`，`release_eligible=false`。
  WSL 本机编译二进制 SHA256 为 `495bc236543619b0c5d98ba3b3ece5912ec0dcb2d9f7dabfeb8e14b389e1d18c`。
- 当前包按官方裁判副本完成 16+16 局同牌换座：模型 1/32 胜、配对总点差 `-92`，模型实际
  出牌决策 1,108，要求模型 fallback 0，非法动作/预算超限 0，最大探针 RSS 4,268 KiB；
  汇总 `reports/official_anchor_package_16x2_summary.json`。这是牌力未达标证据，不是平台发布许可。

## 2026-10-02 5080 候选复核（仍未发布）

RTX 5080 本机训练条件保持可用：PyTorch `2.11.0+cu128`、CUDA sm_120，训练脚本只读取
`bc-v2-full` 的 train/validation 分片，训练 provenance 明确记录 `test_used=false`。
新增 `train/train_factorized.py --pass-mass-beta`，并完成 beta `0.25`、`0.50` 两个 8 epoch
候选；二者均完成 Python/C++ 特征与网络 parity、单文件导出和官方裁判 16+16 换座，结果仍为
负点差，未替换 anchor、未创建或上传 Bot：

- `factorized005_pass025`：模型 1/32 胜，总点差 `-88`（`-2.75/局`），payload
  `a0e27a2a83086e9afc20e5fc78ee0fe1f1e699331b9b6fbb9029036662d53800`；详见
  `reports/factorized005_pass025_embedded_16x2_summary.json`。
- `factorized005_pass050`：模型 2/32 胜，总点差 `-84`（`-2.625/局`），payload
  `14600990cf8bf0b86abd0d03e9c8a700a32212c44d9dcc895cbf37fcb430dd8b`；详见
  `reports/factorized005_pass050_embedded_16x2_summary.json`。

重新以当前官方裁判副本复核历史 `multi2` 候选：seed `20261020` 为 6/32 胜、总点差 `-56`
（`-1.75/局`），seed `20261120` 为 8/32 胜、总点差 `-41`（`-1.28125/局`）；合并仍低于
零点差，故它是当前“研究中最强”候选而不是发布 anchor。其嵌入 C++17 包已通过本机 GCC 7.2
编译和 1 局官方 oracle 端到端回放，证据为 `dist/oxbot-bc-v2-full-fp32-multi2.cpp`、
`reports/multi2_embedded_official_16x2_summary.json` 与
`reports/multi2_gcc72_embedded_onegame_official.json`。

（历史策略，早于后文私有 version 1 创建）发布门槛要求模型配对点差稳定达到非负，且完成
BotZone 私有编译/1 秒/RSS/实战验收；当时门槛通过前不创建 Bot、不上传源码或模型，也不推进
Windows 人机、微信小程序和移动端。后续在用户明确授权下已创建私有 version 1，但仍不公开、
不加入天梯、不替换 anchor；平台验收未完成前，Windows 人机和商业化端继续暂停。

补充筛选：`multi_weight=3.0`（seed `20261001`）验证最佳 NLL `1.264967`、choice `44.362%`，
劣于 `multi2`，不做整局裁判；`multi_weight=2.0` 更换 seed `20261120` 的最佳 NLL
`1.229805`、choice `39.416%`，同样淘汰。两者均保留为可追溯研究产物，未替换 anchor、未上传。

## 2026-10-02 5080 训练环境与 rollout-Q 隔离审计

在 WSL2 训练环境运行 `train/check_environment.py`（使用临时诊断输出，未覆盖正式依赖
锁定文件）通过：Python `3.12.3`、PyTorch `2.11.0+cu128`、CUDA `12.8`、NVIDIA
GeForce RTX 5080（compute capability `12.0`，BF16 可用）。4 步合成前向/反向耗时
`1.339 s`，峰值分配 `225.55 MiB`；运行时 `nvidia-smi` 显示总显存 `16,303 MiB`、
空闲约 `11,507 MiB`，PyTorch allocator 查询空闲约 `14,953 MiB`。这只是训练环境证据，
不代表任何候选的牌力或上线资格。报告为
`reports/training_environment_diagnostic_20261002.json`，临时 freeze 为
`reports/requirements.freeze.diagnostic.txt`。

rollout-Q 研究训练使用 `train/train_rollout_q.py`，不改变 `CandidateModel` 或 C++ 契约。
脚本启动时强制 train/validation 路径不同，分别校验生成器报告、JSONL SHA、版本/规则契约、
`test_used=false`，并拒绝 train/validation 的重复 `(game_seed, game_index, seat,
event_index)`；脚本没有 test 输入路径。当前 `candidate_rollouts_train32.jsonl`（32 局、
384 条 state）与 `candidate_rollouts_val16.jsonl`（16 局、192 条 state）均只用于研究，
其 provenance 仍基于附件裁判 `裁判代码-修正版.py`（SHA256
`fa63589d3f69ce9127093cec417f1635d8cc17205d80e70f03e44bc6809fc622`），不是官方裁判副本。
因此下一轮应先用官方裁判语义重新生成并审计 train/validation，不能把这批旧数据的训练
指标解释为官方规则下的提升。

在不覆盖 anchor 的前提下，seed `20261005` 的安全长训练命令为：

```bash
/home/ggcle/.venvs/oxbot/bin/python train/train_rollout_q.py \
  --train data/selfplay/candidate_rollouts_train32.jsonl \
  --train-report reports/selfplay_candidate_rollouts_train32.json \
  --validation data/selfplay/candidate_rollouts_val16.jsonl \
  --validation-report reports/selfplay_candidate_rollouts_val16.json \
  --init models/bc-v2-full-fp32 \
  --output models/research-rollout-q-train32-ms-seed5 \
  --epochs 6 --batch-size 32 --lr 5e-5 --seed 20261005 \
  --device cuda --fp32 --pair-weight 1.0 --value-weight 0.1 \
  --bc-weight 0.1 --max-pairs 64 --margin 0.1
```

`--output` 必须是全新目录（脚本对已有 `best.pt` fail-closed）；当前 anchor
`models/bc-v2-full-fp32/best.pt` 的 SHA256 为
`5661204b6b286bc17af6f847443cb72cfbbad8574568d32096b53ac75869cd82`，只能作为只读初始化。
训练完成后只接受 `training.json` 中 `status=completed`、`test_used=false`、
`release_artifact=false`、`init_checkpoint_sha256` 与上述 anchor 一致且 checkpoint SHA 已记录
的结果；在多 seed、官方裁判同牌换座和点差门槛通过前，不导出发布包、不改 anchor、不上传
BotZone。

Learner provenance hardening (same date): `train_rollout_q.py` now binds each rollout report
and new row to the paired `rollout_rules_contract` + `oracle_sha256` identity. Known legacy
attachment reports remain readable as `botzone-attachment-fa63589d-v1`, but a train/validation
run cannot mix that identity with `botzone-official-910cba94-v1`. Before model initialization,
the loader rejects row, game `(game_seed, game_index)`, and deal `game_seed` overlap. Official
train64/validation32 passed this audit (768/384 rows, 64/32 games, no overlaps). The opt-in
tie-aware listwise objective and its bounded official evaluation are documented in
`reports/rollout_q_robust_comparison.md`; its research artifact remains `release_artifact=false`.

## 2026-10-02 官方语义扩充 rollout-Q 与 5080 候选复核

在 RTX 5080（PyTorch `2.11.0+cu128`、CUDA `12.8`、sm_120）上，先以当前 anchor
`bc-v2-full-fp32` 生成并合并了新的官方语义 rollout。训练集为 64 局/768 个 play state/
2,933 个反事实标签，文件 SHA256 为
`94dee48fae37b92e13be290e4f44cc18160f8d03c1c696dda3be53e264df13b4`；验证集为独立 16 局/
192 个 state/717 个标签，文件 SHA256 为
`e628ec12c09d12b0ec5223a0ac94a93c35fba87a61b0060f24e29cc249c1eb25`。训练 seed 范围为
`20261020`、`20271020`、`20281020`、`20291020` 各 16 局，验证 seed 为
`20361020`、`20371020` 各 8 局；两侧无 row/game/deal 交集，均为 `test_used=false`，
`rollout_rules_contract=botzone-official-910cba94-v1`，裁判 SHA256 仍为
`910cba94244106b68535b8bee67631b476241a9924bd1789dacbbf217fb1e895`。完整审计见
`reports/official_rollout_q_train64_val16_audit.md`。

使用 8 epoch、FP32、listwise weight `0.25`、seed `20261009` 训练的研究候选为
`models/research-rollout-q-official64-v2-listwise025-val16-seed20261009/best.pt`，
checkpoint SHA256 为
`28427a2ea7e51b7e786aee9cb35e4639481bc809df27d3ab2121a1b369f999a2`；训练 provenance
记录 `test_used=false`、`release_artifact=false`，最佳验证 epoch 8 的 pair accuracy
为 `44.03%`、selected accuracy `67.71%`、value MAE `0.9518`。导出 payload SHA256 为
`b471ecf299d37e2e12107e8772a97652a962179bf6f3990af584ec75266d6bf5`，单文件包
`dist/oxbot-research-rollout-q-official64-v2-listwise025-val16-seed20261009.cpp`
为 2,344,516 bytes，SHA256 为
`63b5e699529732d9e073632e9333d7ae620e07480dd6715093c14ed8914711d0`，本机 GCC 13
C++17/O2 编译通过；Python/C++ 特征 parity 为 52 组/5,923 动作、最大误差 0，网络 parity
最大误差 `3.814697e-6`，stdin 保持打开的 6 项协议测试通过。

该候选在官方裁判、固定两组独立 seed 的同牌换座中没有稳定优势：seed `20269020` 的
32 盘总点差 `+2`（16 胜），配对 bootstrap 每盘估计 `+0.0625`、95% CI
`[-0.40625, 0.625]`；seed `20270020` 的 32 盘总点差 `-5`（12 胜），估计 `-0.15625`、
95% CI `[-0.71875, 0.4375]`。两组合计 64 盘总点差 `-3`、28 胜，所有模型请求均实际
使用该 payload，非法动作、fallback 和预算超限均为 0；本机短进程 benchmark 的最大
目标样例约 91 ms、峰值 RSS 约 116 MiB，但这不是 BotZone 1 秒 CPU/RSS 证明。候选因此
保留为研究 artifact，不替换 anchor、不上传 BotZone；官方 G++ 7.2、平台私有对局和上线
门槛仍未完成，Windows 人机、微信小程序和移动端继续暂停。

同一官方 train64/val16 数据的另一训练 seed 由独立训练过程得到
`models/research-rollout-q-official64-16`（checkpoint SHA256
`2d376049d96a0799b3ac213e705ccfb5607d8e0a480d9ba63d6bc36c09681715`）。其导出 payload
SHA256 为 `0cf8ff6bb036efe053c4ff293f5974cbf11eac2c9c5112ed078d9edda64e77d1`，单文件包
`dist/oxbot-research-rollout-q-official64-16-e3.cpp` 为 2,344,506 bytes；官方 seed
`20269020` 的 32 盘配对点差为 `-7`、模型胜率 15/32，bootstrap 每盘估计 `-0.21875`
且区间跨 0。它同样仅作研究对照，未替换 anchor。

说明：扩充生成过程按新 seed 重新发布了 `official_rollouts_train64.jsonl`，所以更早的
`research-rollout-q-official64-listwise025` checkpoint 中记录的旧 train64 文件 SHA 与当前
路径不再相同；该旧 checkpoint 保留作历史对照，不能把当前文件路径误当作其可复现实验输入。
本节的 `v2-val16` 与 `official64-16-e3` 均记录了当前 train64/val16 SHA，后续只以这两者
或重新生成并锁定的新数据版本作为训练输入。

随后在 RTX 5080 上完成了同一官方 anchor-policy rollout 数据的独立 seed 对照：
`models/research-rollout-q-official-anchor64-lw025-seed20261008`，8 epoch、FP32、
batch 32、`listwise_weight=0.25`，训练 64 局/768 rows，验证 16 局/192 rows。训练
provenance 绑定 `botzone-official-910cba94-v1`、裁判 SHA
`910cba94244106b68535b8bee67631b476241a9924bd1789dacbbf217fb1e895`，train/validation
文件 SHA 分别为 `94dee48fae37b92e13be290e4f44cc18160f8d03c1c696dda3be53e264df13b4` 与
`e628ec12c09d12b0ec5223a0ac94a93c35fba87a61b0060f24e29cc249c1eb25`，`test_used=false`、
`release_artifact=false`，初始化 anchor SHA 为
`5661204b6b286bc17af6f847443cb72cfbbad8574568d32096b53ac75869cd82`。checkpoint SHA 为
`08b8cbaeb60e5ea14ebecf23290cc073e210f95225b2958400883cf3432e1ced`，峰值显存约
313.6 MiB；验证 pair accuracy `44.19%`（初始 `44.03%`）、value MAE `0.9549`（初始
`2.1893`），但 selected accuracy 从 `100%` 降至 `67.19%`，没有显示可据此晋级的策略
提升。该结果仅作为可复现实验记录，不导出、不替换 anchor、不上传 BotZone，也未读取
held-out test；下一步仍需候选目标改进后再做官方裁判同牌换座。

同一数据随后完成了 pair-only 消融 `models/research-rollout-q-official-anchor64-paironly-seed20261011`：
关闭 tie-aware listwise 项（`listwise_weight=0`），其余初始化、数据和训练权重保持一致。
checkpoint SHA 为 `bb2b8a3cddba96a0523d5a131164498b5d20d82eff8bc8e3cea7b2e327ed5ed7`，
`test_used=false`、`release_artifact=false`，训练约 23.71 秒，峰值显存约 314.0 MiB。
验证 pair accuracy 保持 `44.03%`（273/620），value MAE 为 `0.9696`，selected accuracy
从初始 `100%` 降至 `67.71%`；与 listwise 对照没有清晰优势。因此 pair-only 也淘汰，
不做官方强度评测、不导出、不替换 anchor、不上传 BotZone。短视 credit sidecar 继续保持
审计状态，不直接并入候选 Q 学习。

## 2026-10-02 私有上传前包固定（提交 version 1 前的历史快照）

为后续 BotZone 私有编译/协议验收，已固定长时 stdin 版本单文件包
`dist/oxbot-bc-v2-full-fp32-anchor-stdin.cpp`：2,402,149 bytes，源码 SHA256
`6d464064525eb483a38e5d48fc56c34e02201486867bf8cb683458996389b928`，嵌入模型仍为
当前 anchor（模型 SHA `cccb8ecf58c88ee51ca4fb231162f90f14521a78d172ed7b74739fbad37f7232`）。
该包使用临时 G++ 7.2.0、C++17/O2/Werror 编译通过，并通过 6 项普通 JSON/stdin-open
协议检查；完整记录见 `reports/botzone_anchor_stdin_package_audit.md`。它仍是
`release_eligible=false` 的私有验收候选，不是公开发布包；平台编译、1 秒时限/RSS、
私有对局和模型加载证据仍待 BotZone 页面操作。

本轮提交 version 1 之前的浏览器快照确认 BotZone 已登录会话真实可用：`OXbot5080Lab` 已存在，Bot ID
`6abf0281e2453c4f471a8531`，GuanDan 游戏 ID `65490c16ec1ab1389702dced`，当前版本 `0`，
`cpp17`、普通 JSON、长时 stdin 已启用，源码未公开。页面可查看源代码、版本比较、Bot
信息/修改、排名趋势、天梯对局和全局保存信息，并提供“创建新 Bot”和版本上传控件。
当时尚未创建新版本或改变线上设置；该快照已被后文用户授权创建的 version 1 状态取代，
平台编译、私有对局和 1 秒/RSS 证据仍缺。

## 2026-10-02 5080 官方 rollout 扩容（历史计划状态，已由后文完成记录取代）

已确认本机 RTX 5080 可持续用于正式训练前的离线数据生成、训练和评测；当前
PyTorch `2.11.0+cu128`、CUDA `12.8`、sm_120 环境仍可用。为解决现有官方
`64/16` rollout 的候选覆盖和 pair 样本过小问题，已预注册新一轮官方裁判语义
扩容：`counterfactuals=8`、`max_states=12`、anchor 模型全席位固定策略，train
与 validation 使用不重叠的新 deal/game seed，所有 shard 均要求
`rollout_rules_contract=botzone-official-910cba94-v1`、`test_used=false`。

当时已先完成 2 局 cf8 启动探针（24 个 play rows、170 个候选标签，报告为
`reports/official_rollouts_probe_cf8.json`），验证新配置能够完整结束并通过官方
裁判。正式首批 8 个 train shard 与 2 个 validation shard 当时正在后台生成（每 shard
16 局；train seed 从 `20401020` 起，validation seed 从 `20601020` 起，彼此及历史
数据不重叠）；后续已完成 SHA、row/game/deal 隔离和聚合审计，结果见下一节 cf8 完成记录。
本段所述“BotZone 版本未改变”是生成期间的历史状态；当前 version 1 状态以文末权威记录为准。

## 2026-10-02 cf8 官方 rollout-Q 候选完成（研究 artifact）

正式 cf8 数据已完成并通过隔离审计：train 为 256 局/3072 rows/22294 labels，validation
为独立 64 局/768 rows/5640 labels；全局 320 个唯一 `game_seed`、3840 个唯一 row identity，
train/validation 无 row、game 或 deal 交集。两侧均绑定官方裁判
`botzone-official-910cba94-v1`，裁判 SHA256 为
`910cba94244106b68535b8bee67631b476241a9924bd1789dacbbf217fb1e895`。数据 SHA256：

- `data/selfplay/official_rollouts_train256_cf8.jsonl`：
  `76f4d02c6368db5b76499e3fbae13b7e49a46eadbaf5a95379920ae8164e92e9`
- `data/selfplay/official_rollouts_val64_cf8.jsonl`：
  `9ba5db133e563fd1a10ced31f48dd9b48107a2c6d7accf0797c94c9030182b05`

这里保留两层契约并明确其边界：`rollout_rules_contract=botzone-official-910cba94-v1`
标识生成标签所用的官方裁判；嵌入模型头与 C++ 网络 ABI 的
`rules_contract=botzone-corrected-fa63589d-v1` 则是既有特征/网络格式契约。两者不是
同一字段，也不是 SHA 错配；模型加载仍按 C++ ABI 契约 fail-closed，训练审计另按官方
裁判 SHA 绑定，上传前不应把两层合并成一个身份。

在本机 RTX 5080（PyTorch `2.11.0+cu128`、CUDA `12.8`）以 FP32 训练 8 epoch，batch 32，
learning rate `5e-5`，pair/value/BC 权重 `1.0/0.1/0.1`，listwise weight `0.25`、
beta `2.0`、margin `0.1`、最多 64 pairs，seed `20261003`。最佳 checkpoint 为
`models/research-rollout-q-official256-cf8-listwise025-seed20261012/best.pt`，
`best_epoch=6`，SHA256
`12604c2aaa08aa33ddee011ec9ff10d09ea9f37210d34bbc5b56536379af86cf`；验证 pair accuracy
`0.481407`、value MAE `0.801894`、selection objective `0.558688`。训练 provenance 明确
`test_used=false`、`release_artifact=false`，初始化 checkpoint 仍为 anchor
`5661204b6b286bc17af6f847443cb72cfbbad8574568d32096b53ac75869cd82`。

候选导出包为
`models/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.bin`（文件 SHA256
`4cbcd14b0090b31b608706e40e7b2326fa75c13941104c91c480baf2725ea3ea`，payload SHA256
`9d83a403011a09b0d2c9ab73fc5bce61a87dd5b015ebe2e6e4991f44f100ddf2`），单文件源码
`dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.cpp` 为
2,344,713 bytes，SHA256
`b799d987d23dad4ce16cb5f596c2065e88f62587b7e048000defb043106970d5`。Python/C++ feature
parity 通过（52 observations、5923 actions、最大误差 0）；trained network parity 通过
（最大误差约 `5.72e-6`，4 组尺寸均 optimized path 与 trace path 一致，破坏性模型校验全拒绝）。

候选已用 GCC 7.2.0、C++17/O2 编译为
`bin/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-gcc72`，二进制
SHA256 `3f76b796c9d100bae8c5cc581e1d985c2dd3d40c27c3d921a475595219840240`；stdin 保持打开的
普通 JSON 协议测试通过，GCC7 多盘结果与 GCC13 相同 seed/team 的 deal、team score、winner
逐项一致。GCC7 官方裁判 seed `20270020` 同牌换座 32 盘中模型胜 22 盘、总点差 `+24`，
配对每盘估计 `+0.75`（bootstrap 95% CI `[0.3117, 1.25]`），非法动作、模型 fallback、
本地预算超限均为 0；记录的 probe 峰值 RSS 约 4.36 MiB、本机短进程 p95 约 21 ms，仅为离线证据，不等价
BotZone CPU 时限/RSS 证明。详见 `reports/official256_cf8_candidate_gcc72_seed20270020_strength.json`。

在官方裁判下，候选与当前 anchor 分别对同一 `RulePolicy`，使用相同 seed、发牌参数和
`deal_key` 做独立同牌换座，再比较两者相对 RulePolicy 的模型边际（不是四席
candidate-vs-anchor 直接对战）：
seed `20269020` 候选相对 anchor 总点差 `+108`（16/16 配对组候选更好），seed `20270020`
为 `+119`（16/16 组更好）。这是固定 schedule 的描述性回归结果，不能替代强对手、私有
平台对局或置信结论；完整候选/anchor 源报告分别见
`reports/official256_cf8_candidate_seed20269020_strength.json`、
`reports/official256_cf8_candidate_seed20270020_strength.json`、
`reports/official256_cf8_anchor_seed20269020_strength.json`、
`reports/official256_cf8_anchor_seed20270020_strength.json`。

本候选仍严格标记为研究 artifact：不读取 held-out test，不替换 anchor，不公开发布或加入天梯。
2026-10-02 已在用户授权下将该单文件创建为 BotZone `OXbot5080Lab` 私有版本 1（描述
`cf8 research candidate; private validation only; no ladder`），页面刷新已确认版本号从 0 变为 1；
源码 SHA、编译器选择、普通 JSON、长时 stdin 与不开源设置均已记录在
`docs/online_verification.md`。当前仍缺 BotZone 平台编译日志、1 秒/RSS/整局私有对局证据，
因此 release gate 仍为 `release_eligible=false`，不替换 anchor。Windows 人机、微信小程序、
iOS/Android 继续暂停。

## 当前权威线上状态（2026-10-02，停止验证码循环后）

- BotZone `OXbot5080Lab`（Bot ID `6abf0281e2453c4f471a8531`）的 GuanDan 私有 version `1`
  已创建并经页面刷新确认；版本描述为 `cf8 research candidate; private validation only; no ladder`。
- 普通 JSON、长时 stdin、不开源和不进天梯设置已核验；版本未公开、未加入天梯、未替换 anchor。
- CAPTCHA 曾尝试两次且均返回 `captcha.wrong`，私有 GuanDan 游戏桌尚未创建；按用户要求已暂停
  后续验证码识别/提交，不再把它作为当前本地工程的阻塞循环。
- BotZone 平台编译日志、1 秒/RSS、模型加载和整局私有对局结果仍为 `UNVERIFIED`；当前优先推进
  本地训练、规则/包审计和候选评测，Windows 人机、微信小程序、iOS/Android 继续暂停。

## 2026-10-02 停止验证码循环后的本地回归

按用户要求未再操作 CAPTCHA、登录或建桌，转而复核 cf8 本地候选。WSL Ubuntu 24.04 的
`tools/build_wsl.sh` 完成 C++17/O2 构建，smoke、JSON、规则和状态测试全部通过；对
`bin/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-gcc72` 运行
`tests/test_protocol_process.py --require-model` 通过 6 项普通 JSON/stdin-open 检查。

使用官方裁判副本 `reports/official_judge_botzone_2026-10-02.py` 做 4 局独立整局回放也通过：
报告为 `reports/official256_cf8_candidate_gcc72_officialjudge_seed20270020_4game_rerun.json`，
413 次决策中 391 次为模型出牌，fallback/非法动作/本地预算超限均为 0，最大探针 RSS
4,360 KiB，本机短进程 p50/p95/p99/max 为 19.47/20.74/25.28/34.14 ms。该结果只证明
本地官方裁判副本和候选进程的一致性，`online_compatibility=unverified`，不能替代 BotZone
平台编译、1 秒 CPU 时限、RSS 或私有对局证据。

官方还牌边界 golden 重跑 13/13 通过（无 mismatch），报告为
`reports/official_return_boundaries_rerun.json`；原始裁判 SHA `910cba...`、去换行规范化
SHA `49a734...` 与既有记录一致。候选源码、模型、包 manifest 和 GCC7 二进制的完整 SHA
绑定见 `docs/cf8_local_audit.md`。

## 2026-10-03 startup-fix version 2 线上状态（当前权威）

恢复此前卡住的上传页面后，独立刷新 BotZone 已确认 `OXbot5080Lab` 的私有 version `2` 存在：
时间 `2026-10-03 00:03:27`，描述为
`cf8 candidate startup fix; deal fast path before model load; JSON C++17`，源码为
`dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-startupfix.cpp`
（2,347,082 bytes，SHA256
`666b5c9583e4fe6c85b3b8e13d963de820a234a810ffa198bf7003761bed2370`）。页面配置仍为
G++ 7.2.0/C++17、普通 JSON、长时 stdin、不开源。

天梯星标控件在刷新后的页面为 active，排名分 `972.40`；没有再次点击星标，以避免网站提示的
积分回归 1000。`查看天梯对局` 页面在 `2026-10-03 00:04:02` 显示
`isyai2026 / OXbot5080Lab` 的 **Bot 版本号 2**，并提供回放链接和 `7.88` 分数影响。
该已完成的 version 2 天梯对局是当前平台“能编译并启动”证据；完整编译日志、平台 CPU/RSS、
模型加载 SHA、超时/崩溃统计仍为 `UNVERIFIED`，不能把一次对局写成完整 P0 验收。

对应 GCC 7.2 二进制为
`bin/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-startupfix-gcc72`
（SHA256 `99eb1c5e5132e732ae202aa4cdd7782b1e8fbd10ec9dcea0c8b2233559b133cb`）；
`tests/test_protocol_process.py --require-model` 通过 6 项 stdin-open 检查。详见
`reports/botzone_startupfix_online_2026-10-03.md`。

## 2026-10-03 FableDan → OXbot C++17 迁移（本地完成）

已接入 `competition/ckpts/real-v2/best.npz` 的独立 FBDN001/FP32 C++ 推理，
并补齐 80 维特征、48 token 词表、512 token 历史、候选语义归并、长时协议及两种历史格式。
训练仍在 FableDan/PyTorch/RTX 5080 环境，部署端为无第三方推理依赖的 C++17。

最终源码 `dist/oxbot-fabledan-real-v2.cpp` 为 184,231 bytes，SHA256
`bcb7182a6b43b2c1b0431a053320e31872fe9925f677cf2e1cb5b80167d501e9`；
配套权重 `data/fabledan_w_85f1341f.fbd` 为 17,391,837 bytes，SHA256
`85f1341f65a9e72d573c2ce636e0eacb0ee1fff17b02b57daab3bab8a264add0`。
GCC 7.2 严格单文件编译、完整 GCC 13 构建、7 项进程协议检查、24 组编码/200 组候选检查通过；
5 组网络前向最大误差 `2.63e-5`。最终 GCC 7.2 二进制通过 5 局官方裁判长时进程测试，
523 次出牌均使用模型，零违规/回退。6 场景进程基准最大约 370 ms，峰值约 53.9 MiB，
均为本机数据。完整身份、报告和边界见 [迁移验收记录](../reports/fabledan_cpp_migration.md)。

本轮未上传或改变线上版本；`release_eligible=false`。用户手动上传步骤见
[C++ 迁移说明](fabledan_cpp.md)，目标平台时限/RSS与相对旧版本的强度仍待独立验收。

2026-10-03～04：按 Claude 自博弈评测包执行 FableDan real-v2 → DMC 单一实验，保留默认 80 维配置和原优化器、Replay、学习参数，总训练预算仍为累计 24 小时。03 日冒烟曾遇 NVIDIA FECS/驱动重置，诊断重跑完成约 6 分钟训练、39.3 万样本、36 次优化器更新，两组各 20 副及官方裁判 8 局零错误，但对 real-v2 的 95% 区间 [-0.225, 0.600] 跨零且不足 1,000 副，未晋级（[冒烟报告](../competition/ckpts/dmc-smoke/eval/20261003-165521-J48w2C/summary.json)，不作正式结果）。正式训练后一次中断与 04 日 01:40 Codex 停止 app-server、重启及随后 WSL 网卡移除的时间相符；恢复运行至 cycle284 后于 12:01 又出现真实 `CUDA error: unknown error`，训练退出码 1，最终保存和正式评测均跳过（[失败日志](../competition/ckpts/dmc-realv2/logs/20261004-102124-f1dc9c14.stderr.log)）。现已备份并校验 cycle280 检查点（36,974,703 样本、4484 优化步、累计 31,881.374 秒，SHA256 `fa68eadb474fdb0a8276e0870cb5458eda173242cc5460463e1b6cb7b8c103a7`），模型/优化器/Replay/RNG 均可在 CPU 完整恢复；重启后独立 GPU FP32、math SDPA、BF16 烟测通过。改用 WMI 服务启动新隐藏监督进程，核实旧终端退出不影响它；补齐独立日志、15 秒监督心跳、训练进度与响应时间、阶段/退出码、互斥锁、终态 Windows 提示及持久通知记录，修复周期检查点误标完成；训练失败禁止评测，正式评测需匹配最终检查点指纹和累计预算。训练安全/运行时 18 项测试、脚本/晋级/评测 27 项测试及 12 子测试、PowerShell 23 项断言通过。04 日 12:20 已从 cycle280 以 all 模式恢复同一轮，剩余预算 15.1441 小时；12:21 已重新进入 collect，样本 36,999,414，优化步 4484，错误日志为空（[本轮启动记录](../competition/ckpts/dmc-realv2/launch-20261004-122047-99d85c57.json)、[运行日志](../competition/ckpts/dmc-realv2/logs/20261004-122047-99d85c57.log)、[错误日志](../competition/ckpts/dmc-realv2/logs/20261004-122047-99d85c57.stderr.log)、[监督状态](../competition/ckpts/dmc-realv2/supervisor_status.json)）。已核实 ACTIVE 的 oxbot 五分钟聊天监控，故障按 run_id 去重；聊天检查需要 Codex 应用运行，关闭应用期间独立监督与落盘记录仍保留。训练结束后按固定 seed 20261101 跑各对手至少 1,000 副同牌换座及官方裁判 200 局，完成并通过置信区间和零错误门槛才可晋级；目前训练与正式验收均未完成，未上传或推送。

2026-10-05：恢复 run `20261004-135550-3a8e51a3` 完成累计 24 小时训练（cycle 769，101,137,981 samples，12,308 optimizer steps），正式评测晋级。候选对 real-v2 为 1,000 副同牌换座、每局分差 95% CI `[+2.582, +2.672]`；对 rule 为 `[+2.360, +2.466]`；官方裁判 200 局错误 0，报告在 `competition/ckpts/dmc-realv2/eval/20261004-135552-911EV3/`。随后生成 Python 包与 FBDN001/C++ 包，Python traditional/keep-running 各 1 局、C++ 数值对齐最大误差 `3.26e-6`、协议与官方裁判本地检查均通过；未上传 BotZone，目标平台 CPU/RSS 与 GCC 7.2 仍待验证。

2026-10-05：主线改为 C++。C++ FableDan 推理优化后与原实现逐位一致：模拟 G++ 7.2 -O2，512 token、120 个候选从 1,693 ms 降到约 300 ms；实际对局用时中位数约 10 ms，p99 约 100 ms（原来分别约 370 ms 和 850 ms，有 BotZone 超时风险）。新增本机 C++ 对 C++ 同牌换座评测 `competition/tools/cpp_duel.py`（官方裁判，多进程分片），以及发布门槛 `competition/scripts/cpp_release_5080_wsl.sh`：对线上 cf8 打 500 副，区间完全大于 0 才打包。下一步：对现任冠军运行发布门槛。

2026-10-07：冠军 C++ 对 cf8 完成 500 副同牌换座，每局 +2.267，95% CI [2.187, 2.346]，裁判错误 0，模型出牌 85057/85057（100%），bot A p99 0.997020 秒、max 1.422591 秒；报告 reports/cpp_release/20261007-000651/。
NOT RELEASED；gate_failures：cf8: candidate p99 0.997s over local limit 0.500s。
走 B：从冠军启动第二轮 24 小时训练，输出 competition/ckpts/dmc-r2，run_id=20261007-001410-aab39940；status=running、stage=training、phase=collect，stderr 0 字节，新候选晋级必须胜过现任冠军。
