# OXbot Roadmap v3：C++ 模型首发，本机 RTX 5080 训练

决策基线：2026-10-01；2026-10-03 按用户要求补充 FableDan 竞赛底座和 C++ 部署路线。

## 不变的决策

- 正式 Bot：C++17、G++ 7.2.0 `-O2`、普通 JSON；支持传统进程和长时 stdin。长时模式每个响应带保活标记，并接收后续增量请求。
- 独立规则内核；附件裁判保留原样，只作为离线 oracle，不复制为产品规则实现。
- 第一个正式版本必须以真实训练模型决定出牌。规则策略用于测试、基线和模型异常降级。
- 本机训练：南邮 BC 冷启动 → C++ 导出与一致性验证 → 首发；随后 DMC 自博弈持续增强。
- 当前竞赛路线使用 FableDan 128 维、4 层 causal Transformer、80 维候选特征和 512 token 历史；旧 OXGDQ001 的 64 维、2 层模型保留作研究对照。两套模型分别编码与加载，合法动作集合与最终合法性检查共同约束输出。
- 上传源码 UTF-8、小于 4 MB；模型版本化，优先用户存储空间。实际路径和配额须登录核验。
- BotZone 模型上线前，只推进 Bot、规则、数据、训练、评测、发布。Windows UI、整局客户端、微信/iOS/Android 不启动。

## 架构边界

普通 JSON → StateMirror → 独立规则核心 → 合法候选 → ModelPolicy → 终检 → response。

训练和 BotZone 共享牌号、规则契约、候选语义和特征契约。Python/PyTorch/CUDA 留在本机训练环境；C++ 推理不依赖这些运行库。

`competition/` 保留 FableDan 训练和 NumPy 参考推理；`competition/tools/export_fabledan_cpp.py`
将 NPZ 转为独立的 FBDN001 外部权重，供 C++ FableDan 编码器与网络加载。FBDN001 与旧
OXGDQ001 不是可互换的模型格式。

`botzone_compat` 是首发运行档。`standard` 当前仅预留名称，不声称已实现；本地整局规则和地方规则等 Bot 上线后再开发。不能把未经验证的 v2 J2/J4 条目当作线上事实。

## 工作清单与门槛

| 阶段 | 工作 | 可验证产物 | 退出条件 |
|---|---|---|---|
| P0 契约 | 获取官方原版、记录 SHA、与附件 diff；协议与本地 oracle | oracle manifest、golden、差异记录 | 原版身份明确；未取得前兼容状态为未验证 |
| P1 核心 | C++ 牌型、配子、全动作生成、状态镜像、普通 JSON、安全策略 | 核心测试、差分工具、完整单局对局工具 | 分层差分/小手牌穷举通过；完整对局牌权一致 |
| P2 数据 | 受限 pickle ETL、实体牌映射、规则差异、整包切分 | 数据卡、数据 SHA、训练/验证/测试清单 | 截断隔离；来源可追溯；无私牌泄漏；标签覆盖率有报告 |
| P3 BC | 真实南邮数据训练、验证、模型导出、C++ 前向 | checkpoint、manifest、二进制、parity报告 | 模型加载校验；浮点输出/动作对齐；模型实际参与决策 |
| P4 上线 | 性能、压力、duplicate、平台编译、私有实战、发布回滚 | 单文件源码、权重、评测与发布记录 | 10,000 局合法性压力；1 秒时限验证；线上兼容与模型加载成功 |
| P5 增强 | DMC、冻结对手池、同牌换座、冠军晋级 | 新模型和对旧版本的对照报告 | 对至少 2,000 副 duplicate 报告得分差及置信区间；通过后替换 |
| 后续 | Windows 整局人机 → 微信小游戏 → iOS/Android | 首发完成后另排期 | 本路线图仅占位 |

55% 单局胜率、60% 整局胜率保留为强度目标；必须给定对手、样本口径和置信区间，整局目标不前置到 BotZone 单局验收。不能拿对随机 Bot 的胜率证明竞技强度。

## 评测口径

- 同牌换座的每组是相关观测；得分差采用按 duplicate 组的配对 bootstrap 置信区间。
- 胜率报告 Wilson 区间时，明确独立观测单位；不得把同组两盘当作独立 Bernoulli 样本夸大精度。
- 单独报告违规、崩溃、超时、p50/p95/p99/max、RSS、模型加载和 fallback 次数。
- 首发规则压力用纯规则策略可以验证规则/协议，不能代替最终模型版本的同样测试。
- 所有源码、规则、数据切分、特征、模型权重、评测种子均记录版本或 SHA。

## 本机环境

已检测 WSL2 Ubuntu 24.04、16 个逻辑 CPU、RTX 5080 16 GB。独立环境位于 WSL `/home/ggcle/.venvs/oxbot`，Python 3.12.3、PyTorch 2.11.0+cu128；sm_120、BF16 前向/反向已通过实际测试，具体依赖见 `train/requirements.lock.txt`。

旧 OXGDQ001 初版模型 153,601 参数，其环境 smoke 峰值约 226 MiB，仅对应 32 batch 的合成数据验证。
FableDan `real-v2` 的实际训练与数据覆盖见 [真题训练记录](../competition/docs/REAL_DATA_TRAINING.md)；
两套记录均不能代替 BotZone 平台时限与 RSS 验收。

## 数据与来源

南邮原始数据只读，按比赛包隔离训练/验证/测试；商用授权证明作为数据卡附件存档，当前本地研发不因凭证未入库重复询问授权。按用户确定的竞赛路线，FableDan 代码已导入 `competition/`，并使用本机训练的 `real-v2` 权重；DanLM 的代码和权重未导入。来源与修改记录见 [OXBOT_CHANGES.md](../competition/OXBOT_CHANGES.md)。

13 万左右事件并不等于等量可监督标签。配子 claim 不确定、平台规则差异和来源质量分别记录，不能用推测的 claim 污染真值。官方原版已取得，官方还牌边界已落实并通过定向 golden；此前使用附件修正版的历史差分报告仍只对该附件成立。

## 当前进度入口

旧模型与线上版本记录见 [progress.md](progress.md)，当前 FableDan 训练记录见
[REAL_DATA_TRAINING.md](../competition/docs/REAL_DATA_TRAINING.md)，C++ 配对产物见
[C++ 迁移说明](fabledan_cpp.md)。FableDan `real-v2` 的真实 BC 训练、C++ 编码与网络接入已完成；
当前重点是最终 C++ 包的官方裁判、协议、性能与平台实战验收，随后再做 duplicate 强度对照。
本清单保留为验收目标，未获得对应报告的门槛不记为完成。
