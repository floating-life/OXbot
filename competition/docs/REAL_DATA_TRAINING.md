# FableDan 真题训练

`competition/train_real.py` 将当前审计后的南邮 `njupt-decision-v3` 转为 FableDan 的原生 80 维动作输入、48 token 词表和 512 token 公开历史。旧 C++ 的 128 维 shards 与这个模型不同，不能直接用于竞赛版训练。

先在 `competition` 中运行转换，再训练。原始 replay 和 `data/processed/njupt` 只读；转换输出和检查点必须放在独立目录。已有输出不覆盖，重新转换请选新的目录。

```bash
python train_real.py prepare --source ../data/processed/njupt --out ../data/processed/fabledan-real-v2 --include-test
python train_real.py train --data ../data/processed/fabledan-real-v2 --source ../data/processed/njupt --out ckpts/real-v2 --epochs 8 --device cuda:0
```

`--include-test` 仅在转换阶段生成独立测试 shards。训练程序只打开 train/validation；梯度来自 train，best 检查点按 validation NLL 选择。训练期间不打开 `test.jsonl` 或 test shards。一次完整 epoch 遍历每个 P 记录恰好一次，末尾不足一批的记录也会训练。每个 epoch 的训练总数、去重总数、BC 数量、NTP 数量和验证覆盖都写入 `training.json`。

当前源数据共 135,572 条决策，包括 132,638 条 P、1,467 条 T、1,467 条 B。train 为 86,939 条，其中 P 为 84,993；validation 为 25,635 条，其中 P 为 25,135；test 为 22,998 条，其中 P 为 22,510。每个 split 必须满足 `input_records = accepted_records + rule_handled_exchange`，全部 P 必须进入训练 shards。T/B 由实际 Bot 的确定性交换规则执行，全部计入覆盖清单并保留在后续 P 的公开历史；不把历史规则冲突的还贡直接当成模型标签。

2026-10-03 的完整 v2 转换用时约 24.9 秒，当前 `data/processed/fabledan-real-v2/manifest.json` 核验结果如下。NTP 样本没有从 shard 丢弃；训练/验证的 BC 指标会明确使用 BC 数量作分母。

| Split | 全部决策 | P 全覆盖 | BC 监督 | 仅公开 NTP | 规则 T/B |
|---|---:|---:|---:|---:|---:|
| train | 86,939 | 84,993 | 84,534 | 459 | 1,946 |
| validation | 25,635 | 25,135 | 24,931 | 204 | 500 |
| test | 22,998 | 22,510 | 22,401 | 109 | 488 |

772 条仅 NTP 的 P 中，762 条的上一手 claim 语义不能从当时的公开信息唯一确定，另 10 条的示范跟牌与当前竞赛规则不兼容。所有 135,572 条可用决策都有明确的模型训练、独立验证/测试或确定性规则归属。

转换使用 `games.jsonl` 中当前 game/deal 的完整公开事件表，核验它与 `PUBLIC_EVENTS` 的完整动作序列相同，严格按 `event_index` 截止到当前动作之前。它包含 T/B，也不会被 ETL 的 128 条 P 快照窗口限制。最后由 FableDan 的规则保留 BOS/LEVEL 和最近 510 个 token。对手手牌、比赛结果、未来动作、私有初始化标记不会进入输入或目标。

真实动作只提供公开牌面，没有可靠的 wildcard claim。转换器枚举该完整牌面所有可用合法 claim，按 `(type, key, size, sorted claim ranks)` 对应到竞赛生成器的规范候选。这保留 claim 多解，也允许原记录与规范生成器选择不同的自然牌花色或 wildcard 替代；不会凭空补一个 claim。编码完全相同的候选只保留一次，BC 优化所有正候选概率之和。

上一手的未知 claim 只有在所有候选具有相同 type/key/size 时，才恢复跟牌所需的这些不变量，并计数 `unknown_previous_semantics_recovered`。未知历史 claim 用保留 token 47 和公开印刷 rank 表示。无法确定上一手语义、或原记录跟牌与当前规则不兼容时，P 记录保留为公开 next-token prediction（NTP）样本：`bc_supported=false`，正标签全部为 false，BC/准确率分母不包含它。每种原因有明确计数和实例；不会把这类记录当作合法行动示范。训练要求 NTP 权重大于 0，保证这些记录确实参与梯度更新。

RTX 5080 默认使用 BF16、batch 128、候选预算 8,192、候选 MLP 分块 2,048。候选预算只拆批，不采样或丢弃候选；单记录超过预算时独立成批。可用 `--batch`、`--candidate-budget`、`--candidate-chunk` 调整显存。

```bash
python train_real.py train --data ../data/processed/fabledan-real-v2 --source ../data/processed/njupt --out ckpts/real-v2 --epochs 8 --resume ckpts/real-v2/latest.pt
```

续训保持原输出目录，保留已完成 epoch、优化器、有效学习率、NTP 权重、随机种子和 best epoch。显式传入 `--lr`、`--ntp-weight` 或 `--seed` 可覆盖对应配置，最终实际参数写入报告和检查点。`--init` 在新目录以已有兼容模型的权重开始新的 BC 训练，不继承旧训练进度或优化器。

每个 epoch 原子保存 `latest.pt/latest.npz`，验证 NLL 改善时保存 `best.pt/best.npz`。PT 使用 FableDan 原生检查点结构；NPZ 可直接走现有 NumPy 推理和上传流程。BC 检查点的 `cycle=0` 用于后续自博弈冷启动；自博弈初始化应使用仅加载权重的方式，以自己的优化器和学习率开始。

选定 best 后，单独评估 held-out test：

```bash
python train_real.py evaluate --data ../data/processed/fabledan-real-v2 --checkpoint ckpts/real-v2/best.pt --split test --out reports/real-v2-heldout.json --device cuda:0
```

数据清单记录源 JSONL、公开历史和竞赛 encoder 的 SHA256；训练检查源码/数据契约、shard 集合及每个 shard 哈希，并核验每轮完整覆盖。NLL 与 top1 只是模仿质量；模型牌力和官方裁判合法性继续使用现有 duplicate 对局和上传前检查验证。
