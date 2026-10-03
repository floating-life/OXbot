# 南邮掼蛋数据审计与 ETL 契约

审计日期：2026-10-01。原始来源目录：`D:\coding\RL\训练数据\南邮`。ETL：`train/etl_njupt.py`；输出：`data/processed/njupt`。训练优先使用该目录的 `njupt-decision-v3`，不要再混用早期 v1/v2。

## 实际产出

原始清单有 47 个压缩包条目，其中 1 个为说明包，46 个为对阵包。每个对阵包包含 3 个完整升级盘；每盘包含多次发牌。这里将对阵包称为 match、完整升级盘称为 game、单次发牌称为 deal，避免将整场升级赛与一副牌混为一谈。

| 项目 | 数量 |
| --- | ---: |
| 对阵包 / 完整升级盘 / 可用发牌局 | 46 / 138 / 1300 |
| 决策总数 | 135572 |
| 出牌 P / 进贡 T / 还贡 B | 132638 / 1467 / 1467 |
| P 中实际出牌 / 过牌 | 64965 / 67673 |
| 隔离文件 | 1 |
| 与修正版裁判冲突的还贡动作 | 17（分布于 16 盘） |
| V 之后排除的重复 P | 11（分布于 4 盘） |
| V 之后保留为结果元数据的 F | 46 |

按 `(match_id, archive_path)` 升序，前 32 个包为 train，随后 7 个为 validation，最后 7 个为 test。同包的三盘、全部发牌与决策始终处于同一 split。拆分不使用赛果或决策标签。

| Split | 包数 | 决策数 | JSONL 字节数 |
| --- | ---: | ---: | ---: |
| train | 32 | 86939 | 408259791 |
| validation | 7 | 25635 | 125443717 |
| test | 7 | 22998 | 108346397 |

`games.jsonl` 为 18614896 字节，`quarantine.jsonl` 为 263 字节。各文件的 SHA-256、源 `.data` 哈希、原始压缩包哈希和统计写入 `manifest.json`。本次 v3 manifest SHA-256：`17d78ac0dd7e849e4c1be3b2acd327880f8df0e5ce503a640c4e9d4b10f07c66`。

## 原始记录含义

- `R(-1, level)` 表示本副的全局级别；随后 `R(0, level0)`、`R(1, level1)` 为两队等级。不是当前行动玩家自己队伍的等级。1162 次跨副转换中，新级别全部等于上一副头游队伍的新等级。
- `I(seat, cards)` 为初始 27 张暗牌。四个座位合计严格包含两副完整牌。
- `T(from, to, card)`、`B(from, to, card)` 是实际转移；回放先检查给牌者牌权，再更新双方牌权。
- `P(seat, list)` 是出牌；`P(seat, 1)` 是过牌。列表不包含 BotZone 的 `claim`。
- `C` 是接风。全量 1157 次 C 都发生于上一手出牌者已经出完、其搭档仍有牌，随后行动者全部是该搭档。ETL 清空上一手限制并恢复领出。
- `V` 是当前完整升级盘结束。解析在第一个 V 停止生成决策。空的 `.data_级别0_级别1` 文件补充末尾等级，空的 `.ros` 文件名补充三盘总比分。
- `F` 只在第三盘 V 后出现，记录每座位的三盘总比分。保留为结果元数据，不作为行动特征。V 后重复 P 只进入审计记录，不进入训练。

唯一隔离文件为 `白马 vs Pass/白马队_Pass队_20201018_140424_0.data`：仅有 3 条 R 和 4 条 I，没有 P 或 V；SHA-256 为 `c2625f2ae0038adfb62c608958540fe7a1d779cf18e11fb4cebc17f25c6d67f9`。其完整同盘版本被正常选中。隔离原因是 `missing_terminal_v`，未修改、移动或删除原文件。

## 牌号与牌权

原始牌号不是 `source_id - 2` 的四花色交错序列。自然牌为四个 13 张花色块：

```text
rank_number = ((source_id - 2) % 13) + 2   # 2..14，14 为 A
source_suit_block = (source_id - 2) // 13
source 54 = 小王；source 55 = 大王
```

进贡数据核验了块 1 为红心。其余花色采用历史顺序方块、黑桃、梅花，即源花色块到 BotZone `[红心, 方块, 黑桃, 梅花]` 的排列为 `[1,0,2,3]`。三种非红心花色的名称排列是约定；一致的全局置换不改变该游戏规则或同花关系。BotZone 自然牌点数顺序为 `A234567890JQK`，其中 `0` 表示 10；标准面号为 `0..53`，物理副本为 `face` 和 `face+54`。

原始数据用相同牌号表示两副中的同花同点牌。ETL 内部为每副牌建立确定性的全局物理实体，贯穿进贡、还贡与出牌，用于牌权核查；公开事件审计文件保留这些实体。所有 138 盘的牌权、出牌顺序、跳过已出完座位与禁止领出过牌检查均通过。

训练特征不能使用该全局实体的人工副本编号：按座位分配的副本编号可能间接反映对手初始持牌。因此 `features.own_hand` 与 `label.cards` 按本方可见多重集独立编码：每个面号第一张为 `face`，第二张为 `face+54`。测试已验证，交换对手暗牌能改变内部全局实体，却不会改变行动者的 features 或 label。学习编码再按 `%54` 使用牌面。

## 决策 JSONL v3

每行由 `schema, match_id, game_id, deal_id, event_index, stage, features, label, provenance, quality_flags, result` 组成。`stage` 为 `play / tribute / return`。所有 features 都是当前行动发生前的信息；`result` 及来源标识只供审计与拆分，不能传入策略网络。

| features 字段 | 含义 |
| --- | --- |
| `seat, team` | 绝对座位和座位奇偶队伍 |
| `level, level_label, team_levels` | R(-1) 数字级别、BotZone 字符级别、两队等级 |
| `own_hand, own_hand_faces` | 本方局部物理牌号；54 维面号计数 |
| `remaining_counts` | 四座位动作前余牌数量；不是对手暗牌 |
| `leading, last_player, last_cards, last_claim` | 是否领出；跟牌目标及可重建 claim。领出时 player=-1、cards=[]、claim=null |
| `history, history_total` | 本副此前 P 的末 128 条与累计条数；每条含 player/action/claim |
| `played_face_counts` | 本副所有已公开 P 的累计 54 维计数，排除 T/B；不受历史窗口截断影响 |
| `public_history` | R/T/B/P/C 的末 64 条紧凑记录；另有 total/truncated 字段 |
| `done_order` | 当前副已经出完的公开顺序 |
| `tribute, resist, exchange_context_known` | 应进贡数、抗贡状态、源信息是否足以恢复 |

历史 `claim`：过牌为 `[]`；不含红心级牌的出牌为其天然面号多重集；包含红心级牌时为 `null`，不猜测万能牌替代语义。`label.claim` 始终为 `null`，明确原始日志缺少这个字段。`label.claim_candidates` 提供候选而非原始真值；BC 必须在当前规则下重建合法动作，并把牌面多重集匹配的全部 claim 作为多正例。

62600 个自然牌候选逐条通过附件裁判 `checkPokerType` 的类型检查，0 个不一致。2365 个动作仍标为 `pending_reconstruction`：2341 个含红心级牌，24 个是天然三带王对等保守分类器未收录的五张动作；这些动作均保留，不能因缺 claim 直接删除整副数据。全部 P 中，21785 次为领出，105936 次跟随已知上一手 claim，4917 次跟随未知上一手 claim。首轮 BC 可跳过最后一类，重新领出后继续使用样本。

v2 的 64 条 P 历史在 3617 个 P 样本中不足以覆盖模型最后 254 个原始 token，共少 35657 个 token。v3 将 P 历史扩大到 128 条；每条 P 最少 3 个 token，因此足以保留 254 个 token。全量 132638 条 P 已核对：v3 截断历史生成的 token 与完整公开历史的最后 254 个 token 完全相同，累计 played counts 也全部一致。128 条是数据保存窗口；网络仍使用 `BOS + last 254 raw tokens + EOS`。

进贡上下文仅使用公开前史。上一副前两名同队时应双贡，否则应单贡；到本副首个 P 时交换阶段已经结束，前面没有 T 则表示抗贡。此推断即时发生于首个 P 之前，不查询对手初始牌，也不向前查看后续动作。

| 应贡数 / 抗贡 | P 决策数 | context_known |
| --- | ---: | --- |
| 0 / false | 13815 | false（各完整盘首副，使用中立值） |
| 1 / false | 53171 | true |
| 1 / true | 3145 | true |
| 2 / false | 47691 | true |
| 2 / true | 14816 | true |

首副缺显式 `tribute/resist` 元数据，保留 unknown 标志；该数据所有首副均从双方 2 级开始，中立 `0/false` 与上线初始局相容。T/B 阶段不用于首轮出牌 BC，其临时余牌数可能是 26/28，不应直接送入要求最多 27 张的出牌特征编码器。

## 与修正版裁判的边界

17 次还贡给出了当前级牌，违反附件 `isValidReturn` 对级牌的显式排除。仅在对应 B 决策上标记 `return_current_level_conflicts_oracle`，不把整盘所有动作误标为冲突，也不默默丢弃数据；盘级摘要另标 `contains_return_current_level_conflicts_oracle`。这批原始人类/历史程序动作不能直接视为 BotZone 规则真值。

牌权与轮转验证不等于所有历史动作已通过 BotZone 全套规则。尤其缺失万能牌 claim、历史三带王对及不同裁判的级牌比较行为，仍需候选生成器按附件裁判终检。原数据 `result` 保留完整盘和三盘赛果，当前 BC 不读取它们作为输入。

## 安全与复现

原始 `.data` 是拼接的协议 3 pickle。受限加载器禁止解析类/全局对象、持久化引用、递归容器和非预期类型；只接受整数、字符串、空值、列表、元组，再做事件结构检查。默认限制为单文件 4 MiB、每文件 20000 个事件、深度 32、序列长度 4096、字符串长度 4096、单事件遍历 100000 个值、整数 64 位。未完成的 pickle 记录不会被当作正常 EOF。

输出路径不得位于原始来源目录内，源 manifest 指向的解压目录也必须位于来源根目录内。每个输出先写独立临时文件，再原子替换；manifest 最后替换并记录全部数据文件哈希。原始压缩包、`.data`、sidecar 和说明文件均未修改。

```powershell
python -m unittest discover -s tests -p test_etl_njupt.py -v
python -X utf8 train/etl_njupt.py --source 'D:\coding\RL\训练数据\南邮' --output data/processed/njupt
```

当前 17 项回归测试通过，覆盖受限 pickle、资源上限、截断、映射、动作前快照、进还贡、局部副本隐私、接风、抗贡公开推断、历史窗口、赛果尾部、轮转、非法牌权、确定性拆分、重复运行哈希稳定及禁止写入源目录。
