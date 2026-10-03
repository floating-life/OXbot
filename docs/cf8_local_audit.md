# cf8 候选本地审计（研究 artifact）

更新时间：2026-10-02（Asia/Shanghai）。本报告只记录 cf8 候选在本机/WSL 的可复核证据；不进行 BotZone 登录、验证码、建桌或上传操作。`release_eligible` 仍为 `false`，不改变 anchor。

## 结论

cf8 候选可以在本地完成单文件启动、普通 JSON 长连接、规则/状态回放和模型推理；已用官方裁判源码副本做离线整局检查，未发现非法动作、模型降级或本地预算超限。C++/Python 特征 parity、训练权重网络 parity、GCC 7.2 产物及协议检查均有记录。

这些结果不等于 BotZone 平台验收：平台实际编译日志、BotZone CPU 计时/RSS、平台模型加载、私有整局回放和账号侧状态仍是 `UNVERIFIED`。本地 WSL wall-clock 和 `/proc` RSS 不能宣称满足 BotZone 的 1 秒/内存限制。

## 候选身份与完整性

| 工件 | 大小 | SHA-256 | 备注 |
|---|---:|---|---|
| [`dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.cpp`](../dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.cpp) | 2,344,713 B | `b799d987d23dad4ce16cb5f596c2065e88f62587b7e048000defb043106970d5` | C++17 单文件，低于 4,000,000 B |
| [`dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.manifest.json`](../dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.manifest.json) | 5,291 B | `c8c7791f4572622f72179c44220660bdabe1e1abf2dcac70e6911b22365402bf` | `release_eligible=false`、`interaction=ordinary_json`、`selection=raw` |
| [`models/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.bin`](../models/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.bin) | 620,482 B | `4cbcd14b0090b31b608706e40e7b2326fa75c13941104c91c480baf2725ea3ea` | 嵌入模型源文件；payload SHA `9d83a403011a09b0d2c9ab73fc5bce61a87dd5b015ebe2e6e4991f44f100ddf2` |
| [`models/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.manifest.json`](../models/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.manifest.json) | 9,078 B | `8ea956b28eca7a5b1c7cbe4186a6cc0b78effedb2d39244edf0d883d0a26db` | `oxbot-causal-candidate-v1`，`raw` 默认策略 |
| `models/research-rollout-q-official256-cf8-listwise025-seed20261012/best.pt` | — | `12604c2aaa08aa33ddee011ec9ff10d09ea9f37210d34bbc5b56536379af86cf` | 最佳 epoch 6；训练 checkpoint，不上传 |
| `bin/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-gcc72` | 856,280 B | `3f76b796c9d100bae8c5cc581e1d985c2dd3d40c27c3d921a475595219840240` | 历史 G++ 7.2.0/C++17/O2 产物；平台编译仍未证实 |
| `bin/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-gcc13` | 886,312 B | `9d8244026eaa95c8a3c9aaf98315cb17916a61c374021ab3b6470a25d7213100` | WSL 当前 g++ 13.3 产物 |
| `reports/official_judge_botzone_2026-10-02.py` | 30,308 B | `910cba94244106b68535b8bee67631b476241a9924bd1789dacbbf217fb1e895` | BotZone 官方裁判副本；rollout/离线整局 oracle |
| `裁判代码-修正版.py` | 30,683 B | `fa63589d3f69ce9127093cec417f1635d8cc17205d80e70e03e44bc6809fc622` | 附件修正版；模型 ABI contract 使用它的规则标识 |
| `bin/core_probe` | 372,976 B | `e3a555675591acb5d944af8d995609deb8fdce5c1b0c2bb1fe3517336e0ccc0d` | 本地规则/状态/模型探针 |

候选 manifest 的 15 个 `source_hashes` 已逐项与当前 `core/`、`botzone/` 文件核对，0 个 mismatch。训练输入仍绑定官方裁判 SHA `910cba94244106b68535b8bee67631b476241a9924bd1789dacbbf217fb1e895`：

- train：256 局、3,072 rows；`data/selfplay/official_rollouts_train256_cf8.jsonl` SHA `76f4d02c6368db5b76499e3fbae13b7e49a46eadbaf5a95379920ae8164e92e9`；
- validation：64 局、768 rows；`data/selfplay/official_rollouts_val64_cf8.jsonl` SHA `9ba5db133e563fd1a10ced31f48dd9b48107a2c6d7accf0797c94c9030182b05`；
- provenance 标记 `test_used=false`、`release_artifact=false`，训练设备 RTX 5080/CUDA，未读取 held-out test。

这里保留两层规则契约：rollout 标签使用 `botzone-official-910cba94-v1`；C++ 模型 ABI/特征使用 `botzone-corrected-fa63589d-v1`。两者不能混为同一 SHA。

## 可复核测试结果

### 1. C++ 基础回归

WSL `tools/build_wsl.sh` 及已有产物检查通过：`json_test`、`rules_test`、`smoke`、`state_test` 均返回 0（分别输出 `json tests passed`、`rules_test ok`、`smoke ok`、`state tests ok`）。这些是规则/JSON/状态核心测试，不是 BotZone 平台测试。

### 2. 单文件编译与普通 JSON 长连接

- 在 WSL g++ 13.3.0 以 `g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic` 重新编译 cf8 源码，临时产物 SHA 为 `9d8244026eaa95c8a3c9aaf98315cb17916a61c374021ab3b6470a25d7213100`，与现有 GCC13 产物一致。
- GCC 7.2 产物已通过同一候选的 C++17/O2 构建记录；本机当前未安装 g++ 7，不能把 WSL g++13 重编译冒充平台 G++7.2。
- `wsl python3 tests/test_protocol_process.py --bot bin/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-gcc72 --require-model`：**6 checks passed**；stdin 保持打开时仍在首个响应后存活，之后干净退出，无 stderr。GCC13 产物同样 **6 checks passed**。

### 3. 特征与网络 parity

- [`reports/official256_cf8_feature_parity.json`](../reports/official256_cf8_feature_parity.json)，文件 SHA `dfb1f785779c9f1766b255bedfe1a15571ca8d0fea972c7629d608db91423a69`：52 个 observation、5,923 个动作，Python/C++ 最大绝对误差 **0**。
- [`reports/official256_cf8_network_parity.json`](../reports/official256_cf8_network_parity.json)，文件 SHA `ce8aaf78970c695be3fb07166b7c4cf98e284ced47c54297f005d6457162f2aa`：真实 cf8 checkpoint SHA `12604c2a...af86cf`、payload SHA `9d83a403...f100ddf2`；history/candidate 四组尺寸最大误差 `3.814697265625e-6`，optimized path 与 trace path 完全相等，argmax 全部一致；截断、错误 magic、checksum、feature/rules/architecture/config 篡改均被拒绝。

### 4. 官方裁判离线回放与协议

- [`reports/official256_cf8_candidate_gcc72_officialjudge_seed20270020_4game_rerun.json`](../reports/official256_cf8_candidate_gcc72_officialjudge_seed20270020_4game_rerun.json)，文件 SHA `0019ea80e413a06f8b64bbfe433949a56e0e9d14d27cf2ab4149ffbced9db776`：官方裁判 SHA `910c...`，4 局/413 decisions，其中 391 次 play 使用模型；模型 payload 前缀始终 `9d83a403011a`，fallback 0、非法动作 0、本地预算超限 0；本次 WSL 短进程 wall-clock p50/p95/p99/max 约 `19.42/20.70/24.73/34.14 ms`，probe 峰值 RSS 约 4,360 KiB。
- 既有 16+16 duplicate strength 记录 [`reports/official256_cf8_candidate_gcc72_seed20270020_strength.json`](../reports/official256_cf8_candidate_gcc72_seed20270020_strength.json)，文件 SHA `cbe86607813a21d18778ca4aa1fc6f968ac64a2955eb53a5528a75de795c4dde`：32 局、相对 `RulePolicy` 模型胜 22、总点差 `+24`，配置模型座位 0 fallback/本地预算超限。它是固定 schedule 的描述性 launch-capability 证据，不是强对手强度结论。
- [`reports/official_return_boundaries_rerun.json`](../reports/official_return_boundaries_rerun.json)，文件 SHA `358f1b4965a16d1648bbe26bd6ecec725c2ff42e8c2436084fa3e168ec5af4be`（既有同内容报告 [`official_return_boundaries.json`](../reports/official_return_boundaries.json) SHA `2113f278a4cdd5d79f4a81f8ec8c6b7439b8688d9ce15701686c4ccdd842824a`）：官方 `isValidReturn` 在 13 个等级边界均无 mismatch，原始裁判 SHA `910c...`、规范化 SHA `49a7346d...`。该测试特别处理官方裁判的 `set_level`/`pointorder` 顺序，不能推出平台验收。

### 5. 纯 Python 合同回归

`python -m unittest tests.test_local_judge tests.test_summarize_strength -v`：10 tests passed。覆盖 team 计分、互补 seat 配对、seed schedule、模型 payload 校验、启动失败报告和 bootstrap 配对契约。

使用附件修正版裁判执行小规模 `tools/verify_rules.py --hands 26 --claims 260 --seed 20261012`：260 random claims、3,591 generated actions、724 comparisons、4 个 exhaustive hands 全部通过。使用官方裁判执行同一脚本时，`isValidReturn` 会通过 oracle 的 `setError` 抛出 `JudgeRejected`；这是该通用随机测试与官方裁判 return API 的调用契约不匹配，不能记录成候选非法动作。官方 return 边界 golden 已单独通过（见上）。

## 未完成与下一步门槛

当前唯一外部阻塞仍是用户已要求暂停的 BotZone 登录/CAPTCHA/私有桌流程。以下项目保持 `UNVERIFIED`，不应在报告中写成“已通过”：

1. BotZone 实际 G++ 7.2.0 编译日志、警告和平台版本号；
2. 平台 1 秒 CPU 时限、RSS/内存和首回合计时；
3. 平台实际模型加载 SHA、整局私有回放、崩溃/超时/非法动作统计；
4. 公开发布、天梯、anchor 替换以及 Windows/微信/iOS/Android 产品化。

待 CAPTCHA/私有桌另行解决后，只需把同一 SHA 的 cf8 包与本报告交叉核对，补齐上述平台证据；在此之前继续将 cf8 作为研究候选保留。

## 2026-10-03 线上补充

上述“未完成”段落是 2026-10-02 的本地审计快照。随后 startup-fix 包已作为 BotZone 私有
version 2 创建，并在 `2026-10-03 00:04:02` 的天梯对局记录中以 **Bot 版本号 2** 出场；
这证明平台至少完成了该版本的编译和启动。版本源码 SHA 为
`666b5c9583e4fe6c85b3b8e13d963de820a234a810ffa198bf7003761bed2370`，GCC7 本地产物 SHA 为
`99eb1c5e5132e732ae202aa4cdd7782b1e8fbd10ec9dcea0c8b2233559b133cb`。完整线上记录见
[`reports/botzone_startupfix_online_2026-10-03.md`](botzone_startupfix_online_2026-10-03.md)。

该对局不提供完整平台编译日志、CPU/RSS、模型加载 SHA 或稳定性统计，因此这些项目仍不能
标记为完整通过，`release_eligible=false` 继续保持。
