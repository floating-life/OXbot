# OXbot 对 FableDan 的修改说明

上游：<https://github.com/lrx0716/FableDan>，提交 `7cc5e31`（2026-06-16）。
许可：Apache License 2.0 + 非商业限制（见 `LICENSE`，原样保留）。按 Apache 2.0 第 4 条，下列文件已被修改，修改内容如下。
用途：仅用于 Botzone 竞赛（非商业）。

## A. 会被官方裁判扣 2 分的 bug（已修复）

在附件裁判（《裁判代码-修正版》）上实测，上游 Bot 以传统模式（即 Botzone 上的实际运行方式）运行：只打 2、不进贡的天梯局 4/4 正常；**单贡 4/4、抗贡 4/4、双贡同点 4/4 全部出错**。

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `botzone/bot_fabledan.py` | 进贡或抗贡后，第一个出牌请求的 history 是 `[{},{},{},{}]`，上游按 `h["player"]` 读取会抛 KeyError；传统模式下每回合都要重放历史，之后每一回合都会崩溃 | 跳过没有 `player` 字段的空项 |
| 2 | 同上 | 抗贡时官方裁判仍会发进贡/还贡请求，并要求回答 `[]`；上游照样出一张牌，被判 INVALID_MOVE | 读取 `global.resist`，抗贡时回答 `[]` |
| 3 | 同上 | 双贡且两张贡牌同点时，上游按 wiki 记账（上游拿末游的贡）；官方裁判是**顺时针**分配：头游拿 (first+1) 的贡，(first+2) 拿 (first+3) 的贡。记错手牌后，会打出自己手里没有的牌 | 按官方裁判的逻辑记账；还贡对象相应修正 |
| 4 | `fabledan/cards.py` | 官方裁判把 10 写作 `"0"`，上游 `level_rank("0")` 直接抛异常 | 同时接受 `"0"`、`"10"`、`"T"` |
| 5 | `fabledan/engine.py` | 还贡可能选到 A 或级牌（裁判会拒绝）；也可能把刚收到的贡牌还回去（裁判按原始发牌校验，会判 NOT_YOUR_POKER） | 只还"自然 2–10 且不是级牌"的牌，并排除刚收到的贡牌 |
| 6 | `botzone/bot_fabledan.py` | 异常兜底时，进贡/还贡直接出 `hand[0]` | 改为走合法的进贡/还贡选择 |

## B. 训练引擎与官方裁判对齐

`fabledan/engine.py::_do_tribute` 改为与官方裁判完全一致：双贡同点按顺时针分配；先出规则——单贡后由 (last+2) 先出，双贡不同点由贡大牌者先出，双贡同点由 (first+1) 先出，抗贡由 first 先出。`tests/test_judge_compat.py` 用裁判程序逐局对比交换后的手牌和先出玩家：150 局全部一致。

`botzone/local_judge.py`（上游自带的模拟器）同步改为官方语义：抗贡时也会发请求；还贡按原始发牌校验。

## C. 提升牌力相关

| 文件 | 改动 | 作用 |
|---|---|---|
| `fabledan/combos.py` | 跟牌时只枚举能压过上家的牌型（同类型 + 炸弹类）；同花顺检测改用位掩码 | 自博弈提速，结果与上游逐位一致 |
| `fabledan/encode.py` | 状态特征每个决策只算一次；分词改为增量式（`tokenize_cached`） | 同上 |
| `fabledan/ring.py` | 每局一个分词缓存；新增 `--ladder-frac`：按比例采样"打 2、不进贡"的天梯默认设定 | 让训练分布对准天梯 |
| `fabledan/evaluate.py` | 新增 duplicate 评测（同牌换座成对比较）、`--ladder-frac` | 用更少的局数得到更可信的强弱比较 |
| `fabledan/train_fast.py` | 新增 `--ladder-frac`、`--export-cycles`（定期导出 `latest.npz`）；训练中评测改为 duplicate，并输出平均分差；不再自动打包 16 MB 的 zip（超出 4 MB 源码限制） | |
| `botzone/bot_fabledan.py` | 合法动作上限 128 → 512：上游在约 0.5% 的局面里会截掉一部分动作，可能连炸弹一起截掉 | 推理时与训练时看到同样的动作集合 |

自博弈单核吞吐（同一台机器、随机 Q）：**3,473 → 8,018 决策/秒（约 2.3 倍）**。`tests/test_fast_paths.py` 在 30,772 个局面上验证出牌枚举、特征和分词三处与上游逐位一致。

## D. 上线工具

- `fabledan/packaging.py`、`botzone/pack_bot.py`：生成只含代码的 zip（约 25 KB），外加带版本号的权重文件 `fabledan_w_<sha8>.npz`，后者上传到用户存储空间；zip 内的 `weights_name.txt` 指明加载哪个权重文件。Bot 首回合 debug 会输出实际加载的权重文件路径。
- `tools/judge_runner.py`：在进程内按 Botzone 的方式驱动**官方裁判程序**，覆盖无贡、单贡、双贡、抗贡、双贡同点（顺时针/逆时针两种座位）以及全部 13 个级牌；支持进程内、传统进程、长时进程三种驱动方式，两种 history 格式；也可以通过裁判做 duplicate 对战。
- `tools/make_random_npz.py`：在没有 PyTorch 的机器上生成随机权重，用来测试真实的推理路径。
- `scripts/*.bat`：Windows 下的环境安装、训练、上传前检查、两个检查点对战。

## E. 已验证 / 未验证

- 已验证（当前工作区）：根目录 pytest **59 passed**（11 个 subtests），竞赛专项 pytest **29 passed**；30,772 个 fast-path parity、官方裁判还贡边界 1,404 个牌/级牌组合、官方裁判 156 局进贡交换均通过。`real-v2` 在 RTX 5080 上完成 8 个 epoch，best 为 epoch 4（validation NLL `0.844537`），held-out test NLL `0.830641`。官方裁判真模型 200 局、按座位排列 40 局均为 0 errors；打包 zip 以真实传统和长时进程各跑 1 局也为 0 errors。
- 这些是本机工程和协议证据，不是 Botzone 平台时限、内存或牌力证明。官方源码副本来自当前工作区已有审计，SHA-256 为 `910cba94…`；附件修正版与官方还贡边界不同，因此上传检查强制使用 `judge/judge_official.py`。
- 尚未完成：DanLM V1 的本地对手接入（其可运行代码是 macOS ARM 二进制）；南邮数据目录目前只有记录/回放，没有 16 个可执行 Bot 包。内置规则机器人和已保存 checkpoint 仍可用于本地评测。
