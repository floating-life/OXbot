# BotZone 线上核验记录（P0/P4）

本文件是登录后逐字段填写的证据表。网页内容只作为第三方事实来源，不能替代用户授权；未填写或没有可复核证据的字段保持 `UNVERIFIED`。文中日期较早的段落保留历史快照；当前权威线上状态见文末的 `2026-10-03 startup-fix version 2` 小节。

## 2026-10-02 私有候选版本更新

用户在提交前明确授权创建 BotZone 私有验收版本。已在已登录会话中上传并提交候选单文件：

- Bot：`OXbot5080Lab`（Bot ID `6abf0281e2453c4f471a8531`）
- 新版本：`1`，创建时间页面显示 `2026-10-02 14:55:25`
- 版本描述：`cf8 research candidate; private validation only; no ladder`
- 源码：`oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.cpp`
- 编译器：`G++ 7.2.0 (-O2, C++17)`，扩展名 `cpp17`
- 普通 JSON：`simpleio=false`
- 长时 stdin：`enable_keep_running=true`
- 开源：关闭；天梯星标：未开启

独立刷新后的 BotZone 版本表已显示版本 `1`，因此“版本创建”已得到平台页面证据。平台编译日志、运行时限/RSS 和私有对局结果仍为 `UNVERIFIED`；在这些证据完成前，不替换 anchor、不公开、不加入天梯。

创建私有 GuanDan 游戏桌时，BotZone 首页要求填写一位 CAPTCHA。此前曾尝试填写两次，均返回
`captcha.wrong`；私有 GuanDan 游戏桌尚未创建。按用户要求，本轮已停止继续识别/提交验证码，
因此线上私有对局验收暂时暂停；这不阻塞本地训练、审计和候选整理。

补充记录（2026-10-02，版本 1 提交后的只读复核）：Computer Use 一度无法以足够置信度确认
Chrome 当前 URL，随后停止浏览器操作；该次复核没有输入密码、上传文件、创建版本或改变已记录
版本 1 的 BotZone 状态。此前已保存的官方游戏页/裁判源码证据仍有效；Bot 列表和版本 1
创建状态已有下方页面证据，平台编译和私有对局字段继续保持 `UNVERIFIED`。

同日只读 HTTP 访问 `https://www.botzone.org.cn/mybots` 返回公共未登录页面（页面含
`?msg=notlogin` 和登录表单）；该请求未携带浏览器会话，也未尝试提交登录信息，不能证明
保存的浏览器会话是否有效。

本轮通过 Codex 内置浏览器的已登录会话核验到：Bot `OXbot5080Lab` 已存在，Bot ID
`6abf0281e2453c4f471a8531`，GuanDan 游戏 ID `65490c16ec1ab1389702dced`，当前最新版本为
`1`，扩展名/语言为 `cpp17`；`simpleio=false`（普通 JSON）、`enable_keep_running=true`
（长时 stdin）、`opensource=false`。Bot 管理视图另显示描述“RTX5080训练模型研究候选；
C++17普通JSON；stdin长时运行；规则终检与降级”；版本 1 的上传描述以上方 version 1 上传记录中的
`cf8 research candidate; private validation only; no ladder` 为准。上述页面证据证明账号会话、
Bot 列表和版本 1 可访问，但不等于平台编译或对局已通过。

## 记录元数据

| 字段 | 值 |
|---|---|
| 抓取时间（Asia/Shanghai） | `2026-10-02`；通过已登录浏览器会话读取 |
| BotZone 游戏页 URL/游戏 ID | `https://www.botzone.org.cn/game/GuanDan` / `65490c16ec1ab1389702dced` |
| 登录账号/昵称（不保存密码） | 已确认登录态；不记录账号名、不保存密码 |
| 页面截图/HTML 证据 | 游戏页“裁判代码”弹窗 DOM；源码副本见下表 |

## 官方裁判与规则

| 字段 | 值/证据 |
|---|---|
| 官方原版裁判下载文件名与版本 | BotZone GuanDan 当前页面源码；页面未提供独立版本号 |
| 下载来源 URL 与页面时间 | [游戏页](https://www.botzone.org.cn/game/GuanDan)；接口见 `reports/official_judge_diff_2026-10-02.md` |
| 保存路径 | `reports/official_judge_botzone_2026-10-02.py` |
| 官方原版 SHA256 | `49a7346ddc0a06ebd8922e582d535fbc6fb5418ddea16b023a5f3384fc9b6fd9`（规范化文本） |
| 与附件 `裁判代码-修正版.py` 的 diff | `reports/official_judge_diff_2026-10-02.md`；还牌边界存在可执行差异 |
| 附件裁判 SHA256 | `fa63589d3f69ce9127093cec417f1635d8cc17205d80e70f03e44bc6809fc622` |
| 兼容结论 | 官方原版已取得但与附件修正版不一致；现有附件 oracle 结果不能代替官方平台验收 |

差异报告必须至少覆盖：请求/响应字段、阶段名称、牌号范围、claim 语义、贡还、抗贡、特殊牌、完成玩家、`pass_on` 和历史窗口。

## Bot 合同

| 字段 | 目标/已知状态 | 页面证据 |
|---|---|---|
| 语言/编译器 | 页面已选 `cpp17`（对应 BotZone G++ 7.2.0）；平台实际编译仍待验收 | 已确认页面设置；平台编译 `UNVERIFIED` |
| 交互方式 | 普通 JSON（`simpleio=false`） | 已确认页面设置 |
| 长时运行 | 开启（`enable_keep_running=true`）；每行响应，不等待 EOF | 已确认页面设置 |
| 回合时限/首回合时限 | 约 1 秒；以页面为准 | `UNVERIFIED` |
| CPU/内存 | 单核 / 页面上限（路线图目标 256 MB） | `UNVERIFIED` |
| 源码大小上限 | `< 4,000,000` bytes | `UNVERIFIED` |
| 模型存储空间/配额/路径 | `UNVERIFIED` | `UNVERIFIED` |

代码侧当前协议假设（必须与官方页面逐项核对）：`requests` 长度比 `responses` 多一；首个 `deal` 携带 `your_id`、27 张 `deliver` 和 `global`；后续阶段为 `deal/tribute/return/play`；出牌为 `[action, claim]`。

## 私有发布验收（仅候选冻结后）

1. 锁定源码单文件、模型二进制、manifest、规则 SHA、训练/评测种子；记录所有 SHA256。
2. version 1 已在用户授权下创建为不公开私有测试版本；后续新版本上传或线上设置变更仍须在操作前取得相应授权。
3. 保存平台编译器版本、完整编译日志、警告/错误、版本 ID。
4. 私有对局至少核验：模型加载 SHA、实际模型决策数、降级数、非法动作、超时、崩溃、p50/p95/p99/max、RSS、回放结果。
5. 只有官方裁判、平台编译、私有对局全部有证据且模型通过牌力门槛，才可将 `release_eligible` 改为 `true`；否则保留 `false` 并可回滚到上一候选。

## 当前本地证据

- RTX 5080 训练环境：[reports/training_environment.json](../reports/training_environment.json)
- 当前线上 anchor 与候选筛选：[docs/progress.md](progress.md)
- 官方原版已取得并完成初步逐行 diff；附件裁判仍仅作离线 oracle。
- 还牌边界已按官方源码补做定向兼容测试，`reports/official_return_boundaries.json` 13/13
  通过；官方 G++ 7.2 和平台私有对局仍未验证。
- 当前重建 anchor 单文件为 2,400,969 bytes，SHA256
  `a9ccf9504f653f68ae89a9934aac6fba77240154bcf3a10f9d19237ed66ef827`；官方裁判副本
  16+16 局同牌换座为模型 0/32 胜、配对点差 `-92`，实际模型决策 1,108，要求模型
  fallback/非法动作/本机预算超限均为 0。汇总见
  `reports/official_anchor_package_16x2_summary.json`。这些结果不等于 BotZone 平台验收，
  manifest 仍为 `release_eligible=false`；这里的“没有上传”仅指该 anchor 包，cf8 候选已作为
  私有 version 1 另行上传。
- 本地 WSL g++13.3 编译不等于 BotZone G++7.2 验收。
- 提交 version 1 之前曾固定协议验收候选 `dist/oxbot-bc-v2-full-fp32-anchor-stdin.cpp`
  （2,402,149 bytes，
  SHA256 `6d464064525eb483a38e5d48fc56c34e02201486867bf8cb683458996389b928`）；它已用临时
  G++ 7.2.0 和 6 项 stdin-open 协议检查通过，但未上传；`release_eligible=false`，只允许作为
  本地私有编译/协议验收参考，不能公开发布。实际私有 version 1 使用上方 version 1 上传记录所列
  cf8 单文件源码。

## 当前线上状态（2026-10-02）

- `OXbot5080Lab` 的 BotZone 私有 version `1` 已存在；普通 JSON、长时 stdin、不开源和不进天梯
  设置已核验。该版本未公开、未加入天梯、未替换 anchor。
- 两次 CAPTCHA 尝试均为 `captcha.wrong`，未创建私有 GuanDan 游戏桌；按用户要求已暂停后续
  验证码操作。
- 平台编译日志、BotZone 1 秒/RSS、模型加载和整局私有对局结果仍为 `UNVERIFIED`。
- 当前继续推进本地训练、规则/包审计和候选评测；Windows 人机、微信小程序、iOS/Android
  仍在平台验收完成前暂停。

## 2026-10-03 startup-fix version 2 与天梯复核

在恢复卡住的上传页面后，独立刷新 Bot 列表确认 `OXbot5080Lab` 已创建私有 version `2`：

- 时间：`2026-10-03 00:03:27`；描述：`cf8 candidate startup fix; deal fast path before model load; JSON C++17`；
- 上传源码：`dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-startupfix.cpp`，
  2,347,082 bytes，SHA256 `666b5c9583e4fe6c85b3b8e13d963de820a234a810ffa198bf7003761bed2370`；
- 页面仍显示 `source.cpp17`、G++ 7.2.0 (-O2, C++17)、普通 JSON、长时运行开启、不开源。

刷新后的“修改 Bot”页显示天梯控件为 active，排名分为 `972.40`。没有再次点击星标，因为页面
明确提示切换会将积分回归 1000。随后“查看天梯对局”页面在 `2026-10-03 00:04:02` 显示
`isyai2026 / OXbot5080Lab` 的 **Bot 版本号 2**，并有回放链接和 `7.88` 分数影响；回放路径为
`/match/6abfde4ce2453c4f471b2a23`。这给出了平台编译并实际启动 version 2 的页面证据。

本机对应的 GCC 7.2 产物为
`bin/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-startupfix-gcc72`，
SHA256 `99eb1c5e5132e732ae202aa4cdd7782b1e8fbd10ec9dcea0c8b2233559b133cb`；
`tests/test_protocol_process.py --require-model` 通过 6 项 stdin-open 检查。完整字段和限制见
[`reports/botzone_startupfix_online_2026-10-03.md`](../reports/botzone_startupfix_online_2026-10-03.md)。

这次对局证明了平台能编译并运行 version 2，但管理页面未提供完整编译日志、平台 CPU/RSS、模型
加载 SHA 或崩溃/超时统计；这些字段仍保持 `UNVERIFIED`，`release_eligible=false` 不变。
