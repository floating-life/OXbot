# OXbot GuanDan

## FableDan 竞赛版

按当前竞赛路线，[competition](competition) 提供 FableDan 训练、NumPy 参考推理和
Python 对照包，面向 Ryzen 7 9800X3D 加 RTX 5080。其 `real-v2` 权重已接入独立的
C++17 FBDN001 推理路径，部署产物见下方“C++ FableDan 迁移包”。Windows 和 WSL
训练脚本、官方裁判检查仍在 `competition/`；旧 OXGDQ001 C++ 模型保留用于对照。

先读 [competition/docs/TRAINING_5080.md](competition/docs/TRAINING_5080.md)，再运行
`competition/scripts/setup_windows.bat` 或 `competition/scripts/setup_wsl.sh`。

当前已实现 C++17 规则核心、状态回放、普通 JSON、候选动作模型和纯 C++ 推理。RTX 5080 已完成
cf8 rollout-Q 候选训练与导出；startup-fix 单文件已在用户授权下创建为 BotZone 私有 version 2，
并在刷新后的天梯页面实际参加了一场对局。平台完整编译日志、1 秒/RSS、模型 SHA 和私有整局
统计仍未齐全，因此继续保持 `release_eligible=false`，不公开、不替换 anchor。
BotZone CAPTCHA/建桌流程按用户要求暂停，当前继续进行本地包审计、规则回归和候选评测。

当前线上验收候选（BotZone 私有 version 2）：
`dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-startupfix.cpp`
（2,347,082 bytes，SHA256
`666b5c9583e4fe6c85b3b8e13d963de820a234a810ffa198bf7003761bed2370`）。
候选与平台状态的完整证据见 `reports/botzone_startupfix_online_2026-10-03.md`、
`docs/online_verification.md`、`docs/progress.md` 和 `docs/cf8_local_audit.md`。

针对 BotZone 长时运行 TLE 的待上传协议修复包：
`dist/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012-keep-running-fix.cpp`
（2,349,844 bytes，SHA256
`c51ef00cdb8351ab43ffa1c3af1a81c22701324554546e99fb9286e65d0998fa`）。
它在每个 JSON 响应后输出保活标记，并在后续 raw request 到达时恢复完整历史；当前线上 version 2
保持不变，需由用户手动上传此新单文件。

## C++ FableDan 迁移包

重建、验证与手动上传步骤见 [C++ 迁移说明](docs/fabledan_cpp.md)。

已新增独立的 C++17 FableDan 推理路径，默认 OXGDQ001 模型和现有线上包不变：

- 源码：[dist/oxbot-fabledan-real-v2.cpp](dist/oxbot-fabledan-real-v2.cpp)，184,231 bytes，SHA256
  `bcb7182a6b43b2c1b0431a053320e31872fe9925f677cf2e1cb5b80167d501e9`。
- 配套用户存储权重：[data/fabledan_w_85f1341f.fbd](data/fabledan_w_85f1341f.fbd)，FBDN001/FP32，
  17,391,837 bytes，文件 SHA256
  `85f1341f65a9e72d573c2ce636e0eacb0ee1fff17b02b57daab3bab8a264add0`。
- 训练原始导出和 FBDN 转换清单在 `competition/ckpts/real-v2/best.npz`、
  `dist/fabledan-real-v2.fbdn.json`；C++ 运行时只读 `data/` 下的 FBDN 文件，不解析 NPZ。

手动上传时，源码与权重必须和同一包的 manifest 对应：C++ 源码上传到 BotZone 的 C++17 版本，
`fabledan_w_85f1341f.fbd` 上传到用户存储并保持文件名，程序通过 `data/` 路径读取。当前包仍标记
`release_eligible=false`，本地 parity/编译通过不等于平台编译或强度验收；不要覆盖线上版本。

## 本地构建

Windows 工作区推荐通过已安装的 WSL Ubuntu 24.04 构建：

```powershell
.\tools\build.ps1
```

或在 WSL 中：

```bash
bash tools/build_wsl.sh
```

产物包括 `bin/oxbot`、离线诊断 probe 和规则/状态/JSON 测试。Bot 每收到一行即输出 JSON
响应和保活标记；stdin 关闭后退出，长时模式则继续接收请求。
通用开发构建默认读取历史 `models/oxbot-bc-v1.bin`；本地可用 `--model 路径` 指定权重，
`--model ''` 显式测试规则策略。BotZone cf8 验收使用独立的嵌入式单文件包，不依赖运行目录模型。

`tools/amalgamate.py` 可生成单文件源码。旧 OXGDQ001 模型可用 `--embed-model 模型.bin`，
FableDan FBDN001 使用 `--model-path data/配套权重.fbd`。打包器强制源码小于 4,000,000 字节，
但不会将打包成功标成上线验收通过。平台完整编译日志、账号实战统计和时限/RSS 仍须继续记录；
一次天梯对局只证明 version 2 已被平台编译并启动。

## 当前边界

- `裁判代码-修正版.py` 保持原样，仅作为离线修正版参考；当前官方源码副本与差异记录见
  `reports/official_judge_botzone_2026-10-02.py` 和 `reports/official_judge_diff_2026-10-02.md`。
- 南邮原始训练数据只读，不进入源码包；比赛包切分、牌权回放、历史信息和不确定 claim 均单独审计。
- Windows UI、微信小游戏、iOS/Android 等工作在 BotZone 模型版本通过后启动。

计划见 `docs/roadmap-v3.md`，实际证据与未完成项见 `docs/progress.md`。
