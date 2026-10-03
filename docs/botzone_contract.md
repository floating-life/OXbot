# BotZone JSON 契约（P0）

正式 Bot 使用普通 JSON、单行输入、单行输出；当前线上候选启用 BotZone 长时运行模式。首次回合输入是完整 envelope，后续回合输入只有当前 request，进程在内存中恢复历史。普通首回合输入形如：

```json
{"requests":[{"stage":"play"}],"responses":[],"data":null,"globaldata":null}
```

实际游戏请求由掼蛋裁判提供；程序只依赖最新 request 和可回放的 `requests/responses`。输出始终是对象，`response` 是本回合动作，`debug` 仅记录短状态信息，不向 stdout 输出其他内容。

阶段响应：

- `deal`：`[]`
- `tribute` / `return`：`[card_id]`，抗贡时为 `[]`
- `play`：`[action, claim]`；跟牌时允许 `[[],[]]`，自由出牌禁止过牌

牌号必须是 `0..107`。牌型由 `claim` 决定；P0 出牌前检查 `action` 牌权、claim 覆盖关系、牌型和压制关系。所有模型动作都经过同一终检，失败时降级到 `RulePolicy`。

BotZone 目标约束：C++17、G++ 7.2.0、`-O2`、单核约 1 秒、内存 256 MB、源码 UTF-8 且小于 4 MB。训练依赖和 CUDA 不进入上传包。

长时运行时，每个 JSON response 后必须立即输出单独一行
`>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` 并 flush；否则平台会把输出视为未结束并判定超时。
首回合仍按完整 `requests`/`responses` envelope 处理，后续只有一条 request，程序将其追加到
进程内历史后再交给同一套状态重建逻辑。
