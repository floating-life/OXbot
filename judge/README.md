# 裁判 oracle 记录

工作区同时保留用户提供的派生修正版和已从 BotZone 登录会话取得的官方源码副本。

- SHA-256：`FA63589D3F69CE9127093CEC417F1635D8CC17205D80E70F03E44BC6809FC622`
- 状态：可用于离线规则审计和 golden 生成；**不是已确认的 BotZone 线上原版**。
- 官方源码副本：`../reports/official_judge_botzone_2026-10-02.py`
- 官方源码规范化 SHA-256：`49a7346ddc0a06ebd8922e582d535fbc6fb5418ddea16b023a5f3384fc9b6fd9`
- 官方来源：BotZone GuanDan 游戏页的“裁判代码”弹窗；游戏 ID `65490c16ec1ab1389702dced`。
- 官方与附件的逐行差异：`../reports/official_judge_diff_2026-10-02.md`
- 约束：不得将“原版与修正版一致”写入测试结论，直到原版证据存在。

官方原版与附件在还牌边界上存在可执行差异；因此附件仍只作离线修正版 oracle，平台兼容性验收必须再用官方原版语义核对。
