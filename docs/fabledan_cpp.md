# FableDan → OXbot C++17

竞赛训练仍使用 `competition/` 中的 FableDan / PyTorch 与 RTX 5080；BotZone 的
规则回放、特征编码、候选评分和响应全部在 C++17 中执行，不需要 Python、NumPy、
CUDA 或第三方推理库。当前导出的是已训练的 `competition/ckpts/real-v2/best.npz`。

## 手动上传当前候选

| 用途 | 文件 | 大小 |
|---|---|---:|
| Bot 源码 | [oxbot-fabledan-real-v2.cpp](../dist/oxbot-fabledan-real-v2.cpp) | 184,231 bytes |
| 用户存储权重 | [fabledan_w_85f1341f.fbd](../data/fabledan_w_85f1341f.fbd) | 17,391,837 bytes |
| 源码和权重绑定清单 | [manifest](../dist/oxbot-fabledan-real-v2.manifest.json) | JSON |

源码 SHA256：`bcb7182a6b43b2c1b0431a053320e31872fe9925f677cf2e1cb5b80167d501e9`。
权重文件 SHA256：`85f1341f65a9e72d573c2ce636e0eacb0ee1fff17b02b57daab3bab8a264add0`。

上传源码时选择 **G++ 7.2.0 / C++17**，使用普通 JSON（不勾选简单交互），启用长时运行。
将 `.fbd` 文件以原文件名上传到 BotZone 用户存储空间；程序读取的相对路径固定为
`data/fabledan_w_85f1341f.fbd`。源码与这份权重须一起使用，不能将 `.npz` 改名代替 `.fbd`。
源码小于 4 MB；权重采用用户存储，不嵌入源码。平台存储配额仍需在上传时确认。

`deal` 的 `response=[]`、`model_status=deferred` 是正常行为。到第一次 `play`，
调试信息应包含以下字段，才能证明真实模型参与了选牌：

```text
version=fabledan-real-v2-cpp-fp32
policy=model
model_status=model_selected
model_sha=ace1339157c5
legality_fallback=0
```

其中 `model_sha` 是权重 **payload** SHA 的前 12 位，与上面的整个文件 SHA 不同。
每条 JSON 后还会输出 `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` 保活标记，后续直接接收
raw request 并恢复完整历史。`deal`、进贡、还贡由阶段规则处理，`play` 由模型评分选牌，
最终仍进行合法性检查；权重加载或推理失败会在 debug 中明确显示降级。

## 本机构建与重新导出

在仓库根目录的 PowerShell 中运行：

```powershell
.\tools\build.ps1
```

这会通过 WSL 构建 `bin/oxbot`、三个 probe 和四项 C++ 测试。通用开发版
`bin/oxbot` 仍沿用旧模型默认路径；要运行 FableDan，可显式传入
`--model data/fabledan_w_85f1341f.fbd`，或使用上表中自带该路径的单文件包。

在 WSL、已激活含 NumPy 的 Python 环境时，可以按以下步骤重新导出和打包。
命令从仓库根目录执行，`npz` 可以换成后续同架构模型；文件名由实际 SHA 派生。

```bash
python competition/tools/export_fabledan_cpp.py \
  competition/ckpts/real-v2/best.npz dist/fabledan-real-v2.fbdn \
  --dtype fp32 --manifest dist/fabledan-real-v2.fbdn.json
oxbot_weight_sha=$(sha256sum dist/fabledan-real-v2.fbdn | cut -c1-8)
oxbot_weight_path="data/fabledan_w_${oxbot_weight_sha}.fbd"
cp dist/fabledan-real-v2.fbdn "$oxbot_weight_path"
python tools/amalgamate.py --output dist/oxbot-fabledan-real-v2.cpp \
  --model-path "$oxbot_weight_path" --candidate-version fabledan-real-v2-cpp-fp32
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
  dist/oxbot-fabledan-real-v2.cpp -o bin/oxbot-fabledan
```

导出器和 C++ 加载器接受本项目四层架构：128 hidden、4 heads、48 个 token、512 token
历史上限。旧80维特征使用FBDN001/version1；新增224维花色/出后结构特征使用
FBDN002/version2，必须搭配重新构建的C++源码，旧单文件包不支持新格式。
新特征、迁移和验收见 [组牌特征v2](../competition/docs/STRUCTURE_FEATURES.md)。
其他架构改变需要同步修改两端，不能只换权重。
导出器也支持 FP16 存储；当前候选固定使用 FP32，以保持本次已验证的数值和 SHA。
新权重需重新执行下列验收，不沿用旧模型的结果。

## 本地验收入口

先执行完整构建，再检查网络、真实进程和官方裁判：

```bash
bash tools/build_wsl.sh
python competition/tools/check_fabledan_encoding.py \
  --probe bin/core_probe --report reports/fabledan_encoding_parity.json
python competition/tools/check_fabledan_candidates.py \
  --probe bin/core_probe --report reports/fabledan_candidate_parity.json
python competition/tools/check_fabledan_cpp.py \
  --npz competition/ckpts/real-v2/best.npz \
  --weights data/fabledan_w_85f1341f.fbd --probe bin/fabledan_probe \
  --tolerance 0.0001 --report reports/fabledan_cpp_network_parity.json
python tests/test_protocol_process.py --bot bin/oxbot-fabledan --require-model
python competition/tools/judge_runner.py \
  --judge competition/judge/judge_official.py \
  --driver 'cpp:bin/oxbot-fabledan' --bot-cwd . \
  --games 2 --scenario mix --positional --require-model --fail-fast \
  --report reports/fabledan_cpp_official_process.json
```

网络检查覆盖最长 512 token、256 个候选，报告保留模型/probe SHA 与误差。
`cpp:` 驱动按真实长时运行协议检查完整牌局；`--require-model` 会拒绝出牌阶段的规则降级，
不会把 `deal` 的延迟加载误判成加载失败。官方裁判小样本只检查集成，不能证明竞技强度。

当前验证及边界见 [C++ 迁移验收记录](../reports/fabledan_cpp_migration.md)。
当前 `release_eligible=false`：目标平台的完整局 CPU 时间、RSS 和相对旧版的强度评测仍未完成。
本机 G++ 7.2 编译与长时进程通过，不能替代 BotZone 上的运行验收。
