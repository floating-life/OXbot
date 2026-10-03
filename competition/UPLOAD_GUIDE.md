> **OXbot 当前路线**：使用 C++17 源码和配套 FBDN001 外部权重。训练继续使用本目录的 Python/PyTorch；Python 源码 zip 保留为对照部署方式。
> 完整重建和手动上传步骤见 [C++ 迁移说明](../docs/fabledan_cpp.md)。

> **C++ 迁移包**：源码为根目录的 [oxbot-fabledan-real-v2.cpp](../dist/oxbot-fabledan-real-v2.cpp)，
> 用户存储权重为 [fabledan_w_85f1341f.fbd](../data/fabledan_w_85f1341f.fbd)。选择 C++17，
> 使用普通 JSON；权重上传时保持文件名，程序通过 `data/fabledan_w_85f1341f.fbd` 读取。
> 源码和权重须与 [manifest](../dist/oxbot-fabledan-real-v2.manifest.json) 的 SHA 记录对应。
> 本地构建与 parity 不代表平台验收，当前仍为 `release_eligible=false`，由用户手动上传并保留旧版本回滚。

> **Python 对照包**：以下各节只适用于 NumPy/Python 版本。默认权重约 16 MB，超过 4 MB
> 源码上限，使用“代码 zip + `fabledan_w_<sha8>.npz`”，完整训练步骤见
> [TRAINING_5080.md](docs/TRAINING_5080.md)。

# Deploying FableDan to Botzone

[Botzone](https://www.botzone.org.cn/) hosts a rated GuanDan ladder. This guide
covers packaging and uploading the Python reference bot; use the paired C++
files above for the OXbot C++ migration.

## 1. Package the submission zip

The bot is pure Python + NumPy (which Botzone's `python3` environment provides),
so no PyTorch is needed at deploy time. From the `competition/` directory:

```bash
# Keep the default transformer weights separate from source
python botzone/pack_bot.py --weights ckpts/real-v2/best.npz --out dist/OXbot-real-v2.zip
# -> dist/OXbot-real-v2.zip       (code only)
# -> dist/fabledan_w_<sha8>.npz  (upload to user storage, keep the filename)
```

The zip contains `__main__.py`, the pure-Python `fabledan/` modules, and
`weights_name.txt`, which pins its matching weight file. Training exports
`best.npz` and `latest.npz`; it does not automatically package a submission.
`--embed-weights` is only suitable for small custom models below the source limit.

## 2. Create the bot

1. Sign in at <https://www.botzone.org.cn> and open **My Bots**.
2. Click **Create a new Bot** and fill in the form:
   - **Name**: e.g. `FableDan`
   - **Game**: **GuanDan**
   - **Source code**: upload `dist/OXbot-real-v2.zip`
   - **Compiler / language**: **python3** (the highest python3 version in the
     list). A `.zip` whose root contains `__main__.py` is Botzone's multi-file
     Python upload format.
   - Leave **"simple interaction" unchecked** — this bot uses the JSON protocol.
3. Upload the paired `dist/fabledan_w_<sha8>.npz` to user storage without renaming it.
4. Submit and inspect the platform's compilation and match results.

## 3. Verify it plays correctly

1. On the bot page, start a test match on GuanDan — you can fill all four seats
   with FableDan (self-play).
2. Open the replay:
   - A complete play-by-play means the protocol is correct.
   - In the first-turn response's `debug` field, look for `model=transformer`
     (trained model) or `model=mlp` (demo weights). `model=rule` means the
     weights were not loaded — see troubleshooting below.
3. When it works, enable it on the ladder (the GuanDan ranked queue schedules
   rated matches automatically).

## 4. Update to a new version

In **My Bots**, open FableDan → **Upload new version**, choose the latest
`dist/OXbot-real-v2.zip` and its paired versioned weights, keeping the compiler on **python3**. The ladder
keeps the bot's history and score; no need to recreate it.

## 5. Runtime environment

- Single-core CPU, 256 MB memory cap.
- Per-turn time limit with a python multiplier; the first turn is relaxed.
- `python3` ships NumPy (this bot's only dependency); no PyTorch.
- With long-running mode enabled, weights load once per process.
- Check first-turn load time and per-decision CPU/RSS on Botzone for the exact package; local timings are not platform measurements.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Python zip compile failure | Select **python3** for this zip; the separate `.cpp` package uses C++17 |
| First-turn timeout | Zip too large or slow weight load — use the separate-weights option |
| `debug` shows `model=rule` | Confirm the user-storage file name matches `weights_name.txt` and is readable under `data/` |
| Illegal-move loss | Capture the match log (full request/response) so it can be reproduced and fixed |
| Zip exceeds source-size limit | Use separate versioned weights; upload `dist/fabledan_w_<sha8>.npz` to user storage without renaming it |
