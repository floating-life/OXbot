# 本机训练路线

本机 RTX 5080（约 16 GB 显存）承担 BC、后续 DMC learner 和离线评测。已建立 WSL2 Ubuntu 24.04 隔离环境 `/home/ggcle/.venvs/oxbot`，Python 3.12.3、PyTorch 2.11.0+cu128；sm_120、BF16 和真实梯度更新已通过环境检查。确切版本见 `requirements.lock.txt`，证据见 `reports/training_environment.json`。训练依赖不进入 BotZone。

## 阶段

1. 受限 pickle ETL、牌号映射、逐局回放和规则差异审计。
2. 按比赛包切分 32/7/7，训练 64 维、2 层 causal Transformer 候选 Q 模型做 BC。
3. 导出版本化权重，逐层对齐 Python/C++，并在单核 CPU 测 p50/p95/p99。
4. C++ headless simulator 产生 DMC rollout；GPU 训练、CPU 多进程 actor，先 1–2 万局烟测，再扩大规模。

模型训练规模可以利用 5080，但 BotZone 首发模型仍以 1 秒时限、源码/存储限制和合法率为硬门槛。

## 当前工具

- `etl_njupt.py`：受限 pickle 解析，逐事件牌权回放，输出动作前信息集，按比赛包切分；原始数据只读。
- `prepare_bc.py`：通过 C++ 核心生成合法候选，多种兼容 claim 均作为正标签；跟牌条件不确定的样本跳过并计数。训练最多采样 128 个负候选；验证/测试保留全部候选。
- `train_bc.py`：只读 train/validation，使用多正例边缘似然训练；按验证集 NLL 保存最佳 checkpoint，测试集不参与挑选。

开发期动作效用探针：`audit_action_utility.py` 和
`build_utility_targets.py` 只读取 NJUPT 的 train/validation，利用最终胜负
构造不进入模型输入的 `team_win` 目标。将其传给 `train_bc.py` 的
`--utility-targets` 并设置 `--utility-alpha` 可做 advantage-weighted BC
筛选；默认 `--utility-alpha 0` 保持旧训练路径。sidecar 的原始 manifest 与
split SHA 会在训练启动时和 prepared manifest 交叉校验，held-out test 不得读取。
- `export_model.py`：导出 SHA、规则/特征/架构契约与 float32 二进制。C++ 在载入时检查全部张量形状和校验和。
- `tests/check_feature_parity.py`、`tests/check_network_parity.py`：分别核对特征和前向；后者支持 `--checkpoint` 验证实际训练权重。

## Candidate continuation sidecar（开发审计）

`build_candidate_credit_targets.py`（文件名沿用早期称呼）可生成
`oxbot-candidate-continuation-targets-v1` sidecar。它只读取 train/validation，沿公开
续演检查候选是否仍允许第一条后续动作；`+1/-1` 表示 continuation-compatible/
incompatible，不能称为 credit 或 Q。使用 `--max-rows-per-split N` 时是审计模式，输出
`trainable=false`；当前 sidecar 尚未接入 `train_bc.py`，也不进入 BotZone 包。审计报告见
`reports/candidate-continuation-target-plan.md`。

### 2026-10-02 工程化对齐与 cont005 筛选

上段描述的是早期 128 行审计输出；当前开发训练使用已对齐的
`reports/candidate_continuation_targets_v1.json`（SHA256
`aef35f97fb0de22a6ca82019f4bafc373cfb190354696e9072248fc715d981e0`）。它只读取
train/validation（`test_used=false`），对齐后 train/validation 分别为 13,512/4,655
行；prepared shard 候选数 mismatch=0，train 另有 2 个不在 prepared shard 的 ID 被过滤。
`train_bc.py` 现在对 sidecar 的 split、ID、候选数和 provenance 做 fail-closed 校验，并
以 lambda=0.05 加入 continuation pairwise loss；`tests/test_continuation_alignment.py`
验证该契约。sidecar 只用于训练和离线诊断，不随 BotZone 源码/模型包发布。

cont005（8 epoch、RTX 5080 FP32）的最佳 checkpoint 是
`models/bc-v2-full-fp32-cont005/best.pt`，SHA256
`671437b6a272b7549d1c8fc8673a474c6f4e2b7418e14ba42bb9b5aca60050d0`；validation NLL
为 `1.190878`，raw/choice accuracy 为 `63.416%/45.752%`，hard-pair accuracy
`73.812%`。导出 `models/oxbot-bc-v2-full-fp32-cont005.bin`（文件 SHA256
`666881c91a149fee097444c6ec9d45bb69c7faa84f04176eb63279e5ccfce007`，payload SHA256
`2b03e119b4e2dc12fa08dee850c4724dd073d5e9b98e6e64d328adcf0db67a6a`）与 Python 网络最大
误差 `3.814697e-6`，真实 validation 128/128 argmax 一致；报告见
`reports/cont005_network_parity.json` 和 `reports/cont005_trained_observation_parity.json`。

固定 seed `20261020` 的 16 组同牌换座、两队共 32 局中，cont005 胜 `0/32`、总点差
`-95`（平均 `-2.96875`，bootstrap 95% CI `[-3.0, -2.90625]`），明确淘汰，不替换
anchor、不生成发布包、不上传 BotZone。当前 anchor `models/bc-v2-full-fp32` 保持不变；
RTX 5080（PyTorch 2.11.0+cu128、sm_120）只承担本机训练和离线评测，训练 sidecar 与
checkpoint 均不上传。
