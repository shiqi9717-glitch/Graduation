# Mistral-7B Held-out Regime Prediction Design

更新时间：2026-05-08

本设计把 `mistralai/Mistral-7B-Instruct-v0.3` 纳入 diagnostic framework 的新 held-out setting。目标不是现在就跑，而是让 Code 部门明确：

- M1: 如何补齐 projection-to-logit diagnostic
- M2: 如何补跑 English `n=100` behavioral closure
- 在 M1/M2 之前，我们依据现有 frozen exploratory evidence 预判它会落在哪个 regime

## 1. Current Frozen Starting Point

### English bridge exploratory (`20260426_143645`)

路径：
[20260426_143645](/Users/shiqi/code/graduation-project/outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_143645)

当前已知：

- `drift delta = -0.250`
- `compliance delta = -0.208`
- `recovery delta = +0.583`
- `damage = 0.375`
- layers `24-31`, `k=2`, `alpha=0.75`, `prompt_variant=english`
- subspace EV 在 layer `27-28` 附近最高，`explained_variance_sum ~ 0.73`
- `direction_coherence ~ 0.15-0.20`

### Chinese bridge exploratory (`20260426_144430`)

路径：
[20260426_144430](/Users/shiqi/code/graduation-project/outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_144430)

当前已知：

- `drift delta = -0.042`
- `compliance delta = -0.042`
- `recovery delta = +0.125`
- `damage = 0.417`
- 与 English 保持相同 layers / `k` / `alpha`
- `prompt_variant=zh_instruction`

### Missing diagnostic artifacts

两条 exploratory run 都缺：

- `projection_alignment_summary.json`
- `projection_alignment_diagnostic.csv`

## 2. Diagnostic-Only Prediction Before M1/M2

在没有 logit-specificity 文件前，最稳的预判是：

- English exploratory: `logit-specific but damage-prone`
- Chinese exploratory: `pressure-prompt unstable / damage-prone`
- Overall Mistral held-out prediction: `logit-specific but damage-prone`, with strong prompt-variant instability

理由：

- 行为方向在 English 上明显是正向的，但 `damage = 0.375` 已经远高于 clean-control 阈值。
- Chinese 变体的行为信号显著变弱，而 damage 反而更高，说明 prompt stability 很差。
- subspace EV `0.70-0.73` 说明方向并非不可定位；真正可疑的是它是否只在 target logit 上“看起来 specific”，但在 behavior 上以高 damage 为代价。

## 3. M1 Design: Projection-to-logit Diagnostic

### Goal

为 English / Chinese 两个已有 frozen run 各自补出：

- `projection_alignment_summary.json`
- `projection_alignment_diagnostic.csv`

### Why M1 can be post-hoc

现有 Mistral run 目录里已经有：

- `hidden_state_records.jsonl`
- `belief_causal_records.jsonl`
- `belief_causal_comparisons.jsonl`
- `belief_causal_subspace_artifact.npz`
- `pressure_pairs_train.jsonl`
- `pressure_pairs_eval.jsonl`

因此 M1 不需要重跑 Mistral 本体；可以做只读 exporter / finalizer。

### Recommended script

优先复用已有：
[scripts/resume_belief_causal_transfer_eval.py](/Users/shiqi/code/graduation-project/scripts/resume_belief_causal_transfer_eval.py)

原因：

- 该脚本会读取现有 `hidden_state_records`、`belief_causal_records`、`belief_causal_comparisons`、`belief_causal_subspace_artifact`
- 若没有 pending eval item，它不会重新跑模型，只会补写 summary / projection artifacts
- 它本身就调用 `run_belief_causal_transfer.py` 里的 `_projection_alignment_outputs(...)`

### M1 command template: English

这条命令不依赖 MPS，可在普通 shell 中做只读补写：

```bash
cd /Users/shiqi/code/graduation-project
./.venv/bin/python scripts/resume_belief_causal_transfer_eval.py \
  --run-dir outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_143645 \
  --model-name mistralai/Mistral-7B-Instruct-v0.3 \
  --device cpu \
  --dtype auto \
  --k 2 \
  --alpha 0.75 \
  --log-level INFO
```

### M1 command template: Chinese

```bash
cd /Users/shiqi/code/graduation-project
./.venv/bin/python scripts/resume_belief_causal_transfer_eval.py \
  --run-dir outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_144430 \
  --model-name mistralai/Mistral-7B-Instruct-v0.3 \
  --device cpu \
  --dtype auto \
  --k 2 \
  --alpha 0.75 \
  --log-level INFO
```

### Expected M1 outputs

输出路径建议直接写回原 run 目录：

- English:
  - `.../20260426_143645/projection_alignment_summary.json`
  - `.../20260426_143645/projection_alignment_diagnostic.csv`
- Chinese:
  - `.../20260426_144430/projection_alignment_summary.json`
  - `.../20260426_144430/projection_alignment_diagnostic.csv`

### What M1 verifies

M1 主要回答三件事：

- `belief_logit_delta` 是否足够强
- `negative_logit_delta` 是否接近 0、显著为负、还是错误地为正
- `specificity_ratio` 是否很高但仍与高 damage 共存

## 4. M2 Design: n=100 Behavioral Closure

### Goal

补一条 English-only 的正式 held-out closure：

- model: `mistralai/Mistral-7B-Instruct-v0.3`
- method: `pressure_subspace_damping`
- layers: `24-31`
- `k=2`
- `alpha=0.75`
- `n=100`
- `prompt_variant=english`

### Why choose English

- English exploratory 的 directional signal 明显强于 Chinese
- Chinese 行已经表现出 prompt-instability boundary，不适合作为第一条 formal held-out closure
- 若 English `n=100` 仍高 damage，framework 就能更稳地把 Mistral 压到 `damage-prone` 而不是“只是中文写坏了”

### Recommended script

直接复用：
[scripts/run_belief_causal_transfer.py](/Users/shiqi/code/graduation-project/scripts/run_belief_causal_transfer.py)

已有 Mistral `n=24` shell 脚本：
[scripts/run_mistral7b_belief_causal_transfer_mps.sh](/Users/shiqi/code/graduation-project/scripts/run_mistral7b_belief_causal_transfer_mps.sh)

M2 只需把 `train_n/eval_n` 扩到 `100/100`，并换输出根目录。

### M2 macOS Terminal command template

这条命令涉及 MPS，应由用户在普通 macOS Terminal 中执行：

```bash
cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name mistralai/Mistral-7B-Instruct-v0.3 \
  --device mps \
  --dtype auto \
  --layers 24-31 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --prompt-variant english \
  --train-n 100 \
  --eval-n 100 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260508 \
  --output-root outputs/experiments/mistral7b_belief_causal_transfer_n100_english \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO
```

### Expected M2 outputs

至少需要：

- `belief_causal_summary.csv`
- `belief_causal_subspace_summary.csv`

按当前脚本能力，实际上还会自动额外给出：

- `projection_alignment_summary.json`
- `projection_alignment_diagnostic.csv`
- `belief_causal_run.json`
- `belief_causal_records.jsonl`
- `belief_causal_comparisons.jsonl`

## 5. Expected Comparison Table

| Setting | Diagnostic signals (before M1/M2) | Predicted regime | What M1/M2 will verify |
| --- | --- | --- | --- |
| Mistral-7B English exploratory n=24 | EV peak `0.73`; coherence `0.15-0.20`; drift/compliance/recovery direction positive; damage `0.375` | logit-specific but damage-prone | M1 will test whether target-logit movement is strong and whether specificity coexists with high damage |
| Mistral-7B Chinese exploratory n=24 | same layer family; much weaker drift/compliance/recovery; damage `0.417` | prompt-unstable, damage-prone boundary | M1 will test whether Chinese row loses target-logit strength or keeps specificity while remaining behaviorally poor |
| Mistral-7B English n=100 planned closure | same layers `24-31`, `k=2`, `alpha=0.75`; English chosen for stronger signal | likely logit-specific but damage-prone | M2 will test whether high damage persists at scale, or whether exploratory damage was a small-sample artifact |

## 6. Decision Logic After M1/M2

### If M1 shows

- strong `belief_logit_delta`
- near-zero or mildly negative `negative_logit_delta`
- very high `specificity_ratio`

and M2 still shows `damage > 0.10`, then Mistral becomes the cleanest held-out example of:

- `logit-specific but damage-prone`

### If M1 shows weak target-logit movement

then Mistral should be downgraded toward:

- `locatable but not controllable`

### If M2 n=100 sharply reduces damage

and keeps directional gains, then Mistral could move upward to:

- `secondary-controllable`

但以当前 frozen n=24 evidence 看，这不是默认预期。

## 7. Bottom Line

当前最稳的设计判断是：

- M1 应先用只读补写方式把 Mistral exploratory rows 的 projection diagnostic 补齐
- M2 再用 English `n=100` 验证“高 damage 是稳定 regime，还是小样本偶然”
- 在 M1/M2 之前，Mistral 的最佳先验标签不是 `clean-controllable`，而是 `logit-specific but damage-prone`
