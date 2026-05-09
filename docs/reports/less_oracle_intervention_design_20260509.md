# Less-oracle Intervention Design for GLM and Llama

更新时间：2026-05-09

本设计回答的问题是：

> GLM / Llama 的 regime classification 是否只是 `baseline_state_interpolation` 单一方法的 artifact？

建议使用的 less-oracle alternative 是：

- `pressure_subspace_damping` (`PSD`)

## 1. Current Artifact Check

### GLM-4-9B

`PSD n=100` 已存在，且是正式 clean protocol：

- [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/belief_causal_summary.csv)
- [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/projection_alignment_summary.json)

关键值：

- `alpha = 0.75`
- `drift_delta = -0.01`
- `compliance_delta = -0.01`
- `recovery_delta = +0.12`
- `damage = 0.06`

### Llama-3.1-8B

`PSD n=100` 已存在，但 formal closure 是 `alpha = 0.5`：

- [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_n100/meta-llama_Llama-3.1-8B-Instruct/20260504_025609/belief_causal_summary.csv)

关键值：

- `alpha = 0.5`
- `drift_delta = +0.03`
- `compliance_delta = +0.01`
- `recovery_delta = -0.03`
- `damage = 0.08`

另有旧 exploratory alpha sweep：

- [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_summary.csv)

该 sweep 覆盖：

- `alpha = 0.15, 0.25, 0.35, 0.5`
- `n = 24`

当前未发现现成 `Llama PSD alpha=0.75 n=100`。

## 2. Design Decision

### GLM

不需要再新跑 exploratory。

理由：

- 已有 `PSD n=100 alpha=0.75`
- 已足以证明 GLM 在 less-oracle 方法下仍不是 clean-controllable

### Llama

优先复用已有 formal + exploratory，而不是先新开一条重复 run。

建议分两层证据：

1. formal row: `n=100, alpha=0.5`
2. exploratory support: `n=24, alpha sweep up to 0.5`

只有当审稿人或下游 insist on matched `alpha=0.75` 时，才补一条：

- `n=24`
- `alpha=0.75`
- same layers `24-31`

## 3. Recommended Readout

统一报告：

- `drift_delta`
- `compliance_delta`
- `recovery_delta`
- `damage`

可选补一个：

- `clean_control_score = -(drift_delta + compliance_delta) + recovery_delta - 10 * damage`

## 4. Expected Regime Conclusion

### GLM

在 PSD 下仍应保持：

- `weak / tradeoff-limited`
- 或 `damage-prone boundary`

关键点不是“完全无效”，而是：

- 有少量正向 movement
- 但 clean window 不成立

### Llama

在 PSD 下仍应保持：

- `locatable but not controllable`
- 或 `not-controllable`

关键点是：

- 行为改善不成立
- damage / null effect 先出现

## 5. Optional Exploratory Backfill for Llama

只有在需要 matched `alpha=0.75` 时，才补这条：

| 参数 | 值 |
| --- | --- |
| model | `meta-llama/Llama-3.1-8B-Instruct` |
| method | `pressure_subspace_damping` |
| layers | `24-31` |
| `k` | `2` |
| `alpha` | `0.75` |
| `n` | `24` |
| prompt variant | `english` |

## 6. Script Reuse

直接复用：

- [scripts/run_belief_causal_transfer.py](/Users/shiqi/code/graduation-project/scripts/run_belief_causal_transfer.py)

### Optional Llama n=24 alpha=0.75 command

```bash
cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device mps \
  --dtype auto \
  --layers 24-31 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --prompt-variant english \
  --train-n 24 \
  --eval-n 24 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260509 \
  --output-root outputs/experiments/llama31_8b_belief_causal_transfer_alpha075_n24 \
  --log-level INFO
```

## 7. Preferred Reporting Strategy

文稿与图表里优先写：

- GLM: PSD `n=100` already confirms less-oracle tradeoff-limited behavior
- Llama: PSD `n=100` already confirms non-controllability; optional matched-alpha backfill is exploratory only

不要为了对齐参数而把 exploratory row 升格到 formal 级别。

## 8. Bottom Line

这份设计的核心不是要求再跑一轮“看起来更对称”的实验，而是先充分复用已有 PSD 证据：

- GLM: 已足够
- Llama: 已足够支撑 regime consistency；如需 `alpha=0.75` 只补一条小型 exploratory

因此最稳的 Architecture 建议是：

- `reuse-first`
- `only backfill Llama alpha=0.75 if strictly needed`
