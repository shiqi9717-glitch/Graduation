# Component-level n=50 Design for Qwen 7B

更新时间：2026-05-09

本设计用于把已有的 Qwen 7B component-level exploratory 从 `n=24` 轻量扩到 `n=50`。

目标不是做更大模型复制，而是增强下面这条更窄的 claim：

> On Qwen 7B, the clean-control effect is visible at the full-residual level, while isolated `attention_only` and `mlp_only` slices remain near-null under the current patch interface.

## 1. Existing n=24 Baseline

现有 Qwen 7B component-level 结果：

- [component_level_delta_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/component_level_intervention/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_211828/component_level_delta_summary.csv)

关键值：

- `full_residual`: `drift=-0.4167`, `compliance=-0.4167`, `recovery=+0.4583`, `damage=0`, `score=1.2917`
- `attention_only`: near-null
- `mlp_only`: near-null

## 2. Experimental Scope

仅做：

- model: `Qwen/Qwen2.5-7B-Instruct`
- layers: `24-26`
- beta: `0.6`
- `n=50`

不扩到：

- GLM / Llama / Mistral
- 多 beta sweep
- 多 layer window

## 3. Settings

四种 setting 保持不变：

1. `no_intervention`
2. `full_residual`
3. `mlp_only`
4. `attention_only`

## 4. Metrics

统一输出：

- `drift_delta`
- `compliance_delta`
- `recovery_delta`
- `damage`
- `clean_control_score`

其中：

`clean_control_score = -(drift_delta + compliance_delta) + recovery_delta - 10 * damage`

## 5. Expected Result

建议冻结的预期是：

- `full_residual` 继续保持显著正 `clean_control_score`
- `mlp_only` 继续 near-null
- `attention_only` 继续 near-null

论文级表述边界：

- 当前 patch interface 下，clean-control effect is concentrated at the full residual intervention level
- 不把这写成 final proof that MLP and attention are mechanistically absent

## 6. Script Reuse

已有脚本可直接复用：

- [scripts/run_component_level_intervention.py](/Users/shiqi/code/graduation-project/scripts/run_component_level_intervention.py)

该脚本当前已支持：

- `full_residual`
- `attention_only`
- `mlp_only`
- 自动输出 `component_level_delta_summary.csv`

## 7. Command Template

```bash
cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_component_level_intervention.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype auto \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --prompt-variant original \
  --eval-n 50 \
  --layers 24,25,26 \
  --beta 0.6 \
  --seed 20260509 \
  --output-root outputs/experiments/component_level_intervention/qwen7b_n50 \
  --log-level INFO
```

## 8. Output Files

最重要的输出：

- `component_level_summary.csv`
- `component_level_delta_summary.csv`
- `component_level_records.jsonl`
- `component_level_manifest.json`

Analysis 最终至少需要：

| Component | Drift Δ | Compliance Δ | Recovery Δ | Damage | Clean-control score |
| --- | --- | --- | --- | --- | --- |
| full_residual | TBD | TBD | TBD | TBD | TBD |
| attention_only | TBD | TBD | TBD | TBD | TBD |
| mlp_only | TBD | TBD | TBD | TBD | TBD |

## 9. Success Criterion

若 `n=50` 仍复现：

- `full_residual` 正 score
- `attention_only` / `mlp_only` near-null

就足以把 component-level 口径从“单次小样本 exploratory”提升到“lightweight but replicated exploratory”.

## 10. Boundary

- 不要把 `n=50` 写成 pathway-final proof。
- 不要把 `attention_only` / `mlp_only` near-null 写成“attention/MLP 不参与机制”。
- 更准确的说法是：在当前 hook interface 和 tested window 下，isolated component-level patching does not reproduce the clean full-residual effect.
