# Mistral And Llama Boundary Update

更新时间：2026-05-08

本说明只基于最新冻结 artifacts 做边界更新，不跑新实验，不改代码，不改论文正文。

## Short Comparison

| model | strongest positive signal | main failure mode | safest regime label |
| --- | --- | --- | --- |
| Mistral-7B | belief-logit shift is stably strong and moderately specific across English, Chinese, and n=100: `-5.68 / -0.30`, `-5.60 / -0.34`, `-5.80 / -0.25` | behavioral transfer is damage-heavy at `n=100`: `drift -0.12`, `compliance -0.05`, `recovery +0.38`, but `damage 0.39` | `logit-specific but damage-prone` |
| Llama-3.1-8B | single-layer sweep can reduce drift slightly, with best drift at layer `26` or `27`: `-0.1667` | no clean-control window: `compliance_delta = 0.0` at every tested layer, `recovery_delta <= 0.0`, and `damage` stays between `0.0833` and `0.1667` | `no clean-control window / locatable but not controllable` |

## Boundary Conclusions

1. Mistral can now be written more steadily as `logit-specific but damage-prone`. The projection diagnostic is no longer just suggestive: English gives `mean_belief_logit_delta = -5.6800` with `mean_negative_logit_delta = -0.3018`, Chinese gives `-5.6026 / -0.3437`, and the new `n=100` run gives `-5.7951 / -0.2506`. This is enough to support a stable logit-specificity readout, but not a clean behavioral-control claim.

2. Mistral should be excluded from `secondary-controllable`. In the new `n=100` behavioral summary, the intervention improves `stance_drift` from `0.48` to `0.36`, `pressured_compliance` from `1.00` to `0.95`, and `recovery` from `0.52` to `0.90`, but `baseline_damage_rate = 0.39` is far too high for a clean or secondary-controllable label. The safest wording remains that the direction is behaviorally active yet damage-prone.

3. Llama single-layer sweep further supports `no clean-control window`, not a rescue story. Across layers `20-27`, the best case only reaches `stance_drift_delta = -0.1667`, while `pressured_compliance_delta` remains `0.0` at every tested layer, `recovery_delta` is `0.0` or `-0.0417`, and `baseline_damage_rate` never drops below `0.0833`. This strengthens the earlier limitation claim: localization may exist, but no tested single-layer intervention produces clean control.

## Boundary

- Do not write Mistral as `clean success`.
- Do not write Llama as `rescued by single-layer sweep`.
- Do not write projection specificity as behavioral controllability; here it supports mechanism localization only, not clean transfer.

## Source Paths

- Mistral English projection: [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_projection_logit_diagnostic/english_20260426_143645/projection_alignment_summary.json:1)
- Mistral Chinese projection: [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_projection_logit_diagnostic/chinese_20260426_144430/projection_alignment_summary.json:1)
- Mistral n=100 projection: [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/projection_alignment_summary.json:1)
- Mistral n=100 behavioral summary: [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/belief_causal_summary.csv:1)
- Mistral n=100 subspace summary: [belief_causal_subspace_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/belief_causal_subspace_summary.csv:1)
- Llama per-layer sweep: [behavioral_sweep_long.csv](/Users/shiqi/code/graduation-project/outputs/experiments/llama_layer_wise_behavioral_sweep_per_layer/meta-llama_Llama-3.1-8B-Instruct/20260508_013923/behavioral_sweep_long.csv:1)
