# Off-target Logit Decomposition

更新时间：2026-05-09

本报告只从已有 per-item records 提取 GLM 和 Mistral 的 target-vs-off-target movement，不跑新实验，不改代码，不改论文正文。

## Operational Definition

- `Target-logit improvement` is defined here as the change, at the `pressured` stage, in the logit assigned to the `no_intervention` baseline target option:
  `intervened_pressured_logit(target) - no_intervention_pressured_logit(target)`.
- `Off-target instability score` is defined here as the number of non-target options whose `baseline`-stage ranking changes between `no_intervention` and `matched_belief_subspace_damping`.
- Damage subtype counts (`overcorrection`, `unstable ranking`, `answer flip`, `margin collapse`) are taken from the existing damage-extraction summaries, while the target/off-target decomposition is computed directly from `belief_causal_records.jsonl`.
- A clean Qwen comparator in the same per-item format was not found under the frozen closure directory, so Qwen is reported as `N/A` here rather than estimated.

## Per-model Summary

| Model | n | Mean target-logit improvement | Mean non-target ranking change | Damage rate | Damage cases: overcorrection | Damage cases: unstable ranking | Damage cases: answer flip | Damage cases: margin collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4-9B | 100 | +0.6272 | 0.06 | 0.06 | 3 | 3 | 0 | 0 |
| Mistral-7B | 100 | +1.8883 | 0.59 | 0.39 | 16 | 23 | 0 | 0 |
| Qwen 7B clean reference | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Scatter Data

Per-item scatter data has been exported to:

- [offtarget_logit_scatter_data_20260509.csv](/Users/shiqi/code/graduation-project/docs/reports/offtarget_logit_scatter_data_20260509.csv:1)

Columns:

| Item ID | Model | Target-logit improvement | Off-target instability score | Is damage? |
| --- | --- | ---: | ---: | --- |

## Key Analysis Paragraph

In damage-prone regimes, the intervention is not inactive: target-logit movement still occurs in the expected direction on average. Under the operational definition above, GLM shows mean target-logit improvement of `+0.6272`, and Mistral shows a larger mean improvement of `+1.8883`. But this movement coexists with off-target ranking perturbation. In GLM, all `6/6` damage cases exhibit non-target ranking instability at baseline, and `3/6` of those damage cases also show positive target-logit improvement at the pressured stage. In Mistral, the same pattern appears at higher severity: all `39/39` damage cases exhibit non-target ranking instability, and `23/39` simultaneously show positive target-logit improvement. This supports the interpretation that damage-prone regimes fail not because the intervention is inactive, but because target-directed movement coexists with non-target ranking perturbation. A directly comparable clean Qwen per-item reference was not available in the frozen closure directory, so this report does not overclaim a same-format Qwen contrast.

## Notes on Scope

- The decomposition above is intentionally operational rather than ontological: it provides a reproducible target-vs-off-target summary from the available JSONL files, but it is not the only possible decomposition.
- Because the `comparisons` and `records` files do not use a perfectly identical labeling surface for every field, this report anchors the target option to the `no_intervention` baseline choice and avoids inferring any extra “true option” metadata beyond what the current artifacts directly expose.

## Source Paths

- GLM records: [belief_causal_records.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/belief_causal_records.jsonl:1)
- GLM comparisons: [belief_causal_comparisons.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/belief_causal_comparisons.jsonl:1)
- Mistral records: [belief_causal_records.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/belief_causal_records.jsonl:1)
- Mistral comparisons: [belief_causal_comparisons.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/belief_causal_comparisons.jsonl:1)
- GLM damage summary: [damage_signal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/damage_mechanism_analysis/GLM-4-9B/20260508_211524/damage_signal_summary.csv:1)
- Mistral damage summary: [damage_signal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/damage_mechanism_analysis/Mistral-7B/20260508_211524/damage_signal_summary.csv:1)
