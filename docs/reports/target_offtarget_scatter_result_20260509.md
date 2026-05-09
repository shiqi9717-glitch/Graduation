# Target-Offtarget Scatter Result

更新时间：2026-05-09

本说明只汇报 R13 的散点图产物与最稳 interpretation，不跑新实验，不改代码，不改论文正文。

## Figure Output

- Figure PDF: [figure_target_offtarget_scatter.pdf](/Users/shiqi/code/graduation-project/docs/papers/figures/figure_target_offtarget_scatter.pdf)
- Unified scatter data: [target_offtarget_scatter_data_20260509.csv](/Users/shiqi/code/graduation-project/docs/reports/target_offtarget_scatter_data_20260509.csv:1)

## Operational Setup

- X axis: `target_logit_improvement = intervened_pressured_logit(target) - no_intervention_pressured_logit(target)`
- Y axis: `off_target_instability_score = number of non-target options whose baseline-stage ranking changes between no_intervention and intervention`
- Filled points mark `baseline_damage = True`

Qwen was included using the same-format PSD `n=100` records from `pressure_subspace_damping_qwen7b_n100`, not from the older closure directory.

## Centroid Summary

| Model | Mean target-logit improvement | Mean off-target instability | Damage rate |
| --- | ---: | ---: | ---: |
| Qwen 7B | +1.0344 | 0.50 | 0.04 |
| GLM-4-9B | +0.6272 | 0.06 | 0.06 |
| Mistral-7B | +1.8883 | 0.59 | 0.39 |

## Interpretation

- The figure supports the narrow claim that `target movement != absence of damage`. Both GLM and Mistral occupy regions with positive target movement, yet they still contain many damage points.
- Mistral is the clearest damage-prone regime in this visualization: it combines the strongest mean target movement (`+1.8883`) with the highest mean off-target instability (`0.59`) and by far the highest damage rate (`0.39`).
- Qwen should not be over-described here as a zero-instability clean corner. Under this common operational definition, Qwen still shows some non-target reshuffling (`mean instability = 0.50`), but its damage rate remains much lower (`0.04`) than Mistral. The safest cross-model reading is therefore that off-target instability is informative but not singly sufficient, while strong target movement alone is also insufficient.

## Caption-safe Wording

> Target-logit improvement versus off-target ranking instability across Qwen 7B PSD, GLM-4-9B, and Mistral-7B. Positive target movement can coexist with non-target ranking perturbation in damage-prone regimes, especially for Mistral. The contrast supports the narrower claim that target-directed movement is informative but not sufficient to rule out damage.

## Boundary

- Do not caption this figure as full causal proof; it is an operational visualization.
- Do not write Qwen here as a zero-instability gold-standard clean case.
- Do not write the figure as showing that instability alone fully determines damage; the safer claim is that target movement alone is insufficient and off-target perturbation remains an important cofactor.

## Source Paths

- Qwen records: [belief_causal_records.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen7b_n100/Qwen_Qwen2.5-7B-Instruct/20260506_181314/belief_causal_records.jsonl:1)
- Qwen comparisons: [belief_causal_comparisons.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen7b_n100/Qwen_Qwen2.5-7B-Instruct/20260506_181314/belief_causal_comparisons.jsonl:1)
- GLM records: [belief_causal_records.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/belief_causal_records.jsonl:1)
- GLM comparisons: [belief_causal_comparisons.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/belief_causal_comparisons.jsonl:1)
- Mistral records: [belief_causal_records.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/belief_causal_records.jsonl:1)
- Mistral comparisons: [belief_causal_comparisons.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/belief_causal_comparisons.jsonl:1)
