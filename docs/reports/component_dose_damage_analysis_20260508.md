# Component, Dose, and Damage Analysis

更新时间：2026-05-09

本说明只基于 C10-C12 的冻结结果做短综合分析，不跑新实验，不改代码，不改论文正文。

## Short Table

| model | best component | best beta | best clean_control_score | damage interpretation |
| --- | --- | --- | ---: | --- |
| Qwen 7B | `full_residual` | `0.8` or `1.0` among newly added points only | `1.3952` on the available dose file; component slice gives `1.2917` at `beta=0.6` | no damage signal in the tested component/dose slices; current evidence is consistent with a clean full-residual window, but this is still exploratory |
| GLM-4-9B | `full_residual` | `0.1` or `0.2` | `0.0417` | damage looks like off-target interference, mainly `3` non-target overcorrections plus `3` unstable option-rankings, with no answer-flip or margin-collapse signature |
| Llama-3.1-8B | `full_residual` | `0.1` or `0.2` | `0.0000` | no separate damage subtype run here; dose sweep shows that stronger beta adds damage before any clean benefit window appears |
| Mistral-7B | N/A in C10/C11 | N/A in C10/C11 | N/A in C10/C11 | damage is dominated by off-target ranking instability and overcorrection: `23` unstable option-rankings and `16` non-target overcorrections at `damage_rate = 0.39` |

## Conclusions

1. C10 supports `full_residual` as the main effective path, but not yet as final pathway proof. In all three models, `full_residual` is the only component with non-trivial positive score: Qwen gets `clean_control_score = 1.2917` with `drift/compliance/recovery = -0.4167 / -0.4167 / +0.4583`, GLM gets `0.75` with `-0.25 / -0.25 / +0.25`, and Llama gets `0.5` with `-0.125 / -0.125 / +0.25`. By contrast, `attention_only` and `mlp_only` are near-null in all three cases, typically `0.0000` to `0.0417`. So the current exploratory read is that whatever useful signal exists is carried by the full residual path rather than an isolated attention-only or MLP-only patch.

2. GLM and Llama now look more steadily like `not a beta-tuning miss`, not merely “beta 没调好”. GLM’s dose-response curve is flat-to-bad at low beta and benefit-damage coupled at high beta: `beta=0.1` or `0.2` gives only `score = 0.0417`, while `beta=1.0` finally improves `drift` to `-0.1667` and `compliance` to `-0.4583` but explodes `damage` to `0.4583`, pushing the score back below zero. Llama is even clearer: `beta=0.1` and `0.2` are pure null (`score = 0.0000`), while stronger beta only adds damage first, reaching `damage = 0.875` at `beta=1.0` with a still-negative score. This makes the failure profile much harder to dismiss as under-tuned hyperparameters.

3. GLM and Mistral damage look more like off-target interference than generic answer collapse. In GLM, the entire `damage_rate = 0.06` is explained by `3` non-target overcorrections and `3` unstable option-rankings, with `0` answer flips and `0` margin collapses. Mistral shows the same subtype but much more severely: `damage_rate = 0.39`, `23` unstable option-rankings, `16` non-target overcorrections, and a positive `mean_baseline_margin_delta_on_damage = 0.8173`, again with `0` answer flips and `0` margin collapses. The current best read is therefore benefit-damage coupling through ranking instability / off-target disruption, not a full-vocab or catastrophic-collapse story.

## Boundary

- Do not write C10’s exploratory component result as final pathway proof.
- Do not write the Qwen C11 file as a fully unified six-point sweep; the current file only contributes the newly added `0.8` and `1.0` points.
- Do not write C12 as full-vocab top-k token-shift evidence; the current evidence is still primarily option-logit, margin, and ranking based.

## Source Paths

- Qwen component: [component_level_delta_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/component_level_intervention/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_211828/component_level_delta_summary.csv:1)
- GLM component: [component_level_delta_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/component_level_intervention/glm4_9b/THUDM_glm-4-9b-chat-hf/20260508_212602/component_level_delta_summary.csv:1)
- Llama component: [component_level_delta_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/component_level_intervention/llama31_8b/meta-llama_Llama-3.1-8B-Instruct/20260508_213347/component_level_delta_summary.csv:1)
- Qwen dose-response: [qwen7b_dose_response_long.csv](/Users/shiqi/code/graduation-project/outputs/experiments/dose_response_sweep/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_214042/qwen7b_dose_response_long.csv:1)
- GLM dose-response: [behavioral_sweep_long.csv](/Users/shiqi/code/graduation-project/outputs/experiments/dose_response_sweep/glm4_9b/THUDM_glm-4-9b-chat-hf/20260508_214719/behavioral_sweep_long.csv:1)
- Llama dose-response: [behavioral_sweep_long.csv](/Users/shiqi/code/graduation-project/outputs/experiments/dose_response_sweep/llama31_8b/meta-llama_Llama-3.1-8B-Instruct/20260508_220426/behavioral_sweep_long.csv:1)
- GLM damage summary: [damage_signal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/damage_mechanism_analysis/GLM-4-9B/20260508_211524/damage_signal_summary.csv:1)
- Mistral damage summary: [damage_signal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/damage_mechanism_analysis/Mistral-7B/20260508_211524/damage_signal_summary.csv:1)
