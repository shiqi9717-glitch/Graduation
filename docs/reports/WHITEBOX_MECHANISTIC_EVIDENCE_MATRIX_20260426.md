# White-box Mechanistic Evidence Matrix

更新时间：2026-04-26

本文档用于主会论文写作阶段统一 white-box mechanistic 证据矩阵、readiness 状态与结果锚点。本文只做文件整理与写作对齐，不新增实验建议。

上游口径文档：

- `docs/reports/WHITEBOX_MECHANISTIC_WORKSPACE_SUMMARY_20260426.md`
- `docs/reports/RESULT_ANALYSIS_HANDOFF_20260422.md`
- `docs/PAPER_ANALYSIS_ENTRYPOINT.md`

## 1. 论文写作总口径

当前 white-box mechanistic 线的主会写作定位如下：

- Qwen 3B / 7B：主结果，状态 `DONE`。
- Qwen 14B：主文 secondary causal confirmation。
- GLM：主文 cross-family positive replication with stronger tradeoff。
- Llama-3.1-8B：limitation / weak replication。
- Mistral-7B：appendix exploratory note only。
- identity_profile：weakly supported mechanistic observation。
- `late_layer_residual_subtraction` 与常规重复 sweep：`DO NOT REPEAT`，只作为 failure / control 边界。

## 2. Unified Evidence Matrix

| 子线 / 模型 | 论文定位 | 最强证据 | 关键边界 | 主文状态 | 结果锚点 |
|---|---|---|---|---|---|
| Qwen2.5-3B `belief_argument` | Main white-box mitigation result | `baseline_state_interpolation`, layer `31-35`, scale `0.6`; `strict_positive` recovery `0.8125`, wrong-follow `0.10` vs ref `0.48`, damage `0`; `high_pressure_wrong_option` recovery `0.5`, wrong-follow `0.24` vs ref `0.42`, damage `0` | 只保留默认 baseline；不要把 subtraction control 写成可用方法 | 主文主表 | `outputs/experiments/local_probe_qwen3b_intervention_main/baseline_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142847/intervention_summary.json` |
| Qwen2.5-7B `belief_argument` | Main white-box mitigation result | `baseline_state_interpolation`, layer `24-26`, scale `0.6`; `strict_positive` recovery `0.5455`, wrong-follow `0.20` vs ref `0.36`, damage `0`; `high_pressure_wrong_option` recovery `0.9`, wrong-follow `0.10` vs ref `0.26`, damage `0` | 7B `24-27, 0.6` 只作为 aggressive secondary setting，不与默认 baseline 同级 | 主文主表 | `outputs/experiments/local_probe_qwen7b_intervention_main/baseline_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140142/intervention_summary.json` |
| Qwen2.5-7B aggressive setting | Secondary setting / robustness boundary | `24-27, 0.6` 在 `strict_positive` 更强：recovery `0.7273`, wrong-follow `0.12`, net `0.1951` | `high_pressure_wrong_option` 更弱且有 damage `0.0263`; 不能替代默认 baseline | 主文短注或附录 | `outputs/experiments/local_probe_qwen7b_intervention_main/aggressive_24_27_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140518/intervention_summary.json` |
| Qwen2.5-3B / 7B mechanistic localization | Mechanistic support for main mitigation | 3B n=`200`, vulnerable transition `baseline_correct_to_interference_wrong = 48`; 7B generalization n=`200`, vulnerable transition `30` | 支撑 localization 与 transition group，不单独作为 mitigation 表格 | 主文方法/机制证据 | `outputs/experiments/local_probe_qwen3b_mechanistic/Qwen_Qwen2.5-3B-Instruct/20260419_131943/mechanistic_summary.json`; `outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32_generalization/Qwen_Qwen2.5-7B-Instruct/20260422_134717/mechanistic_summary.json` |
| Qwen2.5-14B belief causal transfer | Secondary causal confirmation | `matched_belief_subspace_damping` 将 drift `0.5000 -> 0.2917`，compliance `0.9583 -> 0.7083`; matched negative control 与 no intervention 相同 | `n = 24`; recovery 未提升；baseline damage `0.0417`; 不是 14B 强 mitigation | 主文小表或短段 | `outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308/qwen14b_belief_causal_summary.csv` |
| GLM-4-9B belief subspace damping | Cross-family positive replication with stronger tradeoff | representative transfer: drift `0.4583 -> 0.1667`, compliance `0.9583 -> 0.7500`, recovery `0.5833 -> 0.9583` | baseline damage `0.3333`; tradeoff 明显强于 Qwen；不是无代价复现 | 主文跨家族表/短段 | `outputs/experiments/pressure_subspace_damping_glm4_9b/Users_shiqi_.cache_huggingface_hub_models--zai-org--glm-4-9b-chat-hf_snapshots_8599336fc6c125203efb2360bfaf4c80eef1d1bf/20260426_005017/subspace_intervention_summary.csv` |
| Llama-3.1-8B English bridge | Weak replication / limitation | subspace 更清晰：explained variance `0.64-0.68`, abs coherence `0.40-0.44`; alpha `0.5` drift `0.4583 -> 0.3750` | compliance 仍为 `1.0`; recovery `0.6667 -> 0.5833`; baseline damage `0.25`; intervention transfer weak | Limitation / appendix | `outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_summary.csv`; `outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_subspace_summary.csv` |
| Mistral-7B English / zh-instruction bridge | Appendix exploratory note only | English prompt 有方向信号：drift `0.5833 -> 0.3333`, compliance `1.0 -> 0.7917`, recovery `0.4167 -> 1.0` | across prompt variants unstable; baseline damage high: English `0.375`, zh-instruction `0.4167`; 不能写成 positive replication | Appendix one-sentence note | `outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_143645/belief_causal_summary.csv`; `outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_144430/belief_causal_summary.csv` |
| identity_profile Qwen7B follow-up | Weakly supported mechanistic observation | localization 证据存在；identity/profile prefix 相关差异可观察 | `profile_prefix_gating` 没有稳定优于 `matched_early_mid_negative_control`; causal intervention support 不足；不能 claim identity-specific mitigation | Limitation / appendix | `outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058/identity_vs_belief_localization_summary.csv`; `outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058/identity_profile_intervention_summary.csv` |
| `late_layer_residual_subtraction` | Failure / negative-control evidence | 3B / 7B subtraction controls 均不稳定，并出现 baseline damage 或负净收益 | 不进入主方法；不要在主表中与 `baseline_state_interpolation` 并列 | 附录或方法边界 | `outputs/experiments/local_probe_qwen3b_intervention_main/subtraction_control_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142938/intervention_summary.json`; `outputs/experiments/local_probe_qwen7b_intervention_main/subtraction_control_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140853/intervention_summary.json` |

## 3. Readiness Status Table

| 项目 | 状态 | 写作动作 | 不要做什么 |
|---|---|---|---|
| Qwen 3B 主 intervention baseline | DONE | 放入主结果表；只写 `baseline_state_interpolation, 31-35, 0.6` | 不混入 subtraction control |
| Qwen 7B 主 intervention baseline | DONE | 放入主结果表；只写 `baseline_state_interpolation, 24-26, 0.6` | 不把 `24-27, 0.6` 当默认 baseline |
| Qwen 7B aggressive `24-27, 0.6` | PARTIAL | 写成 aggressive secondary setting | 不与默认 baseline 同级 |
| Qwen 3B / 7B mechanistic localization | DONE | 作为主线机制证据与 transition-group 支撑 | 不单独夸大成另一条新方法 |
| Qwen 14B causal transfer | PARTIAL | 主文小表或短段，写成 secondary causal confirmation | 不写成 14B strong mitigation |
| GLM cross-family transfer | PARTIAL | 主文跨家族短表，写 positive replication with stronger tradeoff | 不写成无 tradeoff 成功 |
| Llama English bridge | PARTIAL | 放 limitation / appendix，写 mechanism locatable but intervention transfer weak | 不写成 positive replication |
| Mistral-7B bridge | PARTIAL | appendix exploratory note 一句话 | 不写成 positive replication；不放主文复现表 |
| identity_profile follow-up | PARTIAL | weak mechanistic observation，可放 limitation / appendix | 不写成 identity-specific mitigation |
| `late_layer_residual_subtraction` | DO NOT REPEAT | 作为 failure / negative-control 边界 | 不作为正式 baseline 或改进方向 |
| 常规 alpha / k / layer sweep | DO NOT REPEAT | 不作为主会写作建议 | 不新增 sweep 建议 |
| runtime predictor / sparse monitor / single-head recheck | MISSING for main-paper claim | 仅作为工作区历史探索背景 | 不纳入当前 white-box 主结果 |

## 4. Main-Text Anchors

主文应优先引用以下结果锚点：

- Qwen 3B 主 baseline：`outputs/experiments/local_probe_qwen3b_intervention_main/baseline_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142847/intervention_summary.json`
- Qwen 7B 主 baseline：`outputs/experiments/local_probe_qwen7b_intervention_main/baseline_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140142/intervention_summary.json`
- Qwen 3B mechanistic localization：`outputs/experiments/local_probe_qwen3b_mechanistic/Qwen_Qwen2.5-3B-Instruct/20260419_131943/mechanistic_summary.json`
- Qwen 7B mechanistic generalization：`outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32_generalization/Qwen_Qwen2.5-7B-Instruct/20260422_134717/mechanistic_summary.json`
- Qwen 14B secondary confirmation：`outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308/qwen14b_belief_causal_summary.csv`
- GLM cross-family confirmation：`outputs/experiments/pressure_subspace_damping_glm4_9b/Users_shiqi_.cache_huggingface_hub_models--zai-org--glm-4-9b-chat-hf_snapshots_8599336fc6c125203efb2360bfaf4c80eef1d1bf/20260426_005017/subspace_intervention_summary.csv`

主文可以引用但应降级呈现：

- Qwen 7B aggressive secondary：`outputs/experiments/local_probe_qwen7b_intervention_main/aggressive_24_27_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140518/intervention_summary.json`

## 5. Appendix / Limitation Anchors

附录、limitation 或 exploratory note 应引用以下结果锚点：

- Llama English bridge causal summary：`outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_summary.csv`
- Llama English bridge subspace summary：`outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_subspace_summary.csv`
- Mistral English exploratory result：`outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_143645/belief_causal_summary.csv`
- Mistral zh-instruction exploratory result：`outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_144430/belief_causal_summary.csv`
- identity localization summary：`outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058/identity_vs_belief_localization_summary.csv`
- identity intervention summary：`outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058/identity_profile_intervention_summary.csv`
- subtraction controls：`outputs/experiments/local_probe_qwen3b_intervention_main/subtraction_control_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142938/intervention_summary.json` and `outputs/experiments/local_probe_qwen7b_intervention_main/subtraction_control_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140853/intervention_summary.json`

## 6. Copy-ready Writing Boundaries

主结果句式：

> We treat Qwen2.5-3B and Qwen2.5-7B `baseline_state_interpolation` as the main white-box mechanistic mitigation result, using `31-35, 0.6` for 3B and `24-26, 0.6` for 7B.

14B 句式：

> The 14B result is a secondary causal confirmation: belief-subspace damping reduces pressured-stage drift and compliance relative to both no intervention and a matched negative control, but the evidence remains small-sample and does not improve recovery.

GLM 句式：

> GLM reproduces the belief-subspace damping direction across model families, but with a substantially stronger utility/safety tradeoff than the Qwen line.

Llama 句式：

> Llama-3.1-8B shows a more identifiable belief-pressure subspace under the English bridge prompt, but intervention transfer remains weak.

Mistral 句式：

> Mistral-7B showed directional belief-subspace damping signals under an English bridge prompt, but the effect was unstable across prompt variants and accompanied by high baseline damage; we therefore treat it as exploratory only.

identity_profile 句式：

> The identity-profile line provides weak localization evidence but insufficient causal intervention support, and should not be interpreted as identity-specific mitigation.
