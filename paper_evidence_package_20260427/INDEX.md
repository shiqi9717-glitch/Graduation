# Evidence Package Index

更新时间：2026-04-27

| Paper section | Claim | Source file | Figure/table | Status | Notes |
|---|---|---|---|---|---|
| Abstract / Main claim | Pressure is not one mechanism | `appendix_materials/frozen_evidence_dossier.md` | Main claim | ready | 冻结表述源 |
| Intro / Framing | Belief pressure and identity/profile pressure should be separated | `appendix_materials/frozen_evidence_dossier.md` | R1 | ready | 使用 dossier 的主命题与 R1 |
| Methods / Metric mapping | Qwen 3B/7B use objective-local proxy mapping | `configs/statistical_closure_README.md` | Metric definition note | ready | 不可与 bridge causal lines 做同义 leaderboard |
| Data / Objective-local subset | Qwen mainline uses fixed objective-local subsets | `datasets/qwen7b_objective_local_sample_set.json` | Figure 2 methods | ready | 7B 子集定义 |
| Data / Objective-local subset | Qwen mainline uses fixed objective-local subsets | `datasets/qwen3b_objective_local_sample_set.json` | Figure 2 methods | ready | 3B 子集定义 |
| Data / Bridge corpus | Full bridge manifest and coverage | `datasets/full_bridge_manifest.json` | Appendix dataset stats | ready | 配合 `datasets/dataset_statistics_bridge_coverage.csv` |
| Main result / Qwen 7B | Qwen 7B is the formal mainline | `results_mainline/qwen7b_mainline_run/intervention_summary.json` | Figure 2 | ready | 默认 `24-26, 0.6` |
| Main result / Qwen 7B held-out | Qwen 7B mainline direction survives held-out validation | `results_mainline/qwen7b_heldout_validation/objective_local_eval_metrics.csv` | Main-text support note | ready | 与主文 held-out validation 表述一致；仍属 objective-local proxy |
| Main result / Qwen 3B | Qwen 3B is the formal replication | `results_mainline/qwen3b_replication_run/intervention_summary.json` | Figure 2 | ready | 默认 `31-35, 0.6` |
| Main result / Effect sizes | Qwen mainline delta + CI | `tables/whitebox_effect_size_table.csv` | Figure 2, Table 1 | ready | 主文首选 effect size 源 |
| Main result / Figure 2 | Qwen mainline intervention effect with CI | `figures/figure2_qwen_mainline_effect_ci.csv` | Figure 2 | ready | 直接作图源数据 |
| Mechanistic support | Qwen 7B late-layer drift is localizable | `results_mainline/qwen7b_mechanistic_summary.json` | Mainline mechanistic support | ready | generalization mechanistic summary |
| Mechanistic support | Qwen 3B late-layer drift is localizable | `results_mainline/qwen3b_mechanistic_summary.json` | Mainline mechanistic support | ready | replication mechanistic summary |
| Cross-scale support | Qwen 14B is secondary causal confirmation | `results_transfer/qwen14b_secondary_confirmation_run/qwen14b_belief_causal_summary.csv` | Table 2 / Figure 3 | ready | 不升级成新的 mainline |
| Cross-family support | GLM is positive replication with stronger tradeoff | `results_transfer/glm4_9b_tradeoff_run/subspace_intervention_summary.csv` | Table 2 / Figure 3 | ready | 必须保留 baseline damage |
| Cross-model summary | Bridge-causal panel source data | `tables/table2_source_data.csv` | Table 2 | ready | 对应 Figure 3 panel B |
| Cross-model summary | Split-panel tradeoff data | `figures/figure3_panelA_qwen_proxy_tradeoff.csv` | Figure 3 | ready | Qwen proxy panel |
| Cross-model summary | Split-panel tradeoff data | `figures/figure3_panelB_bridge_causal_tradeoff.csv` | Figure 3 | ready | bridge / transfer panel |
| Limitation / Llama | Llama is weak replication / limitation | `results_llama_diagnostics/llama_limitation_summary.md` | Figure 4, Table 5 | ready | 可直接引用中英文表述 |
| Limitation / Llama | Projection norm and logit diagnostics | `results_llama_diagnostics/projection_norm_summary.json` | Figure 4 | ready | 包含 norm、fraction、logit delta |
| Limitation / Llama | Projection vs stance drift correlation | `results_llama_diagnostics/projection_vs_stance_drift_correlation.csv` | Figure 4 | ready | 诊断表格源 |
| Limitation / Llama | Behavioral closure | `results_llama_diagnostics/behavioral_closure.csv` | Table 5 | ready | alpha sweep 主行为结果 |
| Limitation / Llama | Table 5 summary source | `tables/table5_source_data.csv` | Table 5 | ready | 方便附录直接取数 |
| Exploratory / Mistral | Mistral is appendix exploratory only | `results_transfer/mistral_english_exploratory_run/belief_causal_summary.csv` | Figure 3 optional / appendix note | ready | 高 damage、跨 prompt 不稳 |
| Exploratory / Mistral | Prompt-variant instability | `results_transfer/mistral_zh_instruction_exploratory_run/belief_causal_summary.csv` | Appendix note | ready | 与 English 结果成对使用 |
| Identity boundary | identity/profile is weak mechanistic observation | `results_identity_profile/qwen7b_identity_followup/identity_profile_intervention_summary.csv` | Table 6 | ready | 不得写成 identity-specific mitigation |
| Identity boundary | prefix localization and matched-gap stats | `results_identity_profile/prefix_localization_summary.csv` | Table 6 | ready | 包含 `prefix_localization_gain` 等 |
| Identity boundary | Table 6 source | `tables/table6_source_data.csv` | Table 6 | ready | 直链附录表格源 |
| Controls / Subtraction | Subtraction controls are method-boundary evidence | `results_controls/qwen7b_subtraction_control/intervention_summary.json` | Appendix robustness table | ready | 7B 对照 |
| Controls / Subtraction | Subtraction controls are method-boundary evidence | `results_controls/qwen3b_subtraction_control/intervention_summary.json` | Appendix robustness table | ready | 3B 对照 |
| Controls / Sanity | Sanity control exists and should stay appendix-only | `results_controls/qwen7b_sanity_control/intervention_summary.json` | Appendix robustness table | ready | 不升格主文 |
| Controls / Formal controls | Qwen 7B formal controls reduce arbitrary-edit and pairing-artifact explanations | `results_controls/qwen7b_formal_controls/formal_controls_aggregate_metrics.csv` | Main-text support note / appendix robustness | ready | 对应主文 formal controls 口径 |
| Controls / Negative controls | Raw negative-control intervention run is archived | `results_controls/qwen7b_negative_controls_raw_run/intervention_summary.json` | Appendix robustness table | ready | 保留与 formal controls 上游衔接 |
| Controls / Stability | Layer/strength stability outputs | `results_controls/qwen_layer_strength_sweep/qwen_mainline_sweep_long.csv` | Appendix robustness table | ready | 稳定性与边界材料 |
| Controls / Stability | Sweep manifest and pivots | `configs/qwen_layer_strength_sweep_manifest.json` | Appendix robustness table | ready | 层/强度 sweep 配置 |
| Prompt templates | Bridge prompt templates | `prompt_templates/bridge_scenarios.jsonl` | Appendix prompt templates | ready | 基础 prompt 证据 |
| Human audit | White-box human audit bundle | `human_audit/whitebox_human_audit_bundle` | Appendix human audit | ready | 包含 rows、manifest、guidelines |
| Human audit | Annotation instructions | `human_audit/annotation_guidelines.md` | Appendix human audit | ready | 给复核人直接用 |
| Appendix / Supporting note | Objective-local metric family, updated controls + held-out support, and GLM sample-size comparability boundaries | `appendix_materials/whitebox_supporting_note.md` | Appendix note / evidence boundary note | ready | 用于补充主文与附录的边界说明，不构成新主结果 |
| Appendix / Internal note | Internal reproducibility note | `appendix_materials/internal_reproducibility_note.md` | Appendix reproducibility | ready | 本 package 内生成说明 |
| Appendix / Draft lineage | Frozen dossier and writing lineage | `appendix_materials/frozen_evidence_dossier.md` | Appendix writing support | ready | 写作部门上游锚点 |
