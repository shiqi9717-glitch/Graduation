# Final Evidence Ledger

更新时间：2026-05-06

本台账只整理当前冻结证据，不跑新实验，不改代码，不改论文正文。每条记录都尽量把数字、路径、口径边界、和论文落点绑在一起。若某条只有点估计而无冻结 CI，本台账明确写成“point-estimate summary only”，不补编额外不确定性细节。

## 1. Qwen 7B Mainline

`experiment_name`: Qwen 7B objective-local mainline closure  
`model`: Qwen/Qwen2.5-7B-Instruct  
`pressure_type`: `belief_argument` under objective-local proxy focus subset  
`n`: 100  
`metric_family`: objective-local proxy  
`intervention_method`: `baseline_state_interpolation`  
`config`: layers `24-26`, `beta=0.6`  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/whitebox_effect_size_table.csv`  
`main_numbers`: drift `-0.1493`; compliance `-0.1595`; recovery `+0.7127`; damage `0.0000`  
`CI/uncertainty`: drift `[-0.2200,-0.0800]`; compliance `[-0.2300,-0.0900]`; recovery `[0.5000,0.9000]`; no baseline damage observed in the tested sample; archived 95% CI reported, but bootstrap repetition count / CI variant not independently re-confirmed in this ledger  
`allowed_claim`: formal mainline; clean objective-local benefit profile; belief-style pressure is partially intervenable in this tested Qwen mainline  
`forbidden_claim`: bridge-causal leaderboard win; universal mitigation; baseline-safe; guaranteed zero damage  
`paper_section`: Section 6; Figure 2; Appendix robustness table

## 2. Qwen 3B Replication

`experiment_name`: Qwen 3B objective-local replication closure  
`model`: Qwen/Qwen2.5-3B-Instruct  
`pressure_type`: `belief_argument` under objective-local proxy focus subset  
`n`: 100  
`metric_family`: objective-local proxy  
`intervention_method`: `baseline_state_interpolation`  
`config`: layers `31-35`, `beta=0.6`  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/whitebox_effect_size_table.csv`  
`main_numbers`: drift `-0.1597`; compliance `-0.2789`; recovery `+0.6906`; damage `0.0000`  
`CI/uncertainty`: drift `[-0.2402,-0.0800]`; compliance `[-0.3700,-0.1900]`; recovery `[0.5000,0.8667]`; no baseline damage observed in the tested sample; archived 95% CI reported, but bootstrap repetition count / CI variant not independently re-confirmed in this ledger  
`allowed_claim`: formal replication of the Qwen mainline direction under the same proxy mapping  
`forbidden_claim`: direct bridge-style merger with Qwen14B / GLM / Llama rows; universally stronger than 7B  
`paper_section`: Section 6; Figure 2; Appendix robustness table

## 3. Qwen Formal Controls

`experiment_name`: Qwen 7B formal controls (`random_direction_control` + `shuffled_label_control`)  
`model`: Qwen/Qwen2.5-7B-Instruct  
`pressure_type`: `belief_argument` under objective-local proxy focus subset  
`n`: 100 per control aggregate  
`metric_family`: objective-local proxy  
`intervention_method`: `random_direction_control`; `shuffled_label_control`  
`config`: layers `24-26`, `beta=0.6`; seeded aggregate control suite  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_formal_controls_qwen7b/20260427_121722`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_formal_controls_qwen7b/20260427_121722/formal_controls_aggregate_metrics.csv`  
`main_numbers`: random-direction drift `+0.0025`, compliance `-0.0060`, recovery `+0.0190`, damage `0.0127`; shuffled-label drift `+0.3949`, compliance `-0.0460`, recovery `+0.2667`, damage `0.5848`  
`CI/uncertainty`: random-direction drift `[-0.0113,0.0179]`, compliance `[-0.0180,0.0000]`, recovery `[0.0000,0.0632]`, damage `[0.0000,0.0337]`; shuffled-label drift `[0.2941,0.4918]`, compliance `[-0.1280,0.0280]`, recovery `[0.2000,0.3400]`, damage `[0.5372,0.6338]`; archived 95% CI reported, but bootstrap repetition count / CI variant not independently re-confirmed in this ledger  
`allowed_claim`: supports “not an arbitrary activation edit” and “not a simple pairing artifact”  
`forbidden_claim`: standalone mitigation family; new main result line  
`paper_section`: Section 6 robustness paragraph; Table `qwen-robustness`; Appendix robustness

## 4. Qwen Held-out Validation

`experiment_name`: Qwen 7B held-out objective-local validation  
`model`: Qwen/Qwen2.5-7B-Instruct  
`pressure_type`: `belief_argument` under objective-local proxy held-out focus slice  
`n`: `n_focus=292` (`n_raw=300`)  
`metric_family`: objective-local proxy  
`intervention_method`: `baseline_state_interpolation`  
`config`: layers `24-26`, `beta=0.6`; archived held-out slice, no retuning  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_qwen7b_heldout_eval/qwen7b_heldout_mainline/20260427_135639`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_qwen7b_heldout_eval/qwen7b_heldout_mainline/20260427_135639`  
`main_numbers`: drift `-0.2000`; compliance `-0.1747`; recovery `+0.6571`; damage `0.0000`  
`CI/uncertainty`: drift `[-0.2522,-0.1511]`; compliance `[-0.2192,-0.1336]`; recovery `[0.5441,0.7667]`; no baseline damage observed in the tested sample; held-out export is archived with 95% CI, but bootstrap repetition count / CI variant are not relitigated in this ledger  
`allowed_claim`: robustness-side validation; not solely driven by the original tuned subset  
`forbidden_claim`: clean split main result; standalone benchmark; strict leaderboard result  
`paper_section`: Section 6 held-out paragraph; Table `qwen-robustness`; Appendix robustness

## 5. Cross-model n=100: Llama

`experiment_name`: Llama-3.1-8B belief causal transfer n=100  
`model`: meta-llama/Llama-3.1-8B-Instruct  
`pressure_type`: `belief_argument` bridge / transfer-style prompt family  
`n`: 100  
`metric_family`: bridge / transfer-style  
`intervention_method`: `matched_belief_subspace_damping`  
`config`: layers `24-31`, `k=2`, `alpha=0.5`  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_n100/meta-llama_Llama-3.1-8B-Instruct/20260504_025609`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_n100/meta-llama_Llama-3.1-8B-Instruct/20260504_025609/belief_causal_summary.csv`  
`main_numbers`: drift `+0.03`; compliance `+0.01`; recovery `-0.03`; damage `0.08`  
`CI/uncertainty`: point-estimate summary only in the current frozen n=100 update note; no archived CI cited in this ledger entry  
`allowed_claim`: locatable but not controllable; limitation / boundary evidence  
`forbidden_claim`: positive replication; strong cross-family success  
`paper_section`: Section 7 cross-model boundary; Section 10 limitations; Section 11 conclusion

## 6. Cross-model n=100: GLM

`experiment_name`: GLM-4-9B belief causal transfer n=100  
`model`: THUDM/glm-4-9b-chat-hf  
`pressure_type`: `belief_argument` bridge / transfer-style prompt family  
`n`: 100  
`metric_family`: bridge / transfer-style  
`intervention_method`: `pressure_subspace_damping` / matched belief subspace damping  
`config`: layers `30-33`, `k=2`, `alpha=0.75`  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/belief_causal_summary.csv`  
`main_numbers`: drift `-0.01`; compliance `-0.01`; recovery `+0.12`; damage `0.06`  
`CI/uncertainty`: point-estimate summary only in the current frozen n=100 update note; no archived CI cited in this ledger entry  
`allowed_claim`: weak-directional / residual-damage tradeoff  
`forbidden_claim`: clean strong success; no-cost cross-family replication  
`paper_section`: Section 7 cross-model boundary; Section 10 limitations; Section 11 conclusion

## 7. Cross-model n=100: Qwen14B

`experiment_name`: Qwen 14B belief causal transfer n=100  
`model`: Qwen/Qwen2.5-14B-Instruct  
`pressure_type`: `belief_argument` bridge / transfer-style prompt family  
`n`: 100  
`metric_family`: bridge / transfer-style  
`intervention_method`: `matched_belief_subspace_damping`  
`config`: matched belief damping, frozen n=100 transfer setting; see run summary for exact layers / alpha bundle  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/qwen14b_belief_causal_transfer_n100/Qwen_Qwen2.5-14B-Instruct/20260504_131013`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/qwen14b_belief_causal_transfer_n100/Qwen_Qwen2.5-14B-Instruct/20260504_131013/belief_causal_summary.csv`  
`main_numbers`: drift `-0.06`; compliance `-0.07`; recovery `+0.03`; damage `0.01`  
`CI/uncertainty`: point-estimate summary only in the current frozen n=100 update note; no archived CI cited in this ledger entry  
`allowed_claim`: secondary positive confirmation; weaker but cleaner than the old n=48 cautionary row  
`forbidden_claim`: new main mitigation line; keep using the old harmful / not recommended headline without update  
`paper_section`: Section 7 cross-scale support; Section 10 limitations; Section 11 conclusion

## 8. Free-form Diagnostic n=50

`experiment_name`: Qwen 7B free-form diagnostic confirm experiment (five conditions)  
`model`: Qwen2.5-7B-Instruct  
`pressure_type`: `belief_argument` free-form pressured / recovery prompts  
`n`: 50  
`metric_family`: free-form diagnostic boundary metrics (`drift_rate`, `wrong_follow_rate`, `readable_rate`, `repetition_rate`, `distinct-1`)  
`intervention_method`: patch-horizon sweep: `no_intervention`; `prefill_only`; `first_token_only`; `first_3_tokens`; `continuous`  
`config`: Qwen 7B mainline late-layer direction patched into open-ended generation with five continuity conditions  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503`  
`output_result_path`: `/Users/shiqi/code/graduation-project/docs/reports/freeform_diagnostic_confirm_boundary_note_20260504.md`; `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/no_intervention/Qwen2.5-7B-Instruct/20260503_150943/run_summary.json`; `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/prefill_only/Qwen2.5-7B-Instruct/20260503_172219/run_summary.json`; `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/first_token_only/Qwen2.5-7B-Instruct/20260503_192855/run_summary.json`; `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/first_3_tokens/Qwen2.5-7B-Instruct/20260503_213547/run_summary.json`; `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/continuous/Qwen2.5-7B-Instruct/20260504_001705/run_summary.json`  
`main_numbers`: pressured `no_intervention` drift `0.36`, wrong-follow `0.42`, readable `0.96`; `prefill_only` drift `0.10`, wrong-follow `0.16`, readable `0.94`; `first_token_only` same directional profile; `first_3_tokens` drift `0.10`, wrong-follow `0.16`, readable `0.42`, repetition `0.58`; `continuous` drift `0.10`, wrong-follow `0.16`, readable `0.00`, repetition `1.00`; recovery side remains directionally similar with quality collapse under longer continuity  
`CI/uncertainty`: descriptive rates only; no CI cited in the frozen diagnostic note; this design does not measure baseline-patched damage  
`allowed_claim`: front-loaded diagnostic effect with continuity-quality tradeoff; diagnostic free-form boundary evidence only  
`forbidden_claim`: clean free-form intervention success; full free-form validation; measured baseline damage  
`paper_section`: Appendix free-form diagnostic; Section 6 boundary sentence; Section 10 limitations

## 9. Intervention Family Comparison

`experiment_name`: Qwen 7B intervention family comparison  
`model`: Qwen/Qwen2.5-7B-Instruct  
`pressure_type`: `belief_argument`  
`n`: mixed: 100 mainline, 100 PSD, 48 old PSD, 108 subtraction / controls  
`metric_family`: mixed; objective-local proxy plus bridge / transfer-style boundary line, not a single leaderboard  
`intervention_method`: `baseline_state_interpolation`; `pressure_subspace_damping`; `late_layer_residual_subtraction`; `random_direction_control`; `shuffled_label_control`  
`config`: mainline `24-26, beta=0.6`; PSD `k=2, alpha=0.75`; subtraction `24-26, beta=0.6`; controls under the archived Qwen 7B control suite  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen7b_n100/Qwen_Qwen2.5-7B-Instruct/20260506_181314`; `/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen7b_clean/Qwen_Qwen2.5-7B-Instruct/20260430_161102`; `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_formal_controls_qwen7b/20260427_121722`; `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`  
`output_result_path`: `/Users/shiqi/code/graduation-project/docs/reports/intervention_family_comparison_20260506.md`  
`main_numbers`: mainline `-0.1493 / -0.1595 / +0.7127 / 0.0000`; PSD n=100 `-0.04 / -0.04 / +0.05 / 0.04`; PSD old n=48 `-0.0417 / 0.0000 / +0.0417 / 0.0417`; subtraction `+0.0115 / 0.0000 / +0.0476 / 0.0460`; random control `+0.0025 / -0.0060 / +0.0190 / 0.0127`; shuffled control `+0.3949 / -0.0460 / +0.2667 / 0.5848`  
`CI/uncertainty`: mainline / controls carry archived CI in their source tables; PSD n=100 and intervention-family comparison note are point-estimate summaries in this ledger; mixed-family comparison is for bounded qualitative comparison only  
`allowed_claim`: implementation-robust but not implementation-agnostic; `pressure_subspace_damping` can be a weaker less-oracle alternative  
`forbidden_claim`: new SOTA steering method; any late-layer linear rule works; head-to-head unified leaderboard across metric families  
`paper_section`: Section 6 boundary sentence; Section 8 implementation-boundary discussion; Appendix intervention-family table

## 10. Prompt-family Generalization

`experiment_name`: Qwen 7B prompt-family generalization check  
`model`: Qwen2.5-7B-Instruct  
`pressure_type`: `belief_argument` with default / authority / majority wording variants  
`n`: 60  
`metric_family`: small wording-family robustness note on objective-local style scoring  
`intervention_method`: no intervention; prompt wording variation only  
`config`: default wording vs `authority/teacher` variant vs `majority/consensus` variant  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_prompt_family_check/20260501/Qwen2.5-7B-Instruct/20260501_231604`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_prompt_family_check/20260501/Qwen2.5-7B-Instruct/20260501_231604/run_summary.json`; `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_prompt_family_check/20260501/Qwen2.5-7B-Instruct/20260501_231604/prompt_family_results.jsonl`  
`main_numbers`: baseline accuracy `0.783`; default pressured accuracy `0.633`, drift `0.283`, wrong-follow `0.283`; authority variant pressured accuracy `0.467`, drift `0.450`, wrong-follow `0.467`; majority variant pressured accuracy `0.600`, drift `0.317`, wrong-follow `0.317`  
`CI/uncertainty`: descriptive subset rates only; no CI cited in the frozen run summary  
`allowed_claim`: not tied to a single wording; same drift direction across the three tested variants  
`forbidden_claim`: full prompt-family invariance; independence from all social-pressure phrasings  
`paper_section`: Section 6 boundary sentence; Appendix prompt-family table

## 11. Identity/Profile Boundary + Human Audit

`experiment_name`: identity/profile boundary evidence plus human audit / expert adjudication support  
`model`: Qwen 7B identity/profile follow-up; human raters across audit materials  
`pressure_type`: `identity_profile` boundary line plus human-readable audit of proxy metrics  
`n`: identity/profile follow-up `see artifact`; human audit `108 items, 3 annotators`; expert adjudication `24 targeted rows`  
`metric_family`: boundary mechanistic follow-up plus human corroboration audit  
`intervention_method`: identity/profile matched-control follow-up; human annotation and expert adjudication  
`config`: identity/profile prefix-gating / matched-control follow-up; human audit on archived rows; expert adjudication targeted to recovery disagreements  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058`; `/Users/shiqi/code/graduation-project/docs/reports/human_audit_quality_report_20260427.md`; `/Users/shiqi/code/graduation-project/docs/reports/human_audit_revised_agreement_table_20260427.csv`; `/Users/shiqi/code/graduation-project/docs/reports/human_audit_expert_adjudication_appendix_note_20260427.md`  
`output_result_path`: `/Users/shiqi/code/graduation-project/docs/reports/human_audit_quality_report_20260427.md`; `/Users/shiqi/code/graduation-project/docs/reports/human_audit_expert_adjudication_appendix_note_20260427.md`  
`main_numbers`: identity/profile localization evidence exists, but causal intervention support remains insufficient; human audit covers `108` rows with `3` annotators and shows stronger proxy-human alignment for drift / compliance / baseline damage than for recovery; expert adjudication on `24` targeted recovery rows yields proxy `1/24`, human majority `10/24`, mixed `1/24`, unresolved `12/24`  
`CI/uncertainty`: appendix-quality corroboration only; no formal CI claim; recovery remains the weakest field on both proxy-human agreement and expert adjudication  
`allowed_claim`: weak human corroboration; identity/profile is a boundary case with localization observations but insufficient causal intervention support  
`forbidden_claim`: formal human validation; solved identity-specific mitigation; expert-adjudicated full recovery success  
`paper_section`: Section 9 identity/profile boundary; Section 10 limitations; Appendix human-audit note

## Bottom Line

当前最稳的证据台账边界是：

- Qwen 7B = formal mainline under objective-local proxy
- Qwen 3B = formal replication under the same proxy family
- Qwen controls + held-out = robustness-side support, not new main results
- Qwen14B / GLM / Llama n=100 = bridge / transfer-style boundary evidence, with updated cross-model wording
- free-form diagnostic / prompt-family / intervention-family / identity-profile / human audit = appendix-strengthening or limitation-boundary material, not primary closure replacements
