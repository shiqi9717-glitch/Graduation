# Final Evidence Ledger

更新时间：2026-05-08

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

## 12. Llama Layer-wise Behavioral Sweep

`experiment_name`: Llama layer-wise behavioral sweep  
`model`: meta-llama/Llama-3.1-8B-Instruct  
`pressure_type`: `belief_argument` bridge / transfer-style prompt family  
`n`: `train_n=100`, `eval_n=100`  
`metric_family`: bridge / transfer-style layer-wise behavioral sweep  
`intervention_method`: `pressure_subspace_damping` / matched belief subspace damping  
`config`: union layers `20-27`; tested windows `20-21`, `22-23`, `24-25`, `26-27`, `20-23`, `24-27`, `20-27`; `k=2`; `alpha in {0.5, 0.75}`  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/llama_layer_wise_behavioral_sweep/meta-llama_Llama-3.1-8B-Instruct/20260507_145643`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/llama_layer_wise_behavioral_sweep/meta-llama_Llama-3.1-8B-Instruct/20260507_145643/behavioral_sweep_long.csv`; `/Users/shiqi/code/graduation-project/outputs/experiments/llama_layer_wise_behavioral_sweep/meta-llama_Llama-3.1-8B-Instruct/20260507_145643/behavioral_sweep_manifest.json`  
`main_numbers`: `alpha=0.5` rows stay near-null on drift/compliance with recovery `-0.03 to -0.04` and damage `0.03`; `alpha=0.75` rows reach at most drift `-0.05`, compliance `-0.03`, but recovery collapses to `-0.24 to -0.33` with damage `0.05 to 0.12`; no tested layer window yields a clean-control window  
`CI/uncertainty`: point-estimate sweep table only; no archived CI cited in this ledger entry  
`allowed_claim`: layer-wise sweep reinforces the Llama limitation boundary; no clean-control window appears in the tested 20-27 sweep  
`forbidden_claim`: hidden clean layer rescue; positive replication; controllable cross-family success  
`paper_section`: Appendix diagnostic framework; Figures A-C; Section 10 limitations

## 13. GLM Layer-wise Behavioral Sweep

`experiment_name`: GLM layer-wise behavioral sweep  
`model`: THUDM/glm-4-9b-chat-hf  
`pressure_type`: `belief_argument` bridge / transfer-style prompt family  
`n`: `train_n=100`, `eval_n=100`  
`metric_family`: bridge / transfer-style layer-wise behavioral sweep  
`intervention_method`: `pressure_subspace_damping` / matched belief subspace damping  
`config`: union layers `30-33`; tested windows `30`, `31`, `32`, `33`, `30-31`, `32-33`, `30-33`; `k=2`; `alpha in {0.5, 0.75}`  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/glm_layer_wise_behavioral_sweep/THUDM_glm-4-9b-chat-hf/20260507_170627`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/glm_layer_wise_behavioral_sweep/THUDM_glm-4-9b-chat-hf/20260507_170627/behavioral_sweep_long.csv`; `/Users/shiqi/code/graduation-project/outputs/experiments/glm_layer_wise_behavioral_sweep/THUDM_glm-4-9b-chat-hf/20260507_170627/behavioral_sweep_manifest.json`  
`main_numbers`: `alpha=0.5` rows are essentially null on drift/compliance with recovery `-0.02 to -0.04` and damage `0.00`; `alpha=0.75` rows achieve at best compliance `-0.02` and recovery `+0.03`, but still incur damage `0.04` and never produce a clean positive net profile; full `30-33` row remains weak with drift `+0.02`, recovery `-0.01`, damage `0.04`  
`CI/uncertainty`: point-estimate sweep table only; no archived CI cited in this ledger entry  
`allowed_claim`: layer-wise sweep supports weak / tradeoff-limited boundary wording; no single GLM layer window upgrades the line into clean controllability  
`forbidden_claim`: clean layer rescue; no-cost cross-family replication; stable controllable GLM window  
`paper_section`: Appendix diagnostic framework; Figures A-C; Section 7 cross-model boundary

## 14. Mistral-7B n=100 Behavioral Closure

`experiment_name`: Mistral-7B belief causal transfer n=100  
`model`: mistralai/Mistral-7B-Instruct-v0.3  
`pressure_type`: `belief_argument` bridge / transfer-style prompt family  
`n`: 100  
`metric_family`: bridge / transfer-style  
`intervention_method`: `pressure_subspace_damping` / matched belief subspace damping  
`config`: frozen n=100 boundary run; see archived run summary for exact retained window / alpha bundle  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/belief_causal_summary.csv`  
`main_numbers`: belief-logit movement remains strong (`-5.80`) but behavioral closure incurs severe baseline damage (`0.39`)  
`CI/uncertainty`: point-estimate summary only in the current frozen boundary update note; no archived CI cited in this ledger entry  
`allowed_claim`: logit-specific but damage-prone boundary evidence; directional movement does not translate into clean control  
`forbidden_claim`: secondary-controllable; clean cross-family replication; weak positive framing  
`paper_section`: Section 6 cross-model boundaries; Section 8 discussion; Appendix Mistral note

## 15. Mistral-7B Projection-to-Logit Diagnostic

`experiment_name`: Mistral-7B projection-to-logit diagnostic  
`model`: mistralai/Mistral-7B-Instruct-v0.3  
`pressure_type`: `belief_argument` archived English / Chinese / n=100 diagnostic bundle  
`n`: mixed frozen settings  
`metric_family`: projection-to-logit diagnostic  
`intervention_method`: archived belief-pressure diagnostic comparison  
`config`: English, Chinese, and n=100 diagnostic readouts combined for bounded mechanism interpretation  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_projection_logit_diagnostic`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_projection_logit_diagnostic/projection_alignment_summary.json`  
`main_numbers`: belief-logit deltas are stably strong across English, Chinese, and n=100 settings (`-5.68`, `-5.60`, `-5.80`); negative-control movement remains moderate (`-0.25`); specificity is moderate (`~23`); projection-drift correlation is weak (`-0.16`)  
`CI/uncertainty`: diagnostic summary only; no CI; this entry supports regime classification rather than behavioral closure by itself  
`allowed_claim`: logit-level signal is present, but it is only moderate in specificity and does not guarantee clean controllability  
`forbidden_claim`: stable clean-control signature; secondary support line; formal predictor confirmation  
`paper_section`: Section 5 diagnostic regime framework; Section 6 cross-model boundaries; Appendix diagnostic framework

## 16. Free-form 7-condition Update

`experiment_name`: Qwen 7B free-form diagnostic update (seven conditions)  
`model`: Qwen/Qwen2.5-7B-Instruct  
`pressure_type`: `belief_argument` free-form pressured / recovery prompts  
`n`: 50  
`metric_family`: free-form diagnostic boundary metrics (`drift_rate`, `wrong_follow_rate`, `readable_rate`, `repetition_rate`, `distinct-1`)  
`intervention_method`: patch-horizon sweep: `no_intervention`; `prefill_only`; `first_token_only`; `first_3_tokens`; `continuous`; `first_5_tokens`; `first_10_tokens`  
`config`: Qwen 7B mainline late-layer direction patched into open-ended generation with seven continuity conditions; five legacy conditions archived under `20260503`, two incremental conditions archived under `20260507`  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic`  
`output_result_path`: `/Users/shiqi/code/graduation-project/docs/reports/freeform_diagnostic_confirm_boundary_note_20260507.md`; `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503`; `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260507`  
`main_numbers`: from `no_intervention` to `prefill_only`, pressured drift drops `0.36 -> 0.10` and wrong-follow drops `0.42 -> 0.16`; `first_token_only` matches that front-loaded gain; `first_3_tokens` keeps the same behavioral rates but degrades readability to `0.42` with repetition `0.58`; `first_5_tokens` and `first_10_tokens` add no further behavioral gain and collapse pressured / recovery readability to `0.00` with repetition `1.00`  
`CI/uncertainty`: descriptive rates only; no CI cited in the frozen seven-condition note; baseline-patched free-form damage remains unmeasured  
`allowed_claim`: front-loaded diagnostic effect with a steep continuity-quality tradeoff; appendix / limitation / boundary evidence only  
`forbidden_claim`: clean free-form intervention success; longer-horizon usable control; measured baseline damage  
`paper_section`: Appendix free-form diagnostic; Section 10 limitations; diagnostic-framework boundary note

## 17. Regime Diagnostic Framework

`experiment_name`: Regime diagnostic framework  
`model`: mixed archived settings (`Qwen 7B`, `Qwen 3B`, `Qwen 14B`, `GLM-4-9B`, `Llama-3.1-8B`)  
`pressure_type`: mixed archived `belief_argument` settings plus bounded free-form and identity/profile boundary rows  
`n`: mixed; core matrix covers 7 settings drawn from the frozen result package  
`metric_family`: diagnostic framework / regime matrix (`target_logit_delta`, `negative_logit_delta`, `specificity_ratio`, `proj_drift_corr`, `baseline_damage`, `intervention_family_agree`, `freeform_stability`)  
`intervention_method`: mixed archived interventions; this is a synthesis framework rather than a new experimental method  
`config`: `7 settings x 7 indicators` core matrix plus supplementary boundary rows; prediction language kept bounded to regime diagnosis rather than formal prediction  
`source_data_path`: `/Users/shiqi/code/graduation-project/docs/reports/regime_diagnostic_matrix_20260507.md`  
`output_result_path`: `/Users/shiqi/code/graduation-project/docs/reports/regime_diagnostic_matrix_20260507.md`; `/Users/shiqi/code/graduation-project/docs/reports/regime_diagnostic_matrix_20260507.csv`  
`main_numbers`: core controllable rows are Qwen 7B mainline, Qwen 3B mainline, Qwen 14B n=100, and bounded free-form `prefill_only`; key counterexamples are GLM and Qwen 7B PSD as high-specificity false positives, plus Llama as `locatable-but-not-controllable`; the framework explicitly treats `specificity` as jointly informative but singly insufficient  
`CI/uncertainty`: framework-level synthesis only; no new CI; depends on the archived point estimates and CI-bearing upstream rows already frozen elsewhere  
`allowed_claim`: bounded diagnostic framework for controllability regimes; useful joint-profile explanation rather than a formal predictor  
`forbidden_claim`: universally valid controllability rule; `specificity` as a necessary structural condition; deployment-ready predictor  
`paper_section`: Section 8 diagnostic-framework discussion; Appendix diagnostic framework; Figures A-C

## 18. Held-out Prediction Result

`experiment_name`: Held-out prediction result  
`model`: mixed held-out settings from the diagnostic framework note  
`pressure_type`: mixed archived settings used for held-out regime comparison  
`n`: 6 held-out settings  
`metric_family`: held-out regime-prediction evaluation  
`intervention_method`: none; protocol evaluates predicted vs observed regime labels from the frozen diagnostic matrix  
`config`: predicted vs observed comparison under `heldout_prediction_protocol_20260507`; reports both strict-match and collapsed-family-match criteria  
`source_data_path`: `/Users/shiqi/code/graduation-project/docs/reports/heldout_prediction_result_20260507.md`  
`output_result_path`: `/Users/shiqi/code/graduation-project/docs/reports/heldout_prediction_result_20260507.md`  
`main_numbers`: strict match `4/6 = 0.667`; collapsed-family match `6/6 = 1.000`; mismatches occur on Qwen 3B mainline (`clean-or-secondary controllable` vs `clean-controllable`) and Qwen 14B n=100 (`clean-or-secondary controllable` vs `secondary-controllable`)  
`CI/uncertainty`: protocol-evaluation summary only; no CI; sample is too small to support a strong predictor claim  
`allowed_claim`: supports a bounded diagnostic hypothesis at the coarse regime-family level  
`forbidden_claim`: formal held-out predictor success; exact regime predictor; standalone benchmark result  
`paper_section`: Section 8 diagnostic-framework discussion; Appendix held-out prediction note

## 19. Authority-pressure Held-out Validation

`experiment_name`: Qwen 7B authority-pressure held-out validation  
`model`: Qwen/Qwen2.5-7B-Instruct  
`pressure_type`: `authority_pressure` held-out pressure-type validation on the belief-argument framework  
`n`: 100  
`metric_family`: mixed diagnostic + bridge-style behavioral closure, interpreted only as a bounded pressure-type held-out validation rather than as an unqualified leaderboard row  
`intervention_method`: belief-argument-derived intervention family applied to authority-pressure held-out data  
`config`: diagnostic reference uses the frozen belief-argument Qwen 7B mainline as the source framework; authority-pressure diagnostic and behavioral closure are archived separately and compared conservatively at the regime level  
`source_data_path`: `/Users/shiqi/code/graduation-project/docs/reports/authority_pressure_heldout_result_20260509.md`; `/Users/shiqi/code/graduation-project/docs/reports/authority_pressure_diagnostic_analysis_20260508.md`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_diagnostic/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_102251/projection_alignment_summary.json`; `/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_behavioral_closure/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_203029/belief_causal_summary.csv`  
`main_numbers`: diagnostic `belief_logit_delta -3.8392`, `negative_logit_delta -0.0054`, `specificity_ratio 713.61`, `proj_drift_corr -0.3050`; behavioral closure `drift_delta -0.3100`, `compliance_delta -0.3100`, `recovery_delta +0.3700`, `baseline_damage 0.0000`; predicted regime `secondary-controllable`; observed regime `clean-controllable`  
`CI/uncertainty`: report-level held-out summary only; no CI is claimed in the frozen short report; result is interpreted at the regime-matching level rather than as a new formal predictor benchmark  
`allowed_claim`: first pressure-type held-out validation for the bounded diagnostic framework; framework on `belief_argument` correctly predicts the controllability direction for `authority_pressure`, with the observed regime at least as strong as predicted  
`forbidden_claim`: universal cross-pressure validation; predictor failure because observed is stronger than predicted; proof that specificity alone establishes behavioral controllability; leaderboard-style merger with cross-model bridge rows  
`paper_section`: Section 5 diagnostic regime framework; Section 8 discussion; Appendix diagnostic framework

## 20. C13 Screening: InternLM2.5-7B

`experiment_name`: InternLM2.5-7B belief pressure-transfer screening  
`model`: InternLM2.5-7B  
`pressure_type`: `belief_argument` pressure-transfer screening under the frozen C13 rubric  
`n`: 100  
`metric_family`: bridge / transfer-style screening summary  
`intervention_method`: `matched_belief_subspace_damping`  
`config`: frozen formal n=100 screening run under the pre-registered C13 rule  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/internlm2_5_7b_chat/trainonly/internlm_internlm2_5-7b-chat`; `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/internlm2_5_7b_chat/n100/internlm_internlm2_5-7b-chat/20260509_031623`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/internlm2_5_7b_chat/n100/internlm_internlm2_5-7b-chat/20260509_031623/belief_causal_summary.csv`  
`main_numbers`: no-intervention `drift 0.00`, `compliance 0.42`, `recovery 1.00`, `damage 0.00`; intervened `drift 0.00`, `compliance 0.42`, `recovery 1.00`, `damage 0.00`; delta `drift 0.00`, `compliance 0.00`, `recovery 0.00`, `damage 0.00`  
`CI/uncertainty`: screening summary only; no CI cited in the current code-side report  
`allowed_claim`: negative screening result  
`forbidden_claim`: partial-positive / second non-Qwen positive
`paper_section`: Section 9 limitations; appendix / evidence-ledger boundary note

## 21. C13 Screening: Yi-1.5-6B

`experiment_name`: Yi-1.5-6B belief pressure-transfer screening  
`model`: Yi-1.5-6B  
`pressure_type`: `belief_argument` pressure-transfer screening under the frozen C13 rubric  
`n`: 100  
`metric_family`: bridge / transfer-style screening summary  
`intervention_method`: `matched_belief_subspace_damping`  
`config`: frozen formal n=100 screening run under the pre-registered C13 rule  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/yi_1_5_6b_chat/trainonly/01-ai_Yi-1.5-6B-Chat`; `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/yi_1_5_6b_chat/n100/01-ai_Yi-1.5-6B-Chat/20260509_033508`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/yi_1_5_6b_chat/n100/01-ai_Yi-1.5-6B-Chat/20260509_033508/belief_causal_summary.csv`  
`main_numbers`: no-intervention `drift 0.56`, `compliance 0.98`, `recovery 0.46`, `damage 0.00`; intervened `drift 0.54`, `compliance 0.93`, `recovery 0.50`, `damage 0.29`; delta `drift -0.02`, `compliance -0.05`, `recovery +0.04`, `damage 0.29`  
`CI/uncertainty`: screening summary only; no CI cited in the current code-side report  
`allowed_claim`: negative screening result  
`forbidden_claim`: partial-positive / second non-Qwen positive
`paper_section`: Section 9 limitations; appendix / evidence-ledger boundary note

## 22. C13 Screening: Baichuan2-7B

`experiment_name`: Baichuan2-7B belief pressure-transfer screening  
`model`: Baichuan2-7B  
`pressure_type`: `belief_argument` pressure-transfer screening under the frozen C13 rubric  
`n`: 100  
`metric_family`: bridge / transfer-style screening summary  
`intervention_method`: `matched_belief_subspace_damping`  
`config`: frozen formal n=100 screening run under the pre-registered C13 rule  
`source_data_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/baichuan2_7b_chat/trainonly/baichuan-inc_Baichuan2-7B-Chat`; `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/baichuan2_7b_chat/n100/baichuan-inc_Baichuan2-7B-Chat/20260509_034823`  
`output_result_path`: `/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/baichuan2_7b_chat/n100/baichuan-inc_Baichuan2-7B-Chat/20260509_034823/belief_causal_summary.csv`  
`main_numbers`: no-intervention `drift 0.33`, `compliance 0.78`, `recovery 0.66`, `damage 0.00`; intervened `drift 0.28`, `compliance 0.68`, `recovery 0.72`, `damage 0.07`; delta `drift -0.05`, `compliance -0.10`, `recovery +0.06`, `damage 0.07`  
`CI/uncertainty`: screening summary only; no CI cited in the current code-side report; near-threshold under the strict rubric but still negative because drift does not satisfy the strict `< -0.05` cutoff  
`allowed_claim`: negative screening result  
`forbidden_claim`: partial-positive / second non-Qwen positive
`paper_section`: Section 9 limitations; appendix / evidence-ledger boundary note

## Bottom Line

当前最稳的证据台账边界是：

- Qwen 7B = formal mainline under objective-local proxy
- Qwen 3B = formal replication under the same proxy family
- Qwen controls + held-out = robustness-side support, not new main results
- Qwen14B / GLM / Llama n=100 = bridge / transfer-style boundary evidence, with updated cross-model wording
- Llama / GLM layer-wise sweeps = no hidden clean rescue window; they sharpen the cross-model boundary rather than upgrade those lines
- free-form seven-condition update = front-loaded effect plus a steeper continuity-quality collapse, still appendix-strength boundary evidence
- regime diagnostic framework + held-out prediction = bounded explanatory scaffold, not a formal controllability predictor
- no second non-Qwen partial-positive was found under the frozen rubric; Baichuan remains a near-threshold boundary row rather than a positive support line
- free-form diagnostic / prompt-family / intervention-family / identity-profile / human audit = appendix-strengthening or limitation-boundary material, not primary closure replacements
