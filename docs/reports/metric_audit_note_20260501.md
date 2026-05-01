# Metric Audit Note

审计范围：
- `src/open_model_probe/intervention.py`
- `scripts/summarize_intervention_metrics.py`
- `scripts/export_whitebox_objective_local_eval.py`
- `scripts/run_whitebox_formal_controls.py`
- `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/README.md`
- `outputs/experiments/local_probe_qwen7b_intervention_main/baseline_24_26_s06/.../intervention_run.json`
- `outputs/experiments/whitebox_qwen7b_heldout_eval/qwen7b_heldout_mainline/20260427_135639/objective_local_eval_manifest.json`
- `docs/reports/WHITEBOX_REPRODUCIBILITY_INDEX_20260428.md`

## Metric Definition Audit

- `stance_drift_delta`: 一致。`README` 定义为 `intervention_rate - reference_rate`，在 `export_whitebox_objective_local_eval.py` 中实现为 `intervention_error_rate - reference_error_rate`；对 Qwen mainline 来说，确实是 interference-induced-error proxy，负值更好。
- `stance_drift_delta` 分母：一致。`intervention.py` 与 `export_whitebox_objective_local_eval.py` 都先筛 `baseline_reference_correct`，再在这个集合上计算 reference/intervention error rate，因此 drift 的分母是 `baseline_correct_den`，不是全部样本。
- `pressured_compliance_delta`: 一致。`README` 定义为 `intervention_rate - reference_rate`，代码实现为 `wrong_follow_intervened - wrong_follow_reference`，负值更好。
- `pressured_compliance_delta` 分母：一致。代码在 `FOCUS_SAMPLE_TYPES = {strict_positive, high_pressure_wrong_option}` 上直接求 wrong-option-follow 平均值，因此 compliance 的分母是全部 focus rows，不要求 baseline 正确。
- `recovery_delta`: 一致。`README` 说明 Qwen mainline 的 reference 为 structural zero；代码里 `recovery_delta` 直接等于 `recovery_rate`，即 reference error rows 上 `intervention_recovers_error` 的比例，因此正值更好。
- `recovery_delta` 分母：一致。代码只在 `reference_error = baseline_correct ∩ interference_induced_error_reference` 上计算 recovery，因此 recovery 的分母是 reference error rows，不是全部样本。
- `baseline_damage_rate`: 一致。代码在 `baseline_correct_den` 上计算 `baseline_damage` 比例，和 `README` 的“intervention baseline-damage rate; lower is better”一致。
- `benefit_sum_no_damage` / `net_effect_score`: 无法确认。它们在 `README` 和 `whitebox_effect_size_table.csv` 中有定义，但不在本次要求核对的两个主代码文件中实现；当前只确认表中符号方向与 `README` 一致。

## Sign / Direction Audit

- 符号方向：一致。`stance_drift_delta` 与 `pressured_compliance_delta` 都是 `intervention - reference`，因此负值更好；`recovery_delta` 是 intervention recovery rate，因此正值更好；`baseline_damage_rate` 越低越好。
- `summarize_intervention_metrics.py`: 一致但范围有限。该脚本只打印 `intervention_recovery_rate`、`wrong_option_follow_rate`、`baseline_damage_rate` 和 `net_recovery_without_damage`，不直接计算 closure delta 或 CI，因此不能单独作为 closure 主表公式来源。

## Bootstrap / CI Audit

- held-out objective-local CI：一致。`export_whitebox_objective_local_eval.py` 使用 item-level bootstrap，默认 `2000` 次，`random.Random(seed)` 重采样 `focus_rows`，并用排序后的 `2.5% / 97.5%` 分位数取 `ci_low/ci_high`，属于 percentile bootstrap，不是 normal approximation。
- formal-controls aggregate CI：一致。`run_whitebox_formal_controls.py` 先对 shared sample IDs 做 item-level bootstrap，再在每次重采样后对各 seed run 的 metric 取平均，最后用排序后的 `2.5% / 97.5%` 分位数取区间，也属于 percentile-style bootstrap，不是 normal approximation。
- frozen closure 主表 CI：无法确认。`whitebox_effect_size_table.csv` 与 `README` 给出了 CI，但当前审计输入未直接包含其生成脚本，因此无法仅凭 `intervention.py` 和 `summarize_intervention_metrics.py` 确认 closure 主表的 bootstrap 次数与实现细节。

## Leakage Detection

Qwen 7B held-out 泄漏判断：`clean`。现有脚本链路显示 `24-26` 与 `0.6` 已在 `scripts/run_qwen7b_intervention_main_mps.sh` 中先固定用于 2026-04-23 的 mainline 和 sanity runs，而 held-out 样本是 2026-04-27 由 `build_whitebox_objective_local_heldout.py` 重新从原始 CMMLU 抽样、并显式排除 frozen mainline sample IDs 后才生成；没有证据表明 held-out 指标被反向用于选层或选强度。

边界说明：这只能说明“没有看到 held-out-assisted parameter selection 的迹象”，不能把 Qwen 7B mainline升级成严格 pre-registered clean split 设计。`24-26 @ 0.6` 仍然是在 mainline curated subset 及其相邻 sanity/secondary runs 上固定下来的默认设置，因此独立性边界应继续保持为 `robustness-side validation`, 而不是更强的 clean-split main result。
