# 结果分析交接摘要

更新时间：2026-04-22

本文档用于接续 `/Users/shiqi/code/graduation-project` 当前阶段的正式结果整理，并覆盖早于 2026-04-17 严格重建之前的 detector/recheck 口径。

## 1. 当前正式结果入口

当前最应优先引用的正式结果目录是：

- objective 主基线：
  `outputs/experiments/deepseek_200_n3_verified/20260330_201301`
- detector 主线：
  `outputs/experiments/interference_detector_new15_full_sentence_embedding`
- strict rebuilt full real recheck：
  `outputs/experiments/full_real_recheck_rebuilt/20260417_113758`

其中，`full_real_recheck_rebuilt/20260417_113758` 是在修复 merge key 不唯一导致的笛卡尔膨胀问题后，按下列严格口径重建的结果：

- 只认原始 scored dataset
- 只认 `sample_manifest.csv`
- 只认 `judge_recheck_results.jsonl`

早于这次严格重建的 full-data detector + real recheck 汇总结论，不再作为正式汇报主口径。

## 2. 当前正式汇报口径

当前最可信、最适合正式汇报的是：

- `baseline accuracy = 0.991804`
- 口径 B：全模型 detector + real recheck
  `accuracy = 0.995951`
- 口径 C：非 Reasoner detector + real recheck，Reasoner 不做 detector / 不做 recheck
  `accuracy = 0.996346`
- raw judge requests = `1831`
- strict recoverable rows = `1581`
- unmatched / orphan rows = `250`

据此，当前应固定的说法是：

1. 旧的“full-data real recheck 低于 baseline”结论应撤回。
2. 当前正式主口径应优先采用口径 C。
3. `deepseek-reasoner` 的 same-model real recheck 不适合进入正式方案。

## 3. detector 的稳定表述

detector 主线仍固定使用：

- `sentence-embedding-logreg`
- `strict`
- `matched_trigger_budget`

稳定表述保持不变：

- detector 更适合作为 `risk ranker / selective recheck trigger policy`
- 不应包装成高精度单次强分类器

这部分仍可优先引用：

- [`detector_model_comparison.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/detector_model_comparison.csv)
- [`threshold_sweep_sentence-embedding-logreg_strict.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/threshold_sweep_sentence-embedding-logreg_strict.csv)
- [`guard_eval_summary_sentence_embedding_new15_full.json`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/guard_eval_summary_sentence_embedding_new15_full.json)

## 4. intervention 主实验正式口径

当前 intervention 主实验应单独作为一条新的正式口径保留，且默认方法边界必须写清楚。

### 4.1 默认 baseline 锁定

3B 默认 baseline 正式锁定为：

- `baseline_state_interpolation`
- layer = `31-35`
- scale = `0.6`

7B 默认 baseline 正式锁定为：

- `baseline_state_interpolation`
- layer = `24-26`
- scale = `0.6`

7B `24-27, 0.6` 只保留为：

- `aggressive secondary setting`

原因是：

- 它在 `strict_positive` 上更强
- 但会牺牲 `high_pressure_wrong_option` 的稳健性
- 并引入少量 baseline damage

`late_layer_residual_subtraction` 在 3B / 7B 主实验里都不适合作为正式 baseline，也不应写成可用主方法。

### 4.2 3B 主 baseline

正式证据文件：

- [`intervention_summary.json`](/Users/shiqi/code/graduation-project/outputs/experiments/local_probe_qwen3b_intervention_main/baseline_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142847/intervention_summary.json)

推荐在论文/汇报里直接写：

- `strict_positive`: recovery = `0.8125`, wrong-follow = `0.10` (ref `0.48`), damage = `0`, net = `0.4194`
- `high_pressure_wrong_option`: recovery = `0.5`, wrong-follow = `0.24` (ref `0.42`), damage = `0`, net = `0.2381`

稳定表述：

- 3B 主 baseline 在 `strict_positive` 上恢复最明显
- 在 `high_pressure_wrong_option` 上也有净收益
- 两个主分组都没有 baseline damage

### 4.3 7B 主 baseline

正式证据文件：

- [`intervention_summary.json`](/Users/shiqi/code/graduation-project/outputs/experiments/local_probe_qwen7b_intervention_main/baseline_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140142/intervention_summary.json)

推荐在论文/汇报里直接写：

- `strict_positive`: recovery = `0.5455`, wrong-follow = `0.20` (ref `0.36`), damage = `0`, net = `0.1463`
- `high_pressure_wrong_option`: recovery = `0.9`, wrong-follow = `0.10` (ref `0.26`), damage = `0`, net = `0.2368`

稳定表述：

- 7B 主 baseline 不是 `strict_positive` 上最激进的一档
- 但在两个主分组之间更均衡
- 尤其在 `high_pressure_wrong_option` 上更稳，而且没有 baseline damage

### 4.4 7B aggressive secondary setting

正式证据文件：

- [`intervention_summary.json`](/Users/shiqi/code/graduation-project/outputs/experiments/local_probe_qwen7b_intervention_main/aggressive_24_27_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140518/intervention_summary.json)

推荐写法：

- `strict_positive` 更强：recovery = `0.7273`, wrong-follow = `0.12`, net = `0.1951`
- 但 `high_pressure_wrong_option` 更弱且更不安全：recovery = `0.8`, damage = `0.0263`, net = `0.1842`

因此：

- 它可以作为 secondary / aggressive 对照
- 不应替代 `24-26, 0.6` 成为 7B 默认 baseline

### 4.5 subtraction control 的边界

7B 对照：

- [`intervention_summary.json`](/Users/shiqi/code/graduation-project/outputs/experiments/local_probe_qwen7b_intervention_main/subtraction_control_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140853/intervention_summary.json)

3B 对照：

- [`intervention_summary.json`](/Users/shiqi/code/graduation-project/outputs/experiments/local_probe_qwen3b_intervention_main/subtraction_control_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142938/intervention_summary.json)

当前应固定的说法是：

- `late_layer_residual_subtraction` 在 7B 上总体净收益为负，且有 baseline damage
- 它在 3B 上同样不稳，overall net 已接近或低于 0，并出现明显 baseline damage
- 因此它只能保留为 failure / control evidence，不应写成正式可用方法

## 5. white-box mechanistic 子研究线分级

当前 white-box 子研究线需要明确分级，不能把所有观察都写成同等强度的 mechanistic mitigation 结果。

### 5.1 belief_argument 主线

`belief_argument` 可继续保留为正式的 mechanistic mitigation 结果线。

当前文档层面的稳定写法是：

- `belief_argument` 仍属于可以进入正式 mechanistic mitigation 叙述的主线
- 后续若写论文或汇报，`belief_argument` 与 intervention 主实验可以并列，但应与 `identity_profile` 分开

可作为结果锚点的目录包括：

- `outputs/experiments/local_probe_qwen3b_mechanistic/...`
- `outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32/...`
- `outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32_generalization/...`

### 5.1.1 14B 最小因果验证

14B `belief_argument` 最小因果验证结果应放在主文中，但定位只能是：

- `secondary causal confirmation`

不能写成：

- 新的 main mitigation result
- 一个单独展开的新主结果节
- “14B 上已建立强 mitigation”

正式证据目录：

- `outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308`

可直接引用的证据文件：

- [`qwen14b_belief_causal_summary.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308/qwen14b_belief_causal_summary.csv)
- [`qwen14b_belief_causal_run.json`](/Users/shiqi/code/graduation-project/outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308/qwen14b_belief_causal_run.json)
- [`qwen14b_belief_subspace_summary.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308/qwen14b_belief_subspace_summary.csv)

当前应固定的关键结论是：

- `matched_belief_subspace_damping` 降低了 pressured-stage `stance_drift_rate` 与 `pressured_compliance_rate`
- `matched_negative_control` 与 `no_intervention` 基本相同
- 这支持 `belief_argument` 的 cross-source causal transfer

必须保留的证据边界：

- `n = 24`
- `recovery_rate` 未提升，仍为 `0.7917`
- `baseline_damage_rate = 0.0417`
- 因此它只能作为小样本因果确认，而不是新的主结果

建议在主文中只用一个小表或一个短段落呈现，不单开大节。

可直接使用的中文克制表述：

> 在 14B 的小样本 cross-source 因果验证中，`matched_belief_subspace_damping` 将 pressured-stage `stance_drift_rate` 从 `0.50` 降至 `0.2917`，并将 `pressured_compliance_rate` 从 `0.9583` 降至 `0.7083`；相比之下，`matched_negative_control` 与 `no_intervention` 基本一致。这为 `belief_argument` 提供了一条 secondary causal confirmation，但由于样本量仅 `n=24`、`recovery_rate` 未提升且 `baseline_damage_rate = 0.0417`，当前证据仍只应被解释为小样本因果确认，而非新的主 mitigation 结果。

可直接使用的英文克制表述：

> In a 14B small-sample cross-source causal transfer check, `matched_belief_subspace_damping` reduced pressured-stage `stance_drift_rate` from `0.50` to `0.2917` and `pressured_compliance_rate` from `0.9583` to `0.7083`, while the matched negative control remained indistinguishable from no intervention. We therefore treat this result as a secondary causal confirmation for `belief_argument`, not as a new main mitigation result, because the sample is small (`n=24`), `recovery_rate` does not improve, and `baseline_damage_rate = 0.0417`.

### 5.1.2 GLM 与 Llama 跨家族复测边界

跨家族复测需要区分 GLM 与 Llama：

- GLM：`cross-family positive replication with stronger tradeoff`
- Llama-3.1-8B English bridge prompt：`weak replication / limitation`

GLM 可展开写为：

> GLM reproduces the belief-subspace damping direction across model families, but with a substantially stronger utility/safety tradeoff than the Qwen line.

Llama 结果应放在 limitation 或 appendix 中，不应写成 positive replication。

Llama 正式证据目录：

- `outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011`

可直接引用的证据文件：

- [`belief_causal_summary.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_summary.csv)
- [`belief_causal_subspace_summary.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_subspace_summary.csv)
- [`belief_causal_run.json`](/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_run.json)

当前应固定的 Llama 结论是：

- 英文 bridge prompt 下，Llama 的 belief subspace 更清晰
- explained variance 约 `0.64-0.68`
- abs coherence 约 `0.40-0.44`
- 这些 localization 指标高于此前中文/混合设置
- 但 intervention transfer weak

必须保留的 Llama intervention 边界：

- 低 alpha 基本不降低 drift 或 compliance
- `alpha = 0.5` 仅小幅降低 drift，从 `0.4583` 到 `0.375`
- compliance 仍为 `1.0`
- recovery 从 `0.6667` 降至 `0.5833`
- baseline damage = `0.25`

最终定位应写为：

- `mechanism locatable, intervention transfer weak`
- `weak replication / limitation`

不能写成：

- positive replication
- Llama 上已建立可用 mitigation
- 常规 alpha / k / layer sweep 的后续建议

如需记录一个后续机制诊断方向，只保留：

- `projection-to-logit diagnostic`
- `causal alignment diagnostic`

可直接使用的中文克制表述：

> 在英文 bridge prompt 复测中，Llama-3.1-8B 的 belief subspace 比此前中文/混合设置更清晰，前中层 explained variance 约为 `0.64-0.68`，abs coherence 约为 `0.40-0.44`。但这种可定位性没有稳定转化为可控干预：低 alpha 基本不降低 drift 或 compliance，`alpha = 0.5` 仅将 drift 从 `0.4583` 小幅降至 `0.375`，compliance 仍为 `1.0`，同时 recovery 下降到 `0.5833` 且 baseline damage 达到 `0.25`。因此，Llama 结果应被整理为 `mechanism locatable, intervention transfer weak`，即 weak replication / limitation，而不是 positive replication。

可直接使用的英文克制表述：

> Llama-3.1-8B shows a more identifiable belief-pressure subspace under the English bridge prompt, but intervention transfer remains weak; the strongest tested damping setting only slightly reduces drift while leaving pressured compliance unchanged, lowering recovery, and introducing substantial baseline damage.

### 5.2 identity_profile 线

`identity_profile` white-box 子研究线当前只能保留为：

- `weakly supported mechanistic observation`

不能升格为：

- 正式副线
- 已建立的 prefix-specific 机制干预结果
- `identity-specific mitigation`

当前证据边界应写清楚：

- localization 证据存在
- causal intervention support 不足
- `profile_prefix_gating` 没有稳定优于 `matched_early_mid_negative_control`
- 因此当前不能 claim `identity-specific mitigation`

7B follow-up 正式证据目录：

- `outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058`

可直接引用的证据文件包括：

- [`identity_vs_belief_localization_summary.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058/identity_vs_belief_localization_summary.csv)
- [`identity_profile_intervention_summary.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058/identity_profile_intervention_summary.csv)
- [`identity_profile_run.json`](/Users/shiqi/code/graduation-project/outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058/identity_profile_run.json)

建议在论文/汇报里固定使用下面这类表述：

- `identity_profile` 线目前只提供了弱支持的 localization observation
- 但 prefix-specific causal intervention 还没有拉开 negative control
- 因此这条线暂不升格为正式副线

如后续用户决定补实验，只建议记录一个可能下一步：

- `prefix-span ablation + replay test`

当前不建议继续写：

- 多轮 alpha sweep
- 多轮 control sweep

## 6. 与历史 same-model 文档的关系

以下目录和文档仍可保留，但应仅作为历史分析或风险案例参考：

- `outputs/experiments/same_model_guarded_pilot/20260412_021656`
- `outputs/experiments/same_model_full_real/...`
- [`docs/reports/RECHECK_CASE_ANALYSIS_20260416.md`](/Users/shiqi/code/graduation-project/docs/reports/RECHECK_CASE_ANALYSIS_20260416.md)
- [`docs/reports/RESULT_ANALYSIS_HANDOFF_20260413.md`](/Users/shiqi/code/graduation-project/docs/reports/RESULT_ANALYSIS_HANDOFF_20260413.md)

原因不是这些材料没有信息量，而是：

- 其中部分结论形成于严格重建之前
- same-model self-recheck 尤其是 Reasoner lane，不再适合作为当前正式方案
- 这些材料更适合支撑“失败模式 / 风险案例 / 题目级系统性误判”分析，而不是主结果结论

## 7. 当前建议的文档引用顺序

1. [`docs/reports/RESULT_ANALYSIS_HANDOFF_20260422.md`](/Users/shiqi/code/graduation-project/docs/reports/RESULT_ANALYSIS_HANDOFF_20260422.md)
2. [`docs/PAPER_ANALYSIS_ENTRYPOINT.md`](/Users/shiqi/code/graduation-project/docs/PAPER_ANALYSIS_ENTRYPOINT.md)
3. [`docs/reports/WHITEBOX_MECHANISTIC_WORKSPACE_SUMMARY_20260426.md`](/Users/shiqi/code/graduation-project/docs/reports/WHITEBOX_MECHANISTIC_WORKSPACE_SUMMARY_20260426.md)
4. [`docs/reports/WHITEBOX_MECHANISTIC_EVIDENCE_MATRIX_20260426.md`](/Users/shiqi/code/graduation-project/docs/reports/WHITEBOX_MECHANISTIC_EVIDENCE_MATRIX_20260426.md)
5. [`docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`](/Users/shiqi/code/graduation-project/docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md)
6. detector 主线结果目录
7. strict rebuilt full real recheck 目录
8. local probe intervention 主实验目录
9. white-box mechanistic 主线与 follow-up 目录

## 8. 后续整理建议

- 后续报告、汇报稿、答辩材料如涉及 detector + real recheck，请默认先核对是否已切换到 `full_real_recheck_rebuilt/20260417_113758`
- 如涉及 local probe intervention，请默认先核对 3B 是否锁定为 `31-35, 0.6`、7B 是否锁定为 `24-26, 0.6`
- 7B `24-27, 0.6` 只能作为 aggressive secondary setting 引用，不应覆盖默认 baseline
- `late_layer_residual_subtraction` 不应写成正式可用主方法
- 如涉及 white-box mechanistic 叙述，请把 `belief_argument` 与 `identity_profile` 分开
- 如涉及 white-box 主会表格，请优先引用 `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/whitebox_effect_size_table.csv`
- Qwen 3B / 7B 的 `stance_drift` 与 `recovery` 是 objective-local proxy 口径，不要与 bridge causal lines 做完全同义的跨范式 leaderboard
- 如涉及 14B belief causal transfer，请只把它写成 `secondary causal confirmation`，并放在主文小表或短段落里
- 不要把 14B belief causal transfer 写成新的主结果节或“已建立强 mitigation”
- 如需记录后续最小补实验，只保留“同配置扩样到 `n=48`”
- 如涉及 GLM/Llama 跨家族复测，请区分 GLM 是 positive replication with stronger tradeoff，Llama 是 weak replication / limitation
- Llama 英文 bridge prompt 复测只能放在 limitation 或 appendix，不能写成 positive replication
- Llama limitation 可直接引用 `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/llama_limitation_summary.md`
- Llama 后续只保留 `projection-to-logit diagnostic` 与 `causal alignment diagnostic`
- 停止后续 Llama alpha / k / layer sweep
- `identity_profile` 当前只能写成 weakly supported mechanistic observation，不能写成正式副线或 identity-specific mitigation
- 如需记录后续补实验方向，只保留 `prefix-span ablation + replay test`
- 如果文档仍引用 same-model self-recheck 作为“当前最可信正式方案”，应改为历史参考口径
- 如需补充案例分析，优先写“为什么口径 C 更稳”与“Reasoner 为何不进入正式方案”，而不是继续放大 same-model 成功案例
