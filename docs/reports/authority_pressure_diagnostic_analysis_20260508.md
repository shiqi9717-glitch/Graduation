# Authority-Pressure Diagnostic Analysis

更新时间：2026-05-08

本说明只回答 diagnostic 层面的问题，不把这轮写成正式 behavioral closure，也不把附带 `belief_causal_summary.csv` 当作正式结果表。

## Conclusions

1. `authority-pressure` 部分沿用了 `belief_argument` 下的 target-logit specificity，但强度更弱。Qwen 7B 主 belief-direction 的冻结参考值大约是 `mean_belief_logit_delta = -5.3871`、`mean_negative_logit_delta = -0.0050`；这轮 authority-pressure diagnostic 给出的是 `-3.8392` 和 `-0.00538`。因此，negative-control 近零这一“特异性骨架”基本保留了，但 target-logit pull 明显减弱，不像原主线那样强。

2. 当前 diagnostic 信号更像 `positive transfer with mismatch`，而不是 clean positive transfer。summary 层面上，`100/100` 个样本都满足 `abs(negative_logit_delta_vs_no) < 0.05`，说明 negative-control side 非常干净；但 item-level 上只有 `23/100` 达到 `belief_logit_delta <= -5`，`80/100` 达到 `<= -3`，同时仍有 `8/100` 出现非负的 belief-logit delta。也就是说，方向并没有完全失配，但 target 侧的强度分布明显更散、更弱，因此最稳妥的诊断口径是 mixed-positive rather than clean transfer.

3. 建议协调器继续推进 C9，但应把它定位成“有诊断理由的 follow-up”，而不是预期中的正式成功复现。当前 diagnostic 已经给出一个值得追的信号：`mean_negative_logit_delta` 仍近零、`corr_pressured_projection_with_stance_drift = -0.3050` 也保持了与主线同向的弱负相关；但 `baseline_projection_as_fraction_of_pressured_projection = 1.2681` 和较弱的 `mean_belief_logit_delta = -3.8392` 一起说明，它更像一个可疑似转移、但尚未完成行为确认的 setting。最稳建议是：可以推进 C9 做正式 behavioral check，但不要把 C8 本身写成 regime 定稿。

## Boundary

- 不把 C8 写成 C9。
- 不把附带 [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_diagnostic/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_102251/belief_causal_summary.csv:1) 写成正式 behavioral closure 结果。
- 不把 projection specificity 直接写成已证实 behavioral controllability。

## Source Paths

- Diagnostic summary: [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_diagnostic/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_102251/projection_alignment_summary.json:1)
- Diagnostic rows: [projection_alignment_diagnostic.csv](/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_diagnostic/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_102251/projection_alignment_diagnostic.csv:1)
- Run manifest: [authority_pressure_diagnostic_manifest.json](/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_diagnostic/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_102251/authority_pressure_diagnostic_manifest.json:1)
- Reference mainline diagnostic: [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_projection_logit_diagnostic/Qwen_Qwen2.5-7B-Instruct/20260501_183852/projection_alignment_summary.json:1)
