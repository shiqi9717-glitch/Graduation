# Authority-Pressure Held-out Result Report

更新时间：2026-05-09

本报告只基于已冻结的 C8 diagnostic 和 C9 behavioral closure 做 held-out result 总结，不跑新实验，不改代码，不改论文正文。

## 1. Diagnostic Signals 对比

| signal | belief_argument (Qwen 7B mainline) | authority_pressure (Qwen 7B) | change |
| --- | ---: | ---: | --- |
| belief_logit_delta | -5.3871 | -3.8392 | -28.7% weaker |
| negative_logit_delta | -0.0050 | -0.0054 | unchanged in practice |
| specificity_ratio | 1083.72 | 713.61 | lower, but still high |
| proj_drift_corr | -0.3899 | -0.3050 | same direction, weaker magnitude |

Interpretation: authority-pressure retains the same near-zero negative-control signature as the belief-argument mainline, but the target-logit pull is weaker. This is exactly the profile that motivated a conservative `secondary-controllable` prediction rather than a full-strength mainline expectation.

## 2. Behavioral Closure 对比

| metric | belief_argument mainline | authority_pressure | change |
| --- | ---: | ---: | --- |
| drift_delta | -0.1493 | -0.3100 | stronger |
| compliance_delta | -0.1595 | -0.3100 | stronger |
| recovery_delta | +0.7127 | +0.3700 | weaker |
| baseline_damage | 0.0000 | 0.0000 | unchanged |

The authority-pressure run satisfies the same clean-controllability screen used elsewhere:

| condition | observed | satisfied? |
| --- | ---: | --- |
| `drift_delta < -0.05` | -0.3100 | yes |
| `compliance_delta < -0.05` | -0.3100 | yes |
| `recovery_delta >= -0.05` | +0.3700 | yes |
| `baseline_damage < 0.10` | 0.0000 | yes |

So the observed authority-pressure regime is best labeled `clean-controllable`, not merely `secondary-controllable`.

## 3. Predicted vs Observed

| setting | diagnostic signals | predicted regime | observed regime | match? | interpretation |
| --- | --- | --- | --- | --- | --- |
| Qwen 7B + authority_pressure | specificity retained; target-logit about 29% weaker than belief-argument mainline | secondary-controllable | clean-controllable | yes | predicted direction is correct; observed behavioral closure is stronger than the conservative forecast |

Why did the observed row come out stronger than predicted? The most stable explanation is not that the intervention itself became intrinsically stronger, but that authority pressure created a larger correction margin. In the no-intervention baseline, authority pressure shows higher susceptibility than the belief-argument mainline: `stance_drift_rate 0.37` vs `0.22`, and `pressured_compliance_rate 0.93` vs `0.76`. That larger baseline distortion gives the same intervention family more room to reduce absolute error. At the same time, the recovery side is still weaker than the belief-argument mainline in absolute delta terms: authority pressure gives `recovery_delta = +0.3700`, versus `+0.7127` in the frozen mainline. That pattern is consistent with the original diagnostic forecast: same direction, weaker intrinsic leverage, but applied to a pressure setting with larger baseline vulnerability.

## 4. Interpretation

- This is the first pressure-type held-out validation for the diagnostic framework.
- The framework built on `belief_argument` pressure correctly predicted the controllability direction for `authority_pressure` on the same model.
- The prediction was conservative rather than wrong: the held-out row landed in a regime that is at least as strong as predicted, while preserving `baseline_damage = 0.00`.

## 5. Bottom Line

Paper-safe wording:

> The diagnostic framework, built on belief-argument pressure, correctly predicts the controllability direction for authority pressure on the same model, with the observed regime at least as strong as predicted.

Do not write:

> The framework universally predicts controllability across all pressure types.

## Source Paths

- C8 diagnostic summary: [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_diagnostic/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_102251/projection_alignment_summary.json:1)
- C9 behavioral summary: [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_behavioral_closure/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_203029/belief_causal_summary.csv:1)
- Mainline ledger reference: [FINAL_EVIDENCE_LEDGER_20260508.md](/Users/shiqi/code/graduation-project/docs/reports/FINAL_EVIDENCE_LEDGER_20260508.md:1)
