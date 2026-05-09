# Held-out Prediction Result

更新时间：2026-05-07

本报告按 `heldout_prediction_protocol_20260507.md` 执行，只使用 R1 产出的诊断矩阵，不跑新实验。

## Predicted vs Observed

| setting | predicted_regime | observed_regime | strict_match | collapsed_family_match |
| --- | --- | --- | --- | --- |
| Qwen 3B `baseline_state_interpolation (31-35, beta=0.6)` | clean-or-secondary controllable | clean-controllable | No | Yes |
| Qwen 14B `matched_belief_subspace_damping n=100` | clean-or-secondary controllable | secondary-controllable | No | Yes |
| Qwen 7B `free-form prefill_only` | front-loaded controllable-but-bounded | front-loaded controllable-but-bounded | Yes | Yes |
| Qwen 7B `free-form first_3_tokens` | quality-limited free-form boundary | quality-limited free-form boundary | Yes | Yes |
| Qwen 7B `free-form continuous` | generation-unstable | generation-unstable | Yes | Yes |
| Qwen 7B `identity/profile follow-up` | boundary / insufficient-transfer-evidence | boundary / insufficient-transfer-evidence | Yes | Yes |

## Match Statistics

- strict match: `4/6 = 0.667`
- collapsed-family match: `6/6 = 1.000`

## Mismatch Review

- Qwen 3B `baseline_state_interpolation (31-35, beta=0.6)`: predicted `clean-or-secondary controllable` vs observed `clean-controllable`.
- Qwen 14B `matched_belief_subspace_damping n=100`: predicted `clean-or-secondary controllable` vs observed `secondary-controllable`.

## Conclusion

The current held-out check supports the bounded diagnostic hypothesis. High target-logit movement plus low damage transfers better than specificity alone, while free-form quality collapse is best predicted on a separate generation-stability axis.