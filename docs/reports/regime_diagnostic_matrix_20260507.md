# Regime Diagnostic Matrix

更新时间：2026-05-07

本报告只基于冻结 source files 计算诊断矩阵，不跑新实验，不改代码，不改论文正文。主表按 `7 settings x 7 indicators` 组织；补充 boundary rows 放在后文。

## Core 7x7 Matrix

| setting | D1 target_logit_delta | D2 negative_logit_delta | D3 specificity_ratio | D4 proj_drift_corr | D5 baseline_damage | D6 intervention_family_agree | D7 freeform_stability | predicted_regime | observed_regime |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Qwen 7B `baseline_state_interpolation (24-26, beta=0.6)` | -5.39 (strong) | -0.0050 (near-zero) | 1083.72 (very-high) | -0.39 (aligned) | 0.0000 (clean) | partial | N/A | specific-but-not-controllable | clean-controllable |
| Qwen 3B `baseline_state_interpolation (31-35, beta=0.6)` | -5.10 (strong) | -0.9303 (supportive) | 5.48 (moderate) | -0.52 (aligned) | 0.0000 (clean) | N/A | N/A | clean-or-secondary controllable | clean-controllable |
| Qwen 14B `matched_belief_subspace_damping n=100` | -7.33 (strong) | -1.4254 (supportive) | 5.14 (moderate) | 0.08 (unclear) | 0.0100 (clean) | N/A | N/A | clean-or-secondary controllable | secondary-controllable |
| GLM-4-9B `pressure_subspace_damping n=100` | -2.87 (moderate) | -0.0011 (near-zero) | 2580.02 (very-high) | -0.45 (aligned) | 0.0600 (residual) | N/A | N/A | weak / tradeoff-limited | weak / tradeoff-limited |
| Llama-3.1-8B `pressure_subspace_damping n=100` | -1.83 (weak) | 0.1716 (bad-positive) | 10.65 (moderate) | -0.29 (weak) | 0.0800 (residual) | N/A | N/A | locatable-but-not-controllable | locatable-but-not-controllable |
| Qwen 7B `pressure_subspace_damping n=100` | -5.90 (strong) | -0.0017 (near-zero) | 3387.65 (very-high) | -0.41 (aligned) | 0.0400 (residual) | partial | N/A | specific-but-not-controllable | weak / tradeoff-limited |
| Qwen 7B `free-form prefill_only` | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | partial | stable | front-loaded controllable-but-bounded | front-loaded controllable-but-bounded |

Trace note: every numeric field is exported to the machine-readable CSV together with its source path and source field name where available.

## Controllability Label Process

本轮使用以下 binary label：

```python
controllability_label = 1 if observed_regime in {
    "clean-controllable",
    "secondary-controllable",
    "front-loaded controllable-but-bounded",
} else 0
```

| setting | observed_regime | label | justification |
| --- | --- | ---: | --- |
| Qwen 7B `baseline_state_interpolation (24-26, beta=0.6)` | clean-controllable | 1 | behavioral closure is directionally positive with low damage |
| Qwen 3B `baseline_state_interpolation (31-35, beta=0.6)` | clean-controllable | 1 | behavioral closure is directionally positive with low damage |
| Qwen 14B `matched_belief_subspace_damping n=100` | secondary-controllable | 1 | behavioral closure is directionally positive with low damage |
| GLM-4-9B `pressure_subspace_damping n=100` | weak / tradeoff-limited | 0 | row remains weak, tradeoff-limited, control-like, or method-boundary only |
| Llama-3.1-8B `pressure_subspace_damping n=100` | locatable-but-not-controllable | 0 | row remains weak, tradeoff-limited, control-like, or method-boundary only |
| Qwen 7B `pressure_subspace_damping n=100` | weak / tradeoff-limited | 0 | row remains weak, tradeoff-limited, control-like, or method-boundary only |
| Qwen 7B `free-form prefill_only` | front-loaded controllable-but-bounded | 1 | short-horizon free-form patch improves behavior while remaining readable |
| Qwen 7B `late_layer_residual_subtraction` | method-boundary failure | 0 | insufficient transfer evidence for clean controllability |
| Qwen 7B `random_direction_control` | null-like control | 0 | row remains weak, tradeoff-limited, control-like, or method-boundary only |
| Qwen 7B `shuffled_label_control` | pairing-artifact damage | 0 | row remains weak, tradeoff-limited, control-like, or method-boundary only |
| Qwen 7B `free-form first_3_tokens` | quality-limited free-form boundary | 0 | quality collapse or boundary status prevents controllable label |
| Qwen 7B `free-form continuous` | generation-unstable | 0 | quality collapse or boundary status prevents controllable label |
| Qwen 7B `identity/profile follow-up` | boundary / insufficient-transfer-evidence | 0 | insufficient transfer evidence for clean controllability |

## Rule-by-Rule Validation

### Rule 1

`target_logit_delta <= -5` is closer to a necessary condition for clean or secondary controllability, but not sufficient.

| status | settings |
| --- | --- |
| supports | Qwen 7B `baseline_state_interpolation (24-26, beta=0.6)`; Qwen 3B `baseline_state_interpolation (31-35, beta=0.6)`; Qwen 14B `matched_belief_subspace_damping n=100` |
| counterexamples / boundary rows | Qwen 7B `pressure_subspace_damping n=100` |

### Rule 2

high `specificity_ratio` is informative but not sufficient for controllability.

| status | settings |
| --- | --- |
| counterexamples / boundary rows | GLM-4-9B `pressure_subspace_damping n=100`; Qwen 7B `pressure_subspace_damping n=100` |

### Rule 3

`negative_logit_delta > 0` is a strong bad signal.

| status | settings |
| --- | --- |
| supports | Llama-3.1-8B `pressure_subspace_damping n=100` |

### Rule 4

`baseline_damage <= 0.02` is necessary for clean control, but not sufficient.

| status | settings |
| --- | --- |
| supports | Qwen 7B `baseline_state_interpolation (24-26, beta=0.6)`; Qwen 3B `baseline_state_interpolation (31-35, beta=0.6)`; Qwen 14B `matched_belief_subspace_damping n=100` |
| counterexamples / boundary rows | GLM-4-9B `pressure_subspace_damping n=100`; Llama-3.1-8B `pressure_subspace_damping n=100`; Qwen 7B `pressure_subspace_damping n=100`; Qwen 7B `late_layer_residual_subtraction`; Qwen 7B `random_direction_control` |

### Rule 5

free-form horizon mainly diagnoses generation stability rather than stronger behavioral gain.

| status | settings |
| --- | --- |
| supports | Qwen 7B `free-form prefill_only` |
| counterexamples / boundary rows | Qwen 7B `free-form first_3_tokens`; Qwen 7B `free-form continuous` |

## Supplementary Boundary Rows

| setting | key available signals | observed_regime |
| --- | --- | --- |
| Qwen 7B `late_layer_residual_subtraction` | damage=0.0460, family=no | method-boundary failure |
| Qwen 7B `random_direction_control` | damage=0.0127, family=no | null-like control |
| Qwen 7B `shuffled_label_control` | damage=0.5848, family=no | pairing-artifact damage |
| Qwen 7B `free-form first_3_tokens` | family=partial, freeform=brittle | quality-limited free-form boundary |
| Qwen 7B `free-form continuous` | family=partial, freeform=unstable | generation-unstable |
| Qwen 7B `identity/profile follow-up` | identity/profile boundary family | boundary / insufficient-transfer-evidence |

## Bottom Line

- `target_logit_delta` strength plus low damage still looks closer to a necessary profile for controllability than specificity alone.
- `specificity_ratio` remains useful but produces major false positives, especially Qwen 7B PSD and GLM.
- free-form stability is a separate axis: `prefill_only` is boundedly usable, while longer horizons collapse into brittle or unstable generation.