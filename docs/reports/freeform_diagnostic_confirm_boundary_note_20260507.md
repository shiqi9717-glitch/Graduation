# Qwen 7B Free-form Diagnostic Confirm Boundary Note

更新时间：2026-05-07

本说明只更新 patch-horizon 对照表与 boundary claim，不跑新实验，不改代码，不改论文正文。

## 7-condition Summary

| condition | pressured drift_rate | pressured wrong_follow_rate | pressured readable_rate | pressured repetition_rate | pressured distinct-1 | recovery drift_rate | recovery wrong_follow_rate | recovery readable_rate | recovery repetition_rate | recovery distinct-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_intervention` | 0.36 | 0.42 | 0.96 | 0.00 | 0.5385 | 0.36 | 0.38 | 0.96 | 0.00 | 0.6165 |
| `prefill_only` | 0.10 | 0.16 | 0.94 | 0.00 | 0.5538 | 0.10 | 0.14 | 0.96 | 0.00 | 0.6103 |
| `first_token_only` | 0.10 | 0.16 | 0.94 | 0.00 | 0.5538 | 0.10 | 0.14 | 0.96 | 0.00 | 0.6103 |
| `first_3_tokens` | 0.10 | 0.16 | 0.42 | 0.58 | 0.4304 | 0.10 | 0.14 | 0.24 | 0.76 | 0.4067 |
| `continuous` | 0.10 | 0.16 | 0.00 | 1.00 | 0.0016 | 0.10 | 0.14 | 0.00 | 1.00 | 0.0016 |
| `first_5_tokens` | 0.10 | 0.16 | 0.00 | 1.00 | 0.0718 | 0.10 | 0.14 | 0.00 | 1.00 | 0.0794 |
| `first_10_tokens` | 0.10 | 0.16 | 0.00 | 1.00 | 0.0041 | 0.10 | 0.14 | 0.00 | 1.00 | 0.0041 |

## Updated Boundary Claim

- 行为改善仍然是前置饱和的：从 `prefill_only` 开始，`pressured drift_rate` 已从 `0.36` 降到 `0.10`，`wrong_follow_rate` 已从 `0.42` 降到 `0.16`；后续延长到 `first_3_tokens`、`first_5_tokens`、`first_10_tokens`、`continuous` 并没有进一步改善这些行为指标。
- 质量退化边界比此前更陡。`first_3_tokens` 还是 brittle boundary：`pressured readable_rate = 0.42`；但 `first_5_tokens` 与 `first_10_tokens` 已经直接塌到 `0.00`，而且 `repetition_rate = 1.00`。这说明 patch-horizon tradeoff 在 `3 -> 5 tokens` 之间已经跨过可读性断崖。
- 因此最稳写法仍然不是 clean free-form intervention success，而是 `front-loaded diagnostic effect with a steep continuity-quality tradeoff`。这条证据适合放 appendix / limitation / boundary note，不应升格为 deployment-level free-form validation。

## Boundary

- 本轮仍然没有 baseline-patched free-form generation，因此不能把 baseline damage 写成已测量。
- 本轮结果支持 late-layer effect 在 open-ended generation 中可见，但不支持 longer-horizon patched generation 可用。
- `first_5_tokens` 和 `first_10_tokens` 的加入，使“tradeoff 更陡”这一点比 2026-05-04 的五条件版本更明确。