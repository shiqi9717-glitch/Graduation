# Free-form Quality Metrics

更新时间：2026-05-09

本报告只从已有 free-form 7-condition `run_summary.json` 提取质量指标，不跑新实验，不改代码，不改论文正文。

## Complete Quality Metrics Table

| Condition | Drift | Wrong-follow | Readable | Repetition | Distinct-1 | Distinct-2 | Max-token hit | Avg length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_intervention` | 0.36 | 0.42 | 0.96 | 0.00 | 0.5385 | 0.8386 | 1.00 | 50.0 |
| `prefill_only` | 0.10 | 0.16 | 0.94 | 0.00 | 0.5538 | 0.9000 | 1.00 | 50.0 |
| `first_token_only` | 0.10 | 0.16 | 0.94 | 0.00 | 0.5538 | 0.9000 | 1.00 | 50.0 |
| `first_3_tokens` | 0.10 | 0.16 | 0.42 | 0.58 | 0.4304 | 0.6680 | 1.00 | 50.0 |
| `first_5_tokens` | 0.10 | 0.16 | 0.00 | 1.00 | 0.0718 | 0.1042 | 1.00 | 50.0 |
| `first_10_tokens` | 0.10 | 0.16 | 0.00 | 1.00 | 0.0041 | 0.0050 | 1.00 | 50.0 |
| `continuous` | 0.10 | 0.16 | 0.00 | 1.00 | 0.0016 | 0.0024 | 1.00 | 50.0 |

All rows above are taken from the `pressured` scenario entries in `summary.by_scenario_condition`. No requested field was missing in the current seven files, so this table contains no `N/A` cells.

## Key Analysis Paragraph

The quality collapse follows a measurable and threshold-like pattern across patch horizons. As patch horizon increases from `prefill_only` to `continuous`, `distinct-1` drops from `0.5538` to `0.0016`, `distinct-2` drops from `0.9000` to `0.0024`, `repetition_rate` rises from `0.00` to `1.00`, and `max_token_hit_rate` stays pinned at `1.00`. The transition is sharp rather than smooth: readability is largely preserved at `first_token_only` (`0.94`), but falls to `0.42` at `first_3_tokens` and then to `0.00` by `first_5_tokens`, where repetition has already saturated at `1.00`. This supports the interpretation that generation stability failure is a threshold-like transition rather than a gradual linear degradation.

## Main-text Table Suggestion

- Main text can keep: `Condition`, `Drift`, `Readable`, `Repetition`, `Distinct-1`.
- Appendix should keep the full table above, including `Distinct-2`, `Max-token hit`, and `Avg length`.

## Source Paths

- [no_intervention run_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/no_intervention/Qwen2.5-7B-Instruct/20260503_150943/run_summary.json:1)
- [prefill_only run_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/prefill_only/Qwen2.5-7B-Instruct/20260503_172219/run_summary.json:1)
- [first_token_only run_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/first_token_only/Qwen2.5-7B-Instruct/20260503_192855/run_summary.json:1)
- [first_3_tokens run_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/first_3_tokens/Qwen2.5-7B-Instruct/20260503_213547/run_summary.json:1)
- [continuous run_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260503/continuous/Qwen2.5-7B-Instruct/20260504_001705/run_summary.json:1)
- [first_5_tokens run_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260507/first_5_tokens/Qwen2.5-7B-Instruct/20260507_192721/run_summary.json:1)
- [first_10_tokens run_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_freeform_diagnostic/20260507/first_10_tokens/Qwen2.5-7B-Instruct/20260507_213615/run_summary.json:1)
