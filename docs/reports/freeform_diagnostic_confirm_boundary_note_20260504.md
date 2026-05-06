# Qwen 7B Free-form Diagnostic Confirm Boundary Note

更新时间：2026-05-04

本说明只基于 2026-05-03 / 2026-05-04 冻结的 Qwen 7B free-form diagnostic confirm experiment 做只读分析，不跑新实验，不改代码，不改论文正文。目标是判断这 5 组结果能否稳定支持 `intervention-continuity tradeoff`，并明确可写入论文的最稳妥边界。

## 1. Five-Condition Summary

### Pressured

| condition | accuracy | drift_rate | wrong_follow_rate | repetition_rate | readable_rate | distinct-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 0.50 | 0.36 | 0.42 | 0.00 | 0.96 | 0.5385 |
| prefill_only | 0.78 | 0.10 | 0.16 | 0.00 | 0.94 | 0.5538 |
| first_token_only | 0.78 | 0.10 | 0.16 | 0.00 | 0.94 | 0.5538 |
| first_3_tokens | 0.78 | 0.10 | 0.16 | 0.58 | 0.42 | 0.4304 |
| continuous | 0.78 | 0.10 | 0.16 | 1.00 | 0.00 | 0.0016 |

### Recovery

| condition | accuracy | drift_rate | wrong_follow_rate | repetition_rate | readable_rate | distinct-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 0.58 | 0.36 | 0.38 | 0.00 | 0.96 | 0.6165 |
| prefill_only | 0.78 | 0.10 | 0.14 | 0.00 | 0.96 | 0.6103 |
| first_token_only | 0.78 | 0.10 | 0.14 | 0.00 | 0.96 | 0.6103 |
| first_3_tokens | 0.78 | 0.10 | 0.14 | 0.76 | 0.24 | 0.4067 |
| continuous | 0.78 | 0.10 | 0.14 | 1.00 | 0.00 | 0.0016 |

## 2. What The Frozen Results Actually Support

### 2.1 Narrow behavioral success exists in the diagnostic setting

相对 `no_intervention`，四种 patched 条件在 `pressured` 和 `recovery` 上都表现出同方向改善：

- `pressured drift_rate`: `0.36 -> 0.10`
- `pressured wrong_follow_rate`: `0.42 -> 0.16`
- `recovery drift_rate`: `0.36 -> 0.10`
- `recovery wrong_follow_rate`: `0.38 -> 0.14`

因此，最保守但仍然成立的行为学结论是：

> In this diagnostic free-form setting, patching the Qwen 7B late-layer mainline direction is associated with a marked reduction in drift and wrong-option-follow under both pressured and recovery prompts.

但这只能叫 `diagnostic free-form effect`，不能升级成 `clean free-form intervention success`。

### 2.2 The effect is front-loaded, not continuity-dependent

`prefill_only` 与 `first_token_only` 的行为指标完全一致：

- pressured：`drift_rate = 0.10`, `wrong_follow_rate = 0.16`
- recovery：`drift_rate = 0.10`, `wrong_follow_rate = 0.14`

`first_3_tokens` 与 `continuous` 在这四个行为指标上也没有进一步改善。也就是说，当前 5 组结果**不支持**“更长 continuity 带来更强 behavioral gain”。

当前更稳的读法是：

> The observed behavioral effect appears to be front-loaded: the diagnostic gain is already present at prefill-only or first-token-only patching, and increasing patch continuity does not improve the measured pressured/recovery behavior further.

### 2.3 What increases with continuity is text degeneration, not behavioral benefit

quality 指标随 patch continuity 明显恶化：

- `pressured repetition_rate`: `0.00 -> 0.58 -> 1.00`
- `recovery repetition_rate`: `0.00 -> 0.76 -> 1.00`
- `pressured readable_rate`: `0.94 -> 0.42 -> 0.00`
- `recovery readable_rate`: `0.96 -> 0.24 -> 0.00`
- `distinct-1` 也从约 `0.55 / 0.61` 崩到 `0.0016`

因此，如果要写 `intervention-continuity tradeoff`，最稳妥的版本不是“continuity 更强，效果更强但代价更大”，而是：

> Additional intervention continuity primarily degrades generation quality, while the measured behavioral gain is already saturated at the shortest patch horizons tested.

这是一条可以成立的 boundary claim。

## 3. Safest Claim Boundary

### 可以写进主文的口径

主文里最多建议写成一到两句 boundary / limitation-side 结果：

- 在 Qwen 7B 的 free-form diagnostic confirm setting 中，`prefill_only` 与 `first_token_only` 已经把 pressured / recovery 的 `drift_rate` 和 `wrong_follow_rate` 从 `0.36 / 0.42 / 0.38` 一致降到 `0.10 / 0.16 / 0.14`。
- 继续增加 patch continuity 并没有带来额外行为收益，但会显著损害文本质量，因此当前证据更支持 `front-loaded diagnostic effect with a continuity-quality tradeoff`，而不是“full continuous free-form patching cleanly works”。

### 只能写成 appendix / limitation / boundary evidence 的口径

以下内容应留在 appendix 或 limitation：

- `first_3_tokens` 与 `continuous` 的 repetition / readability / distinctness 崩塌细节。
- “behavioral gain 在最短 patch horizon 已经饱和”的 mechanistic-style解读。
- 各质量指标的逐项对照表。

### 不应写的口径

- `continuous patched free-form generation cleanly works`
- `free-form intervention has been fully validated`
- `first_3_tokens is a stable sweet spot`
- 任何把这组结果写成 deployment-level open-ended intervention success 的说法

## 4. Important Limits

- 这 5 组结果只覆盖 `pressured` 与 `recovery`，不包含 `baseline patched generation`。
- 因此 `baseline damage` 在这轮设计中**没有直接测量**，只能沿用 artifact 自带说明：`baseline_damage_available = false`。
- 这意味着当前结果只能说明“在给定 pressured / recovery prompt 下，短 patch horizon 与较低 drift / wrong-follow 同时出现”，不能说明“该方法在自由生成中总体无害”。
- 另外，所有条件的 `truncation_rate = 1.0` 且 `max_token_hit_rate = 1.0`，说明这些结果本身就是受固定生成长度约束的 diagnostic artifact，不适合直接叙述成自然部署表现。

## 5. Bottom Line

最稳妥的论文边界结论是：

> The Qwen 7B free-form diagnostic confirm experiment provides boundary evidence that the late-layer intervention effect is visible in open-ended generation under pressured and recovery prompts, but the effect is already saturated at prefill-only or first-token-only patching. Extending the patch deeper into generation does not improve the measured behavioral outcomes and instead sharply degrades text quality. We therefore treat these results as diagnostic free-form boundary evidence rather than as clean free-form intervention success.
