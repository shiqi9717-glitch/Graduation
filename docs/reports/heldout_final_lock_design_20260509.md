# Held-out Final Lock Design

更新时间：2026-05-09

本设计用于冻结最后 `2-3` 个新 held-out prediction，并在不与已有 held-out 重复的前提下，形成最终 lock。

目标不是再发散，而是把 diagnostic framework 从“解释已有结果”推进到“冻结预测后再对照 closure”。

## 1. Selection Principle

新 held-out 应满足至少一项：

- 新 model family
- 新 prompt family / pressure wording
- 新但已可闭环的 behavioral closure

避免：

- 重复已有 Qwen mainline / Qwen14B / GLM / Llama / Mistral 正式 held-out
- 依赖尚未存在的数据生成链

## 2. Recommended Final Lock Rows

建议冻结以下 3 个：

### H1: Baichuan2-7B-Chat `n=100`

理由：

- 新模型，且 closure 已存在
- 当前行为 profile 为 borderline partial-positive
- projection diagnostic 也已存在，可做真正 diagnostic-first lock

现有路径：

- [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/baichuan2_7b_chat/n100/baichuan-inc_Baichuan2-7B-Chat/20260509_034823/belief_causal_summary.csv)
- [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/second_positive_screening_n100/baichuan2_7b_chat/n100/baichuan-inc_Baichuan2-7B-Chat/20260509_034823/projection_alignment_summary.json)

### H2: Qwen 7B authority-pressure closure

理由：

- 不是新模型，但是真正的新 prompt family / pressure wording line
- 已有独立 diagnostic 和 closure
- 最适合验证“diagnostic framework 是否能跨 pressure wording 工作”

现有路径：

- [projection_alignment_summary.json](/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_diagnostic/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_102251/projection_alignment_summary.json)
- [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/authority_pressure_behavioral_closure/qwen7b/Qwen_Qwen2.5-7B-Instruct/20260508_203029/belief_causal_summary.csv)

### H3: One new model from A12

优先顺序：

1. `Phi-3-mini-128k-instruct`
2. `Gemma-2-9B-IT`
3. `DeepSeek-V2-Lite-Chat` if locally runnable

理由：

- 只有当 A12 出现 partial-positive 或 clear diagnostic signal 时，它才值得进入 final lock

## 3. Why Not Prioritize New Pressure Type

`social_consensus` 或 `emotional` 目前没有现成生成链或冻结 artifact。

因此：

- 可作为 future extension
- 不适合作为这一轮 final lock 主轴

更稳妥的替代是：

- 用已经存在的 `authority_pressure` 做 new prompt-family held-out

## 4. Freeze-then-Compare Workflow

每个 held-out row 都遵守同一顺序：

1. 冻结 diagnostic signals
2. 写 `predicted regime`
3. 再读取或运行 behavioral closure
4. 记录 `observed regime`
5. 计算 match

## 5. Suggested Prediction Table

建议先写：

- `docs/reports/heldout_final_lock_predictions.md`

格式：

| Held-out setting | Diagnostic signals | Predicted regime | Rationale |
| --- | --- | --- | --- |
| Baichuan2-7B-Chat n=100 | target delta moderate, neg delta near-zero, low damage but not zero | weak / tradeoff-limited or borderline partial-positive | borderline non-Qwen candidate |
| Qwen7B authority pressure | strong target movement, low damage, wording shift not family shift | controllable under wording transfer | same mechanistic family under new prompt line |
| New model from A12 | TBD | TBD | TBD |

再写：

- `docs/reports/heldout_final_lock_results.md`

格式：

| Held-out setting | Predicted regime | Observed regime | Match? | Note |
| --- | --- | --- | --- | --- |

## 6. Concrete Predicted Regimes

### Baichuan2

现有 signal：

- `drift_delta = -0.05`
- `damage = 0.07`
- `mean_belief_logit_delta ≈ -2.21`
- `mean_negative_logit_delta ≈ -0.0037`

建议冻结预测：

- `borderline partial-positive / weak low-damage improvement`

### Qwen authority pressure

现有 signal：

- behavioral closure 非常强：`0.37 -> 0.06` drift, `0.93 -> 0.62` compliance, `0.59 -> 0.96` recovery, `damage = 0`
- diagnostic target movement 明显：`mean_belief_logit_delta ≈ -3.84`

建议冻结预测：

- `controllable under prompt-family transfer`

### A12 new model

只有满足以下任一情况才纳入 final lock：

- `n=50` 已 partial-positive
- 或 diagnostic signals 极强且值得扩到 `n=100`

否则不要为了凑数硬塞进 final lock。

## 7. Match Criteria

建议两级：

- `strict match`
- `collapsed-family match`

collapsed families:

- controllable
- weak / tradeoff-limited
- damage-prone
- boundary / not-controllable

## 8. Success Condition

如果最终能完成：

- Baichuan2
- Qwen authority pressure
- 一个 A12 新模型或明确放弃第三条

就足够构成 final lock。

不要求为了凑满 `3` 条而强行引入未准备好的 pressure type。

## 9. Bottom Line

这一轮 final lock 的推荐顺序是：

1. `Baichuan2-7B-Chat`
2. `Qwen7B authority_pressure`
3. `A12` 中唯一真正值得扩到 formal held-out 的新模型

如果 A12 新模型全部 negative，那么 final lock 就只保留前两条，并如实说明第三条未进入 formal held-out。
