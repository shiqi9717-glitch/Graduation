# Diagnostic Framework Evaluation

更新时间：2026-05-08

本报告只基于当前 frozen 结果做理论贡献边界评估，不跑新实验，不改代码，不改论文正文。输入主要来自：

- `docs/reports/regime_diagnostic_matrix_20260507.md`
- `docs/reports/heldout_prediction_result_20260507.md`
- `docs/reports/controllability_predictor_analysis_20260506.md`
- `docs/reports/freeform_diagnostic_confirm_boundary_note_20260507.md`
- `docs/papers/figures/figureA_layer_diagnostic_profile.pdf`
- `docs/papers/figures/figureB_layer_behavioral_effect.pdf`
- `docs/papers/figures/figureC_layer_clean_control_score.pdf`

## Executive Readout

当前 diagnostic framework 最稳的理论贡献，不是提出一个可泛化的 controllability predictor，而是给出一个 **bounded diagnostic framework**：

> controllability is better explained by the joint profile of target-logit effect, projection-drift alignment, low damage, and intervention-family agreement than by specificity alone, but no single feature is sufficient.

这比旧 Section 8 命题更强的地方在于：它不仅解释正例，也能解释为什么高-specificity 行会失败，尤其是 Qwen 7B PSD 和 GLM 这两个关键反例。

## 1. Framework 的适用边界

## 1.1 在哪些 settings 上成立

当前框架在以下 setting 上成立得最好：

- Qwen 7B `baseline_state_interpolation (24-26, beta=0.6)`
- Qwen 3B `baseline_state_interpolation (31-35, beta=0.6)`
- Qwen 14B `matched_belief_subspace_damping n=100`
- Qwen 7B `free-form prefill_only`（仅作为 bounded free-form 边界）

这些行共同满足的不是某一个单特征，而是一个联合特征轮廓：

- `target_logit_delta` 足够强
- `baseline_damage` 低
- `projection_drift_corr` 至少不与行为 readout 明显冲突
- intervention 的 behavioral closure 与表示层诊断方向大体一致

其中最稳的子结论是：

- `target_logit_delta` 强度加低 damage，比 `specificity_ratio` 更接近 controllability 的必要条件
- free-form setting 只能支持 `front-loaded diagnostic effect`，不能支持长程可用的 free-form control

## 1.2 在哪些 settings 上失败

当前框架失败或只能弱成立的 setting 主要有四类：

- **high-specificity false positives**
  - GLM-4-9B `pressure_subspace_damping n=100`
  - Qwen 7B `pressure_subspace_damping n=100`
- **locatable-but-not-controllable boundary**
  - Llama-3.1-8B `pressure_subspace_damping n=100`
- **method-boundary / control rows**
  - Qwen 7B `late_layer_residual_subtraction`
  - `random_direction_control`
  - `shuffled_label_control`
- **free-form horizon failure**
  - `first_3_tokens`
  - `first_5_tokens`
  - `first_10_tokens`
  - `continuous`

## 1.3 失败原因：数据不足 vs rule 本身有缺陷

### A. rule 本身有缺陷的部分

最明确有缺陷的是把 `specificity_ratio` 写成 controllability 的必要或近充分条件。

证据：

- GLM 与 Qwen 7B PSD 都有极高 specificity，但行为上都不是 clean controllable
- held-out 检查也没有支持“single-feature predictor”写法
- `controllability_predictor_analysis_20260506.md` 已显示，high specificity 更像会产生假阳性，而非稳定预测 controllability

因此：

- 旧写法 `specificity is a necessary structural condition` 应视为过强
- 新框架必须明确写成 `jointly informative, singly insufficient`

### B. 主要是数据不足的部分

有些位置不是 rule 明显错误，而是当前数据不足以把 failure mechanism 进一步判定清楚：

- Qwen 14B 目前正式收口仍偏小样本，且更像 `secondary causal confirmation`
- Figure A 当前只能支持 `retained-window diagnostic profile`，不能支持完整逐层 projection sweep 结论
- free-form 没有 baseline-patched 对照，因此不能把它升级成 clean deployment-style claim

因此，数据不足主要限制的是：

- 框架的强预测口径
- 某些失败机制的细分归因
- 更细颗粒度的层级结论

但它**不改变**当前总边界：框架适合作为 bounded diagnostic explanation，不适合作为 formal predictor。

## 2. 与旧 Section 8 proposition 的对比

## 2.1 旧命题

旧命题可概括为：

> specificity is a necessary structural condition

这一路径的问题在于，它默认 near-zero negative movement 和高 specificity 至少接近 controllability 的必要条件。

## 2.2 新框架

当前 frozen 结果支持的更稳版本是：

> specificity + target-logit effect + projection-drift alignment + low damage + intervention-family agreement jointly predict controllability, but no single feature is sufficient.

还需要进一步降调成：

> jointly diagnose controllability regimes

而不是：

> formally predict controllability

## 2.3 新框架比旧命题好在哪里

### 1. 它能吸收反例，而不是被反例击穿

旧命题最容易被以下两行击穿：

- Qwen 7B PSD：very-high specificity，但 weak / tradeoff-limited
- GLM：very-high specificity，但 weak / tradeoff-limited

新框架允许我们说：

- specificity 只说明“方向可能较专一”
- 但缺少低 damage、稳定 family agreement、以及更完整的 behavior-level closure 时，仍然不会转化成 clean control

### 2. 它更贴近当前论文主线

旧命题容易把论文带成“我们找到一个 predictor / steering recipe”。

新框架更贴近当前主线：

- pressure-type-specific linear steerability
- localization/controllability separation
- bounded mechanistic evidence

也就是：diagnostic 不是为了部署一个 universal rule，而是为了说明 **为什么有些 pressure/model/intervention 组合可控，有些只能被定位，不能被 cleanly 控制**。

### 3. 它自然容纳 free-form boundary

旧命题较难解释为什么 `prefill_only` 有效果，但 `first_5_tokens` 和 `first_10_tokens` 会直接掉到 `readable_rate = 0.00`。

新框架则可以把 free-form horizon 当成独立诊断轴：

- behavioral improvement 可以前置饱和
- generation stability 会在更长 horizon 急剧崩塌

这正说明 controllability 不是单一数值特征，而是多轴边界结构。

## 3. 建议 Section 8 / Discussion 的最终 framing

下面给出 3 个候选 framing，按 claim strength 递增排列。

## Candidate A: 最稳

**Claim strength: conservative / recommended**

> We evaluate whether representation-level and logit-level diagnostics can distinguish controllability regimes in the current frozen intervention set. The main result is bounded: high target-logit movement together with low damage tracks controllable rows better than specificity alone, while several high-specificity rows remain behaviorally non-controllable. This supports a diagnostic framework for controllability regimes, not a formal predictor.

适用场景：

- Section 8 主体
- Discussion 总结段
- reviewer rebuttal 的主口径

优点：

- 几乎不怕 reviewer 用 GLM、Qwen 7B PSD、free-form failure 反打
- 能自然接上 `localization does not imply controllability`

## Candidate B: 中等强度

**Claim strength: moderate**

> Controllability is jointly associated with target-logit effect, projection-behavior alignment, and low collateral damage, whereas specificity alone produces clear false positives. The resulting framework explains both clean Qwen successes and cross-family boundary cases, but remains limited to the current intervention families and frozen settings.

优点：

- 更强调 “joint profile”
- 适合 discussion 中对诊断框架做一层正向概括

风险：

- `jointly associated` 比较稳，但若写成 `jointly predict` 就会偏强

## Candidate C: 偏强，不推荐直接主打

**Claim strength: aggressive / not recommended as primary framing**

> We identify a multi-feature diagnostic signature of controllability across intervention families.

风险：

- 容易被理解为已经得到较稳定、可外推的 predictor
- 与 held-out `strict match = 0.667`、free-form quality collapse、Figure A 非完整逐层 sweep 之间存在张力

## Recommended Framing

推荐使用 **Candidate A** 作为最终主口径。

最稳原因：

- 与 held-out 结果一致：支持 bounded diagnostic hypothesis，但不支持强 predictor
- 与主线一致：强调 regime distinction，而非 universal mitigation
- 与 failure cases 一致：允许 Llama、GLM、Qwen 7B PSD 成为贡献性边界，而不是坏消息

## 4. 对论文整体叙事的增益

当前 diagnostic framework 最有价值的理论增益，不是单独新增一个 Section 8 小结论，而是帮助整篇论文完成下面这个升级：

- 从 “我们找到一个可减掉的 direction”
- 升级为 “pressure-related effects are not uniformly controllable, and diagnostics help separate locatable, controllable, and tradeoff-limited regimes”

这和当前主文最稳 framing 是一致的：

- `Pressure is not uniformly linearly steerable.`
- `localization does not imply controllability.`
- `belief_argument exhibits a late-layer, partially intervenable drift in the Qwen line.`

## 5. Bottom Line

- 当前 diagnostic framework 的最佳定位是：`bounded diagnostic framework for controllability regimes`
- 它优于旧 Section 8 proposition，因为它能解释正例，也能吸收高-specificity 反例
- 它不应被写成 formal predictor，不应把 specificity 写成必要结构条件
- 最终推荐写法应把重点放在：`jointly informative, singly insufficient`
