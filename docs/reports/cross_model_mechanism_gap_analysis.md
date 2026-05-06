# Cross-Model Mechanism Gap Analysis

更新时间：2026-05-01

本报告只基于当前冻结 artifact 做跨模型对照分析，不跑新实验，不改实验结果。目标是解释为什么 Qwen 的 pressure direction 更可控，而 Llama 的方向更难转化为行为控制；同时评估 GLM 的方向性为何伴随更强 tradeoff。

## Scope And Boundary

- Qwen 7B / 3B mainline closure 仍属于 `objective-local proxy` metric family。
- Qwen 14B / GLM / Llama / Mistral / clean-protocol rows 属于 `bridge / transfer-style` family。
- 因此 Table C 只做并列对照，不把 Qwen 7B / 3B mainline 与 bridge rows 当成同一无注释 leaderboard。
- GLM 当前没有同格式 frozen projection-to-logit summary；因此关于 GLM 的机制差距只能结合 layer-wise concentration 与 behavioral closure 讨论，不能补写未冻结的 logit-specific 结论。

## Table A. Projection-to-Logit Cross-Model Comparison

| model | n | mean belief logit delta | mean negative logit delta | belief/negative logit ratio | baseline projection norm | pressured projection norm | baseline as fraction of pressured | projection vs stance drift corr | projection vs compliance corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 7B | 48 | -5.3871 | -0.0050 | 1083.51 | 70.9992 | 69.7820 | 1.0174 | -0.3899 | 0.1857 |
| Qwen 3B | 48 | -5.1019 | -0.9303 | 5.48 | 22.2496 | 43.7769 | 0.5082 | -0.5243 | 0.2753 |
| Llama 3.1 8B | 24 | -0.7889 | 0.0646 | 12.21 | 5.5327 | 9.1910 | 0.6812 | -0.3961 | n/a |

Notes:

- Qwen 7B 的关键不是 projection norm 大，而是 `belief logit delta = -5.3871` 且 `negative logit delta = -0.0050`，几乎 logit-null 的 negative channel 说明该方向高度特异。
- Qwen 7B 的 `baseline / pressured = 1.0174`，说明 belief direction 在 baseline 与 pressured 状态下都强烈存在；它更像“结构性已有方向”，而不是“pressure 才把它激活出来”。
- 三个模型的 `projection vs stance drift corr` 都是负值：Qwen 7B `-0.3899`，Qwen 3B `-0.5243`，Llama `-0.3961`。因此更大的 pressured projection 并不预测更大的 drift。
- Llama 的 `belief/negative ratio = 12.21` 不低，但其 belief logit delta 只有 `-0.7889`，说明“比例看起来还行”并不足以转化成强 controllability；绝对 logit effect 很弱。

## Table B. Layer-wise Concentration Comparison

提取规则：

- Qwen / GLM：从 `pressure_subspace_summary.csv` 中筛 `pressure_type = belief_argument`，并保留 `belief_argument_subspace` 或 `philpapers_belief_argument_subspace`。
- Llama：从 `belief_causal_subspace_summary.csv` 中筛 `pressure_type = belief_argument`，对应列名为 `direction_abs_coherence`，等价于 Qwen / GLM 表中的 `direction_cosine_abs_mean`。
- 为避免同一 layer 因不同 subspace 命名重复出现，以下 `top-3 layers` 采用“每层取 belief-specific 行中的最大 explained_variance_sum”后的唯一层排序。

| model | top-1 layer + explained variance | top-3 layers | abs coherence at top-1 layer | late-layer concentration evidence | n_pairs used |
| --- | --- | --- | ---: | --- | ---: |
| Qwen 7B | layer 24, `0.9448` | 24, 25, 26 | 0.3700 | 高度集中在 24-26，与 formal mainline late-layer window 直接重合；未见强 mid-layer 竞争进入 top-3。 | 24 |
| Qwen 3B | layer 31, `0.9595` | 31, 32, 33 | 0.6322 | 高度集中在 31-33，并继续延伸到 34-35；与 3B mainline 31-35 window 高一致。 | 24 |
| Llama 3.1 8B | layer 24, `0.6759` | 24, 25, 26 | 0.4406 | 也有 late-layer concentration，但 explained variance 明显低于 Qwen；更像“可定位方向”，不直接说明“可行为控制方向”。 | 24 |
| GLM-4-9B | layer 32, `0.9352` | 32, 33, 31 | 0.5510 | top layers 同样集中，且主要在 31-33；当前数据不支持把其高 damage 简单归因于 subspace 分散。 | 24 |

Notes:

- Qwen 7B / 3B 的集中带与各自主线选层吻合，支持“mainline 选层并非任意”。
- Llama 也可定位到 late layers，但其 top-1 explained variance `0.6759` 明显弱于 Qwen 7B `0.9448` 和 Qwen 3B `0.9595`。
- GLM 的 concentration 并不弱，因此“GLM tradeoff 高”不能仅靠“层不集中”解释；当前 frozen files 更支持“方向存在，但 baseline manifold 干扰更大”的保守解释。

## Table C. Behavioral Closure Cross-Model Comparison

| model_group | evidence_level | metric_family | n_items | stance_drift_delta + CI | pressured_compliance_delta + CI | recovery_delta + CI | baseline_damage_rate + CI | net_effect_score |
| --- | --- | --- | ---: | --- | --- | --- | --- | ---: |
| Qwen-7B main baseline | formal_mainline | objective-local proxy | 100 | -0.1493 [-0.2200, -0.0800] | -0.1595 [-0.2300, -0.0900] | 0.7127 [0.5000, 0.9000] | 0.0000 [0.0000, 0.0000] | 1.0214 |
| Qwen-3B replication | formal_mainline_replication | objective-local proxy | 100 | -0.1597 [-0.2402, -0.0800] | -0.2789 [-0.3700, -0.1900] | 0.6906 [0.5000, 0.8667] | 0.0000 [0.0000, 0.0000] | 1.1292 |
| Qwen-14B secondary causal confirmation | secondary_causal_confirmation | bridge / transfer-style | 24 | -0.2068 [-0.4167, 0.0000] | -0.2484 [-0.4167, -0.0833] | 0.0000 [0.0000, 0.0000] | 0.0416 [0.0000, 0.1250] | 0.4136 |
| Qwen-3B clean protocol | clean_protocol_positive_confirmation | bridge / transfer-style | 48 | -0.3750 [-0.5210, -0.2290] | -0.4170 [-0.5630, -0.2710] | 0.0833 [0.0000, 0.1880] | 0.0417 [0.0000, 0.1040] | 0.8333 |
| Qwen-7B clean protocol | clean_protocol_weak_boundary | bridge / transfer-style | 48 | -0.0417 [-0.1040, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0417 [0.0000, 0.1040] | 0.0417 [0.0000, 0.1040] | 0.0417 |
| Qwen-14B n=48 | cautionary_boundary | bridge / transfer-style | 48 | 0.0417 [-0.1040, 0.1880] | -0.1042 [-0.2080, 0.0000] | -0.1042 [-0.2500, 0.0630] | 0.1042 [0.0210, 0.1880] | -0.1458 |
| GLM-4-9B cross-family positive replication | cross_family_positive_replication_stronger_tradeoff | bridge / transfer-style | 24 | -0.2932 [-0.5417, -0.0417] | -0.2095 [-0.4167, 0.0000] | 0.3769 [0.1667, 0.5833] | 0.3325 [0.1667, 0.5417] | 0.5472 |
| Llama-3.1-8B weak replication | weak_replication_limitation | bridge / transfer-style | 24 | -0.0824 [-0.2917, 0.1250] | 0.0000 [0.0000, 0.0000] | -0.0847 [-0.2917, 0.1250] | 0.2499 [0.0833, 0.4167] | -0.2523 |
| Mistral-7B appendix exploratory | appendix_exploratory | bridge / transfer-style | 24 | -0.2491 [-0.5000, 0.0000] | -0.2094 [-0.3750, -0.0417] | 0.5836 [0.3750, 0.7917] | 0.3769 [0.2083, 0.5833] | 0.6652 |
| identity_profile representative boundary row | boundary_evidence_insufficient_intervention_support | identity/profile boundary metrics | 36 | n/a | n/a | n/a | n/a | n/a |

## Core Findings

1. **Logit specificity is the clearest mechanistic separator, but it works only when the belief effect is both strong and specific.**
   Qwen 7B has `mean belief logit delta = -5.3871`, which is about `6.8x` the magnitude of Llama’s `-0.7889`, while its `mean negative logit delta = -0.0050` is effectively zero. That yields a ratio of `1083.51`, far above Qwen 3B’s `5.48` and Llama’s `12.21`. The best summary is therefore not “high ratio alone,” but “strong plus specific”: Qwen 7B behaves much more like a usable control axis than Llama does.

2. **Projection magnitude does not predict controllability.**
   Qwen 7B has a very large pressured projection norm (`69.7820`) compared with Llama (`9.1910`), but both models show negative projection-vs-drift correlation: Qwen 7B `-0.3899`, Llama `-0.3961`, and Qwen 3B is even more negative at `-0.5243`. This means larger pressured loading onto the extracted direction does not imply larger behavioral drift, and conversely reducing behavior cannot be explained by “shrinking projection magnitude” alone. The frozen data point to logit specificity, not raw projection strength, as the more behaviorally relevant signal.

3. **Qwen 7B’s belief direction appears structurally present even without pressure.**
   Its baseline projection norm (`70.9992`) is slightly larger than its pressured norm (`69.7820`), giving `baseline / pressured = 1.0174`. This contrasts with Qwen 3B (`0.5082`) and Llama (`0.6812`), where the pressured state loads more strongly than baseline. The most conservative reading is that in Qwen 7B the relevant belief direction is already strongly represented in baseline computation, and pressure mainly exploits an existing axis rather than creating a new one. This helps explain why damping can be highly logit-specific in Qwen 7B even though projection magnitude itself does not track item-level drift.

4. **Late-layer localization is necessary but not sufficient for control.**
   Qwen 7B concentrates at layers `24, 25, 26` with top explained variance `0.9448`; Qwen 3B concentrates at `31, 32, 33` with top explained variance `0.9595`. Llama also localizes late, with top layers `24, 25, 26`, but its top explained variance is only `0.6759`, and its closure row remains weak: drift `-0.0824`, compliance `0.0000`, recovery `-0.0847`, damage `0.2499`, net effect `-0.2523`. So “direction is locatable” does not imply “direction is behaviorally controllable.” Llama is the clearest current example of a locatable-but-not-reliably-actionable direction.

5. **GLM’s tradeoff looks structural, but current files do not support a simple “diffuse subspace” story.**
   GLM’s closure row shows meaningful directionality: drift `-0.2932`, compliance `-0.2095`, recovery `0.3769`, net effect `0.5472`. But damage is high at `0.3325`, much worse than Qwen 7B/3B mainline `0.0000`. At the same time, GLM’s top layers are fairly concentrated at `32, 33, 31`, with top explained variance `0.9352` and top-layer coherence `0.5510`. That means the present frozen evidence does **not** justify saying “GLM tradeoff is high because its layer-wise direction is diffuse.” A more defensible current reading is that GLM has a usable directionality signal, but damping that direction appears to interfere more strongly with the baseline manifold. Because no frozen GLM projection-to-logit summary is included here, this remains a data limitation rather than a settled mechanism claim.

## Short Closing

Across the current frozen artifacts, the main cross-model gap is not “whether a direction exists,” but “whether the extracted direction is strong, specific, and behaviorally isolatable.” Qwen 7B is the cleanest case of a direction that is both logit-strong and logit-specific, while Qwen 3B is the cleanest case where pressure more visibly increases loading onto that direction. Llama shows that late-layer localization alone is not enough, and GLM shows that cross-family directionality can coexist with substantial baseline damage.
