# Failure Case Analysis

更新时间：2026-05-08

本报告只基于当前 frozen 结果分析 failure cases，不跑新实验，不改代码，不改论文正文。主要输入来自：

- `docs/reports/regime_diagnostic_matrix_20260507.md`
- `docs/reports/heldout_prediction_result_20260507.md`
- `docs/reports/controllability_predictor_analysis_20260506.md`
- `docs/reports/freeform_diagnostic_confirm_boundary_note_20260507.md`
- `docs/papers/figures/figureA_layer_diagnostic_profile.pdf`
- `docs/papers/figures/figureB_layer_behavioral_effect.pdf`
- `docs/papers/figures/figureC_layer_clean_control_score.pdf`

## Executive Readout

当前最重要的 failure 不是“方法失效”，而是两类不同边界：

- **GLM / Qwen 7B PSD 型**：方向看起来很 specific，也有一定 target-logit effect，但 clean behavioral controllability 不出现，说明 `specificity != controllability`
- **Llama 型**：方向可以被定位，但 target-logit push 弱、negative control 方向还会反向泄漏，说明 `localization != controllability`

因此 failure case 的理论价值，不是要求我们马上扩实验，而是帮助论文把 controllability boundary 说清楚。

## 1. GLM Failure 机制假说

## 1.1 现象

GLM-4-9B `pressure_subspace_damping n=100` 的 frozen 指标为：

- `specificity_ratio = 2580.02`（very-high）
- `target_logit_delta = -2.87`（moderate）
- `projection_drift_corr = -0.45`（aligned）
- `behavioral drift delta = -0.01`
- `behavioral compliance delta = -0.01`
- `baseline_damage = 0.06`

这是一条很关键的 high-specificity false positive。

## 1.2 机制假说

### Hypothesis G1: cross-family transfer gap

虽然 GLM 上能看到与 belief-pressure 相关的表示方向，但这个方向与 Qwen 主线中可干预的 decision geometry 不同，因此 representation-level alignment 不能稳定转化成 behavior-level closure。

这会表现为：

- 有一定负向 `projection_drift_corr`
- 也有 moderate 的 `target_logit_delta`
- 但最终行为变化很弱

也就是说，GLM 可能支持“可定位的 pressure-related direction”，但不支持“与当前 intervention family 匹配的 clean control manifold”。

### Hypothesis G2: baseline manifold structure mismatch

GLM 的基线表征流形可能比 Qwen 更脆弱或更拥挤，因此即便方向特异性高，真正施加干预时也更容易伴随 collateral shift。

这与当前 `baseline_damage = 0.06` 是一致的：

- damage 不算灾难性
- 但已经高于 clean-control rows
- 足以把原本 moderate 的正向效应压平

### Hypothesis G3: intervention method mismatch

GLM failure 也可能不是“方向不存在”，而是当前 `pressure_subspace_damping` 这套 intervention family 对 GLM 不匹配。

如果是这种情况，那么失败更应解释为：

- diagnostic signal exists
- but this intervention family does not transfer cleanly across model family

这对论文反而是有利的，因为它进一步支持：

> pressure-related localization may transfer more easily than controllability.

## 1.3 哪些可用当前数据检验

当前 frozen 数据已经能做的，不需要新实验：

- 检查 GLM 在 Figure A / B / C 中是否呈现“layer-wise profile 看起来像有方向，但 clean-control score 始终不起”的分离
- 对照 Qwen 7B PSD：
  - 两者是否共享 `very-high specificity + aligned corr + non-clean behavior`
  - 如果共享，则更支持“旧 rule 有结构性假阳性”
- 对照 Qwen 14B：
  - Qwen 14B 的 `specificity` 不高，但 target-logit 更强、damage 更低，说明 GLM failure 不是因为“specificity 不够”

## 1.4 当前数据还不能判定的部分

仅靠现有 frozen 结果，还不能区分以下两种更细机制：

- 是 family-level geometry gap 为主
- 还是 GLM 上当前 intervention 参数不合适为主

因此，GLM 最稳写法不是“机制已知”，而是：

> GLM provides a high-specificity but weakly controllable boundary case, consistent with cross-family transfer limits and/or intervention-family mismatch.

## 2. Llama Failure 机制假说

## 2.1 现象

Llama-3.1-8B `pressure_subspace_damping n=100` 的 frozen 指标为：

- `projection_drift_corr = -0.29`（moderate / weakly aligned）
- `target_logit_delta = -1.83`（weak）
- `negative_logit_delta = +0.1716`（bad-positive）
- `behavioral drift delta = +0.03`
- `behavioral compliance delta = +0.01`
- `baseline_damage = 0.08`

这条是最典型的 `locatable-but-not-controllable` 边界。

## 2.2 机制假说

### Hypothesis L1: direction is locatable but not behaviorally decisive

Llama 上的 `projection_drift_corr = -0.29` 说明某种 belief-related direction 是可以定位到的，但它对最终行为决策的控制力不够强，因此只能形成 localization，不能形成 effective control。

这正对应论文主线中的：

> localization does not imply controllability.

### Hypothesis L2: belief-pressure direction is not sufficiently specific in Llama

`negative_logit_delta = +0.1716` 是当前最强的坏信号之一，说明干预并没有只压 belief target，反而对 negative control 方向发生了反向泄漏。

这意味着 Llama 上的问题可能不是“完全没有方向”，而是：

- 方向不够干净
- 或方向在 logits 上并不 selective

因此，Llama failure 比 GLM 更像“定位得到，但 logit-level selectivity 和 behavior-level transfer 都不够”。

### Hypothesis L3: intervention parameter mismatch

Llama 当前使用的是 `alpha=0.5`，而不是与部分 Qwen / GLM 设置完全相同的参数。理论上也可能存在参数未对齐问题。

但在当前 frozen 证据里，最稳结论仍然不是“参数还没调好”，而是：

- 即便已有一定定位证据
- target-logit effect 仍弱
- negative control 方向仍为正
- damage 也不低

所以把它写成“只差调参”会过强。

## 2.3 哪些可用当前数据检验

当前 frozen 数据已经能做的：

- 利用 Figure A / B / C 检查：
  - layer-wise diagnostic profile 是否显示有局部 alignment
  - 但 behavior effect 和 clean-control score 没有跟上
- 与 GLM 对照：
  - GLM 是 `alignment + moderate target-logit + weak behavior`
  - Llama 是 `some alignment + weak target-logit + bad-positive negative control`
  - 这有助于把两类 failure 分开，而不是统称“跨家族失败”
- 与 Qwen 14B / Qwen 3B 对照：
  - Qwen 正例并不需要极高 specificity
  - 但需要更强 target-logit push 和更低 damage

## 2.4 当前数据还不能判定的部分

现有数据不足以严格区分：

- Llama 的问题主要来自 representation specificity 不足
- 还是主要来自 intervention hyperparameter mismatch

因此最稳写法是：

> Llama provides a boundary case where a pressure-related direction appears locatable, but its logit-level selectivity and behavioral controllability remain weak under the current intervention family.

## 3. Failure Case 的统一理论价值

GLM 和 Llama 不是同一种 failure，它们共同支持的是一个更强的 separation：

- GLM / Qwen 7B PSD 说明：`specificity does not imply controllability`
- Llama 说明：`localization does not imply controllability`

如果把两者合并看，当前 diagnostic framework 最有价值的不是“找到一个强 predictor”，而是把 failure 按机制边界分型：

- **specific-but-not-controllable**
- **locatable-but-not-controllable**
- **front-loaded-but-not-stable**

这会让 failure case 成为 paper 的机制证据，而不是附带坏消息。

## 4. 是否需要新实验

## 4.1 当前结论是否能只靠现有数据成立

可以。

就 innovation framing 而言，现有 frozen 数据已经足够支持：

- bounded diagnostic framework
- specificity 的假阳性边界
- localization/controllability separation
- free-form continuity-quality tradeoff

因此，**不需要为了写清当前理论边界而强行补新实验**。

## 4.2 如果必须补跑，最小清单是什么

如果后续别的部门坚持要补一个最小验证清单，建议只考虑：

1. Llama 的 projection-to-logit / causal alignment diagnostic formalization
2. Qwen14B `n=48` 正式补齐
3. identity/profile 的最小 prefix-span ablation + replay

不建议为了 failure case 再开：

- 大规模 cross-family sweep
- 新 intervention family 扩展
- 把 GLM / Llama 重新包装成正向 replication 的尝试

## 5. Bottom Line

- GLM failure 最稳解释是：`high-specificity false positive under cross-family and/or intervention-family transfer limits`
- Llama failure 最稳解释是：`locatable but weakly controllable, with weak target-logit effect and bad-positive negative control leakage`
- 这两类 failure 共同把论文从“steering success report”提升为“controllability boundary paper”
- 当前最好的写法不是补强 predictor claim，而是把 failure 当作机制分型证据
