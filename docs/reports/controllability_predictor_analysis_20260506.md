# Controllability Predictor Analysis

更新时间：2026-05-06

本报告只测试：已有诊断特征是否能预测现有 intervention family 的 `behavioral controllability`。它不提出新的 predictor，不跑新实验，不改代码，不改论文正文。

## 3.1 Controllability Label 表

本轮采用如下二值标签：

```python
clean_controllability = (
    drift_delta < -0.05 and
    compliance_delta < -0.05 and
    recovery_delta >= -0.05 and
    baseline_damage < 0.10
)
```

满足则 `label = 1`，否则 `label = 0`。

| 模型 | 干预方法 | drift Δ | compliance Δ | recovery Δ | damage | label |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 7B | `baseline_state_interpolation (24-26, 0.6)` | -0.1493 | -0.1595 | +0.7127 | 0.0000 | 1 |
| Qwen 7B | `baseline_state_interpolation aggressive (24-27, 0.6)` | -0.1839 | -0.1852 | +0.7619 | 0.0115 | 1 |
| Qwen 7B | `pressure_subspace_damping (k=2, alpha=0.75) n=100` | -0.0400 | -0.0400 | +0.0500 | 0.0400 | 0 |
| Qwen 7B | `late_layer_residual_subtraction` | +0.0115 | 0.0000 | +0.0476 | 0.0460 | 0 |
| Qwen 7B | `random_direction_control` | +0.0025 | -0.0060 | +0.0190 | 0.0127 | 0 |
| Qwen 7B | `shuffled_label_control` | +0.3949 | -0.0460 | +0.2667 | 0.5848 | 0 |
| Qwen 3B | `baseline_state_interpolation (31-35, 0.6)` | -0.1597 | -0.2789 | +0.6906 | 0.0000 | 1 |
| Qwen 3B | `pressure_subspace_damping (k=2, alpha=0.75)` | -0.3750 | -0.4170 | +0.0833 | 0.0417 | 1 |
| Qwen 14B | `matched_belief_subspace_damping n=100` | -0.0600 | -0.0700 | +0.0300 | 0.0100 | 1 |
| GLM-4-9B | `pressure_subspace_damping (k=2, alpha=0.75) n=100` | -0.0100 | -0.0100 | +0.1200 | 0.0600 | 0 |
| Llama-3.1-8B | `pressure_subspace_damping (k=2, alpha=0.5) n=100` | +0.0300 | +0.0100 | -0.0300 | 0.0800 | 0 |

Notes:

- `Qwen 3B pressure_subspace_damping` 当前工作区未定位到独立 `n=100` artifact；本表沿用现有 frozen clean-protocol row（`n=48`）做标签，后文会单独标注这一点。
- controls / subtraction 没有对应的完整 projection/layer-wise 诊断文件，因此在特征表中会标 `N/A`。

## 3.2 预测特征表

| 模型 | 干预方法 | belief_logit_delta | negative_logit_delta | specificity_ratio | baseline_projection_norm | pressured_projection_norm | baseline_pressured_ratio | projection_drift_corr | top1_explained_variance | layer_concentration | label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 7B | `baseline_state_interpolation (24-26, 0.6)` | -5.3871 | -0.0050 | 1083.72 | 70.9992 | 69.7820 | 1.0174 | -0.3899 | 0.8582 | 2.5586 | 1 |
| Qwen 7B | `baseline_state_interpolation aggressive (24-27, 0.6)` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 1 |
| Qwen 7B | `pressure_subspace_damping (k=2, alpha=0.75) n=100` | -5.8990 | -0.0017 | 3387.65 | 65.9136 | 75.2256 | 0.8762 | -0.4086 | 0.8252 | 2.4155 | 0 |
| Qwen 7B | `late_layer_residual_subtraction` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| Qwen 7B | `random_direction_control` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| Qwen 7B | `shuffled_label_control` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| Qwen 3B | `baseline_state_interpolation (31-35, 0.6)` | -5.1019 | -0.9303 | 5.48 | 22.2496 | 43.7769 | 0.5082 | -0.5243 | 0.8179 | 2.4784 | 1 |
| Qwen 3B | `pressure_subspace_damping (k=2, alpha=0.75)` | -5.1019 | -0.9303 | 5.48 | 22.2496 | 43.7769 | 0.5082 | -0.5243 | 0.8179 | 2.4784 | 1 |
| Qwen 14B | `matched_belief_subspace_damping n=100` | -7.3272 | -1.4254 | 5.14 | 48.9202 | 61.7447 | 0.7923 | +0.0777 | 0.5382 | 1.9084 | 1 |
| GLM-4-9B | `pressure_subspace_damping (k=2, alpha=0.75) n=100` | -2.8733 | -0.0011 | 2580.02 | 9.0481 | 11.9808 | 0.7552 | -0.4459 | 0.5958 | 2.1284 | 0 |
| Llama-3.1-8B | `pressure_subspace_damping (k=2, alpha=0.5) n=100` | -1.8273 | +0.1716 | 10.65 | 5.4071 | 10.8760 | 0.4972 | -0.2861 | 0.5068 | 1.9317 | 0 |

Notes:

- `top1_explained_variance` 来自对应 subspace summary 中 top layer 的 `top1_explained_variance`。
- `layer_concentration` 定义为 belief-related rows 中 top-3 unique layers 的 `explained_variance_sum` 之和。
- controls / subtraction / aggressive secondary 目前没有对应的完整 projection + layer-wise diagnostic artifact，因此严格按要求标为 `N/A`。

## 3.3 预测分析

### 1. Logit specificity 是否预测 controllability？

按当前 frozen sample，**如果把 specificity 定义成 `abs(belief / negative)`，答案是否定的**。

- 两个最强反例都是 `label = 0`，但 specificity 极高：
  - Qwen 7B `pressure_subspace_damping n=100`: `specificity_ratio = 3387.65`, `label = 0`
  - GLM `pressure_subspace_damping n=100`: `specificity_ratio = 2580.02`, `label = 0`
- 相比之下，几个 `label = 1` 的正例反而只有中等 specificity：
  - Qwen 3B mainline: `5.48`
  - Qwen 3B pressure_subspace_damping: `5.48`
  - Qwen 14B n=100: `5.14`

这意味着旧命题里“negative-control logit movement near zero”这一条，**不能直接升级为跨组合 predictor**。在现有样本里，near-zero negative delta 更像是会产生高-specificity 假阳性，而不是稳定预测 controllability。

`belief_logit_delta` 比 ratio 略好，但也不是充分条件：

- Llama `-1.83`、GLM `-2.87` 都较弱，且确实 `label = 0`
- 但 Qwen 7B PSD n=100 也有很强的 `belief_logit_delta = -5.90`，依然 `label = 0`

所以，**“强 + 特异”作为事后解释仍有启发性，但不足以单独预测 clean controllability**。

### 2. Projection norm 是否预测 controllability？

基本不预测。

- Qwen 7B PSD n=100 是 `label = 0`，但 `pressured_projection_norm = 75.23`，比 Qwen 14B 正例的 `61.74` 还高，也远高于 Qwen 3B 正例的 `43.78`
- Qwen 7B mainline 正例的 `baseline_projection_norm = 70.9992` 很高，但这并不能区分它和 Qwen 7B PSD n=100 这种高投影负例

因此，projection norm 更像“方向活跃度”特征，而不是 controllability 判别器。

### 3. Layer concentration 是否预测 controllability？

只能部分预测，不能单独成立。

- 正例里确实有高 concentration：
  - Qwen 7B mainline: `2.5586`
  - Qwen 3B mainline / PSD: `2.4784`
- 但负例 Qwen 7B PSD n=100 也很高：`2.4155`
- 反过来，Qwen 14B 是正例，但 `layer_concentration = 1.9084`，反而低于 GLM 负例的 `2.1284`

所以 localization / concentration 也不是充分条件。它更适合说明“方向可定位”，不适合单独说明“方向可 cleanly control”。

### 4. 哪个特征最能区分 controllable vs not controllable？

在当前只有 `7` 个具备完整特征的组合里，做一个极简单的单特征阈值扫描，结果是：

- `negative_logit_delta` 最强，当前样本上可做到 `7/7`
- `belief_logit_delta`、`baseline_projection_norm`、`pressured_projection_norm`、`layer_concentration` 都只能到 `6/7`
- `specificity_ratio` 也只有 `6/7`

但这里最重要的不是 `7/7` 本身，而是**方向和原命题相反**：

- 当前最优切分更接近 “`negative_logit_delta` 需要明显低于 0”，
- 而不是 “`negative_logit_delta` 越接近 0 越好”。

因此，现有数据并不支持把 Section 8 的旧 proposition 原封不动升级成 predictor。若要升级，命题本身必须改写。

### 5. 是否有反例？

有，而且很关键。

- **specificity 高但 label=0**
  - Qwen 7B PSD n=100: `specificity_ratio = 3387.65`, `label = 0`
  - GLM n=100: `specificity_ratio = 2580.02`, `label = 0`
- **specificity 不高但 label=1**
  - Qwen 14B n=100: `specificity_ratio = 5.14`, `label = 1`
  - Qwen 3B mainline / PSD: `5.48`, `label = 1`

所以 ratio 并不能单调排序 controllability。最清楚的现实是：**高 specificity 可以出现在 clean control，也可以出现在 weak / tradeoff-heavy / not-controllable rows。**

## 3.4 Held-out 验证

当前具备完整 predictor features 的组合只有 `7` 个，其中真正可用于 “Qwen 7B + Qwen 3B train / Llama + GLM test” 的训练行只有 `4` 个，而且 Qwen 3B mainline / PSD 两行共享同一套诊断特征。因此：

> insufficient data for formal held-out validation

不过可以做一个极简 toy rule 作为定性检查：

- 在可用的 Qwen train rows 上，规则 `negative_logit_delta <= -0.005 -> controllable` 可把训练集分开
- 用这个规则看 held-out：
  - Llama: `+0.1716` -> predict `0`，与 label 一致
  - GLM: `-0.0011` -> predict `0`，与 label 一致
  - Qwen 14B: `-1.4254` -> predict `1`，与 label 一致

但这条 toy rule 只是在极小样本上的定性现象，不能视为 formal validation，也不能直接写进主文作为稳定 predictor。

## 3.5 最终结论

> Logit specificity is similarly predictive of behavioral controllability as representation-level localization alone in the current frozen sample, not more predictive. The strongest single predictor is negative_logit_delta rather than specificity_ratio, with the key finding that the clearest false positives for the old Section 8 proposition are precisely the high-specificity but not-controllable Qwen 7B pressure-subspace and GLM rows.

## Practical Readout

如果要把 Section 8 从“事后解释”往“可预测框架”推进，当前最稳的写法不是：

> controllability appears when negative-control logit movement stays near zero

而应改成更保守的测试性表述：

> We test whether existing diagnostic features predict controllability, and find that no single representation-level or logit-specificity feature cleanly generalizes across the current intervention families. In particular, near-zero negative-control logit movement is not sufficient: some of the highest-specificity rows remain behaviorally non-controllable.

这会比直接宣称 “we propose a controllability predictor” 稳得多，也更符合当前 frozen evidence。
