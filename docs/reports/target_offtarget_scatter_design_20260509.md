# Target-Movement vs Off-target-Instability Scatter Design

更新时间：2026-05-09

本设计定义一张可直接用于主文或 appendix 的散点图，用来可视化：

- Qwen 7B PSD `n=100`
- GLM-4-9B PSD `n=100`
- Mistral-7B PSD `n=100`

在 `target-directed improvement` 与 `off-target instability` 上的机制差异。

## 1. Figure Goal

图的核心命题是：

> Damage-prone regimes are not inactive. They can show target-logit improvement and still fail because off-target instability rises at the same time.

预期视觉分布：

- Qwen: 右下角或偏右下，`high target movement + low instability`
- GLM: 中右到右上，`some target movement + modest instability`
- Mistral: 明显右上，`strong target movement + high instability`

## 2. Axes Definition

### X axis: target-logit improvement

定义：

`target_logit_improvement = intervened_pressured_logit(target) - no_intervention_pressured_logit(target)`

解释：

- 正值更好
- 目标选项 `target` 以 `no_intervention` baseline 里的正确/目标选项为锚

### Y axis: off-target instability score

首选定义：

`off_target_instability_score = number of non-target options whose ranking changes between no_intervention and intervention`

这是当前 `offtarget_logit_decomposition_20260509.md` 已采用的 operational definition。

若某个 model row 缺更细 rank data，则可退化为近似：

- `damage flag` 二值
- 或 `baseline margin change` 的绝对值

但主图应优先保持三模型同一口径。

## 3. Point Encoding

每个 item 一个点。

编码建议：

- color:
  - Qwen 7B = teal
  - GLM = orange
  - Mistral = red
- shape:
  - Qwen = circle
  - GLM = square
  - Mistral = triangle
- alpha transparency: `0.45-0.60`

Damage cases 额外强调：

- `is_damage = True` 用实心点
- `is_damage = False` 用空心点或更低不透明度

## 4. Data Sources

### Qwen 7B

使用：

- [belief_causal_records.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen7b_n100/Qwen_Qwen2.5-7B-Instruct/20260506_181314/belief_causal_records.jsonl)
- [belief_causal_comparisons.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen7b_n100/Qwen_Qwen2.5-7B-Instruct/20260506_181314/belief_causal_comparisons.jsonl)

### GLM

使用：

- [belief_causal_records.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/belief_causal_records.jsonl)
- [belief_causal_comparisons.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/belief_causal_comparisons.jsonl)

### Mistral

使用：

- [belief_causal_records.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/belief_causal_records.jsonl)
- [belief_causal_comparisons.jsonl](/Users/shiqi/code/graduation-project/outputs/experiments/mistral7b_belief_causal_transfer_n100/mistralai_Mistral-7B-Instruct-v0.3/20260508_012308/belief_causal_comparisons.jsonl)

已有 GLM / Mistral scatter-ready extraction：

- [offtarget_logit_scatter_data_20260509.csv](/Users/shiqi/code/graduation-project/docs/reports/offtarget_logit_scatter_data_20260509.csv:1)

Qwen 需要用同一 extraction logic 补一份同格式 row，然后与 GLM / Mistral 合并。

## 5. Required Intermediate CSV Schema

建议 Analysis 先统一生成：

`docs/reports/target_offtarget_scatter_data_20260509.csv`

列：

| Column | Meaning |
| --- | --- |
| `item_id` | item id |
| `model` | `Qwen 7B` / `GLM-4-9B` / `Mistral-7B` |
| `target_logit_improvement` | x axis |
| `off_target_instability_score` | y axis |
| `is_damage` | bool |
| `damage_subtype` | optional label |
| `baseline_margin_change` | optional secondary diagnostic |

## 6. Plot Layout

建议一页一图，主图为单 panel scatter。

附加元素：

- x=0 竖线
- y=0 横线
- 每个模型一条 centroid marker
- 右上角 legend

可选加边际摘要：

- 每模型 `mean x`, `mean y`, `damage rate`

## 7. PDF Output

建议输出：

- `docs/reports/figures/target_offtarget_scatter_20260509.pdf`

若图用于论文编译，可同步导出：

- `docs/reports/figures/target_offtarget_scatter_20260509.png`

## 8. Caption Draft

建议 caption 核心句：

> Qwen concentrates in a high-target / low-instability region, whereas GLM and especially Mistral show target-directed movement that coexists with off-target ranking instability, consistent with benefit-damage coupling rather than inactive intervention.

## 9. Expected Interpretation

### Qwen

- 目标 logit movement 强
- instability 低
- 与 `partial-positive / low-damage` 更一致

### GLM

- 平均 target improvement 为正，但幅度中等
- instability 不高但 damage case 集中在非目标 ranking 扰动
- 对应 `weak/tradeoff-limited`

### Mistral

- target movement 更强
- instability 显著更高
- 对应 `damage-prone`

## 10. Boundary

- 不把该图写成 full causal proof；它是 target-vs-off-target operational visualization。
- 若 Qwen 的 per-item extraction 与 GLM/Mistral 字段命名略有差异，必须先统一口径后再合图。
- 若 Qwen 同格式 row 无法及时补齐，则主图暂时只画 GLM/Mistral，Qwen 作为 planned comparator 明确标注。
