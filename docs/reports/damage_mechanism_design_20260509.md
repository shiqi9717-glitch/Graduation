# Damage Mechanism Analysis Design

更新时间：2026-05-09

本设计用于解释为什么某些模型能得到 target-side benefit，但同时出现明显 baseline damage。当前重点对象：

- GLM-4-9B
- Mistral-7B

## 1. Current Starting Point

### GLM n=100

- drift `-0.01`
- compliance `-0.01`
- recovery `+0.12`
- damage `0.06`
- 已有：
  - `projection_alignment_diagnostic.csv`
  - `belief_causal_records.jsonl`
  - `belief_causal_comparisons.jsonl`

### Mistral n=100

- drift `-0.12`
- compliance `-0.05`
- recovery `+0.38`
- damage `0.39`
- 已有：
  - `projection_alignment_diagnostic.csv`
  - `belief_causal_records.jsonl`
  - `belief_causal_comparisons.jsonl`

## 2. What Is Already Available

当前 per-item 数据里已经有：

- `answer_logits` for `A/B/C/D`
- `correct_wrong_margin`
- `baseline_damage`
- `belief_logit_delta_vs_no`
- `negative_logit_delta_vs_no`

当前还没有的关键字段：

- full-vocab `top_token_logits before/after`
- richer free-text degeneration signals

因此，A10 可以立即完成：

- baseline margin collapse analysis
- option-ranking instability analysis
- target benefit vs negative-control movement analysis

但若要做真正的 `top-k token shift`，Code 部门需要补跑或补记录。

## 3. Analysis Modules

### D1. Negative-control logit trajectory

目标：

- 检查 target benefit 上升时，negative-control movement 是否同步扩大

现有数据能做的版本：

- 直接在 `projection_alignment_diagnostic.csv` 上分析：
  - `belief_logit_delta_vs_no`
  - `negative_logit_delta_vs_no`
  - `baseline_damage`

建议输出：

- per-item scatter: `x = target benefit`, `y = negative-control movement`
- damage cases 高亮

解释重点：

- GLM 若 `negative-control` 近零但 damage 仍高，说明 damage 可能来自 `other logits` 而不是这个 control axis
- Mistral 若 `negative-control` 已明显移动，则说明 damage 可能和更广泛的 logit distortion 耦合

### D2. Baseline answer margin analysis

目标：

- 检查 damage case 是否主要对应 `baseline correct answer margin collapse`

现有数据即可做：

- 从 `belief_causal_records.jsonl` 取：
  - `method = no_intervention`, `scenario = baseline`, `correct_wrong_margin`
  - `method = matched_belief_subspace_damping`, `scenario = baseline`, `correct_wrong_margin`

建议派生字段：

- `baseline_margin_before`
- `baseline_margin_after`
- `baseline_margin_delta = after - before`
- `is_damage_case`

预期解释：

- Mistral 更像 `large target benefit + baseline margin collapse`
- GLM 更可能是 `small target benefit + modest but frequent off-target perturbation`

### D3. Option-ranking instability

目标：

- 分析 damage case 是否表现为 `A/B/C/D` 选项排序突变，而非单纯正确项微降

现有数据即可做：

- 用 `answer_logits` 重建四选项排序
- 比较 baseline before/after 的：
  - top-1 是否翻转
  - correct option rank 是否下降
  - wrong non-target option 是否上升到 top-1

建议标签：

- `answer_flip`
- `margin_collapse_without_flip`
- `non_target_overcorrection`
- `unstable_option_ranking`

### D4. Top-k logit shift

目标：

- 看 damage 是否来自四选项之外的大范围 token disruption

现有数据不足：

- 当前 `belief_causal_records.jsonl` 只有 `answer_logits`
- 没有 `top_token_logits` 或 full-vocab top-k snapshot

Code 需要补的最小字段：

- `top_token_logits` for each `(method, scenario, item)`
- 推荐 `top_k = 20`

最小实现方式：

- 在 `run_belief_causal_transfer.py` 的 per-scenario record 写入与 `LocalProbeRunner` 一致的 `top_token_logits`

### D5. Damage case clustering

目标：

- 给 damage cases 做定性分桶，而不是只报一个比例

对当前 multiple-choice data，推荐主分桶：

- `answer_flip`
- `margin_collapse`
- `non_target_overcorrection`
- `unstable_option_ranking`

仅在未来补到更丰富 raw outputs 后，才扩展：

- `format degeneration`
- `semantic drift`
- `refusal`

原因：

- 当前 belief causal transfer 是 option-logit style multiple-choice output
- 直接谈 `format degeneration / refusal` 容易超出已测数据

## 4. Required Data Table

| Analysis item | Current source | Already available? | If missing, what Code should add |
| --- | --- | --- | --- |
| target benefit vs negative-control movement | `projection_alignment_diagnostic.csv` | Yes | N/A |
| baseline margin collapse | `belief_causal_records.jsonl` | Yes | N/A |
| option-ranking instability | `belief_causal_records.jsonl` | Yes | N/A |
| top-k token shift | per-item records | No | add `top_token_logits` before/after |
| qualitative free-text degeneration | raw text outputs | No | only relevant if a free-form damage branch is added |

## 5. Output Tables

### Damage taxonomy table

| Model | Damage type | Frequency | Associated signal | Example ID | Interpretation |
| --- | --- | ---: | --- | --- | --- |

建议 `Associated signal` 填：

- strong `baseline_margin_delta < 0`
- large `negative_logit_delta_vs_no`
- high projection increase
- option-rank instability

### Signal summary table

| Model | Target-logit benefit | Negative-control movement | Baseline-margin change | Damage rate | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |

建议聚合字段：

- `mean_belief_logit_delta_vs_no`
- `mean_negative_logit_delta_vs_no`
- mean / median `baseline_margin_delta` on damage cases
- overall `baseline_damage_rate`

## 6. Concrete Extraction Plan For Analysis

### GLM

1. 从 `projection_alignment_diagnostic.csv` 提取每个 item 的：
   - `belief_logit_delta_vs_no`
   - `negative_logit_delta_vs_no`
   - `baseline_damage`
2. 从 `belief_causal_records.jsonl` 配对 baseline before/after margin
3. 对 damage cases 计算：
   - mean margin collapse
   - option flip frequency

### Mistral

1. 同样读取 projection diagnostic 与 records
2. 重点比较：
   - damage cases 的 `baseline_margin_delta`
   - 是否存在强 target benefit 但 baseline margin 被压穿 0 的模式

## 7. Expected Hypotheses

### GLM

预期更像：

- `off-target logit disruption`

也就是：

- `negative-control` 本身不一定大幅移动
- 但四选项排序被小幅系统性扰动
- damage 是分散性的，不一定表现为极端 margin collapse

### Mistral

预期更像：

- `baseline answer margin collapse`

也就是：

- target-side benefit 很强
- 但 baseline correct margin 在 patched baseline 上被显著压低
- damage cases 更可能集中为 correct-to-wrong flip

## 8. Code Supplement Request

若要把 A10 做到更强论文级解释，Code 部门应补一版 records，至少新增：

- `top_token_logits`
- `ranked_options`
- `correct_option_rank`
- `wrong_option_rank`

推荐文件名：

- `belief_causal_records_with_topk.jsonl`

但在这一步之前，Analysis 部门已经可以先做 margin / option-ranking / damage clustering 的第一版。

## 9. Bottom Line

这份设计的核心是把 `damage` 从单一比例拆成机制类型：

- GLM 更可能是 `distributed off-target disruption`
- Mistral 更可能是 `margin-collapse-dominated damage`

如果这个分解成立，就能把 diagnostic framework 从“有没有 damage”进一步推进到“damage 是怎么来的”。
