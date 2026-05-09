# Component-level Mechanism Analysis Design

更新时间：2026-05-09

本设计用于回答审稿人可能提出的机制质疑：

> 你的 clean-control window 到底来自 MLP、attention，还是只是在 full residual stream 上看到的 diffuse effect？

目标是设计 `MLP-only vs attention-only vs full-residual vs no-intervention` 对比实验，不跑实验，只给 Code / Analysis 可执行协议。

## 1. Intervention Components

### C0. No intervention

- 不做任何 patch
- 作用：behavioral baseline / damage baseline

### C1. Full residual

- 当前已有 `baseline_state_interpolation`
- hook 粒度：decoder block 输出后的 residual stream
- 对 baseline 场景：直接用 baseline residual final-token state
- 对 pressured 场景：`(1-β) * pressured_residual + β * baseline_residual`

### C2. Attention-only

- 只替换 attention contribution，保留 MLP contribution 不变
- 推荐 hook 点：`self_attn.o_proj` 的 `register_forward_pre_hook`
- 对 baseline 场景：baseline attention output
- 对 pressured 场景：`(1-β) * pressured_attn + β * baseline_attn`

### C3. MLP-only

- 只替换 MLP contribution，保留 attention contribution 不变
- 推荐 hook 点：`mlp.down_proj` 的输出 `register_forward_hook`
- 对 baseline 场景：baseline MLP output
- 对 pressured 场景：`(1-β) * pressured_mlp + β * baseline_mlp`

## 2. Recommended Hook Semantics

### Attention-only

当前 `model_runner.py` 已经具备两块可复用基础：

- `_capture_attention_pre_o_proj(...)`
- `patch_attention_head_output(...)` / `patch_attention_head_outputs_multi(...)`

因此 attention-only 最自然的做法不是重写 residual patch，而是新增：

- `capture_attention_output(prompt, target_layers)`：
  - 返回每层 final-token 的 merged attention output
- `patch_attention_output_multi(prompt, layer_patch_map, ...)`：
  - 在 `o_proj` pre-hook 处直接替换 final-token merged attention vector

### MLP-only

当前仓库没有对 MLP contribution 的等价 capture / patch helper。

推荐新增：

- `capture_mlp_output(prompt, target_layers)`：
  - 在 `mlp.down_proj` 输出处 capture final-token low-dim vector
- `patch_mlp_output_multi(prompt, layer_patch_map, ...)`：
  - 在 `mlp.down_proj` forward hook 输出处替换 final-token vector

为什么选 `down_proj` 输出：

- 语义最直接：这就是被加回 residual 的 MLP contribution
- 维度与 residual stream 一致，插值公式可直接复用
- 比在 `gate_proj` / `up_proj` 输入处 patch 更稳定，也更少架构依赖

## 3. Patch Tensor Construction

建议复用现有 `baseline_state_interpolation` 语义，不引入新数学定义。

### Full residual

- baseline: `R_base`
- pressured: `(1-β) * R_press + β * R_base`

### Attention-only

- baseline: `A_base`
- pressured: `(1-β) * A_press + β * A_base`

### MLP-only

- baseline: `M_base`
- pressured: `(1-β) * M_press + β * M_base`

这里的关键原则是：

- 只改变 component target
- 不改变插值数学
- 这样才能把结果解释成 `same intervention geometry, different pathway locus`

## 4. Implementation Recommendation

推荐路线：

- 保留现有 `intervention.py` 里的 4 个 method 不动
- 在 `model_runner.py` 新增 component capture / patch helper
- 写一个独立脚本：
  - `scripts/run_component_level_intervention.py`

不建议直接把 component-level 分支硬塞进现有 `run_local_probe_intervention.py`，原因是：

- 会把主线脚本语义搞复杂
- component-level 实验仍属 exploratory mechanism branch
- 更适合在单独脚本里写清 `component = full_residual / attention_only / mlp_only`

## 5. Code Changes Needed

### `model_runner.py`

建议新增：

- `capture_attention_output(...)`
- `capture_mlp_output(...)`
- `patch_attention_output_multi(...)`
- `patch_mlp_output_multi(...)`

### `intervention.py`

二选一：

1. 轻改 `intervention.py`
   - 新增 `attention_only_interpolation`
   - 新增 `mlp_only_interpolation`
2. 或完全不改
   - 在新脚本里直接构造 component-specific `layer_patch_map`

架构建议：

- 优先选方案 2
- 即：不扩 `SUPPORTED_METHODS`
- 把 component-level patch tensor 生成留在新脚本内

这样最不容易污染当前主线代码路径。

## 6. Experiment Grid

| Model | Regime role | Layers | β | n | Components |
| --- | --- | --- | ---: | ---: | --- |
| Qwen 7B | clean-control anchor | `24-26` | `0.6` | `24` | no-intervention, full-residual, attention-only, mlp-only |
| GLM-4-9B | specificity / damage overlap | `30-33` | `0.6` | `24` | no-intervention, full-residual, attention-only, mlp-only |
| Llama-3.1-8B | localization without clean control | `20-27` | `0.6` | `24` | no-intervention, full-residual, attention-only, mlp-only |
| Mistral-7B | optional high-damage extension | `24-31` | `0.6` | `24` | no-intervention, full-residual, attention-only, mlp-only |

说明：

- `β=0.6` 统一对齐 Qwen mainline，不追求各模型 individually tuned optimum
- `n=24` 先做 exploratory，主要看 component attribution 是否清晰

## 7. Output Table

Analysis 部门的最终汇总表建议直接用：

| Model | Regime | Layer | Component | Drift Δ | Compliance Δ | Recovery Δ | Damage | Clean-control score | Interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

其中：

`clean_control_score = -(drift_delta + compliance_delta) + recovery_delta - 10 * damage`

符号约定：

- `drift_delta` 负向更好
- `compliance_delta` 负向更好
- `recovery_delta` 正向更好
- `damage` 越低越好

## 8. Expected Hypotheses

### Qwen 7B

- 若 `MLP-only ≈ full residual`，说明 clean-control 主要沿 MLP pathway 承载
- 若 `attention-only` 只保留弱效果，说明 attention 更像定位 / routing，而不是主要 corrective pathway

### GLM

- 若 `MLP-only` 带来主要 benefit 和主要 damage，说明 damage 可能是 MLP off-target side effect
- 若 `attention-only` 也高 damage，则 damage 可能是更 diffuse 的 residual coupling

### Llama

- 若 full residual、attention-only、mlp-only 全都没有 clean-control window，则最强支持 `localization exists, but component-clean control is absent`
- 若某个 component 略好，也只应写成 weak mechanistic clue，不应直接升级成 replication

### Mistral

- 若加入 Mistral，最可能看到的是 `benefit + damage` 同时集中在某一 component
- 这会帮助区分 “damage 来自 MLP margin collapse” 还是 “damage 来自 attention-side routing disruption”

## 9. Suggested Script Interface

建议新脚本参数：

```text
--model-name
--component {full_residual,attention_only,mlp_only}
--layer-configs
--beta-values
--sample-file
--n-items
--output-root
```

输出应至少包含：

- `component_level_records.jsonl`
- `component_level_summary.csv`
- `component_level_manifest.json`

## 10. Bottom Line

这组实验不是要替换当前主线，而是要把主线再机制化一层：

- full residual 告诉我们“有没有 clean-control”
- attention-only / mlp-only 告诉我们“clean-control 更像由哪个 component 承担”
- 如果 Qwen 的 clean-control 主要落在 MLP-only，而 GLM / Llama 不成立，就能把跨模型 boundary 再向机制层推进一步
