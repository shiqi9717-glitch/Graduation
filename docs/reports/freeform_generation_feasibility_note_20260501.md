# Free-form Generation Feasibility Note

更新时间：2026-05-01

本说明只评估技术可行性与最小实现范围，不改现有代码，不跑实验，不改论文正文。

## 1. Executive Conclusion

结论分两层：

- 如果目标是 **真正的 activation-edited free-form generation**，推荐方案是 **A: 自定义 autoregressive decode loop + per-step residual patching**。
- 如果目标只是以最小代价补一个“方向一致”的 sanity check，我更推荐 **降级方案**：只做 `no_intervention` 的 free-form baseline / pressured / recovery 生成，再用人工或 LLM-as-judge 做语义层面的 drift / compliance / recovery 粗判。

最终建议：
- full patched free-form generation: `No-Go for immediate implementation`
- downgraded no-intervention free-form sanity: `Go`

原因不是原理上做不到，而是当前仓库的 patching 逻辑只覆盖 **单次 forward 的最后 token 位置**，没有任何自回归 continuation hook 已经接好。

## 2. Current State

当前关键边界如下：

- [model_runner.py](/Users/shiqi/code/graduation-project/src/open_model_probe/model_runner.py:765) 的 `patch_final_token_residuals_multi(...)` 通过 `register_forward_hook` 把指定层的 `hidden_states[:, -1, :]` 替换成预先算好的 patch tensor，然后只读取当前步最后位置的 option logits。
- [intervention.py](/Users/shiqi/code/graduation-project/src/open_model_probe/intervention.py:135) 的 `build_layer_patch_map(...)` 对 `baseline_state_interpolation` 的实现是：
  - baseline 场景直接用 `baseline_tensor`
  - 非 baseline 场景用 `(1 - alpha) * interference_tensor + alpha * baseline_tensor`
- [run_local_probe_intervention.py](/Users/shiqi/code/graduation-project/scripts/run_local_probe_intervention.py:252) 当前只构造 `baseline` / `interference` prompt，并把结果写成 `predicted_answer` / `margin` 风格记录。
- [build_human_audit_freeform_sanity.py](/Users/shiqi/code/graduation-project/scripts/build_human_audit_freeform_sanity.py:505) 已明确声明：当前 package 没有 activation-edited free-form decoding hook。

因此，当前仓库没有现成的“patched model continues generating text”路径。

## 3. Option Assessment

### A. 自定义 generate loop + per-step residual patching

这是唯一真正正确的方案。

实现思路：
- 先用现有 `build_layer_patch_map(...)` 为给定 item + scenario 计算固定的 layer patch tensor。
- 写一个新的 generation helper，在每个 decode step 都调用一次模型 forward。
- 在指定层注册 forward hook，把当前 step 最后一个 token 位置的 residual 改成 patch tensor。
- 从当前 step logits 采样下一个 token，再把新 token append 到 `input_ids`，循环直到 EOS 或 `max_new_tokens`。

为什么这是最小正确方案：
- `baseline_state_interpolation` 的干预语义是“在每个生成步都把当前 residual stream 拉回 baseline-aligned state”。
- 只在 prompt final token 上 patch 一次，不能保证后续 token 仍在干预轨道上。

技术判断：
- 与当前 `patch_final_token_residuals_multi(...)` 的 hook 机制兼容。
- 不需要改 intervention 数学定义，只需要给 `LocalProbeRunner` 新增一个“step-wise generation with hooks”入口。

### B. 只 patch prompt final token，再调用标准 `model.generate()`

不推荐。

原因：
- 这只影响 prompt 编码阶段最后一个位置的 hidden state。
- 后续 continuation token 会在未干预的隐藏状态轨迹上继续滚动。
- 它更像“single-step nudge”，不是 paper 当前 `baseline_state_interpolation` 的真实 deployment analogue。

可以把它当极弱 exploratory trick，但不应作为正式 sanity check 结果。

### C. `logits_processor` / `generate()` 钩子间接实现

不推荐作为主方案。

原因：
- `logits_processor` 只能在 logits 层面改输出分布，拿不到中间 residual stream。
- 论文当前 intervention 是 hidden-state edit，不是 output-layer reweighting。
- 即便勉强接入 `generate()` 的内部 hooks，最后也会退化成“自己管理每步 forward”的复杂实现，本质还是 A。

结论：
- C 不能干净复用当前 white-box intervention 定义。

## 4. Recommended Minimal Implementation

如果必须进入实现阶段，我建议的最小实现是：

1. 在 `LocalProbeRunner` 新增一个专用方法，例如：
   - `generate_with_final_token_residual_patches(...)`
2. 该方法内部：
   - 复用 `_encode_prompt(...)`
   - 复用 `_resolve_model_parts(...)`
   - 复用当前 `patch_final_token_residuals_multi(...)` 的 hook 写法
   - 但把单次 forward 改成 autoregressive loop
3. 新增一个小型脚本，例如：
   - `scripts/run_qwen7b_freeform_sanity.py`
4. 该脚本只支持：
   - `Qwen/Qwen2.5-7B-Instruct`
   - `baseline_state_interpolation`
   - layers `24-26`
   - scale `0.6`
   - belief-argument pressure
   - `strict_positive` + `high_pressure_wrong_option` 抽样

这样能把改动面限制在一个 runner 方法和一个任务脚本。

## 5. Files And Functions To Touch

如果实现 full patched free-form generation，最小改动建议如下。

### 必改

- [src/open_model_probe/model_runner.py](/Users/shiqi/code/graduation-project/src/open_model_probe/model_runner.py:765)
  - 参考 `patch_final_token_residuals_multi(...)`
  - 新增一个 autoregressive generation 方法
  - 预计新增位置：`patch_final_token_residuals_multi(...)` 后面最自然

- [scripts/run_local_probe_intervention.py](/Users/shiqi/code/graduation-project/scripts/run_local_probe_intervention.py:252)
  - 不建议直接混进现有主线脚本
  - 更建议抽出一个新脚本复用其 sample loading / `build_layer_patch_map(...)` / output writing 逻辑

### 复用但可不改

- [src/open_model_probe/intervention.py](/Users/shiqi/code/graduation-project/src/open_model_probe/intervention.py:135)
  - `build_layer_patch_map(...)` 现有实现已经足够
  - 不需要为 free-form generation 重新定义 patch tensor

- [scripts/build_human_audit_freeform_sanity.py](/Users/shiqi/code/graduation-project/scripts/build_human_audit_freeform_sanity.py:505)
  - 当前只是记录 limitation
  - 真进入实现后可以更新说明，但不是第一步必须改的地方

## 6. Estimated Workload

full patched free-form generation 的最小工程量大致是：

- `model_runner.py`: 约 `120-180` 行新代码
- 新脚本 `run_qwen7b_freeform_sanity.py`: 约 `140-220` 行新代码
- 如果要加 JSONL schema、失败统计、截断统计，再加 `40-80` 行

整体可粗估为：
- 新增代码 `260-480` 行
- 核心新增函数 `2-3` 个
- 现有函数实质性修改 `0-1` 个

这不是“大重构”，但也明显不是“一小时补丁”。

## 7. Risk Estimate

### 性能

相对当前 option-logit 主线，per-step patching 会明显更慢。

原因：
- 当前主线每个场景基本只做一次 forward
- free-form generation 要做 `prompt_prefill + 每个生成 token 一次 forward`
- Qwen 7B 在本仓库里默认是 `MPS + eager + float32`，本来就不是最快的 decode 组合
- 每步还要经过 Python hook

粗略倍率：
- 相对单次 option-logit forward，单条样本成本大约会上升到 `50x-150x`
- 相对“未 patch 的普通 generate loop”，patched 版本大约再慢 `1.2x-1.8x`

### 输出退化

存在中等风险。

主要风险：
- 重复
- 提前 EOS
- 空泛自我修正
- recovery 场景比 pressured 场景更 noisy

原因：
- 这里的 patch tensor 来自选择题 final-token state，不是为了开放生成专门拟合的 trajectory
- 把固定 residual anchor 强行施加到每个生成步，可能破坏局部语言建模流畅性

### MPS 耗时

对 `60 items x 5 generations = 300 generations` 的完整配置，我建议用区间估计，而不是单点估计。

如果平均实际输出长度在 `80-120 new tokens`：
- 较乐观：`1.5-2.5` 小时
- 更保守：`2.5-4` 小时

如果很多样本接近 `max_new_tokens=256`：
- 可能接近 `4-6` 小时

因此，原先“30-60 分钟”更像未 patch 或显著更短输出的估计，不适合作为 full patched generation 的默认预期。

## 8. Recommended Downgrade

如果目标是“证明方向一致”，更简单且更稳的替代方案是：

1. 只做 `no_intervention` 的 free-form generation
2. 场景保留：
   - baseline
   - pressured
   - recovery
3. 从同一批 60 个 item 生成真实文本
4. 用人工或 LLM-as-judge 标：
   - pressured 是否出现 pressure-following 语义
   - recovery 是否较 pressured 更接近独立判断
5. 再把这个方向性结论与 option-logit 主线做定性对照

这个降级方案不能证明：
- intervention 在 free-form 下同样有效

但它可以支持：
- pressure 在 free-form 下确实会诱发可读的 stance drift / compliance 语义
- recovery 的语义恢复比 pressured 更接近 baseline
- option-logit 主线并非纯粹“无行为可读性”的 artifact

从论文防守角度，这已经能回答一部分“multiple-choice artifact”质疑，而且实现成本远低于 full patched generation。

## 9. Go / No-Go

我的建议是：

- **对 full patched free-form generation：No-Go**
  - 不是因为不可做
  - 而是因为它需要新增自定义 decode 基础设施，耗时与不稳定性都不适合当前“最小 sanity check”目标

- **对 downgraded no-intervention free-form sanity：Go**
  - 能更快产出 reviewer-facing sanity evidence
  - 风险更低
  - 与当前 bounded-mechanism 叙事并不冲突

如果后续确实要做 full patched free-form，我建议把它定位成：
- appendix-side exploratory extension
- 明确写成 `engineering-heavy exploratory sanity check`

而不是当前主线必须补齐的 blocker。
