# Cross-Model Belief Transfer Boundary Update

更新时间：2026-05-06

本说明只基于当前冻结的三组 `n=100` belief causal transfer 输出与已知旧 small-sample 结果做对照分析，不跑新实验，不改代码，不改论文正文。目标是更新 Llama / GLM / Qwen14B 的 cross-model 边界口径，尤其重新处理 Qwen14B 从旧 `n=48` 到新 `n=100` 的定位变化。

## 1. N=100 Point Estimates

| model | n_items | no_intervention drift | patched drift | drift delta | no_intervention compliance | patched compliance | compliance delta | no_intervention recovery | patched recovery | recovery delta | baseline_damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | 100 | 0.52 | 0.55 | +0.03 | 0.99 | 1.00 | +0.01 | 0.57 | 0.54 | -0.03 | 0.08 |
| GLM-4-9B | 100 | 0.51 | 0.50 | -0.01 | 0.98 | 0.97 | -0.01 | 0.52 | 0.64 | +0.12 | 0.06 |
| Qwen2.5-14B | 100 | 0.41 | 0.35 | -0.06 | 0.94 | 0.87 | -0.07 | 0.70 | 0.73 | +0.03 | 0.01 |

读法约定：

- `drift / compliance` 负向 delta 更好。
- `recovery` 正向 delta 更好。
- `baseline_damage` 越低越好。
- 当前输入只有 point-estimate summary，没有新的 frozen closure CI；因此以下更新只能写成 boundary update / readout，不应直接替换旧 frozen closure 的正式统计地位。

## 2. Comparison To Older Small-Sample Readouts

### Llama: n=100 vs old n=24

旧 `n=24`（alpha=`0.5`）：

- drift: `0.4583 -> 0.3750`, delta `-0.0833`
- compliance: `1.0000 -> 1.0000`, delta `0.0000`
- recovery: `0.6667 -> 0.5833`, delta `-0.0833`
- damage: `0.2500`

新 `n=100`：

- drift: `0.52 -> 0.55`, delta `+0.03`
- compliance: `0.99 -> 1.00`, delta `+0.01`
- recovery: `0.57 -> 0.54`, delta `-0.03`
- damage: `0.08`

最稳结论：

- `compliance` 不改善、`recovery` 下降、存在 `damage`，这一点与旧口径一致。
- 旧 `n=24` 里 drift 有轻微改善，但新 `n=100` 反而转为轻微变差，因此“Llama 可控”不但不能加强，反而应更明确压回 `locatable but not controllable`。
- 换句话说，`n=100` 没有把 Llama 推向 positive replication，而是把它更稳地固定在 limitation / boundary 位。

### Qwen14B: n=100 vs old n=48

旧 `n=48`：

- drift: `0.4167 -> 0.4583`, delta `+0.0417`
- compliance: `0.9375 -> 0.8333`, delta `-0.1042`
- recovery: `0.7292 -> 0.6250`, delta `-0.1042`
- damage: `0.1042`

新 `n=100`：

- drift: `0.41 -> 0.35`, delta `-0.06`
- compliance: `0.94 -> 0.87`, delta `-0.07`
- recovery: `0.70 -> 0.73`, delta `+0.03`
- damage: `0.01`

最稳结论：

- Qwen14B 的口径必须更新。旧 `n=48` 的 `harmful / not recommended` 已不再适合作为当前 headline readout。
- 新 `n=100` 在四个关键维度上都比旧 `n=48` 更正向：drift 从变差转为改善，recovery 从下降转为上升，damage 从 `0.1042` 降到 `0.01`。
- 同时也不宜把它夸大成“clean strong success”。`recovery delta = +0.03` 仍偏小，改善幅度总体弱于 Qwen 3B clean 的强正向线。
- 因此更稳的更新是：`secondary positive confirmation with residual tradeoff`，或者更简洁地写成 `secondary positive confirmation`，并在句尾保留“not mainline-strength”边界。

### GLM: n=100 vs older narrative

当前没有同协议、同目录、同 freeze 层级的旧 clean `n=24` 结果可直接逐项对齐，因此不能给出像 Llama / Qwen14B 那样的严格前后数值比较。

新 `n=100` point estimates：

- drift delta `-0.01`
- compliance delta `-0.01`
- recovery delta `+0.12`
- damage `0.06`

最稳读法：

- GLM 继续体现 `improvement + damage tradeoff` 结构。
- 但新 `n=100` 的 damage (`0.06`) 比早期 frozen closure 中用于 cross-family叙事的高-damage representative row 温和得多，因此不宜继续沿用过强的“damage-heavy”口气。
- 也不宜把它升级成 clean strong success，因为 drift / compliance 改善幅度都很小，最主要的正向项是 `recovery +0.12`。
- 因此最稳写法仍是：`weak positive / damage tradeoff`，或者 `cross-family weak positive with residual damage tradeoff`。

## 3. Safest Cross-Model Writing

如果要给论文部门提供最稳的 cross-model 一句话排序，建议写成：

- `Qwen14B n=100`: secondary positive confirmation
- `GLM n=100`: weak positive with residual damage tradeoff
- `Llama n=100`: boundary / limitation; locatable but not controllable

如果要写成完整一点的段落，建议用下面这种结构：

> Among the three updated n=100 transfer evaluations, Qwen14B now provides the clearest secondary positive confirmation: drift and compliance both improve, recovery turns slightly positive, and baseline damage drops to 0.01. GLM retains only a weak positive profile, with small drift/compliance gains, a moderate recovery gain, and non-trivial residual damage. Llama remains a limitation case: despite a locatable direction, the intervention does not improve compliance, slightly worsens drift, reduces recovery, and still incurs damage.

## 4. Claim Boundary

### 可以写进主文的口径

- `Qwen14B` 可以从旧 `harmful / not recommended` 更新为 `secondary positive confirmation`。
- `GLM` 可以保留为 `weak positive with tradeoff`，强调其改善较弱、仍有 residual damage。
- `Llama` 应继续作为 `locatable but not controllable` 的 limitation / boundary 例子。

### 只能写成 appendix / note 的口径

- “这三组 n=100 已经重排了 cross-family hierarchy”的强叙事。
- 任何把 point-estimate update 直接当成新的 formal closure 表替代品的写法。
- 关于 GLM 与旧 `n=24` 是否完全一致的精确说法，因为当前缺少同协议旧目录。

### 不应再写的口径

- `Llama is a strong positive replication`
- `GLM is a clean strong success`
- `Qwen14B remains harmful / not recommended` as the current headline readout
- 任何把 Qwen14B n=100 直接升格成与 Qwen 3B / 7B formal mainline 同级的说法

## 5. Bottom Line

这轮 `n=100` 更新最重要的变化不是“所有模型都更强了”，而是 cross-model 边界更清楚了：

- Llama 更稳地落在 `locatable but not controllable`。
- GLM 仍是 `weak positive + residual tradeoff`。
- Qwen14B 则应从旧 small-sample 的 cautionary / harmful readout，更新为 `secondary positive confirmation with residual tradeoff boundary`。

这就是当前最稳妥、也最不容易过度承诺的 cross-model 写法。
