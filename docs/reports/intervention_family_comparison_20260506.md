# Qwen 7B Intervention Family Comparison

更新时间：2026-05-06

本说明只基于 Qwen 7B intervention family 的现有冻结结果做简短对比，不跑新实验，不改代码，不改论文正文。目标是回答三件事：

- `baseline_state_interpolation` 是否仍是最强主 baseline
- `pressure_subspace_damping` 是否可作为 `less-oracle alternative`
- “controllability 不依赖单一实现”能否成立，以及能成立到什么边界

## Method Comparison

注意：`baseline_state_interpolation` 主线使用的是 `objective-local proxy` metric family；`pressure_subspace_damping` 使用的是 `bridge / transfer-style` clean protocol family。下表可用于 family 内外的定性对照，但不应被读成一个无注释统一 leaderboard。

| method | n | drift delta | compliance delta | recovery delta | damage | 定位 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_state_interpolation` mainline (`24-26`, `0.6`) | 100 | -0.1493 | -0.1595 | +0.7127 | 0.0000 | formal mainline；当前最强主 baseline |
| `pressure_subspace_damping` n=100 (`k=2`, `alpha=0.75`) | 100 | -0.04 | -0.04 | +0.05 | 0.04 | less-oracle alternative；弱正向、附带小 tradeoff |
| `pressure_subspace_damping` old n=48 (`k=2`, `alpha=0.75`) | 48 | -0.0417 | 0.0000 | +0.0417 | 0.0417 | weak / boundary evidence；与 n=100 大体同向 |
| `late_layer_residual_subtraction` (`24-26`, `0.6`) | 108 | +0.0115 | 0.0000 | +0.0476 | 0.0460 | method-boundary failure；不可作为可用主方法 |
| `random_direction_control` | 108 | +0.0025 | -0.0060 | +0.0190 | 0.0127 | null-like control；支持“不是任意 activation edit” |
| `shuffled_label_control` | 108 | +0.3949 | -0.0460 | +0.2667 | 0.5848 | pairing-artifact control；高 damage，对照用途 |

## Three Short Conclusions

1. `baseline_state_interpolation` 仍然是最强主 baseline。  
   在当前冻结 evidence 里，它仍是唯一同时满足“大幅 drift/compliance 改善、强 recovery、零 observed damage”的 Qwen 7B formal mainline：`drift -0.1493`、`compliance -0.1595`、`recovery +0.7127`、`damage 0.0000`。不论从效应强度还是 evidence level 看，它都明显强于 `pressure_subspace_damping`。

2. `pressure_subspace_damping` 可以写成 `less-oracle alternative`，但只能是更弱的 secondary line。  
   新 `n=100` 相对旧 `n=48` 没有变弱，反而更整齐一些：`drift -0.04`、`compliance -0.04`、`recovery +0.05`、`damage 0.04`；旧 `n=48` 为 `drift -0.0417`、`compliance 0.0000`、`recovery +0.0417`、`damage 0.0417`。因此它可以支撑“Qwen 7B 的 controllability 不只出现在一个 oracle-heavier implementation 里”，但这条线仍远弱于 formal mainline，且始终伴随小幅 damage。

3. “controllability 不依赖单一实现”可以有限成立，但不能扩写成“任何实现都有效”。  
   目前能支持的最稳版本是：`baseline_state_interpolation` 与 `pressure_subspace_damping` 两种不同 intervention family 都在 Qwen 7B 上给出同方向改善；而 `late_layer_residual_subtraction` 没有复现这种模式，`random_direction_control` 近似 null，`shuffled_label_control` 则伴随极高 damage。换句话说，Qwen 7B 的 controllability 不是单一实现幻觉，但它也绝不是“换任何 late-layer rule 都能成功”。

## Boundary

- 不要写成 `new SOTA steering method`。
- 不要夸大 `pressure_subspace_damping`；它最多是 `less-oracle alternative` 或 `secondary positive line`，不是新的 mainline baseline。
- `late_layer_residual_subtraction`、`random_direction_control`、`shuffled_label_control` 都只应用作对照 / robustness / method-boundary 证据。
- 不要把 `pressure_subspace_damping` 与 `baseline_state_interpolation` 直接写成同一 metric family 下的 head-to-head winner/loser 比赛。

## Bottom Line

最稳的论文写法是：

> For Qwen 7B, `baseline_state_interpolation` remains the strongest formal baseline intervention. A cleaner but weaker `pressure_subspace_damping` line now shows modestly positive transfer at both n=48 and n=100, supporting a limited less-oracle alternative rather than a replacement mainline. At the same time, subtraction and control variants fail to reproduce the main benefit profile, so the current evidence supports implementation-robust but not implementation-agnostic controllability.
