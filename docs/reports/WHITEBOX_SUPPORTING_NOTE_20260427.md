# White-box Supporting Note / Artifact Boundary

更新时间：2026-04-27

本说明用于为 white-box sycophancy 论文当前主文口径提供 supporting note / appendix note 级别的边界说明。本文只整合已有冻结结果与新增审稿支撑材料，不新增实验，不改动 frozen 数字，也不把这些说明扩写成新的主结果段落。

## 1. Objective-local Metric Family Note

Qwen 3B / 7B white-box mainline 结果属于 **objective-local metric family**，而不是 bridge-style causal metric family。这个边界应在主文、图表 caption 和附录中保持明确。

在冻结 statistical-closure 结果中，Qwen mainline 的指标定义为：

- `stance_drift`：`interference-induced-error proxy`
- `pressured_compliance`：`wrong-option-follow proxy`
- `recovery_delta`：相对于 `structural-zero recovery reference` 的 recovery 差值
- `baseline_damage_rate`：在当前测试样本中观测到的 intervention baseline damage rate

这些指标适用于 Qwen 本地主线，因为它们建立在 objective multiple-choice probe setup 上，存在显式 `correct option` 与 `pressure-promoted wrong option`。但它们不等同于 bridge causal 线中使用的 paired baseline / pressured / recovery behavioral metrics。因此：

- Qwen 3B / 7B 不应与 Qwen 14B、GLM、Llama、Mistral 直接合并成单一、无注释的 leaderboard-style 比较。
- 更稳的展示方式是：
  - Qwen 3B / 7B：`objective-local mainline / replication panel`
  - Qwen 14B / GLM / Llama / Mistral：`bridge causal / transfer panel`

一个相关的 artifact boundary 是样本口径：

- robustness / sweep / controls 原始导出里常见 `n=108`
- frozen closure 主效应表里对应 `n=100`

这不是缺失或回填错误，而是分析口径刻意不同：

- `n=108 raw` 保留了 `strict_positive`、`high_pressure_wrong_option` 和 `control`
- `n=100 closure` 的 formal mainline effect size 只保留两类 pressure-relevant 子集，即 `strict_positive + high_pressure_wrong_option`

因此，`n=108 raw -> n=100 closure` 应被理解为 **formal estimand restriction**，而不是样本不完整。

## 2. Updated Controls + Held-out Support Artifact Note

新增 formal controls 与 held-out evaluation 明显加强了 Qwen mainline 的审稿防御力，但它们的作用是 **supporting artifact support**，不是替换或重写 frozen mainline effect sizes。

这批新 artifact 目前支持三条更稳的说法。

### 2.1 Not an arbitrary activation edit

Qwen 7B 与 Qwen 3B 的 `random_direction_control` 都没有复现 mainline 所呈现的“低 drift、低 pressured compliance、强 recovery、低 damage”组合。

Qwen 7B formal controls：

- `stance_drift_delta = 0.0025`
- `pressured_compliance_delta = -0.0060`
- `recovery_delta = 0.0190`
- `baseline_damage_rate = 0.0127`

Qwen 3B formal controls：

- `stance_drift_delta = -0.0115`
- `pressured_compliance_delta = -0.0280`
- `recovery_delta = 0.1462`
- `baseline_damage_rate = 0.0423`

因此，当前最稳的表述是：

> The Qwen mainline is not reproduced by arbitrary late-layer activation edits under the same layer range and intervention strength.

### 2.2 Not a simple pairing artifact

`shuffled_label_control` 也没有复现 mainline 的干净 tradeoff，相反，它在 Qwen 7B 与 Qwen 3B 上都引入了明显 baseline damage。

Qwen 7B shuffled-label control：

- `baseline_damage_rate = 0.5848`

Qwen 3B shuffled-label control：

- `baseline_damage_rate = 0.4115`

即便 shuffled-label control 在某些单项指标上出现局部移动，它也没有恢复出 mainline 的“低 damage + 稳定收益”结构。因此，controls 目前能支持的不是“完全因果识别”，而是更窄但很关键的一条：

> The Qwen mainline cannot be explained as a simple pairing artifact under matched layer range and intervention strength.

### 2.3 Not solely driven by the original tuning subset

Qwen 7B held-out objective-local evaluation 在新的 held-out sample 上保留了 mainline 方向：

- `stance_drift_delta = -0.2000`
- `pressured_compliance_delta = -0.1747`
- `recovery_delta = 0.6571`
- `baseline_damage_rate = 0.0`

这条 held-out 结果并不会把 Qwen 主线升级成 bridge-causal claim，但它显著缓解了以下攻击：

> the positive result only exists on the original tuning subset

目前最稳的写法是：

> The Qwen 7B mainline generalizes to a held-out objective-local sample under the same metric family.

以及：

> This held-out result should still be treated as robustness-side validation rather than as a pre-registered clean-split main result.

关于 `baseline_damage_rate = 0.0`，当前仍应保持审慎：

- 可以写：`no baseline damage was observed in the tested sample`
- 不要写：`baseline-safe`、`guaranteed zero damage` 或类似过强表述

支持这些说法的 artifact 路径如下：

- Qwen 7B formal controls  
  `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_formal_controls_qwen7b/20260427_121722`
- Qwen 3B formal controls  
  `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_formal_controls_qwen3b/20260427_125304`
- Qwen 7B held-out evaluation  
  `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_qwen7b_heldout_eval/qwen7b_heldout_mainline/20260427_135639`
- Qwen layer / strength sweep assets  
  `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_sweep_assets/qwen_layer_strength_sweep/20260427_121108`

关于 CI 与 bootstrap，当前最稳的附录写法是：

- held-out objective-local export：`item-level percentile bootstrap, 2000 iterations`
- formal controls aggregate：`shared sample ID item-level percentile bootstrap, 2000 iterations`
- frozen `whitebox_effect_size_table.csv`：在当前审计输入下，bootstrap 实现细节 `unable to confirm`

因此不要再写：

- `frozen closure bootstrap is fully verified`
- `Qwen 7B held-out proves a clean split main result`

## 3. GLM Sample-size Comparability Note

GLM 结果最适合作为 **cross-family directional support under stronger tradeoff**，而不是 clean replication。这里需要单独说明样本量可比性的边界。

GLM 整个 experiment family 包含 sweep / transfer 结构，但主文中最直接可比的结果行，仍然是 frozen closure 中用于 cross-family comparison 的 representative rows：

- `philpapers belief_argument -> nlp_survey belief_argument`
- representative setting：`k=2, alpha=0.75`
- formal closure sample size：`n=24`

也就是说，artifact family 本身可以更大，但主文最常引用、最直接可比的 GLM rows 仍然是 small-sample rows。它们确实提供了正向方向信号：

- drift 下降
- pressured compliance 下降
- recovery 上升

但同时它们也伴随显著 baseline damage，因此主文应写成：

> most directly comparable rows remain small-sample and tradeoff-sensitive

以及：

> The GLM line provides cross-family directional support for belief-subspace damping, but the most directly comparable rows remain small-sample and tradeoff-sensitive, with materially higher baseline damage than the Qwen mainline.

这条写法的好处是：

- 保留正向 directional signal
- 不夸大样本量可比性
- 不掩盖 GLM 的 baseline-damage tradeoff

对应 frozen closure 结果：

- `n = 24`
- `stance_drift_delta = -0.2932`
- `pressured_compliance_delta = -0.2095`
- `recovery_delta = 0.3769`
- `baseline_damage_rate = 0.3325`

因此，GLM 的主文定位仍应保持为：

> cross-family directionality with stronger tradeoff

而不是：

> clean cross-family replication
