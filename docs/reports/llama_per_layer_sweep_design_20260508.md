# Llama Per-layer Behavioral Sweep Design

更新时间：2026-05-08

本设计专门针对 `Llama-3.1-8B-Instruct` 的逐层 behavioral intervention sweep。目标是验证下面这条机制假说：

> localization window may exist, but clean-control window may still be absent.

## 1. Starting Point

现有冻结 evidence：

- subspace summary 路径：
  [belief_causal_subspace_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_subspace_summary.csv)
- behavioral summary 路径：
  [belief_causal_summary.csv](/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/belief_causal_summary.csv)

当前已知：

- localization range: layers `20-27`
- EV peak: layer `24`, `top1_explained_variance ≈ 0.894`
- direction coherence: `-0.08` 到 `-0.16`
- full window behavioral row (`24-31`, `k=2`, `alpha=0.5`):
  - drift `+0.03`
  - compliance `+0.01`
  - recovery `-0.03`
  - damage `0.08`

这组起点本身已经很像：

- `locatable but not controllable`

但还缺一个更细的 mechanistic check：是否在某个窄单层 window 中，clean-control 会重新出现。

## 2. Sweep Parameter Table

| 参数 | 值 | 理由 |
| --- | --- | --- |
| 方法 | `pressure_subspace_damping` | 与已有 Llama n=100 / alpha sweep family 保持一致 |
| 层范围 | 逐层 sweep：`20`, `21`, `22`, `23`, `24`, `25`, `26`, `27` | 直接测试 clean-control window 是否存在于某个窄层 |
| k | `2` | 与已有配置一致 |
| alpha | `0.5` | 与已有 full-window frozen row 一致 |
| n | `24` | exploratory first pass；先不升到 `n=100` |
| train source | `philpapers2020` | 与现有 English sweep 一致 |
| eval source | `nlp_survey` | 与现有 English sweep 一致 |
| prompt_variant | `english` | 与当前 Llama exploratory 主线一致 |
| output root | `outputs/experiments/llama31_8b_per_layer_behavioral_sweep` | 与现有 artifact family 分离 |

## 3. Output Format

输出 CSV schema 与 Qwen sweep 保持可比：

`model,layer_config_name,beta,num_samples,pressured_compliance_delta,baseline_damage_rate,recovery_delta,stance_drift_delta,net_recovery_without_damage`

字段解释：

- `model`: `Llama-3.1-8B`
- `layer_config_name`: `20`, `21`, ..., `27`
- `beta`: 保持和 Qwen sweep 一致的列名，但在这里实值等于 `alpha=0.5`
- 其余字段按 Qwen sweep 现有 exporter 口径直接计算

## 4. Script Status

与上轮不同，这次已有可直接复用的 sweep 脚本：

- [scripts/run_cross_model_behavioral_layer_sweep.py](/Users/shiqi/code/graduation-project/scripts/run_cross_model_behavioral_layer_sweep.py)

该脚本已经支持：

- `--model-name`
- `--train-source` / `--eval-source`
- `--prompt-variant`
- `--train-n` / `--eval-n`
- `--k`
- `--alpha-values`
- `--layer-configs`
- 输出 `behavioral_sweep_long.csv`

因此 A6 不需要 Code 部门先写新脚本，只需要直接给出单层 layer-configs 并执行。

## 5. macOS Terminal Command Template

这条命令涉及 MPS，应由用户在普通 macOS Terminal 中执行：

```bash
cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_cross_model_behavioral_layer_sweep.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device mps \
  --dtype auto \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --prompt-variant english \
  --train-n 24 \
  --eval-n 24 \
  --method pressure_subspace_damping \
  --k 2 \
  --alpha-values 0.5 \
  --layer-configs '20=20;21=21;22=22;23=23;24=24;25=25;26=26;27=27' \
  --seed 20260508 \
  --output-root outputs/experiments/llama31_8b_per_layer_behavioral_sweep \
  --log-level INFO
```

## 6. Expected Readout

Analysis 部门拿到 `behavioral_sweep_long.csv` 后，应直接检查每一层是否满足：

- `stance_drift_delta < 0`
- `pressured_compliance_delta < 0`
- `recovery_delta > 0`
- `baseline_damage_rate < 0.05`

可再加一个综合列：

- `net_recovery_without_damage = recovery_delta - baseline_damage_rate`

## 7. Hypotheses

### Hypothesis H1

如果 Llama 在任一单层都没有出现：

- drift 改善
- compliance 改善
- recovery 上升
- damage `< 0.05`

则支持：

- `clean-control window 缺失`

### Hypothesis H2

如果某个单层，例如 `24` 或 `25`，出现弱 clean-control：

- 说明 localization 与 controllability 的最佳层并不一定等于旧 full-window bundle
- 也说明 “full window fails” 不等于 “all narrow windows fail”

### Hypothesis H3

如果单层 sweep 全部比 full window 伤害更大：

- 说明 Llama 的方向可能更依赖 distributed bundle
- 但 distributed bundle 仍不能 cleanly control behavior

## 8. Interpretation Guide

### Result pattern A

所有单层都不满足 clean-control 条件：

- strongest claim: `localization window exists but clean-control window is absent`

### Result pattern B

1-2 个单层有轻微 drift/compliance 改善，但 recovery 不上升或 damage 偏高：

- safest claim: `localization and controllability are partially dissociated, with no clean-control window`

### Result pattern C

某个单层出现弱 clean-control：

- safest claim: `clean-control may be layer-misaligned relative to the old full window, but remains weaker and less stable than Qwen`

## 9. Bottom Line

这轮 sweep 的核心不是要把 Llama 救成正向 replication，而是把 limitation 机制化：

- 若单层 sweep 仍失败，就更强地支持 `locatable but not controllable`
- 若窄层出现局部改善，也能支持“localization 与 clean-control 不在同一窗口”这一更细的机制边界
