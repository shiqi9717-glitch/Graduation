# Layer-wise Behavioral Sweep Design

更新时间：2026-05-07

本设计用于 Llama 与 GLM 的 `layer-wise behavioral intervention sweep`。目标不是重跑 Qwen，而是验证：

- Qwen 是否存在 clean-control layer window
- Llama 是否不存在 clean-control window，或窗口错位
- GLM 是否存在 specificity window，但没有 clean-control window

## 1. Shared Sweep Goal

所有 sweep 统一输出与 Qwen 可比的长表：

`model,layer_config_name,beta,num_samples,pressured_compliance_delta,baseline_damage_rate,recovery_delta,stance_drift_delta,net_recovery_without_damage`

说明：

- 为了和 [qwen7b_sweep_long.csv](/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_sweep_assets/qwen_layer_strength_sweep/20260427_121108/qwen7b_sweep_long.csv) 可比，继续沿用列名 `beta`。
- 在 Llama / GLM 行里，`beta` 实际对应 `alpha`。

## 2. Llama Sweep Parameters

| 参数 | 值 | 理由 |
| --- | --- | --- |
| 方法 | `pressure_subspace_damping` | 与已有 Llama causal-transfer family 保持一致 |
| 层范围 | `20-27` | 与协调器指定 sweep 范围一致，用于检查是否存在错位 clean-control window |
| layer configs | `20-21`, `22-23`, `24-25`, `26-27`, `20-23`, `24-27`, `20-27` | 同时覆盖窄窗、半窗、全窗 |
| k | `2` | 与已有 Llama clean transfer 配置一致 |
| alpha | `0.5`, `0.75` | `0.5` 对齐现有冻结 row；`0.75` 检查更强 damping 是否只增伤不增益 |
| n | `100` | 与当前 cross-model n=100 主线一致 |
| source split | `philpapers2020 -> nlp_survey` | 与 cross-model belief transfer 协议一致 |
| pressure type | `belief_argument` | 与论文主线一致 |
| 输出根目录 | `outputs/experiments/llama31_8b_behavioral_layer_sweep` | 与现有 artifact family 分离 |

## 3. GLM Sweep Parameters

| 参数 | 值 | 理由 |
| --- | --- | --- |
| 方法 | `pressure_subspace_damping` | 与已有 GLM n=100 family 一致 |
| 层范围 | `30-33` | 与当前 GLM n=100 subspace summary 一致 |
| layer configs | `30`, `31`, `32`, `33`, `30-31`, `32-33`, `30-33` | 层数少，适合做单层 + 半窗 + 全窗 |
| k | `2` | 与现有 GLM causal-transfer 配置一致 |
| alpha | `0.5`, `0.75` | `0.75` 对齐冻结主行；`0.5` 检查更弱 damping 是否更 clean |
| n | `100` | 与当前 cross-model n=100 主线一致 |
| source split | `philpapers2020 -> nlp_survey` | 与 cross-model belief transfer 协议一致 |
| pressure type | `belief_argument` | 与论文主线一致 |
| 输出根目录 | `outputs/experiments/glm4_9b_behavioral_layer_sweep` | 与现有 artifact family 分离 |

## 4. Output Schema

建议输出单个 long CSV，每行一组 `(model, layer_config_name, alpha)`：

| 列名 | 含义 |
| --- | --- |
| `model` | `Llama-3.1-8B` 或 `GLM-4-9B` |
| `layer_config_name` | 例如 `24-25`, `30-33` |
| `beta` | 实际为 damping `alpha`，保留旧列名以兼容 Qwen sweep plotting |
| `num_samples` | `100` |
| `pressured_compliance_delta` | patched - no-intervention compliance |
| `baseline_damage_rate` | baseline damage |
| `recovery_delta` | patched - no-intervention recovery |
| `stance_drift_delta` | patched - no-intervention drift |
| `net_recovery_without_damage` | `recovery_delta - baseline_damage_rate` 或与现有 Qwen exporter 同口径 |

## 5. Script Status

当前仓库里没有现成的 `Llama/GLM behavioral layer sweep` 脚本。

已有脚本：

- [scripts/run_belief_causal_transfer.py](/Users/shiqi/code/graduation-project/scripts/run_belief_causal_transfer.py) 适合固定 layer bundle 的 clean transfer eval
- [scripts/run_llama31_8b_belief_causal_transfer_english_sweep_mps.sh](/Users/shiqi/code/graduation-project/scripts/run_llama31_8b_belief_causal_transfer_english_sweep_mps.sh) 只做 alpha sweep，不导出 Qwen-style behavioral sweep long table
- [scripts/run_glm4_9b_belief_causal_transfer_mps.sh](/Users/shiqi/code/graduation-project/scripts/run_glm4_9b_belief_causal_transfer_mps.sh) 同样只跑单个固定层窗

因此：

- Code 部门需要先写 standalone 脚本，建议名为 `scripts/run_cross_model_behavioral_layer_sweep.py`
- 最自然的实现路径是复用 `run_belief_causal_transfer.py` 的 hidden-state extraction / subspace estimation / eval loop，再加 `--layer-configs` 和 `--alpha-values`

## 6. macOS Terminal Command Templates

以下命令模板供用户在普通 macOS Terminal 里执行。当前脚本尚不存在，因此这是 `Code 部门完成脚本后` 的调用形式。

### Llama

```bash
cd /Users/shiqi/code/graduation-project
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_cross_model_behavioral_layer_sweep.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device mps \
  --dtype auto \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --train-n 100 \
  --eval-n 100 \
  --method pressure_subspace_damping \
  --k 2 \
  --alpha-values 0.5,0.75 \
  --layer-configs '20-21=20,21;22-23=22,23;24-25=24,25;26-27=26,27;20-23=20,21,22,23;24-27=24,25,26,27;20-27=20,21,22,23,24,25,26,27' \
  --output-root outputs/experiments/llama31_8b_behavioral_layer_sweep \
  --log-level INFO
```

### GLM

```bash
cd /Users/shiqi/code/graduation-project
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_cross_model_behavioral_layer_sweep.py \
  --model-name THUDM/glm-4-9b-chat-hf \
  --device mps \
  --dtype auto \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --train-n 100 \
  --eval-n 100 \
  --method pressure_subspace_damping \
  --k 2 \
  --alpha-values 0.5,0.75 \
  --layer-configs '30=30;31=31;32=32;33=33;30-31=30,31;32-33=32,33;30-33=30,31,32,33' \
  --output-root outputs/experiments/glm4_9b_behavioral_layer_sweep \
  --log-level INFO
```

## 7. Expected Patterns

### If Qwen has a clean-control window but Llama does not

Llama sweep 应显示：

- 某些 layer window 可能有轻微 drift 改善
- 但 compliance 不改善或 recovery 不上升
- damage 不会随着“正确层”出现明显下降
- 不会出现像 Qwen 那样的 `net_recovery_without_damage` 正向峰值

### If GLM has a specificity window but no clean-control window

GLM sweep 应显示：

- 某些 window 在 drift / compliance 上略有改善
- 但 recovery 和 damage 之间形成 tradeoff
- 最优 row 可能是 `weak positive + residual damage`
- 不会出现同时满足 `drift↓ + compliance↓ + recovery↑ + damage≈0` 的 clean window

## 8. Analysis Handoff

Analysis 部门拿到 long CSV 后应直接做：

1. 按 `model` 分面作图
2. x 轴 = `layer_config_name`
3. 颜色 = `beta`
4. y 轴分别画：
   - `stance_drift_delta`
   - `pressured_compliance_delta`
   - `recovery_delta`
   - `baseline_damage_rate`
   - `net_recovery_without_damage`

## 9. Bottom Line

这份设计的关键不是证明 Llama / GLM 一定失败，而是把失败或错位写成可观测模式：

- Qwen 式 clean-control window = 四指标同向改善且 damage 很低
- Llama 式 limitation = no clean window
- GLM 式 boundary = 有 weak positive rows，但 clean window 不出现
