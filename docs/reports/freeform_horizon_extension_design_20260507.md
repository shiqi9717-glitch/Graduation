# Free-form Patch-horizon Extension Design

更新时间：2026-05-07

本设计在现有 5 个 condition 的基础上，只新增：

- `first_5_tokens`
- `first_10_tokens`

不改论文正文，不跑实验；只为 Code 部门明确补跑配置。

## 1. New Conditions

| condition name | patch_generation_steps | patch_step_count | 说明 |
| --- | --- | ---: | --- |
| `first_5_tokens` | `True` | `5` | 与 `first_3_tokens` 完全同构，只把 horizon 从 3 提到 5 |
| `first_10_tokens` | `True` | `10` | 同上，再检查更长 front-loaded horizon 是否已经逼近 `continuous` 的质量退化 |

## 2. Parameters Held Constant

以下参数与现有 5 conditions 保持完全一致：

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- 层：`24-26`
- `beta=0.6`
- 样本数：`n=50`
- prompt family：当前 free-form pressured / recovery 模板
- 生成参数：`temperature=0`, `do_sample=False`, `max_new_tokens=50`
- repetition penalty：`1.0`
- readability judge：`heuristic`
- 输出格式：与现有 `run_summary.json` 和 `prompt_results.jsonl` 一致

## 3. What Stays The Same vs What Changes

### Same as `first_3_tokens`

- 同一脚本：[scripts/run_qwen7b_freeform_patched.py](/Users/shiqi/code/graduation-project/scripts/run_qwen7b_freeform_patched.py)
- 同一 patch family：`baseline_state_interpolation`
- 同一场景：`pressured`, `recovery`
- 同一 summary 字段：`drift_rate`, `wrong_follow_rate`, `readable_rate`, `repetition_rate`, `distinct_1`

### Different from `first_3_tokens`

- `patch_mode` 新增两个枚举值
- `_patch_config_from_mode()` 需要支持 `5` 和 `10`
- 输出目录名新增 `first_5_tokens` 与 `first_10_tokens`

## 4. Current Script Status

现有脚本已经支持：

- `continuous`
- `prefill_only`
- `first_token_only`
- `first_3_tokens`
- `no_intervention`

当前缺的不是新脚本，而是同一脚本里的两个小改动：

1. `argparse` 的 `--patch-mode` choices 增加 `first_5_tokens`, `first_10_tokens`
2. `_patch_config_from_mode()` 增加：
   - `first_5_tokens -> (True, 5)`
   - `first_10_tokens -> (True, 10)`

## 5. macOS Terminal Command Templates

以下命令应由用户在普通 macOS Terminal 中执行。前提是 Code 部门先完成上述脚本小改动。

### first_5_tokens

```bash
cd /Users/shiqi/code/graduation-project
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_qwen7b_freeform_patched.py \
  --mode full \
  --sample-file outputs/experiments/local_probe_qwen7b_intervention_main_inputs/qwen7b_intervention_main_sample_set.json \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype auto \
  --beta 0.6 \
  --patch-mode first_5_tokens \
  --diagnostic-confirm \
  --max-new-tokens 50 \
  --temperature 0 \
  --repetition-penalty 1.0 \
  --readability-judge heuristic \
  --strict-positive-n 30 \
  --high-pressure-n 20 \
  --output-root outputs/experiments/qwen7b_freeform_diagnostic/20260507/first_5_tokens \
  --log-level INFO
```

### first_10_tokens

```bash
cd /Users/shiqi/code/graduation-project
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_qwen7b_freeform_patched.py \
  --mode full \
  --sample-file outputs/experiments/local_probe_qwen7b_intervention_main_inputs/qwen7b_intervention_main_sample_set.json \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype auto \
  --beta 0.6 \
  --patch-mode first_10_tokens \
  --diagnostic-confirm \
  --max-new-tokens 50 \
  --temperature 0 \
  --repetition-penalty 1.0 \
  --readability-judge heuristic \
  --strict-positive-n 30 \
  --high-pressure-n 20 \
  --output-root outputs/experiments/qwen7b_freeform_diagnostic/20260507/first_10_tokens \
  --log-level INFO
```

## 6. Expected Output Format

期望输出与现有 2026-05-03 五个 run 完全同构：

- `run_summary.json`
- `generations.jsonl`
- `hidden_state_arrays/`

`run_summary.json` 中必须继续包含：

- `drift_rate`
- `wrong_follow_rate`
- `readable_rate`
- `repetition_rate`
- `distinct_1`
- `answer_extractable_rate`
- `truncation_rate`

## 7. Expected Result Hypotheses

### Behavioral effect

- `first_5_tokens` 与 `first_10_tokens` 的行为指标应继续接近 `first_3_tokens`
- 若当前 front-loaded claim 成立，它们不应显著优于 `prefill_only` / `first_token_only`

### Quality tradeoff

- `first_5_tokens` 的质量应劣于 `first_3_tokens`，但优于 `continuous`
- `first_10_tokens` 应进一步向 `continuous` 靠近，但未必完全崩到 `readable_rate = 0`

### Failure interpretation

- 如果 `first_5_tokens` 的 `readable_rate` 已接近 `0`，则说明 patch-horizon quality tradeoff 比当前估计更陡
- 如果 `first_10_tokens` 仍保持高 `readable_rate`，则说明 horizon boundary 可能在 `10` token 之后才明显出现

## 8. Analysis Handoff

Analysis 部门拿到新 outputs 后，应与现有五行一起画：

- x 轴：`patch horizon`
- y 轴：
  - `drift_rate`
  - `wrong_follow_rate`
  - `readable_rate`
  - `repetition_rate`
  - `distinct_1`

重点只回答两件事：

- 行为收益是否在最短 horizon 已饱和
- 质量退化是否从 `3 -> 5 -> 10 -> continuous` 呈单调恶化

## 9. Bottom Line

这次扩展不是为了再证明一次“patch 有效”，而是为了更精确地定位 free-form tradeoff 的 horizon：

- `first_5_tokens` 与 `first_10_tokens` 应帮助判断 quality collapse 是突然发生，还是渐进发生
- 若行为收益仍不提高，而质量继续恶化，就能更强地支持 `front-loaded effect + patch-horizon quality tradeoff`
