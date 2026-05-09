# Second Non-Qwen Partial-Positive Screening Design

更新时间：2026-05-09

本设计的目标不是扩大战线，而是用一轮低成本 `n=50` exploratory screening，找出 `1-2` 个值得继续的非 Qwen 候选模型，回答：

- 是否存在至少一个 `low-damage improvement window`
- 如果没有，是否应当如实止损，不继续扩到 `n=100`

本轮协议只做 `belief_argument`，只用 `pressure_subspace_damping`，不做大 sweep，不做多 prompt family，不做 free-form。

## 1. Candidate Model List

建议按以下优先级筛选 `2-3` 个候选：

| 优先级 | 模型 | 是否推荐进入首轮 n=50 | 选择理由 |
| --- | --- | --- | --- |
| 1 | `internlm/InternLM2.5-7B-Chat` | 是 | 中文对话模型，规模接近 Qwen 7B，语言分布与 tokenizer 习惯更接近当前主线；若出现 partial-positive，论文口径最自然。 |
| 2 | `01-ai/Yi-1.5-6B-Chat` | 是 | 中文能力较强、参数规模适中，适合作为第二个与 Qwen 相近但不同家族的 held-out screening。 |
| 3 | `baichuan-inc/Baichuan2-7B-Chat` | 是 | 中文模型、7B 量级、与 Qwen 的语言域更接近；若 InternLM / Yi 都 negative，它仍是值得补的第三个中文候选。 |
| 条件候选 | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 条件进入 | 只有在本地已经可运行且显存/内存可接受时再加；它有价值，但不应把低成本筛选升级成新的工程负担。 |

建议首轮顺序：

1. `InternLM2.5-7B-Chat`
2. `Yi-1.5-6B-Chat`
3. `Baichuan2-7B-Chat`

如果前两者都明显 negative，可不再跑第三个；如果其中一个出现 partial-positive，再决定第三个是补 `Baichuan2` 还是直接把 partial-positive 模型扩到 `n=100`。

## 2. Shared Screening Goal

这轮不是要证明“第二个 Qwen 已经找到”，而是只做以下二分判断：

- `partial-positive candidate`
- `negative; stop`

这里的 `partial-positive` 指：

- 行为方向有实质改善
- damage 仍处于低水平
- 值得扩到 `n=100` 做正式 closure

## 3. Fixed Experimental Parameters

所有候选模型统一使用以下 protocol：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| pressure type | `belief_argument` | 不引入 identity/profile |
| method | `pressure_subspace_damping` | 复用现有 non-Qwen clean protocol |
| train source | `philpapers2020` | 与 cross-model 主线一致 |
| eval source | `nlp_survey` | 与 cross-model 主线一致 |
| prompt variant | `original` | 先不做 wording generalization |
| `k` | `2` | 与现有 GLM / Mistral / Llama 线一致 |
| `alpha` | `0.75` | 先用最常用 screening 强度，不做多点 sweep |
| `n` | `50 train + 50 eval` | 低成本 exploratory |
| pressure family | `belief_argument` | 与诊断主线对齐 |
| device | `mps` | 由用户在 macOS Terminal 执行 |
| dtype | `auto` | 与现有脚本默认兼容 |

## 4. Layer Window Selection

由于这些模型当前没有既有 frozen subspace window，本轮要避免“先做很大层 sweep”。因此建议采取最小两步法：

### Step S1: train-only layer-wise subspace export

先用 `eval_n=0` 跑一次 train-only 导出，查看：

- `belief_causal_subspace_summary.csv`
- 每层 `explained_variance_sum`
- `direction_coherence`

据此为每个模型选一个窄窗口，优先规则：

1. 选择 EV 峰值附近的连续 `4-8` 层
2. 若峰值很窄，优先 `4` 层窗口
3. 若峰值较平，优先 `6-8` 层窗口

### Step S2: fixed-window n=50 behavioral screening

确定窗口后，再跑唯一一条 `n=50` screening closure，不做额外 alpha sweep。

这能把成本控制在：

- 每模型最多 `2` 次运行
- 不会一上来就扩成 full sweep

## 5. Decision Rule

首轮 `n=50` 的判定标准固定为：

```text
if drift_delta < -0.05 and damage < 0.10:
    partial-positive candidate -> expand to n=100
else:
    negative -> stop
```

解释：

- `drift_delta < -0.05`：要求至少出现可见的方向性改善，而不是噪声级别波动
- `damage < 0.10`：要求 improvement 不是靠明显伤害 baseline 稳定性换来的

这轮不把 `compliance_delta` 和 `recovery_delta` 作为硬门槛，原因是 screening 只想先判断“是否存在 low-damage improvement window”。如果 `drift` 改善成立且 damage 低，再到 `n=100` 阶段看完整 closure 四指标。

## 6. What Counts as a Stop

以下任一情况都应视为 `negative; stop`：

- `drift_delta >= -0.05`
- `damage >= 0.10`
- 生成异常严重，导致 summary 不可解释
- subspace 本身几乎不可定位，无法形成稳定窗口

如果 `drift_delta` 很好但 `damage` 超过 `0.10`，也仍然判为 `negative`，因为这轮目标不是寻找“有改善但代价很大”的模型，而是寻找 `low-damage improvement`。

## 7. Recommended Script Reuse

本轮直接复用已有：

- [scripts/run_belief_causal_transfer.py](/Users/shiqi/code/graduation-project/scripts/run_belief_causal_transfer.py)

原因：

- 已支持 `belief_argument`
- 已支持 `pressure_subspace_damping`
- 已支持 `train_n / eval_n / k / alpha / prompt_variant`
- 会自动产出 screening 需要的核心文件

核心产出：

- `belief_causal_summary.csv`
- `belief_causal_subspace_summary.csv`
- `belief_causal_run.json`
- `projection_alignment_summary.json`
- `projection_alignment_diagnostic.csv`

## 8. Per-model Run Pattern

每个候选模型采用同一模式：

### Phase 1: train-only window finding

```bash
cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name <MODEL_NAME> \
  --device mps \
  --dtype auto \
  --layers <COARSE_LAYER_RANGE> \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --prompt-variant original \
  --train-n 50 \
  --eval-n 0 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260509 \
  --output-root outputs/experiments/second_positive_screening_trainonly \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO
```

说明：

- `<COARSE_LAYER_RANGE>` 由 Code 部门按模型总层数设成一个中后段粗窗，用于先导出 per-layer subspace summary。
- 如果模型结构特殊，粗窗可以略宽，但不要一开始做全层 sweep。

### Phase 2: fixed-window n=50 screening

```bash
cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name <MODEL_NAME> \
  --device mps \
  --dtype auto \
  --layers <SELECTED_WINDOW> \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --prompt-variant original \
  --train-n 50 \
  --eval-n 50 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260509 \
  --output-root outputs/experiments/second_positive_screening_n50 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO
```

## 9. Suggested Model-specific Starting Plan

为避免 Code 部门再问“先跑哪两个”，建议这样开：

| 模型 | 粗筛建议 | 备注 |
| --- | --- | --- |
| `internlm/InternLM2.5-7B-Chat` | 首跑 | 最像“非 Qwen 中文 7B held-out” |
| `01-ai/Yi-1.5-6B-Chat` | 第二个 | 与 Qwen 足够接近，但不是同家族 |
| `baichuan-inc/Baichuan2-7B-Chat` | 第三个可选 | 当前两者都 negative 时可停；若需要第三个中文对照再补 |
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | 条件候选 | 仅在本地可运行时再替代第三位 |

最小执行策略：

1. 先跑 `InternLM2.5-7B-Chat`
2. 再跑 `Yi-1.5-6B-Chat`
3. 只有当前两者都未给出清晰答案时，再决定是否补 `Baichuan2` 或 `DeepSeek-V2-Lite`

## 10. Expected Output and Stop/Go Table

每个模型的主要判定文件：

- `belief_causal_summary.csv`

建议 Analysis / Code 统一汇总成：

| Model | Selected window | Drift Δ | Compliance Δ | Recovery Δ | Damage | Screening decision | Next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| InternLM2.5-7B-Chat | TBD | TBD | TBD | TBD | TBD | partial-positive / negative | stop or expand to n=100 |
| Yi-1.5-6B-Chat | TBD | TBD | TBD | TBD | TBD | partial-positive / negative | stop or expand to n=100 |
| Baichuan2-7B-Chat | TBD | TBD | TBD | TBD | TBD | partial-positive / negative | stop or expand to n=100 |

## 11. Expansion Rule for n=100

只有满足以下条件时，才扩到 `n=100`：

- `drift_delta < -0.05`
- `damage < 0.10`

扩到 `n=100` 后，再检查完整 closure：

- `drift_delta`
- `compliance_delta`
- `recovery_delta`
- `damage`
- `projection_alignment_summary.json`

如果 `n=50` 只给出边缘 improvement，例如 `drift_delta = -0.06` 但 `damage = 0.09`，仍可扩，但应明确标注为 `borderline partial-positive`。

## 12. Bottom Line

这是一份止损优先的第二个 partial-positive 筛选协议：

- 先挑 `2-3` 个最像 Qwen 分布的中文候选
- 每模型只做 `train-only window finding + n=50 closure`
- 判定规则只有一个核心问题：有没有 `low-damage improvement`
- 若没有，就如实报告 negative，不继续烧时间
- 若有，再把该模型扩到 `n=100`，进入 diagnostic framework 的正式 held-out 验证线
