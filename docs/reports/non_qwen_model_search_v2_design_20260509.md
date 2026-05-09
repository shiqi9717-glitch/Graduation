# Non-Qwen Model Search V2 Design

更新时间：2026-05-09

本设计用于继续寻找第 `1-2` 个新的非 Qwen 候选模型，但严格保持低成本筛选策略：

- 先检查本地缓存
- 先做 `train-only subspace extraction`
- 再做单条 `n=50 behavioral closure`
- 若 negative，立即停止

当前已知旧筛选结果：

- `InternLM2.5-7B-Chat`: 零效应
- `Yi-1.5-6B-Chat`: `damage ≈ 0.29`
- `Baichuan2-7B-Chat`: borderline partial-positive，`drift_delta = -0.05`, `damage = 0.07`

## 1. Local Cache Check

建议先检查这三个新候选是否已在本地缓存：

- `deepseek-ai/DeepSeek-V2-Lite-Chat`
- `google/gemma-2-9b-it`
- `microsoft/Phi-3-mini-128k-instruct`

当前工作区可直接确认：

- `Baichuan2-7B-Chat` 已缓存
- 上面三个新候选当前未在本地 Hugging Face cache 中发现现成 snapshot

因此本轮“最方便本地跑”的排序应改成：

1. `microsoft/Phi-3-mini-128k-instruct`
2. `google/gemma-2-9b-it`
3. `deepseek-ai/DeepSeek-V2-Lite-Chat` 仅在本地已可运行时进入

## 2. Recommended New Candidates

### Primary candidate: `microsoft/Phi-3-mini-128k-instruct`

理由：

- 体量最小，最适合低成本 exploratory
- 与 Qwen family 明显不同，可作为真正跨 family held-out
- 即使 negative，也能较快止损

### Secondary candidate: `google/gemma-2-9b-it`

理由：

- 不同于 Qwen / GLM / Llama / Mistral 的另一条 family
- 规模仍在可接受范围内
- 若出现 low-damage improvement，会比继续堆中文 7B 更有论文说服力

### Conditional candidate: `deepseek-ai/DeepSeek-V2-Lite-Chat`

理由：

- 中文能力和 mixture-of-experts 架构都很有吸引力
- 但当前未确认本地可运行，不应在本轮低成本筛选里优先投入

结论：

- 默认先跑 `Phi-3-mini`
- 再跑 `Gemma-2-9B-IT`
- 只有在 `DeepSeek-V2-Lite-Chat` 已就绪时，才可替换掉 `Gemma`

## 3. Shared Experimental Parameters

| 参数 | 值 |
| --- | --- |
| pressure type | `belief_argument` |
| method | `pressure_subspace_damping` |
| `k` | `2` |
| `alpha` | `0.75` |
| `train source` | `philpapers2020` |
| `eval source` | `nlp_survey` |
| `prompt variant` | `original` |
| `train_n` | `50` |
| `eval_n` | `50` |
| device | `mps` |
| dtype | `auto` |

## 4. Protocol

### Step 1: train-only subspace extraction

目的：

- 先导出 `belief_causal_subspace_summary.csv`
- 从中后段 coarse layer range 里选一个 `4-8` 层窗口

窗口选择规则：

1. 优先 EV 峰值附近连续层
2. 若峰值平缓，优先 `6-8` 层
3. 若峰值尖锐，优先 `4` 层

### Step 2: n=50 behavioral closure

只跑一条 fixed-window closure，输出：

- `belief_causal_summary.csv`
- `projection_alignment_summary.json`

不做额外 alpha sweep，不做多窗口 sweep。

## 5. Decision Rule

```text
if drift_delta < -0.05 and damage < 0.10:
    partial-positive -> expand to n=100
else:
    negative -> stop
```

说明：

- 这轮只看有没有 `low-damage improvement`
- 不满足就停止，不继续烧时间

## 6. Script Reuse

直接复用：

- [scripts/run_belief_causal_transfer.py](/Users/shiqi/code/graduation-project/scripts/run_belief_causal_transfer.py)

## 7. Command Templates

### Cache check

```bash
cd /Users/shiqi/code/graduation-project
find ~/.cache/huggingface/hub -maxdepth 2 -type d \
  \( -iname '*DeepSeek-V2-Lite-Chat*' -o -iname '*gemma-2-9b-it*' -o -iname '*Phi-3-mini-128k-instruct*' \) \
  2>/dev/null
```

### Train-only

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
  --output-root outputs/experiments/non_qwen_model_search_v2/trainonly \
  --log-level INFO
```

### n=50 closure

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
  --output-root outputs/experiments/non_qwen_model_search_v2/n50 \
  --log-level INFO
```

## 8. Recommended Execution Order

1. `Phi-3-mini-128k-instruct`
2. `Gemma-2-9B-IT`
3. `DeepSeek-V2-Lite-Chat` only if already runnable locally

## 9. Expected Outcome Table

| Model | Why selected | Expected best-case read | Stop/Go criterion |
| --- | --- | --- | --- |
| Phi-3-mini | cheapest new family | weak but possibly low-damage improvement | `drift_delta < -0.05` and `damage < 0.10` |
| Gemma-2-9B-IT | stronger non-Qwen family contrast | either clean negative or tradeoff-limited | same |
| DeepSeek-V2-Lite | attractive if locally ready | possible Chinese-family partial-positive | same |

## 10. Bottom Line

本轮默认不再优先追更多中文 7B，而是：

- 用 `Phi-3-mini` 做最快的新增 held-out screening
- 用 `Gemma-2-9B-IT` 做更强 family shift
- 把 `DeepSeek-V2-Lite` 保留为条件候选

如果两者都 negative，就如实停止。
