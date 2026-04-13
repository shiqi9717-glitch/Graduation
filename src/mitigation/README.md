# Lightweight Interference Detector

这个模块是当前 objective 主评测链路之外的一个独立改进模块，目标是提供一个零重依赖、可本地运行的最小 interference detector 闭环。

当前推荐把它理解为：

- `risk ranker`
- `selective recheck trigger policy`

而不是一个高精度的单次强分类器。

## 数据来源

- 输入来自当前仓库已经跑完的 objective inference / judge 结果
- 同时兼容：
  - 旧 5-arm 设计：`t0 / t_placebo / t1 / t2 / t3`
  - 新 15-condition 设计：`ctrl_base / ctrl_text_placebo / ctrl_letter_placebo / a*_c*_w*`
- 主实验推荐默认只使用 `new15`

## 标签定义

- `strict_label`
  - 更保守，要求 baseline/control 稳定，且当前样本表现出更明显的 wrong-option follow / sycophancy 信号
- `relaxed_label`
  - 更宽松，适合早期探索

这两个标签都是自动构造的，不依赖人工标注。

## 三个主要命令

### 1. build-dataset

从已有 objective 结果构造 detector 数据集：

```bash
PROJECT_ROOT="$(pwd)"
MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig" \
./.venv/bin/python scripts/run_interference_detector.py \
  build-dataset \
  --design-version new15 \
  --output-dir outputs/experiments/interference_detector_mvp
```

### 2. train

训练 detector：

```bash
PROJECT_ROOT="$(pwd)"
MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig" \
./.venv/bin/python scripts/run_interference_detector.py \
  train \
  --dataset outputs/experiments/interference_detector_mvp/interference_strict_split.csv \
  --label-mode strict \
  --model-kind text \
  --recommended-threshold-policy matched_trigger_budget \
  --output-model outputs/experiments/interference_detector_mvp/text_strict.pkl
```

推荐的 `model-kind`：

- `text`
  - 主结果，最接近真实可部署 detector
- `structured-safe`
  - 轻量可解释 baseline，只允许当前样本即时可得的浅层特征
- `structured-oracle`
  - 上界参考，带信息优势，不应作为可部署 detector 主结果

### 3. score

对 detector 数据集离线打分：

```bash
PROJECT_ROOT="$(pwd)"
MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig" \
./.venv/bin/python scripts/run_interference_detector.py \
  score \
  --model-path outputs/experiments/interference_detector_mvp/text_strict.pkl \
  --dataset outputs/experiments/interference_detector_mvp/interference_strict_split.csv \
  --output-file outputs/experiments/interference_detector_mvp/text_strict_scored.csv
```

默认会优先使用模型 artifact 中保存的 `recommended_threshold`。

当前默认推荐阈值策略是 `matched_trigger_budget`，因为毕业论文阶段更适合把 detector 作为 `trigger policy` 来讲，而不是单纯追求 `best_f1`。

## Guard Eval

当前 guard eval 是：

- `offline selective re-check simulation`

也就是利用已有离线数据模拟：

- first answer
- detector score
- if high risk -> re-check proxy
- final answer

它用于验证 detector 是否有 `potential utility`，不是实时在线 guard 系统。

示例：

```bash
PROJECT_ROOT="$(pwd)"
MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig" \
./.venv/bin/python scripts/run_interference_detector.py \
  guard-eval \
  --dataset outputs/experiments/interference_detector_mvp/text_strict_scored.csv \
  --recheck-source baseline_answer
```

## 4. report

把一个实验目录收成论文可直接引用的 summary：

```bash
PROJECT_ROOT="$(pwd)"
MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig" \
./.venv/bin/python scripts/run_interference_detector.py \
  report \
  --experiment-dir outputs/experiments/interference_detector_mvp \
  --label-mode strict
```

这个 summary 会尽量自动发现同目录下的：

- dataset summary
- strict / relaxed dataset
- detector artifact
- scored dataset
- guard eval comparison

并统一输出一个 `interference_experiment_report_strict.json`，用于汇总：

- strict / relaxed 样本数
- 类别分布
- arm / model 分布
- detector dev/test 指标
- overall trigger rate
- high-pressure wrong-option trigger rate
- strict 正类 recall
- strict 负类误报率

这里的 `high-pressure wrong-option` 当前固定定义为：

- `explicit_wrong_option=1`
- `is_control=0`
- `authority_level>=1`
- `confidence_level>=1`

也就是优先对应最强施压的 wrong-option 条件。

## 输出文件说明

- `interference_samples_full.jsonl`
  - 完整 detector 样本
- `interference_strict_split.csv`
  - strict 标签样本
- `interference_relaxed_split.csv`
  - relaxed 标签样本
- `interference_dataset_summary.json`
  - 数据集规模、设计版本、模型来源、条件分布摘要
- `*.pkl`
  - detector artifact
- `*_scored.csv`
  - detector 打分结果，包含 `interference_score / predicted_label / trigger_recheck`
- `interference_experiment_report_*.json`
  - 用于论文和结果汇报的统一实验摘要

## 当前局限

- `structured-safe` 仍然只是轻量基线，不等价于真正的在线 guard 模块
- `guard-eval` 目前是离线模拟，不是实时在线二次调用系统
- `text` detector 目前是字符 n-gram Naive Bayes，能跑通但上限有限
- `report` 汇总的是当前目录下可发现的结果，不会替代真正的在线部署系统
