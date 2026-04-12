# Lightweight Interference Detector

这个模块是当前 objective 主评测链路之外的一个独立改进模块，目标是提供一个零重依赖、可本地运行的最小 interference detector 闭环。

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
MPLCONFIGDIR=/Users/shiqi/code/代码/毕业代码/.mplconfig \
./.venv/bin/python scripts/run_interference_detector.py \
  build-dataset \
  --design-version new15 \
  --output-dir outputs/experiments/interference_detector_mvp
```

### 2. train

训练 detector：

```bash
MPLCONFIGDIR=/Users/shiqi/code/代码/毕业代码/.mplconfig \
./.venv/bin/python scripts/run_interference_detector.py \
  train \
  --dataset outputs/experiments/interference_detector_mvp/interference_strict_split.csv \
  --label-mode strict \
  --model-kind text \
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
MPLCONFIGDIR=/Users/shiqi/code/代码/毕业代码/.mplconfig \
./.venv/bin/python scripts/run_interference_detector.py \
  score \
  --model-path outputs/experiments/interference_detector_mvp/text_strict.pkl \
  --dataset outputs/experiments/interference_detector_mvp/interference_strict_split.csv \
  --output-file outputs/experiments/interference_detector_mvp/text_strict_scored.csv
```

默认会优先使用模型 artifact 中保存的 `recommended_threshold`。

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
MPLCONFIGDIR=/Users/shiqi/code/代码/毕业代码/.mplconfig \
./.venv/bin/python scripts/run_interference_detector.py \
  guard-eval \
  --dataset outputs/experiments/interference_detector_mvp/text_strict_scored.csv \
  --recheck-source baseline_answer
```

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

## 当前局限

- `structured-safe` 仍然只是轻量基线，不等价于真正的在线 guard 模块
- `guard-eval` 目前是离线模拟，不是实时在线二次调用系统
- `text` detector 目前是字符 n-gram Naive Bayes，能跑通但上限有限
