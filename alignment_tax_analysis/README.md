# 阿谀奉承特质量化评估系统 (Alignment Tax Analysis)

## 项目目标
本项目用于量化评估大语言模型的阿谀奉承（Sycophancy）倾向，提供完整评估闭环：

1. 数据扰动（生成原题/扰动题对照）
2. 被测模型推理（生成回答）
3. AI 裁判打分（LLM-as-a-Judge）
4. 统计分析与出表（可直接用于论文）

## 当前阶段
- Phase 1：推理模块（已完成）
- Phase 2：裁判打分模块（已完成）
- Phase 3：主控串联流水线（已完成）
- Phase 4：统计分析与报表模块（已完成）

## 目录结构
```text
alignment_tax_analysis/
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── src/
│   ├── data_pipeline.py
│   ├── inference/
│   ├── judge/
│   └── stats/
├── scripts/
│   ├── run_data_pipeline.py
│   ├── run_inference.py
│   ├── run_judge_pipeline.py
│   ├── run_analysis.py
│   └── run_full_pipeline.py
└── README.md
```

## 环境准备
1. 创建虚拟环境并安装依赖
```bash
pip install -r requirements.txt
```

2. 配置 API Key（复制 `.env.example` 到 `.env`）
```env
GLOBAL_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

如果你用 OpenAI/Anthropic/Qwen 等推理，也请同时配置对应的 key。

## 运行方式总览

### 1) 一键全流程（推荐）
```bash
python scripts/run_full_pipeline.py ^
  --raw-input-file ../sycophancy_database/sycophancy_on_political_typology_quiz.jsonl ^
  --model deepseek-chat ^
  --provider deepseek ^
  --batch-size 10 ^
  --concurrency 5 ^
  --output-root data/results/full_pipeline
```

说明：
- 自动串联：扰动 -> 推理 -> 裁判 -> 统计
- 输出在 `data/results/full_pipeline/<timestamp>/`

### 2) 分阶段运行

数据扰动：
```bash
python scripts/run_data_pipeline.py
```

推理：
```bash
python scripts/run_inference.py ^
  --input-file data/processed/sycophancy_ready_to_test.csv ^
  --input-format csv ^
  --question-column question ^
  --model deepseek-chat ^
  --provider deepseek ^
  --batch-size 10 ^
  --concurrency 5 ^
  --output-format all
```

裁判打分：
```bash
python scripts/run_judge_pipeline.py ^
  --input-file data/results/inference/<your_inference_results>.jsonl ^
  --input-format jsonl ^
  --batch-size 10 ^
  --concurrency 5 ^
  --output-format all
```

统计分析：
```bash
python scripts/run_analysis.py ^
  --input-file data/results/judge/<your_judge_results>.jsonl ^
  --input-format jsonl ^
  --output-dir data/results/analysis
```

## 小批量快速跑通（最低成本）
建议先用 10 条数据验证链路，确认无误后再放大规模。

### 方案 A：一键链路 + 跳过扰动（最快）
先准备一个小推理输入文件（CSV/JSONL，约 10 条），然后：
```bash
python scripts/run_full_pipeline.py ^
  --skip-perturbation ^
  --inference-input-file data/processed/sycophancy_ready_to_test.csv ^
  --model deepseek-chat ^
  --provider deepseek ^
  --batch-size 5 ^
  --concurrency 2
```

### 方案 B：已有推理结果，直接评测+出表
```bash
python scripts/run_full_pipeline.py ^
  --skip-perturbation ^
  --skip-inference ^
  --inference-results-file data/results/inference/<your_inference_results>.jsonl ^
  --batch-size 5 ^
  --concurrency 2
```

### 方案 C：已有裁判结果，直接统计出表
```bash
python scripts/run_full_pipeline.py ^
  --skip-perturbation ^
  --skip-inference ^
  --skip-judge ^
  --judge-results-file data/results/judge/<your_judge_results>.jsonl
```

## 统计报表说明
`run_analysis.py` 和 `run_full_pipeline.py` 最终会输出：

- `final_report_*.csv`：按模型汇总的主报表
- `group_metrics_*.csv`：按 `model_name + question_type` 分组指标
- `delta_metrics_*.csv`：`original vs perturbed` 的绝对差值统计（可计算时）
- `analysis_summary_*.json`：分析过程与文件索引

核心指标包括：
- 均值（Mean）
- 方差（Variance）
- 标准差（Std）
- 分组样本数（n）
- 扰动差值绝对值（Delta abs）

## 常见问题
1. `GLOBAL_API_KEY 未设置`  
请检查 `.env` 是否在项目根目录，且 key 是否有效。

2. JSONL/CSV 字段不匹配  
推理输入至少要能提取问题文本（如 `question` 列）；裁判输入要能提取 `question` 和 `answer/response_text`。

3. 如何确认跑通  
先看命令行最终 summary，再检查 `data/results/...` 下是否生成了 `judge_results_*` 和 `final_report_*`。
