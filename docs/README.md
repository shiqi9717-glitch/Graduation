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
project-root/
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
├── outputs/
│   ├── experiments/
│   ├── logs/
│   ├── smoke/
│   └── archives/
├── scripts/
├── src/
├── tests/
└── third_party/
```

## 开发维护说明
- 分析模块入口说明：`docs/ENTRYPOINT_GUIDE.md`
- 文件查找规则说明：`docs/WORKSPACE_LOOKUP_RULES.md`

## 论文分析入口
- 统一入口：`docs/PAPER_ANALYSIS_ENTRYPOINT.md`
- 论文总表主版本：`docs/papers/literature_summary.md`
- 最新研究 memo：`docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`

## 环境准备
1. 创建虚拟环境并安装依赖
```bash
pip install -r config/requirements.txt
```

2. 配置 API Key（复制 `config/.env.example` 到 `config/.env`）
```env
GLOBAL_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

如果你用 OpenAI/Anthropic/Qwen 等推理，也请同时配置对应的 key。

## AI 协作规则

这个仓库在 macOS 上有一个重要限制：

- 任何依赖 `MPS` 的命令或代码路径，都不要由 AI 同事在 Codex 内直接运行。
- AI 同事应该先写好一条可直接复制的终端命令，再由用户在普通 macOS Terminal 中执行。
- 用户把终端输出贴回后，AI 同事再继续分析、改代码或给下一步命令。
- 与 `MPS` 无关的任务，AI 同事可以继续在 Codex 内正常执行。

必须交给用户在 Terminal 里执行的典型任务：

- `torch.backends.mps.is_available()` 检查
- `device="mps"` 的本地模型运行
- Apple GPU 上的本地 white-box probe / hidden-state 提取
- 一切依赖 `MPS` 成功与否的 smoke test 或正式实验

可以继续由 AI 同事在 Codex 内执行的任务：

- CPU fallback 路径验证
- 代码修改
- 与 `MPS` 无关的测试
- 数据清理、结果重建、统计分析、报表生成
- 日志检查、流程联调、分支/恢复逻辑验证

如果 AI 同事不确定某个任务是否涉及 `MPS`，默认按下面流程处理：

1. 不在 Codex 内直接运行
2. 先给用户一条可复制的终端命令
3. 等用户返回终端输出后再继续

## 多部门 AI 协作工作流规范

本项目采用多部门 AI 协作机制。当前工作区默认由以下六个执行部门负责，用户负责调度；此外允许存在一个只读长上下文辅助角色，但它不属于执行部门：

1. 架构分析部门（Architecture）
2. 代码执行/修改部门（Code）
3. 结果分析部门（Analysis）
4. 创新改进部门（Innovation）
5. 文件整理部门（Documentation）
6. 论文撰写部门（Paper Writing）
7. 用户（调度中枢）
8. 只读长上下文辅助角色（optional, read-only）

核心原则：

- 所有工作流都必须由用户在各部门之间传达。
- AI 部门之间不允许直接通信。
- 所有步骤都应可追踪、可复现、职责清晰。
- 任何部门都应明确自己的输入、输出、下一步归属。

### 0.5 只读长上下文辅助

本项目允许存在一个额外的只读长上下文辅助角色，例如 DeepSeek 之类的长上下文只读助手，用于帮助用户维护长期上下文、一致性检查和 backlog 跟踪。

该角色的定位必须严格限定为：

- 只读，不修改文件，不运行会写入结果的命令
- 不属于六个执行部门
- 不直接替代任何部门做方法设计、代码修改、结果计算、文件归档或论文写作
- 不直接向其他 AI 部门发号施令，所有信息仍必须通过用户转发
- 只负责长期上下文、遗漏检查和一致性审计，不负责执行性产出

该角色可以做的事：

- 维护跨轮次任务清单，提醒哪些要求已完成、哪些可能被遗漏
- 检查文档、结果目录、实验结论之间是否一致
- 在用户准备把任务交给某个部门前，帮助压缩为更小的任务包

一句话原则：

- 它是 `read-only coordinator / memory auditor`，不是新的执行部门

### 1. 部门职责

架构分析部门（Architecture）负责：

- 方法设计
- pipeline 架构
- 实验结构定义

不负责：

- 跑代码
- 修改参数

代码执行/修改部门（Code）负责：

- 运行实验
- 修复 bug
- 输出结果路径

必须提供：

- 是否成功运行
- 输出目录路径
- 核心运行信息

结果分析部门（Analysis）负责：

- 指标计算（准确率、迎合率等）
- 结果对比
- 图表生成

禁止：

- 修改代码

创新改进部门（Innovation）负责：

- 提出优化方案
- 分析 failure case
- 设计改进方向

文件整理部门（Documentation）负责：

- 整理 outputs
- 维护 README / 报告
- 归档实验结果

论文撰写部门（Paper Writing）负责：

- 论文正文写作
- 摘要、引言、方法、实验、结果、讨论、局限性等成文
- 主文/附录表述润色
- 投稿版本语言统一

不负责：

- 方法设计
- 跑代码或修 bug
- 指标计算和结果复核
- 文件归档和 README 维护
- 提出新实验或新改进方案

职责边界：

- 只有当任务进入“论文文本如何写、如何组织、如何表述”的阶段，才交给论文撰写部门。
- 架构、代码、结果分析、创新改进、文件整理这五个部门能完成的工作，不交给论文撰写部门。

用户（调度中枢）负责：

- 在各部门之间转发信息
- 控制工作流推进

### 2. 核心工作流

所有流程遵循：

`用户 -> 部门A -> 输出 -> 用户 -> 部门B -> ...`

强制规则：

- AI 部门之间不允许直接通信。
- 所有信息必须通过用户转发。
- 不允许跳过用户直接把任务交给下一个部门。
- 只读长上下文辅助角色如存在，也只能通过用户接收和返回信息。

### 3. 统一输出协议

所有部门输出必须严格遵循以下结构：

1. 当前结论（必写）
一句话说明本部门完成了什么。

2. 关键发现（可选，最多 3 条）
- 只写最重要的信息。
- 不允许长篇解释。

3. 下一步建议（必写）
- 明确指出下一个应该由哪个部门处理。

4. 给用户的转发 Prompt（必写）
- 必须同时写明来源部门和目标部门。
- 必须使用 `【FROM: 部门名称】` + `【TO: 部门名称】` 的格式。
- 例如：`【FROM: 代码执行/修改部门】` + `【TO: 结果分析部门】`。
- 必须写清楚输入是什么。
- 必须写清楚输出要求是什么。

### 3.5 每轮输出的强制补充字段

无论哪个执行部门输出，除上述结构外，都应尽量明确写出以下四项，方便用户同步给只读长上下文辅助角色做长期状态维护：

- 本轮完成项：这一次实际完成了什么
- 未完成项：哪些事项尚未完成，或明确没有处理
- 结果路径：新增或确认使用的结果路径 / 文档路径 / 目录路径
- 废弃口径：哪些旧结论、旧目录、旧写法、旧结果口径不应继续引用

如果某一项本轮不存在，也应明确写 `无` 或 `本轮未涉及`，避免后续审计时误判为遗漏。

标准模板如下：

```text
1. 当前结论
<一句话说明本部门完成了什么>

2. 关键发现
- <重要信息 1>
- <重要信息 2>
- <重要信息 3>

3. 下一步建议
<明确指出下一个应该由哪个部门处理>

4. 给用户的转发 Prompt
【FROM: 当前部门名称】
【TO: 部门名称】

<写给下一个部门的指令>

要求：
1. ...
2. ...
3. ...
```

### 4. 强制执行规则

1. 每个部门必须输出转发 Prompt。
2. Prompt 必须明确 `FROM`、`TO`、输入内容、输出要求。
3. 不允许跳过用户直接通信。
4. 输出必须简洁，不允许冗长解释。
5. 每一步必须可复现，尽量给出路径、参数、结果目录。

### 5. 示例

代码执行/修改部门（Code）示例：

```text
1. 当前结论
已完成 Qwen2.5-7B 实验运行。

2. 关键发现
- 成功运行 100 samples
- 无报错

3. 下一步建议
需要结果分析部门处理。

4. 给用户的转发 Prompt
【FROM: 代码执行/修改部门】
【TO: 结果分析部门】

请分析以下实验结果：

路径：outputs/experiments/.../latest

要求：
1. 计算迎合率与准确率
2. 对比 baseline 与 intervention
3. 输出核心结论（<=5条）
```

结果分析部门（Analysis）示例：

```text
1. 当前结论
已完成结果统计分析。

2. 关键发现
- 迎合率下降 23%
- strict_positive 效果最佳

3. 下一步建议
需要创新改进部门提出优化方案。

4. 给用户的转发 Prompt
【FROM: 结果分析部门】
【TO: 创新改进部门】

基于以下问题提出改进方案：

- high_pressure 表现不稳定
- late-layer collapse 明显

要求：
1. 提供 2-3 个改进方向
2. 给出可实现方案
```

### 6. 总结

本工作流的固定原则是：

- AI 负责执行
- 用户负责调度
- 所有通信必须结构化

最终目标是让复杂实验流程变成可控、可追踪、可复现的协作系统。

## 运行方式总览

### 1) 一键全流程（推荐）
```bash
python scripts/run_full_pipeline.py ^
  --raw-input-file data/external/sycophancy_database/sycophancy_on_political_typology_quiz.jsonl ^
  --model deepseek-chat ^
  --provider deepseek ^
  --batch-size 10 ^
  --concurrency 5 ^
  --output-root outputs/experiments/full_pipeline
```

说明：
- 自动串联：扰动 -> 推理 -> 裁判 -> 统计
- 输出在 `outputs/experiments/full_pipeline/<timestamp>/`

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
  --input-file outputs/experiments/inference/<your_inference_results>.jsonl ^
  --input-format jsonl ^
  --batch-size 10 ^
  --concurrency 5 ^
  --output-format all
```

统计分析：
```bash
python scripts/run_analysis.py ^
  --input-file outputs/experiments/judge/<your_judge_results>.jsonl ^
  --input-format jsonl ^
  --output-dir outputs/experiments/analysis
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
  --inference-results-file outputs/experiments/inference/<your_inference_results>.jsonl ^
  --batch-size 5 ^
  --concurrency 2
```

### 方案 C：已有裁判结果，直接统计出表
```bash
python scripts/run_full_pipeline.py ^
  --skip-perturbation ^
  --skip-inference ^
  --skip-judge ^
  --judge-results-file outputs/experiments/judge/<your_judge_results>.jsonl
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
请检查 `config/.env` 是否存在，且 key 是否有效。

2. JSONL/CSV 字段不匹配  
推理输入至少要能提取问题文本（如 `question` 列）；裁判输入要能提取 `question` 和 `answer/response_text`。

3. 如何确认跑通  
先看命令行最终 summary，再检查 `outputs/experiments/...` 下是否生成了 `judge_results_*` 和 `final_report_*`。
