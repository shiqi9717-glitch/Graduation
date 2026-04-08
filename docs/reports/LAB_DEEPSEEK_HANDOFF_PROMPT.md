# 给实验室机器上 DeepSeek coding 的交接话术

把下面这段话直接发给实验室机器上的 DeepSeek coding 即可：

```markdown
请帮我在这台实验室本地 32B 模型机器上部署并运行一个中文大模型阿谀奉承评估项目。目标不是重新开发，而是尽量复现已有流程，并在本地 32B 模型上跑一遍与主实验一致的 500 题 objective pipeline。

## 目标

请运行一套完整的 objective 实验流程，要求：

- 题量：500
- arms：`T0 / T_placebo / T1 / T2 / T3`
- Monte Carlo：`N=5`
- 完整流程：`inference -> judge -> analysis`

## 优先方案

优先直接复用已经生成好的 500 题 prompt 文件，而不是重新从 `third_party/CMMLU-master` 抽样。因为这样可以和现有 DeepSeek 主实验保持完全同题集，结果更可比。

如果我已经把下面这个文件复制到你所在机器，请直接使用它作为 inference 输入：

- `objective_cmmlu_prompts.jsonl`

也就是说，请使用：

- `--skip-perturbation`
- `--inference-input-file <objective_cmmlu_prompts.jsonl>`

## 项目结构

你会看到一个项目工作区目录。里面已经有：

- 主脚本：`scripts/run_full_pipeline.py`
- 通用本地运行脚本：`scripts/run_local_32b_main_study.sh`
- requirements：`config/requirements.txt`

## 你需要做的事情

1. 安装 Python 依赖
2. 确认本地 32B 服务的 OpenAI-compatible API 地址
3. 把本地模型名、API base、API key（如果需要）填到运行命令里
4. 优先按“复用现有 500 题 prompt 文件”的方式运行
5. 如果运行过程中发现本地服务对并发敏感，请主动把并发调低，例如：
   - `BATCH_SIZE=1`
   - `CONCURRENCY=1`

## 推荐运行方式

优先尝试：

```bash
ROOT_DIR=/path/to/workspace \
LOCAL_MODEL_NAME=your-local-32b-model \
LOCAL_API_BASE=http://127.0.0.1:8000/v1 \
LOCAL_API_KEY=EMPTY \
EXISTING_PROMPTS=/path/to/objective_cmmlu_prompts.jsonl \
./scripts/run_local_32b_main_study.sh
```

## 输出目标

请把结果完整输出到一个新的结果目录中，并告诉我：

1. `T0 Accuracy`
2. `T_placebo Sycophancy Rate`
3. `T3 Sycophancy Rate`
4. `Invalid Rate`
5. 是否出现明显的 `T0 -> T3` accuracy drop
6. 如果 pipeline 失败，请告诉我是：
   - API 接口问题
   - 模型输出格式问题
   - judge 提取问题
   - 并发/吞吐问题

## 重要要求

- 不要修改原有历史结果
- 尽量少改动代码
- 先以“跑通现有流程”为目标
- 如果必须修改代码，请优先做最小改动，并说明原因
```
