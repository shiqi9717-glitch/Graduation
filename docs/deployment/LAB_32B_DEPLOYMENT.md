# 实验室 32B 本地模型部署说明

这份说明用于把当前中文大模型阿谀奉承评估项目迁移到实验室本地 32B 机器上，并复现与 DeepSeek 主实验相同的 500 题、5 个 arm（`T0 / T_placebo / T1 / T2 / T3`）完整流程。

## 目标

在实验室本地 32B 模型上运行与当前主实验一致的 objective pipeline：

- 500 道题
- 5 个 arm：`T0 / T_placebo / T1 / T2 / T3`
- Monte Carlo `N=5`
- 输出 inference / judge / analysis 全流程结果

## 推荐迁移方式

推荐优先使用 **方案 A：直接复用已经生成好的 500 题 prompt 文件**。  
原因：

- 与当前 DeepSeek 主实验题集完全一致
- 不需要在实验室机器上重新生成 500 题
- 迁移体积很小，现有 prompt 文件约 1.39 MB
- 更容易做公平对比

---

## 方案 A：复用现有 500 题 prompt 文件（推荐）

### 需要带过去的内容

1. 项目整个工作区目录
2. 已生成好的 500 题 prompt 文件：
   - `outputs/experiments/main_study_deepseek_500/20260315_030954/perturbation/objective_cmmlu_prompts.jsonl`

### 运行逻辑

在实验室机器上：

- 不重新做 perturbation
- 直接把上面的 `objective_cmmlu_prompts.jsonl` 作为 inference 输入
- 让本地 32B 模型跑完整的 inference -> judge -> analysis

### 优点

- 与当前 DeepSeek 主实验完全同题集
- 结果最可比
- 最省时间

---

## 方案 B：重新生成 500 题

如果实验室机器上也有完整的 `third_party/CMMLU-master`，也可以重新生成 500 题。

### 需要带过去的内容

1. 项目整个工作区目录
2. `third_party/CMMLU-master/` 数据目录

### 运行逻辑

在实验室机器上重新做：

- perturbation
- inference
- judge
- analysis

### 缺点

- 500 题抽样会重新生成
- 与当前 DeepSeek 主实验不完全同题集
- 不适合做最严格的一一对比

因此，这个方案只建议在你无法方便复制 prompt 文件时使用。

---

## 实验室机器前置要求

1. Python 环境（建议 3.10+）
2. 能创建虚拟环境
3. 本地 32B 服务提供 **OpenAI-compatible API**
   - 例如：
     - `http://127.0.0.1:8000/v1`
     - 或实验室自定义兼容接口
4. 本地模型能够处理中文选择题

## Python 依赖

在项目目录下安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
```

## 最推荐运行方式

项目里已经准备好了一个通用脚本：

- `scripts/run_local_32b_main_study.sh`

这个脚本支持两种模式：

1. 使用已有 prompt 文件
2. 重新从 `third_party/CMMLU-master` 生成 500 题

---

## 推荐命令：直接复用现有 500 题 prompt 文件

```bash
ROOT_DIR=/path/to/workspace \
LOCAL_MODEL_NAME=your-local-32b-model \
LOCAL_API_BASE=http://127.0.0.1:8000/v1 \
LOCAL_API_KEY=EMPTY \
EXISTING_PROMPTS=/path/to/objective_cmmlu_prompts.jsonl \
./scripts/run_local_32b_main_study.sh
```

### 你需要替换的参数

- `ROOT_DIR`
- `LOCAL_MODEL_NAME`
- `LOCAL_API_BASE`
- `LOCAL_API_KEY`
- `EXISTING_PROMPTS`

---

## 备选命令：重新从 CMMLU 生成 500 题

```bash
ROOT_DIR=/path/to/workspace \
LOCAL_MODEL_NAME=your-local-32b-model \
LOCAL_API_BASE=http://127.0.0.1:8000/v1 \
LOCAL_API_KEY=EMPTY \
RAW_INPUT_DIR=/path/to/third_party/CMMLU-master \
./scripts/run_local_32b_main_study.sh
```

如果不传 `EXISTING_PROMPTS`，脚本就会自动走“重新生成 500 题”的路径。

---

## 运行后的结果

默认输出目录：

```bash
outputs/experiments/local_32b_main_study_500/
```

会生成完整的：

- inference
- judge
- analysis
- case study highlights

## 建议的实验室对比方式

为了和当前主实验最公平对比，建议优先比较：

1. `T0 Accuracy`
2. `T_placebo Sycophancy Rate`
3. `T3 Sycophancy Rate`
4. `T0 -> T3` 的 accuracy drop
5. `ATE_T3 >> ATE_placebo`
6. `Invalid Rate`

## 注意事项

1. 本地 32B 模型如果吞吐较低，可以把：
   - `BATCH_SIZE`
   - `CONCURRENCY`
   调小

2. 如果本地服务对并发敏感，可先尝试：

```bash
BATCH_SIZE=1
CONCURRENCY=1
```

3. 当前 objective pipeline 会统一：

- 使用 `T0 / T_placebo / T1 / T2 / T3`
- 使用 Monte Carlo 重复采样
- 使用 rule-based judge
- 输出 Accuracy / Sycophancy Rate / Invalid Rate

所以实验室结果会和现有主流程保持一致口径。
