# 开发入口说明

本文档用于说明当前项目里 `src/analyzer` 与 `src/stats` 的“推荐入口 / 兼容入口”关系，方便后续开发时避免继续沿用语义含混的旧名字。

## 1. 总体原则

- `src/stats/`：放统计分析、报表、pair 级案例报告这类“统计产物导向”模块。
- `src/analyzer/`：放面向主流程编排时更高层、更贴近研究任务语义的分析入口。
- 旧文件名先保留为兼容层，避免现有脚本或历史命令立即失效。
- 新开发优先使用“推荐入口”，不要再新增对兼容层的依赖。

## 2. 当前推荐入口

### 2.1 统计分析主入口

推荐：
- `[src/stats/analyzer.py](/Users/shiqi/code/graduation-project/src/stats/analyzer.py)`

可读性更明确的 analyzer 侧别名：
- `[src/analyzer/stats_analyzer.py](/Users/shiqi/code/graduation-project/src/analyzer/stats_analyzer.py)`

主类：
- `StatsAnalyzer`

适用场景：
- 读取 judge 结果
- 生成 `final_report_*`
- 生成 `group_metrics_*`
- 生成 `analysis_summary_*`

### 2.2 Objective 主实验案例抽取入口

推荐：
- `[src/analyzer/objective_case_extractor.py](/Users/shiqi/code/graduation-project/src/analyzer/objective_case_extractor.py)`

主类：
- `ObjectiveCaseStudyExtractor`

兼容别名：
- `HighValueCaseExtractor`

适用场景：
- CMMLU objective 多臂实验
- 从 `inference_results_*.jsonl` 和 `judge_results*.jsonl` 中抽取
- `collapse cases`
- `resilient cases`

### 2.3 Pair 级案例报告入口

推荐：
- `[src/stats/pair_case_report.py](/Users/shiqi/code/graduation-project/src/stats/pair_case_report.py)`

核心对象：
- `SideRecord`
- `PairRecord`

适用场景：
- 原题 / 扰动题成对对比
- 基于 `delta` 的极端案例报告
- 旧式 subjective / pair-style case study

## 3. 兼容入口

这些文件现在仍然可用，但不建议新代码继续直接依赖：

### 3.1 Legacy `analyzer` wrapper

文件：
- `[src/analyzer/analyzer.py](/Users/shiqi/code/graduation-project/src/analyzer/analyzer.py)`

作用：
- 仅作为旧导入路径的兼容层
- 实际转发到 `src.analyzer.stats_analyzer`

### 3.2 Legacy objective case extractor wrapper

文件：
- `[src/analyzer/case_extractor.py](/Users/shiqi/code/graduation-project/src/analyzer/case_extractor.py)`

作用：
- 兼容旧的 `HighValueCaseExtractor` 导入
- 实际转发到 `src.analyzer.objective_case_extractor`

### 3.3 Legacy pair case extractor wrapper

文件：
- `[src/stats/case_extractor.py](/Users/shiqi/code/graduation-project/src/stats/case_extractor.py)`

作用：
- 兼容旧的 `case_extractor` 名字
- 实际转发到 `src.stats.pair_case_report`

## 4. 推荐导入方式

### 4.1 统计分析

推荐：

```python
from src.stats import StatsAnalyzer
```

也可以：

```python
from src.analyzer.stats_analyzer import StatsAnalyzer
```

不推荐新增：

```python
from src.analyzer.analyzer import StatsAnalyzer
```

### 4.2 Objective 案例抽取

推荐：

```python
from src.analyzer.objective_case_extractor import ObjectiveCaseStudyExtractor
```

兼容可用但不推荐新增：

```python
from src.analyzer.case_extractor import HighValueCaseExtractor
```

### 4.3 Pair 级案例报告

推荐：

```python
from src.stats.pair_case_report import PairRecord, load_side_records, generate_report
```

兼容可用但不推荐新增：

```python
from src.stats.case_extractor import PairRecord, load_side_records, generate_report
```

## 5. 当前脚本对应关系

- `[scripts/run_analysis.py](/Users/shiqi/code/graduation-project/scripts/run_analysis.py)` 使用 `StatsAnalyzer`
- `[scripts/run_full_pipeline.py](/Users/shiqi/code/graduation-project/scripts/run_full_pipeline.py)` 使用 `StatsAnalyzer` + `ObjectiveCaseStudyExtractor`

这意味着：
- `run_full_pipeline.py` 已经切到新的、更清楚的 objective 案例入口
- 主流程不再依赖语义模糊的 `src.analyzer.case_extractor`

## 6. 以后新增代码时的约束

- 如果是“统计汇总 / 显著性分析 / 报表导出”，优先放在 `src/stats/`
- 如果是“面向主实验流程的高层分析入口”，优先放在 `src/analyzer/`
- 不要再创建新的同名文件，如第二个 `case_extractor.py`、第二个 `analyzer.py`
- 优先使用能体现任务语义的名字，例如：
  - `objective_case_extractor.py`
  - `pair_case_report.py`
  - `stats_analyzer.py`

## 7. 一句话记忆

- `StatsAnalyzer` 是统计总分析器。
- `ObjectiveCaseStudyExtractor` 是 objective 主实验案例抽取器。
- `pair_case_report` 是旧式 pair-delta 案例报告工具。
- 旧名字还能跑，但新代码不要继续往旧名字上堆。
