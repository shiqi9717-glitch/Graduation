# 工作区文件查找规则

这份说明用于统一结果分析部门、代码部门、架构分析部门在当前工作区里的找文件方式。

## 一句话规则

- 看代码：去 `src/`
- 看运行脚本：去 `scripts/`
- 看配置：去 `config/`
- 看测试：去 `tests/`
- 看原始/处理后数据：去 `data/`
- 看实验结果：去 `outputs/experiments/`
- 看运行日志：去 `outputs/logs/`
- 看 smoke 结果：去 `outputs/smoke/`
- 看文档和阶段说明：去 `docs/`
- 看第三方基准：去 `third_party/`

## 当前目录职责

```text
project-root/
├── config/         # 配置、依赖、.env 样例
├── data/           # 原始数据、处理后数据、样例数据、外部数据
├── docs/           # 项目说明、状态文档、部署文档、论文资料
├── outputs/        # 所有运行产物
│   ├── experiments/
│   ├── logs/
│   ├── smoke/
│   └── archives/
├── scripts/        # 运行入口、批处理脚本、实验脚本
├── src/            # 核心源码
├── tests/          # 单测、集成测试、手工验证脚本
└── third_party/    # 第三方基准仓库
```

## 查文件的新规则

### 1. 查代码实现

- 业务代码统一从 `src/` 开始查
- 新增 mitigation / detector 相关代码在 `src/mitigation/`
- 分析和统计优先看 `src/stats/`、`src/analyzer/`

### 2. 查实验结果

- 所有正式实验结果、离线 detector 产物、guarded pilot 结果，统一优先到 `outputs/experiments/` 下查
- 不再默认从 `outputs/` 根目录直接找实验目录
- 新生成的实验目录也应该尽量挂到 `outputs/experiments/<experiment_name>/`

### 3. 查日志

- 运行日志统一看 `outputs/logs/`
- 不再把根目录 `logs/` 当成正式日志入口

### 4. 查数据

- 原始数据：`data/raw/`
- 处理后数据：`data/processed/`
- 小样本和调试样例：`data/samples/`
- 外部数据集：`data/external/`
- 第三方 benchmark 仓库：`third_party/CMMLU-master/`

### 5. 查文档

- 项目主说明：`docs/README.md`
- 阶段状态：`docs/PROJECT_STATUS.md`
- 分析入口说明：`docs/ENTRYPOINT_GUIDE.md`
- 论文与文献：`docs/papers/`

## 不作为正式查找入口的内容

下面这些内容属于本地运行或编辑器环境，不应作为“正式找文件路径”：

- `.venv/`
- `.pytest_cache/`
- `.vscode/`
- `.mplconfig/` 里的缓存文件
- 根目录临时 `tmp_*`

## 对三个部门的统一要求

- 结果分析部门：后续查结果先从 `outputs/experiments/` 开始，不要再默认扫 `outputs/` 根目录
- 代码部门：新增脚本默认输出路径优先写到 `outputs/experiments/...` 或 `outputs/logs/...`
- 架构分析部门：以后画目录结构图、数据流图、模块关系图时，以本文件定义的目录职责为准
