# 论文分析统一入口

更新时间：2026-04-13

本文档用于统一新工作区 `/Users/shiqi/code/graduation-project` 下的论文分析入口。

旧工作区 `/Users/shiqi/code/代码/毕业代码` 仅作为迁移背景保留说明，不再作为后续开发或查找文件的正式入口。

## 1. 主版本结论

- 已核对旧工作区与新工作区的 `docs/papers/literature_summary.md`，当前两者内容一致。
- 因此，后续论文总表的唯一主版本统一为：`docs/papers/literature_summary.md`
- 不再继续以旧口径的 `papers/literature_summary.md` 作为主入口；本项目当前实际保存 20 篇核心论文 PDF 的目录是：`docs/papers/`

## 2. 推荐阅读顺序

1. `docs/PAPER_ANALYSIS_ENTRYPOINT.md`
2. `docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`
3. `docs/papers/literature_summary.md`
4. `docs/PROJECT_STATUS.md`
5. `src/mitigation/README.md`

## 3. 当前研究主张

当前更稳妥的项目判断是：

- 当前 interference detector 更像 risk ranker / trigger policy，不像高置信强分类器。
- 当前瓶颈更像“过程可观测性不足”，而不是“模型结构天然无解”。
- 比起继续追求单次 final-answer 分类，下一阶段更值得转向 `process proxy / measurement / audit + recheck`。

详细展开见：`docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`

## 4. 文献组织新主线

后续所有论文分析，统一按以下四条线组织：

1. `measurement / label design`
2. `process monitoring / proxy signals`
3. `audit + recheck / oversight`
4. `causal / mechanistic mitigation`

对应综述重组已写入：

- `docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`
- `docs/papers/literature_summary.md`

## 5. 与当前项目代码和结果的直接锚点

优先引用以下文件，不要脱离当前仓库现状空谈：

- 数据扰动与条件设计：`src/data/local_data_perturber.py`
- detector / re-check 模块说明：`src/mitigation/README.md`
- detector 主线正式目录：`outputs/experiments/interference_detector_new15_full_sentence_embedding`
- detector 主线摘要：`outputs/experiments/interference_detector_new15_full_sentence_embedding/guard_eval_summary_sentence_embedding_new15_full.json`
- detector 主线阈值扫表：`outputs/experiments/interference_detector_new15_full_sentence_embedding/threshold_sweep_sentence-embedding-logreg_strict.csv`
- detector grid 对照：`outputs/experiments/interference_detector_new15_full_detector_grid`
- 固定模型真实 guarded pilot：`outputs/experiments/deepseek_guarded_pilot/20260412_011945`
- same-model self-recheck pilot：`outputs/experiments/same_model_guarded_pilot/20260412_021656`

这里建议后续写作时始终分开三类证据：

- `offline proxy simulation`
- `fixed-model real pilot`
- `same-model real pilot`

## 6. 下一阶段最值得推进的切口

优先做：

1. 把 `new15` 条件从“final-answer 行为检测”推进到“过程代理信号测量框架”
2. 为 detector 数据集补充 `baseline -> arm -> recheck` 的多阶段观测字段
3. 把当前 guard-eval 从“离线选择性重查模拟”扩展成更明确的 audit / recheck 研究接口

一句话概括：

> 下一阶段不应把 detector 包装成“已经能稳定识别干扰的分类器”，而应把它定义为“用于触发复查和分配审计预算的风险排序器”。
