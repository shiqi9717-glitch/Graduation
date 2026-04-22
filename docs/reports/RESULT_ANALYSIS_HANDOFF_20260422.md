# 结果分析交接摘要

更新时间：2026-04-22

本文档用于接续 `/Users/shiqi/code/graduation-project` 当前阶段的正式结果整理，并覆盖早于 2026-04-17 严格重建之前的 detector/recheck 口径。

## 1. 当前正式结果入口

当前最应优先引用的正式结果目录是：

- objective 主基线：
  `outputs/experiments/deepseek_200_n3_verified/20260330_201301`
- detector 主线：
  `outputs/experiments/interference_detector_new15_full_sentence_embedding`
- strict rebuilt full real recheck：
  `outputs/experiments/full_real_recheck_rebuilt/20260417_113758`

其中，`full_real_recheck_rebuilt/20260417_113758` 是在修复 merge key 不唯一导致的笛卡尔膨胀问题后，按下列严格口径重建的结果：

- 只认原始 scored dataset
- 只认 `sample_manifest.csv`
- 只认 `judge_recheck_results.jsonl`

早于这次严格重建的 full-data detector + real recheck 汇总结论，不再作为正式汇报主口径。

## 2. 当前正式汇报口径

当前最可信、最适合正式汇报的是：

- `baseline accuracy = 0.991804`
- 口径 B：全模型 detector + real recheck
  `accuracy = 0.995951`
- 口径 C：非 Reasoner detector + real recheck，Reasoner 不做 detector / 不做 recheck
  `accuracy = 0.996346`
- raw judge requests = `1831`
- strict recoverable rows = `1581`
- unmatched / orphan rows = `250`

据此，当前应固定的说法是：

1. 旧的“full-data real recheck 低于 baseline”结论应撤回。
2. 当前正式主口径应优先采用口径 C。
3. `deepseek-reasoner` 的 same-model real recheck 不适合进入正式方案。

## 3. detector 的稳定表述

detector 主线仍固定使用：

- `sentence-embedding-logreg`
- `strict`
- `matched_trigger_budget`

稳定表述保持不变：

- detector 更适合作为 `risk ranker / selective recheck trigger policy`
- 不应包装成高精度单次强分类器

这部分仍可优先引用：

- [`detector_model_comparison.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/detector_model_comparison.csv)
- [`threshold_sweep_sentence-embedding-logreg_strict.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/threshold_sweep_sentence-embedding-logreg_strict.csv)
- [`guard_eval_summary_sentence_embedding_new15_full.json`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/guard_eval_summary_sentence_embedding_new15_full.json)

## 4. 与历史 same-model 文档的关系

以下目录和文档仍可保留，但应仅作为历史分析或风险案例参考：

- `outputs/experiments/same_model_guarded_pilot/20260412_021656`
- `outputs/experiments/same_model_full_real/...`
- [`docs/reports/RECHECK_CASE_ANALYSIS_20260416.md`](/Users/shiqi/code/graduation-project/docs/reports/RECHECK_CASE_ANALYSIS_20260416.md)
- [`docs/reports/RESULT_ANALYSIS_HANDOFF_20260413.md`](/Users/shiqi/code/graduation-project/docs/reports/RESULT_ANALYSIS_HANDOFF_20260413.md)

原因不是这些材料没有信息量，而是：

- 其中部分结论形成于严格重建之前
- same-model self-recheck 尤其是 Reasoner lane，不再适合作为当前正式方案
- 这些材料更适合支撑“失败模式 / 风险案例 / 题目级系统性误判”分析，而不是主结果结论

## 5. 当前建议的文档引用顺序

1. [`docs/reports/RESULT_ANALYSIS_HANDOFF_20260422.md`](/Users/shiqi/code/graduation-project/docs/reports/RESULT_ANALYSIS_HANDOFF_20260422.md)
2. [`docs/PAPER_ANALYSIS_ENTRYPOINT.md`](/Users/shiqi/code/graduation-project/docs/PAPER_ANALYSIS_ENTRYPOINT.md)
3. [`docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`](/Users/shiqi/code/graduation-project/docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md)
4. detector 主线结果目录
5. strict rebuilt full real recheck 目录

## 6. 后续整理建议

- 后续报告、汇报稿、答辩材料如涉及 detector + real recheck，请默认先核对是否已切换到 `full_real_recheck_rebuilt/20260417_113758`
- 如果文档仍引用 same-model self-recheck 作为“当前最可信正式方案”，应改为历史参考口径
- 如需补充案例分析，优先写“为什么口径 C 更稳”与“Reasoner 为何不进入正式方案”，而不是继续放大 same-model 成功案例
