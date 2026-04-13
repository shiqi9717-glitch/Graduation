# 结果分析交接摘要

更新时间：2026-04-13

本文档用于在新工作区 `/Users/shiqi/code/graduation-project` 下继续承接当前正式结果整理，并统一结果分析部门的主线口径。

## 1. 已核对的正式结果目录

以下目录均已确认位于 `outputs/experiments/` 下，可作为正式入口继续分析：

1. objective 主基线：
   `outputs/experiments/deepseek_200_n3_verified/20260330_201301`
2. cross-model objective：
   `outputs/experiments/cross_model_200_reuse_deepseek_chat/20260331_021630`
3. detector 主线：
   `outputs/experiments/interference_detector_new15_full_sentence_embedding`
4. detector grid 对照：
   `outputs/experiments/interference_detector_new15_full_detector_grid`
5. 固定模型真实 guarded pilot：
   `outputs/experiments/deepseek_guarded_pilot/20260412_011945`
6. same-model self-recheck pilot：
   `outputs/experiments/same_model_guarded_pilot/20260412_021656`

## 2. 当前正式分析口径

- detector 主线固定使用 `sentence-embedding-logreg` + `strict`
- 主 operating point 固定使用 `matched_trigger_budget`
- `relaxed` 和 `hybrid` 目前不作为主线
- `broader645` 扩标签实验已撤回，不作为正式基线
- guarded eval 仍需明确区分：
  - `offline proxy simulation`
  - `real recheck pilot`

其中，detector 主线目录内的 [`CURRENT_BASELINE_NOTE.md`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/CURRENT_BASELINE_NOTE.md) 已明确说明：`broader-645` 临时扩标签实验已撤回。

## 3. detector 主线整理

正式目录：
`outputs/experiments/interference_detector_new15_full_sentence_embedding`

应优先引用的正式产物：

- [`detector_model_comparison.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/detector_model_comparison.csv)
- [`threshold_sweep_sentence-embedding-logreg_strict.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/threshold_sweep_sentence-embedding-logreg_strict.csv)
- [`guard_eval_comparison_sentence_embedding_new15_full.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/guard_eval_comparison_sentence_embedding_new15_full.csv)
- [`guard_eval_summary_sentence_embedding_new15_full.json`](/Users/shiqi/code/graduation-project/outputs/experiments/interference_detector_new15_full_sentence_embedding/guard_eval_summary_sentence_embedding_new15_full.json)

### 3.1 detector 本体能力

在 `strict` 标签下，`sentence-embedding-logreg` 的 `matched_trigger_budget` 对应：

- threshold = `0.55`
- test precision = `0.0672`
- test recall = `0.4706`
- test F1 = `0.1176`
- ROC-AUC = `0.6770`
- detector trigger rate = `0.0881`

这组结果支持当前更稳妥的说法：

- 该 detector 更适合作为 `risk ranker / trigger policy`
- 暂不应包装成高精度的单次强分类器

### 3.2 offline selective recheck simulation

`guard_eval_summary_sentence_embedding_new15_full.json` 的模式为：
`offline_selective_recheck_simulation`

在主线 operating point `matched_trigger_budget` 下：

- threshold = `0.55`
- raw accuracy = `0.9918`
- guarded accuracy = `0.9982`
- raw wrong-option-follow rate = `0.00820`
- guarded wrong-option-follow rate = `0.00181`
- trigger rate = `0.0603`
- strict positive recall = `0.7791`
- strict negative false positive rate = `0.0543`

当前可以稳定沿用的结论是：

- 在离线代理模拟里，主线 detector 的确能用较低追加调用预算换来明显的错误跟随率下降
- 但 detector 自身 precision 很低，因此主张应落在“预算分配和复查触发”而不是“精确识别”

## 4. same-model guarded pilot 整理

正式目录：
`outputs/experiments/same_model_guarded_pilot/20260412_021656`

应优先引用的正式产物：

- [`guarded_eval_summary.json`](/Users/shiqi/code/graduation-project/outputs/experiments/same_model_guarded_pilot/20260412_021656/guarded_eval_summary.json)
- [`guarded_eval_by_group.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/same_model_guarded_pilot/20260412_021656/guarded_eval_by_group.csv)
- [`sample_manifest.csv`](/Users/shiqi/code/graduation-project/outputs/experiments/same_model_guarded_pilot/20260412_021656/sample_manifest.csv)

该目录的模式为：
`real_same_model_guarded_recheck_pilot`

在主线 operating point `matched_trigger_budget` 下：

- num samples = `120`
- triggered samples = `81`
- trigger rate = `0.675`
- raw accuracy = `0.6750`
- guarded accuracy = `0.7917`
- raw wrong-option-follow rate = `0.3250`
- guarded wrong-option-follow rate = `0.1667`
- recheck changed answer rate = `0.4815`
- recheck changed to correct rate = `0.3210`
- recheck changed to wrong rate = `0.1605`

当前可直接采用的解释：

- 真实 same-model recheck pilot 显示，guarded recheck 不是纯粹的“保守兜底”，它会带来真实的答案改写
- 改写总体净收益为正，但同时存在把正确答案改坏的风险
- 因此真实 pilot 的叙述必须与 offline proxy simulation 分开，不能混写成同一类证据

### 4.1 分组观察

`guarded_eval_by_group.csv` 里值得优先跟踪的现象：

- `strict_positive` 组从 `0.0000` 提升到 `0.6667`
- `hard_negative` 组从 `1.0000` 降到 `0.8481`
- `deepseek-chat` 从 `0.4091` 提升到 `0.9091`
- `deepseek-reasoner` 从 `1.0000` 降到 `0.5000`
- `qwen-max` 从 `0.7647` 提升到 `0.9706`
- `qwen3-max` 从 `0.5385` 提升到 `0.7436`

因此，same-model guarded recheck 的下阶段重点不只是“总体是否提升”，还包括：

- 哪些模型从 recheck 中显著获益
- 哪些模型在真实复查里出现反向伤害
- 是否需要模型级阈值或模型级 recheck policy

## 5. 与固定模型真实 guarded pilot 的区分

固定模型真实 guarded pilot 的正式目录为：
`outputs/experiments/deepseek_guarded_pilot/20260412_011945`

后续在文档和论文里建议保持以下区分：

- `deepseek_guarded_pilot/20260412_011945`：固定模型真实 recheck pilot
- `same_model_guarded_pilot/20260412_021656`：same-model self-recheck pilot
- `interference_detector_new15_full_sentence_embedding`：offline proxy simulation 的 detector 主线评估

不要把这三类证据合并写成一个“guarded eval”结论。

## 6. 迁移残留项与更新建议

### 6.1 已确认的残留项

目前发现的显式旧工作区路径残留主要在文档里：

- [`docs/ENTRYPOINT_GUIDE.md`](/Users/shiqi/code/graduation-project/docs/ENTRYPOINT_GUIDE.md) 原先多处仍指向旧工作区绝对路径，现已更新为新工作区路径
- [`docs/PAPER_ANALYSIS_ENTRYPOINT.md`](/Users/shiqi/code/graduation-project/docs/PAPER_ANALYSIS_ENTRYPOINT.md) 仍保留“承接旧工作区”的说明，这部分是历史背景，不算错误，但后续新增内容应只给新工作区正式入口

另外，部分实验生成物的 JSON 摘要仍内嵌旧工作区路径，例如：

- detector 主线 summary
- same-model pilot summary

这些属于历史运行产物，建议：

- 不手工改写结果文件内容，避免破坏产物可追溯性
- 如需完全清理旧路径，应通过重新生成 summary 的方式完成

### 6.2 后续建议

1. 后续所有结果型文档都应优先链接 `outputs/experiments/...`
2. 若脚本仍需要自动发现实验结果，默认 glob 应从 `outputs/experiments/**` 开始，而不是 `outputs/**`
3. guarded 相关文档建议单独拆出：
   - offline proxy simulation
   - fixed-model real pilot
   - same-model real pilot
4. 后续若写论文主文或 rebuttal，建议把 detector 叙述固定为：
   `risk ranking + selective recheck trigger policy`

