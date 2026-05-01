# White-box Sycophancy Paper Evidence Package

更新时间：2026-04-27

本目录用于给论文撰写部门和审稿复现提供一个固定入口，避免继续在项目工作区中到处找文件。

约定：

- 本 package 只整理、索引、链接现有材料，不修改实验结果，不改代码。
- 目录中的大多数结果文件使用符号链接指向原始冻结产物，便于保持单一事实来源。
- 若审稿复现需要回溯原始运行上下文，请优先查看 `configs/`、`results_*` 与 `appendix_materials/`。

证据边界：

- Qwen 3B / 7B 主线使用 objective-local proxy 口径，不应与 bridge causal lines 做完全同义 leaderboard。
- Qwen 14B 只作为 secondary causal confirmation。
- Qwen clean protocol 新增结果中，3B 可作为 positive confirmation 副主线；7B 只作为 behaviorally weak boundary evidence；14B `n=48` 只作为 cautionary / boundary evidence。
- GLM 只作为 cross-family positive replication with stronger tradeoff。
- Llama 只作为 weak replication / limitation，不是 positive replication。
- Mistral 只作为 appendix exploratory，不是 positive replication。
- identity/profile 只作为 weak mechanistic observation / boundary。

## Package Layout

- `configs/`
  - 冻结运行配置、`run.json`、统计口径说明、sweep manifest。

- `datasets/`
  - objective-local sample sets、bridge manifest、dataset coverage、held-out subset builder。

- `results_mainline/`
  - Qwen 7B formal mainline、Qwen 3B formal replication、Qwen 7B held-out validation、mechanistic summaries。

- `results_transfer/`
  - Qwen 14B secondary confirmation、Qwen 3B clean positive confirmation、Qwen 7B clean boundary evidence、Qwen 14B `n=48` cautionary evidence、GLM tradeoff、Llama bridge、Mistral exploratory。

- `results_controls/`
  - formal controls、subtraction controls、sanity controls、negative controls raw run、aggressive secondary、layer/strength stability。

- `results_identity_profile/`
  - identity/profile follow-up、matched controls、late-layer negative control、n=36 summary。

- `results_llama_diagnostics/`
  - projection norm summary、projection/drift correlation、alpha sweep rows、behavioral closure、limitation summary。

- `figures/`
  - Figure 2/3/4 source data、figure asset manifest、figure notes。

- `tables/`
  - 主文与附录表格源数据，包括 `whitebox_effect_size_table.csv`、Table 2、Table 5、Table 6。

- `prompt_templates/`
  - bridge scenarios 与相关 prompt-side protocol tables。

- `human_audit/`
  - white-box human audit bundle、annotation guidelines、paper summary。

- `appendix_materials/`
  - 冻结证据包、主文草稿、appendix-ready notes、supporting note、metric audit note、internal reproducibility note。

## High-priority Files

- 统计收口主表：
  - [whitebox_effect_size_table.csv](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/tables/whitebox_effect_size_table.csv)

- Qwen 主线 Figure 2 源数据：
  - [figure2_qwen_mainline_effect_ci.csv](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/figures/figure2_qwen_mainline_effect_ci.csv)

- Qwen 7B formal controls：
  - [qwen7b_formal_controls](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/results_controls/qwen7b_formal_controls)

- Qwen 7B held-out validation：
  - [qwen7b_heldout_validation](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/results_mainline/qwen7b_heldout_validation)

- 跨模型 Figure 3 / Table 2 源数据：
  - [figure3_panelB_bridge_causal_tradeoff.csv](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/figures/figure3_panelB_bridge_causal_tradeoff.csv)
  - [table2_source_data.csv](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/tables/table2_source_data.csv)

- Qwen clean protocol / appendix boundary note：
  - [qwen_clean_protocol_appendix_note.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/appendix_materials/qwen_clean_protocol_appendix_note.md)
  - [qwen3b_clean_positive_confirmation_run](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/results_transfer/qwen3b_clean_positive_confirmation_run)
  - [qwen7b_clean_boundary_run](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/results_transfer/qwen7b_clean_boundary_run)
  - [qwen14b_n48_cautionary_run](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/results_transfer/qwen14b_n48_cautionary_run)

- Llama limitation 与 Figure 4 源数据：
  - [llama_limitation_summary.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/results_llama_diagnostics/llama_limitation_summary.md)
  - [figure4_llama_projection_summary.csv](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/figures/figure4_llama_projection_summary.csv)
  - [figure4_llama_projection_diagnostic.csv](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/figures/figure4_llama_projection_diagnostic.csv)

- identity/profile Table 6 源数据：
  - [table6_source_data.csv](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/tables/table6_source_data.csv)

- 总索引：
  - [INDEX.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/INDEX.md)

- Supporting note / evidence boundary note：
  - [whitebox_supporting_note.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/appendix_materials/whitebox_supporting_note.md)

- Metric audit / CI-leakage boundary note：
  - [metric_audit_note_20260501.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/appendix_materials/metric_audit_note_20260501.md)

## What To Open First

1. [INDEX.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/INDEX.md)
2. [configs/statistical_closure_README.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/configs/statistical_closure_README.md)
3. [appendix_materials/frozen_evidence_dossier.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/appendix_materials/frozen_evidence_dossier.md)
4. [appendix_materials/internal_reproducibility_note.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/appendix_materials/internal_reproducibility_note.md)
5. [appendix_materials/whitebox_supporting_note.md](/Users/shiqi/code/graduation-project/paper_evidence_package_20260427/appendix_materials/whitebox_supporting_note.md)

## Main-text Alignment Notes

当前 package 已和主文口径对齐：

- Qwen mainline 仍定义为 objective-local proxy。
- Qwen 14B 仍是 secondary causal confirmation。
- Qwen clean protocol 中：3B 可作为 positive confirmation 副主线；7B 只作为 appendix / boundary evidence；14B `n=48` 只作为 cautionary / boundary evidence。
- GLM 仍是 stronger-tradeoff cross-family support。
- Llama 仍是 limitation。
- identity/profile 仍是 boundary case。
- Mistral 仍是 appendix exploratory。

当前 package 也已吸收 2026-05-01 的 metric numerator/denominator + leakage 审计边界：

- `stance_drift_delta`、`pressured_compliance_delta`、`recovery_delta`、`baseline_damage_rate` 的定义与分母口径已确认一致。
- held-out objective-local export 的 CI 可以保守写成：`item-level percentile bootstrap, 2000 iterations`。
- formal controls aggregate 的 CI 可以保守写成：`shared sample ID item-level percentile bootstrap, 2000 iterations`。
- frozen `whitebox_effect_size_table.csv` 的 bootstrap 实现细节当前只能写成 `unable to confirm from the audited inputs`，不要写成 `fully verified`。
- Qwen 7B held-out 的泄漏审计结论是 `clean`，但论文与 package 口径仍只能保持为 `robustness-side validation`，不能升级成 `pre-registered clean split main result`。

此外，当前 package 已补入主文已可安全引用的两组 7B 材料：

- `results_controls/qwen7b_formal_controls`
  - 对应“formal controls”，用于说明主线结果不是任意 activation edit 或简单 pairing artifact。

- `results_mainline/qwen7b_heldout_validation`
  - 对应“held-out validation”，用于说明默认 7B 主线方向不局限于原始调参子集。

## Supporting Note Placement

`appendix_materials/whitebox_supporting_note.md` 是当前 package 中最合适的归档位置。它应被视为 `appendix note / evidence boundary note`，而不是新的主结果文档。

当前它覆盖三部分内容：

- objective-local metric family note
- updated controls + held-out support artifact note
- GLM sample-size comparability note

它的作用是补强主文与附录的表述边界，而不是替换 frozen effect sizes、figure assets 或主文主表。

另外，`appendix_materials/qwen_clean_protocol_appendix_note.md` 用于归档 2026-04-30 clean protocol belief causal transfer 的最新收束口径。它只用于 appendix / evidence boundary 对齐，不应被读成对现有 14B 主文 secondary line 的升级或替代。

`appendix_materials/metric_audit_note_20260501.md` 则用于归档 metric numerator/denominator、bootstrap / CI、以及 held-out leakage 的最新审计边界。它的作用是收紧表述，不是把 frozen closure bootstrap 升格为 fully verified。
