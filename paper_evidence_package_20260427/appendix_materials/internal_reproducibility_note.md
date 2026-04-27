# Internal Reproducibility Note

更新时间：2026-04-27

本文档说明本 evidence package 如何对应到当前工作区中的冻结白盒机制证据。

## Scope

本 package 服务两个目标：

- 论文撰写：主文、附录、答辩材料不再需要在项目里到处找文件。
- 审稿复现：审稿人或内部复核者可以快速定位 metric definition、frozen runs、figure/table source data、prompt templates 和 human audit bundle。

## Reproducibility Principles

- 本 package 不复制大体量结果目录，而是用符号链接指向原始冻结产物。
- 统计口径优先锚定 `whitebox_mechanistic_statistical_closure/20260426_175452`。
- 主文图表优先锚定 `whitebox_mechanistic_figure_assets/20260426_230559`。
- 写作边界优先锚定冻结的 dossier、matrix、workspace summary。

## Required Boundary Conditions

- Qwen 3B / 7B 的主文数字是 objective-local proxy，不是 bridge-causal leaderboard。
- Qwen 14B 只作为 secondary causal confirmation。
- GLM 只作为 positive replication with stronger tradeoff。
- Llama 只作为 weak replication / limitation，不继续建议 alpha / k / layer sweep。
- Mistral 只作为 appendix exploratory，不继续建议 sweep。
- identity/profile 只作为 weak mechanistic observation / boundary。

## Key Frozen Anchors

- Statistical closure:
  - `configs/statistical_closure_README.md`
  - `tables/whitebox_effect_size_table.csv`

- Figure assets:
  - `figures/figure2_qwen_mainline_effect_ci.csv`
  - `figures/figure3_panelA_qwen_proxy_tradeoff.csv`
  - `figures/figure3_panelB_bridge_causal_tradeoff.csv`
  - `figures/figure4_llama_projection_summary.csv`
  - `figures/figure4_llama_projection_diagnostic.csv`

- Mainline runs:
  - `results_mainline/qwen7b_mainline_run`
  - `results_mainline/qwen3b_replication_run`

- Transfer runs:
  - `results_transfer/qwen14b_secondary_confirmation_run`
  - `results_transfer/glm4_9b_tradeoff_run`
  - `results_transfer/llama_english_bridge_run`
  - `results_transfer/mistral_english_exploratory_run`
  - `results_transfer/mistral_zh_instruction_exploratory_run`

- Identity follow-up:
  - `results_identity_profile/qwen7b_identity_followup`

## Package Status

- `README.md` and `INDEX.md` are package-native files generated for navigation.
- All other linked evidence should be treated as frozen upstream sources unless the user explicitly requests a refreshed package.
