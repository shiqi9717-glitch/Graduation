# Paper Draft Index

更新时间：2026-04-26

本文档记录 `docs/papers/` 下的论文写作阶段产物。论文写作部门负责正文和投稿文本；实验设计、代码运行、结果复核、文件归档仍由对应部门负责。

## White-box Mechanistic Paper Drafts

当前 white-box mechanistic 主线论文写作产物：

- `WHITEBOX_MECHANISTIC_PAPER_SKELETON_20260426.md`
  - 用途：论文 skeleton / 结构规划稿。
  - 适用：给论文撰写部门继续展开章节结构、主文叙事和附录安排。

- `WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.md`
  - 用途：Markdown 主文初稿。
  - 适用：正文语言、段落逻辑、结果叙述、limitation 表述继续打磨。

- `WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex`
  - 用途：LaTeX 主文初稿。
  - 状态：当前 LaTeX 初稿已通过 `pdflatex` 编译检查。
  - 适用：投稿格式、LaTeX 表格/引用/版式继续整理。

证据边界必须保留：

- Mistral-7B 只作为 appendix exploratory，不得写成 positive replication。
- Llama-3.1-8B 只作为 weak replication / limitation，不得写成 positive replication。
- Qwen 3B / 7B 的 objective-local proxy 不应与 bridge causal lines 做完全同义 leaderboard。

主要证据包：

- `docs/reports/WHITEBOX_MECHANISTIC_EVIDENCE_DOSSIER_20260426.md`
- `docs/reports/WHITEBOX_MECHANISTIC_EVIDENCE_MATRIX_20260426.md`
- `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`
