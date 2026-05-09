# Final Wording Revision List

## Fix 1: Abstract specificity claim
- 行号: lines 34-35
- 当前文本: `"Across a frozen white-box evidence package spanning Qwen, GLM, Llama, and Mistral families, we identify a structural condition under which a pressure-related direction is behaviorally controllable. Controllability arises when a direction both aligns with a behaviorally actionable logit axis and exhibits high logit-level specificity, producing a strong belief-logit shift while leaving unrelated logits largely unaffected."`
- 问题: `specificity` 反例已存在。Qwen 7B `pressure_subspace_damping` at `n=100` has `specificity_ratio = 3387.65` but remains non-controllable; GLM `n=100` also has very high specificity while remaining a weak-positive / residual-damage case. 当前写法把 post-hoc clue 写成了 condition-like rule。
- 建议改为: `"Across a frozen white-box evidence package spanning Qwen, GLM, Llama, and Mistral families, we test a bounded diagnostic hypothesis about when a pressure-related direction becomes behaviorally controllable. Logit-level alignment with a behaviorally actionable axis appears informative in the cleanest controllable rows, but high specificity alone does not guarantee behavioral control."`
- 判定: `WEAKEN`

## Fix 2: Section 8 “necessary structural condition”
- 行号: lines 460-463
- 当前文本: `"These observations suggest that logit-level alignment with a behaviorally actionable axis is a necessary but not sufficient condition for controllability. ... We therefore retain logit specificity as a necessary structural condition---controllability requires that the direction be logit-effective---while acknowledging that specificity alone does not guarantee behavioral controllability. The intervention-family comparison (Appendix) further supports this condition..."`
- 问题: 这里有两层过强表述。第一层把当前分析写成“necessary but not sufficient condition”；第二层更直接地把 `logit specificity` 升格为 `necessary structural condition`。根据 predictor analysis，高-specificity 负例已经存在，因此不应保留必要条件语气。
- 建议改为: `"These observations support a useful bounded diagnostic hypothesis rather than an established necessary condition. Logit-level alignment with a behaviorally actionable axis appears informative in the cleanest controllable rows, but the current intervention-family evidence shows that neither specificity nor any single diagnostic feature yet generalizes into a stable predictor of behavioral control. The intervention-family comparison (Appendix) further supports this bounded reading..."`
- 判定: `REMOVE`（针对 “We therefore retain logit specificity as a necessary structural condition...”） / `WEAKEN`（针对前后 framing）

## Fix 3: GLM projection-diagnostic wording
- 行号: lines 453, 493, 555
- 当前文本: 
  - `"GLM is retained as a directionality-with-damage case rather than a logit-level mechanism comparison, because no frozen projection-to-logit diagnostic is available."`
  - `"GLM-4-9B is included as a directionality-with-damage case rather than a logit-level mechanism comparison, due to the absence of a frozen projection-to-logit diagnostic."`
  - `"No frozen projection-to-logit summary is currently archived for GLM, so the cross-model mechanism-gap discussion does not make a logit-level comparability claim for that model family."`
- 问题: 这些说法已经过时。GLM 的 `projection_alignment_summary.json` 现在是冻结证据的一部分，predictor analysis 已经使用它。真正该保留的是证据等级边界，而不是“没有数据”这个旧原因。
- 建议改为: `"GLM-4-9B now has an archived projection-to-logit diagnostic, but it should still be read as a weak-positive / residual-damage boundary case rather than as a clean controllability success. The GLM diagnostic is informative for cross-model comparison, yet it does not overturn the paper's broader boundary result that logit-level features do not cleanly predict controllability across intervention families."`
- 判定: `REMOVE`

## Fix 4: Section 6 free-form “further confirms”
- 行号: line 318
- 当前文本: `"A free-form patched generation diagnostic further confirms that the late-layer intervention effect is visible in open-ended generation, though the behavioral gain saturates at the shortest patch horizons while extended patching degrades text quality (Appendix~\\ref{sec:appendix_freeform_patched_diagnostic})."`
- 问题: `further confirms` 稍强，容易把这条 appendix 证据读成更接近 free-form success 或 deployment-side validation。当前 frozen note 的最稳定位是 `diagnostic free-form boundary evidence`。
- 建议改为: `"A free-form patched generation diagnostic provides diagnostic corroboration that the late-layer intervention effect is visible in open-ended generation, though the behavioral gain saturates at the shortest patch horizons while extended patching degrades text quality (Appendix~\\ref{sec:appendix_freeform_patched_diagnostic})."`
- 判定: `WEAKEN`

## Fix 5: Prompt-family + intervention-family artifact citation
- 行号: lines 317, 319, 730-734, 752-773
- 当前文本:
  - `"A small wording-family check on the Qwen 7B mainline subset further suggests ... (Appendix~\\ref{sec:appendix_robustness})."`
  - `"An intervention-family comparison (Appendix) further shows ..."`
  - Appendix prompt-family and intervention-family paragraphs currently report results without an explicit frozen-source artifact anchor in the surrounding prose.
- 问题: 证据本身已经在 appendix，但当前写法仍缺少更明确的 frozen-source / artifact-registry linkage。审稿人如果追问 run provenance，需要更顺手的引用入口。
- 建议改为: 
  - 在主文一句话后补 `source listed in the Appendix Artifact Registry`
  - 或在 appendix 两段末尾补一句：`Frozen source labels are listed in the Appendix Artifact Registry.`
  - 如需更细，可在 Artifact Registry 新增 prompt-family / intervention-family labels，但不要把本地绝对路径写进主文。
- 判定: `NEED_TABLE_OR_CITATION`

## Fix 6: Section 8 “controllability appears strongest when...”
- 行号: lines 490-492
- 当前文本: `"Under the frozen evidence boundary, controllability appears strongest when a pressure-related direction combines a sizable belief-logit effect with high specificity and limited off-target interference. Qwen 7B fits that profile most clearly; Qwen 3B shows a related but not identical pattern; and Llama satisfies localization but not useful control."`
- 问题: 这句又把读者带回 feature-bundle predictor 的方向，和前面的 predictor-boundary 段落打架。Qwen 的 clean controllable rows exhibit this bundle, but the bundle does not yet generalize.
- 建议改为: `"Under the frozen evidence boundary, Qwen's cleanest controllable rows exhibit a combination of sizable belief-logit effect, comparatively high specificity, and limited off-target interference. That bundle is informative for interpreting the Qwen mainline, but it does not yet generalize into a stable predictor of controllability across models or intervention families."`
- 判定: `WEAKEN`

## Fix 7: Conclusion causal-explanation sentence
- 行号: lines 596-600
- 当前文本: `"The central implication is structural rather than empirical. Pressure-induced sycophantic behavior should not be modeled as a single linearly steerable phenomenon, even when it appears behaviorally similar across prompts, pressure types, or model families. The failure of uniform controllability is not incidental, but arises from misalignment between representation-level signals and behaviorally actionable logit axes."`
- 问题: 前两句成立，但最后一句已经接近 settled causal explanation。按照 audit，更稳的写法应把它收回成 bounded mechanism hypothesis，而不是 hard causal diagnosis。
- 建议改为: `"The central implication is structural rather than merely empirical. Pressure-induced sycophantic behavior should not be modeled as a single linearly steerable phenomenon, even when it appears behaviorally similar across prompts, pressure types, or model families. A bounded mechanism hypothesis consistent with the present evidence is that uniform controllability fails when representation-level pressure signals do not map cleanly onto behaviorally actionable logit axes."`
- 判定: `WEAKEN`

## Fix 8: Conclusion “controllability emerges only when...”
- 行号: lines 600-601
- 当前文本: `"Under the current evidence package, controllability emerges only when a pressure-related direction satisfies two conditions simultaneously: it must produce a sufficiently strong belief-logit shift, and it must remain logit-specific, avoiding substantial off-target interference. Qwen provides a clean case where both conditions hold; Llama provides a counterexample where localization does not translate into control; and GLM illustrates that directional alignment alone can come with substantial damage tradeoffs."`
- 问题: 这重复了 Abstract 的过强 rule-like framework，而且在 predictor analysis 之后已经不够稳。Qwen rows can illustrate the bundle; they do not validate it as a general condition.
- 建议改为: `"Under the current evidence package, Qwen's clean controllable rows combine strong belief-logit movement with comparatively high specificity and limited off-target interference. Llama shows that localization can fail to translate into control, and GLM shows that directional alignment can coexist with residual damage. Taken together, these comparisons support a bounded mechanism-boundary account rather than a single controllability rule."`
- 判定: `WEAKEN`
