# Claim-to-Evidence Audit

更新时间：2026-05-06

审查对象：`/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex`

本审计只做 claim-to-evidence 对照，不跑新实验，不改代码，不改论文正文。判定标签含义：

- `PASS`: 当前冻结证据足以支持
- `WEAKEN`: 方向对，但当前写法偏强，需要降调
- `MOVE_TO_LIMITATION`: 只能放到 limitation / boundary 层
- `REMOVE`: 当前冻结证据不足，不建议保留
- `NEED_TABLE_OR_CITATION`: 证据存在，但正文需要更清楚的表格、路径或 artifact 引用

---

## Abstract

Claim: “pressure-induced sycophantic behavior does not admit a single, uniformly effective linear intervention”  
Verdict: `PASS`  
Reasoning: Qwen 7B / 3B mainline为正向，Llama n=100 为 `locatable but not controllable`，GLM n=100 为 `weak positive with residual damage tradeoff`，identity/profile 仍是边界线。这条总述与冻结证据一致，也没有越过 universal mitigation 边界。

Claim: “we identify a structural condition under which a pressure-related direction is behaviorally controllable”  
Verdict: `WEAKEN`  
Reasoning: Section 8 的旧 proposition 已被 `/Users/shiqi/code/graduation-project/docs/reports/controllability_predictor_analysis_20260506.md` 明确收紧。当前 frozen data 不支持把它写成稳定的 structural condition，更不支持把它升级成 predictor-like rule。更稳的表述应是“we test a bounded diagnostic hypothesis about controllability conditions”。

Claim: “Controllability arises when a direction both aligns with a behaviorally actionable logit axis and exhibits high logit-level specificity...”  
Verdict: `WEAKEN`  
Reasoning: 这是本轮最明显的过强点。高 specificity 反例已经存在：Qwen 7B `pressure_subspace_damping n=100` 的 `specificity_ratio = 3387.65` 但 `label = 0`；GLM n=100 的 `specificity_ratio = 2580.02` 也仍是 `label = 0`。因此 specificity 不能写成“arises when”式的判定条件，最多是 bounded post-hoc clue。

Claim: “The Qwen mainline satisfies this condition ... Llama violates this condition ... GLM further illustrates ...”  
Verdict: `WEAKEN`  
Reasoning: Qwen cleanly positive、Llama limitation、GLM damage tradeoff 这三分法是对的；问题在于把它们归到一个过强的“condition”下。建议保留现象层对照，去掉过强的必要/充分语气。

---

## Introduction

Claim: “pressure-induced sycophancy is not a mechanistically uniform phenomenon”  
Verdict: `PASS`  
Reasoning: 这正是当前冻结证据最稳的总命题，且与跨 pressure type、跨 model family、跨 intervention family 的边界材料一致。

Claim: “belief-style pressure in the Qwen mainline exhibits a late-layer drift that is both detectable and partially intervenable under an objective-local protocol”  
Verdict: `PASS`  
Reasoning: Qwen 7B / 3B closure 与 layer-wise / projection diagnostics 支持这一句，且客观限定在 objective-local protocol 内，没有混 metric family。

Claim: “GLM introduces substantial damage tradeoffs; Llama locatable without reliable control; identity/profile does not exhibit the same intervention profile”  
Verdict: `PASS`  
Reasoning: 这三条都与当前更新后的 frozen boundary 一致。

Claim: “we identify the conditions under which linear intervention succeeds”  
Verdict: `WEAKEN`  
Reasoning: 目前更稳的口径是“we identify bounded evidence and structural boundaries under which success and failure come apart”。“conditions under which linear intervention succeeds”太像已经得到可预测框架，而 predictor analysis 的结论是否定性的。

Claim: main claim quote at lines 59-62  
Verdict: `PASS`  
Reasoning: 这段主 claim 仍然是当前最稳的总收束，没有越过 tested linear intervention families 的边界。

---

## Section 6: Qwen Mainline

Claim: Qwen mainline is a clean positive case under the objective-local proxy mapping  
Verdict: `PASS`  
Reasoning: closure numbers、controls、held-out、以及正文对 proxy family 的提醒是匹配的。风险 9“混 metric family”在这里被处理得比较好。

Claim: held-out is robustness-side support rather than a separate benchmark family  
Verdict: `PASS`  
Reasoning: lines 304-308 的口径和冻结要求完全一致，没有把 held-out 升格成 standalone benchmark 或 strict leaderboard result。

Claim: “the strong recovery deltas should be treated as a weaker human-readable dimension than the main drift/compliance gains”  
Verdict: `PASS`  
Reasoning: 与 human audit / expert adjudication 的 frozen readout 一致，尤其 recovery 是最弱字段这一点已被保留。

Claim: wording-family drift is not tied to a single pressure intro template  
Verdict: `PASS`  
Reasoning: `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_prompt_family_check/20260501/Qwen2.5-7B-Instruct/20260501_231604/run_summary.json` 支持 default `0.283`、authority `0.450`、majority `0.317` 三行同方向。  
Follow-up: `NEED_TABLE_OR_CITATION`  
Reasoning: 论文 appendix table 有数字，但 Artifact Registry 目前没有单独列出 prompt-family run。最终收口时建议补 artifact label 或路径引用。

Claim: “A free-form patched generation diagnostic further confirms that the late-layer intervention effect is visible in open-ended generation...”  
Verdict: `WEAKEN`  
Reasoning: 方向上对，但“further confirms”偏强。当前最稳口径是 `/Users/shiqi/code/graduation-project/docs/reports/freeform_diagnostic_confirm_boundary_note_20260504.md` 的 `diagnostic free-form boundary evidence`。它不应被读成 universal free-form success，也不能当 deployment-level validation。

Claim: intervention-family comparison shows the result is not a single-implementation artifact  
Verdict: `PASS`  
Reasoning: `/Users/shiqi/code/graduation-project/docs/reports/intervention_family_comparison_20260506.md` 支持“implementation-robust but not implementation-agnostic”。  
Follow-up: `NEED_TABLE_OR_CITATION`  
Reasoning: 正文一句话已成立，但若要保留，建议在 appendix 或 artifact registry 明示这份 comparison note 的 frozen source。

Claim: “These results should be read as establishing a clean case where controllability conditions are satisfied”  
Verdict: `WEAKEN`  
Reasoning: “clean case”可以保留，但“conditions are satisfied”会把读者带回 Section 8 的过强 proposition。建议改成“clean case where controllability is observed under the tested objective-local intervention family”。

---

## Section 7: Cross-scale and Cross-family Support

Claim: Qwen14B n=100 is secondary causal confirmation, not a new main mitigation line  
Verdict: `PASS`  
Reasoning: 这正是更新后的正式口径，也修正了旧 n=48 的过强 cautionary headline。

Claim: GLM n=100 is weak positive with residual damage tradeoff, not a clean replication  
Verdict: `PASS`  
Reasoning: 与当前 frozen update 一致，也避免了把 GLM 写成 strong replication。

Claim: cross-model figure split avoids merging proxy and bridge-style rows into a single unqualified leaderboard  
Verdict: `PASS`  
Reasoning: 这条对风险 9 的处理是正确的，建议保留。

Claim: Section 7 sufficiently communicates the Llama negative result  
Verdict: `WEAKEN`  
Reasoning: Section 7 本身只写 Qwen14B / GLM，Llama 放在 Section 9。结构上没错，但从“cross-model boundary”角度看，摘要、Section 7、Limitations 三处都应明显提醒 Llama n=100 negative result。当前主文本够用，但不是特别醒目。

---

## Section 8: What Makes a Pressure Direction Controllable?

Claim: “One distinguishing factor is logit specificity rather than direction strength alone”  
Verdict: `WEAKEN`  
Reasoning: 相比 projection norm，specificity 确实更有解释力；但相对于“predict controllability”这个更强目标，它并不稳定。更稳的写法是“logit specificity is more informative than projection magnitude alone in some key comparisons, but it does not cleanly generalize.”

Claim: Qwen 7B’s large belief-logit delta and near-zero negative delta identify a strong and highly specific direction  
Verdict: `PASS`  
Reasoning: 这是对单个 Qwen 7B diagnostic 的局部描述，数字本身准确。

Claim: Table 8 excludes GLM from logit-level comparison because no frozen projection-to-logit diagnostic is available  
Verdict: `REMOVE`  
Reasoning: 这已经被新的 frozen artifact 否定。`/Users/shiqi/code/graduation-project/outputs/experiments/glm4_9b_belief_causal_transfer_n100/THUDM_glm-4-9b-chat-hf/20260504_125002/projection_alignment_summary.json` 已存在，且 predictor analysis 已用到它。当前文本在这点上过时。

Claim: “logit-level alignment with a behaviorally actionable axis is a necessary but not sufficient condition for controllability”  
Verdict: `WEAKEN`  
Reasoning: predictor analysis 已经说明 no single feature cleanly generalizes。把“necessary but not sufficient”写得这么硬，会超出当前 frozen evidence。更稳的说法是“a useful bounded diagnostic hypothesis, not an established necessary condition.”

Claim: “We therefore retain logit specificity as a necessary structural condition...”  
Verdict: `REMOVE`  
Reasoning: 这是本节最该删或重写的句子。既然高-specificity 负例已存在，就不应继续把 specificity 升格为 necessary structural condition。

Claim: Qwen satisfies both alignment and specificity, Llama does not, GLM is a specificity-with-tradeoff counterexample  
Verdict: `WEAKEN`  
Reasoning: 作为 narrative contrast 可以保留，但不要再组织成单一 law-like framework。尤其 GLM 现在是直接反例，不该再被用来为“specificity is necessary”背书。

Claim: “logit-level specificity appears more behaviorally informative than projection magnitude alone”  
Verdict: `PASS`  
Reasoning: 在 Qwen vs Llama 这组对照里这是成立的，而且比当前更强的 predictor claim 安全得多。

Claim: baseline/pressured ratio 1.02 implies Qwen 7B direction is structurally present rather than pressure-created  
Verdict: `PASS`  
Reasoning: 这是一条 bounded mechanistic hypothesis，正文已经保留了 model-specific 限定，没有越界成 universal law。

Claim: predictor-boundary paragraph (“no single feature cleanly generalizes”)  
Verdict: `PASS`  
Reasoning: 这与 `/Users/shiqi/code/graduation-project/docs/reports/controllability_predictor_analysis_20260506.md` 完全一致，应该保留并上提，而不是被前面的强 claim 抵消。

Claim: “controllability appears strongest when a pressure-related direction combines a sizable belief-logit effect with high specificity and limited off-target interference”  
Verdict: `WEAKEN`  
Reasoning: 这句比 predictor-boundary 段落更强，会再次把读者引向 sufficiency / near-sufficiency。可以改成“Qwen’s cleanest controllable rows exhibit this combination, but the feature bundle does not yet generalize into a stable predictor.”

Overall Section 8 verdict: `WEAKEN`  
Reasoning: 本节最有价值的内容是“old proposition fails as a general predictor”，而不是“we found the right controllability law”。最终收口时，应该让 predictor-boundary 段落统领全节。

---

## Section 10: Limitations

Claim: Qwen mainline is objective-local proxy and should not be merged with bridge-style rows  
Verdict: `PASS`  
Reasoning: 对风险 9 的处理正确，而且写得清楚。

Claim: Qwen14B / GLM / Llama n=100 are larger transfer-style boundary evidence, not replacements for the mainline  
Verdict: `PASS`  
Reasoning: 与当前冻结更新一致。

Claim: “No frozen projection-to-logit summary is currently archived for GLM...”  
Verdict: `REMOVE`  
Reasoning: 这句已经过时，与现有 `projection_alignment_summary.json` 冲突，必须在最终收口时修正。

Claim: Llama remains a limitation and localization alone does not establish controllability  
Verdict: `PASS`  
Reasoning: 风险 3 处理得很好，Llama n=100 的负结果也写清楚了。

Claim: identity/profile remains unresolved and should not be described as identity-specific mitigation  
Verdict: `PASS`  
Reasoning: 与正式口径一致，风险 8 处理正确。

Claim: human audit and expert adjudication are weak corroboration only  
Verdict: `PASS`  
Reasoning: 没有把 recovery 写成 human-validated success，也没有把 audit 升格成 formal human validation。

Claim: free-form diagnostic is diagnostic boundary evidence rather than clean free-form success  
Verdict: `PASS`  
Reasoning: 风险 4 处理正确，而且 wording 已经够保守。

---

## Section 11: Conclusion

Claim: “pressure-induced sycophantic behavior is not a mechanistically uniform phenomenon under the tested setup”  
Verdict: `PASS`  
Reasoning: 这是全篇最稳的一句话之一。

Claim: “misalignment between representation-level signals and behaviorally actionable logit axes” is the structural explanation  
Verdict: `WEAKEN`  
Reasoning: 这条解释方向合理，但目前还不足以写成核心定论。因为 Section 8 的 predictor analysis 已经说明现有 feature set 不能稳定 generalize。建议保留为 bounded mechanism hypothesis，而不是结论句的硬核因果解释。

Claim: “controllability emerges only when ... strong belief-logit shift + logit specificity”  
Verdict: `WEAKEN`  
Reasoning: 这里重复了 Abstract 和 Section 8 的过强框架。当前 frozen evidence 更支持“Qwen 的 clean controllable rows exhibit this pattern, but the pattern is not yet a stable predictor across intervention families.”

Claim: “This paper provides an initial answer...”  
Verdict: `PASS`  
Reasoning: 只要后面收成“linear steerability is conditional, model-dependent, and pressure-type-specific”这类边界性总结，就是当前最稳的 conclusion 口径。

---

## Risk Checklist

### 1. 是否把 Qwen 结果写成 universal mitigation

Verdict: `PASS with one caveat`  
Reasoning: Introduction、Section 6、Limitations、Conclusion 大体都守住了 “mechanism-boundary study” 口径。唯一需要收紧的是 Abstract / Section 8 / Conclusion 中对“controllability condition”的过强概括。

### 2. 是否把 Qwen14B / GLM 写成 strong replication

Verdict: `PASS`  
Reasoning: 当前主文已经把 Qwen14B 写成 `secondary causal confirmation`，把 GLM 写成 `weak positive with residual damage tradeoff`。

### 3. 是否把 Llama n=100 负结果写清楚

Verdict: `PASS`  
Reasoning: Section 10 明确写了 `+0.03 / +0.01 / -0.03 / 0.08` 的负结果，整体口径清楚。

### 4. 是否把 free-form diagnostic 写成 universal free-form success

Verdict: `PASS with minor weakening opportunity`  
Reasoning: Limitations 已经写成 diagnostic boundary evidence。Section 6 的 “further confirms” 建议再弱一点。

### 5. 是否把 predictor 写成 sufficient predictor

Verdict: `WEAKEN`  
Reasoning: Abstract、Section 8、Conclusion 仍有这个风险。应让 predictor-boundary 段落成为主导口径。

### 6. 是否把 specificity 写成 sufficient condition

Verdict: `WEAKEN / REMOVE`  
Reasoning: 这是当前最大的口径风险。高-specificity 反例已存在，不能继续写成 “arises when” 或 “necessary structural condition”。

### 7. 是否把 recovery 写成 strong human-validated success

Verdict: `PASS`  
Reasoning: Section 6 和 Limitations 都保留了 recovery weaker / human-correspondence lower 的边界。

### 8. 是否把 identity_profile 写成 solved mitigation

Verdict: `PASS`  
Reasoning: 当前正文对 identity/profile 的处理很稳。

### 9. 是否混合了不同 metric family

Verdict: `PASS`  
Reasoning: Figure 3 split-panel、Section 6 proxy note、Limitations 第一段都明确区分了 objective-local proxy 与 bridge / transfer-style rows。

### 10. 是否在标题、摘要或结论中超过 “tested linear intervention families” 的证据边界

Verdict: `WEAKEN`  
Reasoning: 标题本身未审，但摘要与结论里关于 controllability proposition 的语言仍略微超过 “tested families” 边界。问题不在 positive/negative results，而在 general law 语气过强。

---

## Highest-Priority Final-Wording Fixes

1. 把 Abstract lines 34-35 的 “identify a structural condition” / “Controllability arises when ...” 降为 bounded diagnostic hypothesis。  
2. 删除或重写 Section 8 中 “necessary structural condition” 的句子。  
3. 更新 Section 8 与 Section 10 里关于 GLM “no frozen projection-to-logit diagnostic available” 的过时说法。  
4. 把 Section 6 对 free-form patched generation 的 “further confirms” 改成更保守的 diagnostic corroboration wording。  
5. 给 prompt-family generalization 与 intervention-family comparison 补 artifact citation 或 appendix source pointer。

## Bottom Line

当前论文 v20260506 的主体边界其实已经比较稳：Qwen mainline、Qwen 3B replication、Qwen14B secondary support、GLM tradeoff、Llama limitation、identity/profile boundary、human audit weak corroboration，这些大框架都成立。真正需要在最终收口前修正的，主要不是数字，而是 Section 8 相关的理论语气：它目前仍像一个已成立的 controllability proposition，但当前 frozen evidence 更支持“旧 proposition 经测试后不能稳定推广”。
