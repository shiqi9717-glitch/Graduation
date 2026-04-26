# Pressure Is Not One Mechanism: Separating Belief Drift from Identity/Profile Pathways in White-box Sycophancy

【FROM: 论文撰写部门】
【TO: 用户】

Status: formal main-text draft v1, converted from `Expanded Main-text Draft v1` in `docs/papers/WHITEBOX_MECHANISTIC_PAPER_SKELETON_20260426.md`.

Evidence status: frozen. This draft does not introduce new experiments, does not change result framing, keeps Mistral as appendix exploratory, and keeps Llama as weak replication / limitation.

Primary evidence:

- `/Users/shiqi/code/graduation-project/docs/reports/WHITEBOX_MECHANISTIC_EVIDENCE_DOSSIER_20260426.md`
- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/README.md`
- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/whitebox_effect_size_table.csv`
- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/llama_limitation_summary.md`

## Abstract

Sycophantic behavior in language models is often treated as a single failure mode: a model changes its answer or stance under user pressure. This paper argues that this framing is too coarse. Using a frozen white-box mechanistic evidence package across Qwen, GLM, Llama, and Mistral variants, we show that pressure-induced behavior follows separable internal pathways. For belief-style pressure, Qwen 7B and Qwen 3B provide the formal mainline evidence: a late-layer belief-pressure drift can be detected and partially damped under the default `baseline_state_interpolation` settings, reducing stance drift and pressured compliance while preserving recovery and introducing no measured baseline damage under the objective-local proxy mapping. Qwen 14B provides secondary causal confirmation, and GLM-4-9B reproduces the damping direction across model families, albeit with a substantially stronger damage tradeoff. In contrast, identity/profile pressure shows weaker causal intervention support: localization evidence exists, but prefix-specific intervention does not stably separate from matched controls. Llama-3.1-8B further clarifies the boundary of the claim: its belief-pressure subspace is locatable under an English bridge prompt, but intervention transfer remains weak, leaving pressured compliance unchanged while reducing recovery and increasing baseline damage. Together, these results support a pressure-type-specific account of sycophancy: belief-style pressure admits a transferable and partially intervenable late-layer drift, whereas identity/profile pressure is not explained by the same linear steering story.

## 1. Introduction

Language models often change their answers when a user applies pressure. In sycophancy evaluations, this behavior is commonly treated as a single phenomenon: a model appears to follow the user's stated preference, identity, or argument even when doing so conflicts with a more stable answer. This behavioral framing is useful for measurement, but it can obscure a mechanistic question that matters for both interpretation and intervention: are all pressure-induced shifts produced by the same internal mechanism?

This paper argues that the answer is no. We study pressure-induced sycophancy through a white-box evidence package that separates belief-style pressure from identity/profile pressure. Belief-style pressure refers to cases where the user supplies an argument or stance cue that pushes the model toward a pressure-aligned belief. Identity/profile pressure refers to cases where the user profile or identity frame changes the social context of the answer. Both can look like pressure at the behavioral level, but the evidence does not support treating them as one shared internal pathway.

Our main claim is:

> Pressure is not one mechanism: belief-style pressure admits a transferable and partially intervenable late-layer drift, while identity/profile pressure follows a different and less linearly steerable pathway.

The positive part of the claim is anchored in the Qwen mainline. Under the default `baseline_state_interpolation` settings, Qwen 7B and Qwen 3B show consistent reductions in stance drift and pressured compliance while preserving recovery and introducing no measured baseline damage under the objective-local proxy mapping defined in the statistical closure notes. These results support the view that belief-style pressure can induce a late-layer drift that is not merely detectable but partially intervenable.

The broader evidence is deliberately bounded. Qwen 14B provides secondary causal confirmation, suggesting that the belief-pressure damping direction is not confined to one model size. GLM-4-9B gives cross-family support, but with a substantially stronger baseline-damage tradeoff. Llama-3.1-8B provides the main boundary case: the belief-pressure direction is more identifiable under an English bridge prompt, yet the tested intervention transfers weakly, leaving pressured compliance unchanged, reducing recovery, and introducing substantial baseline damage. Mistral-7B remains appendix exploratory rather than formal replication.

The negative part of the claim is equally important. The identity/profile line provides weak localization evidence but insufficient causal intervention support. We therefore do not claim identity-specific mitigation, and we do not treat identity/profile pressure as solved by the same linear belief-damping story. This distinction is the paper's central contribution: not a claim of universal pressure mitigation, and not a claim of first-ever sycophancy steering, but a pressure-type-specific account of when a white-box pressure direction is detectable, transferable, and behaviorally useful.

The paper proceeds as follows. Section 2 situates the work relative to behavioral sycophancy measurement and activation-level intervention work, emphasizing that our novelty lies in pressure-type separation rather than in generic steering. Section 3 defines the pressure types, metrics, and evidence levels used throughout the paper. Section 4 presents the white-box mechanism map. Section 5 reports the Qwen mainline. Section 6 summarizes cross-scale and cross-family support. Section 7 discusses the Llama limitation. Section 8 explains the identity/profile boundary. Section 9 states limitations, and Section 10 concludes.

## 2. Related Work And Positioning

Prior work on sycophancy often begins from behavioral measurement: does the model follow the user, agree with an incorrect premise, or shift its answer under social or argumentative pressure? This behavioral lens is essential because it defines the failure mode that users experience. However, behavioral similarity does not guarantee mechanistic identity. Two prompts can both induce sycophantic-looking answers while relying on different internal features, layers, or causal pathways.

Activation-level intervention work provides another important backdrop. It shows that model behavior can sometimes be shifted by modifying internal states. Our focus is different. We do not frame the paper as a first demonstration that sycophancy can be steered. Instead, we ask whether different forms of pressure instantiate the same internal mechanism. The evidence suggests they do not.

This distinction shapes the paper's novelty claim. We treat sycophancy mitigation as pressure-type-specific rather than as a single universal steering target. Belief-style pressure shows evidence of a late-layer drift that is partially controllable in the Qwen mainline, while identity/profile pressure does not inherit that conclusion. The key contribution is therefore mechanism separation: belief pressure and identity/profile pressure differ in internal representation and intervention response.

## 3. Setup: Pressure Types, Metrics, And Evidence Levels

We organize the evidence around pressure type rather than around a single aggregate sycophancy score. The first pressure type, `belief_argument`, captures settings where the user introduces a belief-like argument or stance cue. In this setting, the core mechanistic hypothesis is that pressure induces a late-layer representational drift that can be partially damped by moving the residual stream toward a baseline state or by damping a matched belief-pressure subspace.

The second pressure type, `identity_profile`, captures settings where a user profile or identity context changes the social frame of the answer. For this pressure type, the evidence supports localization observations but does not support a stable causal intervention claim. We therefore use identity/profile results as a boundary on generalization rather than as a mitigation success.

The primary behavioral quantities come from the frozen statistical closure. `stance_drift_delta` is defined as the intervention rate minus the reference rate, so negative values indicate less pressure-induced drift. `pressured_compliance_delta` is also intervention rate minus reference rate, so negative values indicate less pressure-aligned compliance. `recovery_delta` measures intervention recovery relative to the reference setup, where positive values indicate improved recovery. `baseline_damage_rate` measures damage on baseline behavior, where lower is better.

For the Qwen 3B and Qwen 7B mainline, the statistical closure uses a special objective-local proxy mapping. In this mapping, stance drift uses an interference-induced-error proxy on strict-positive, high-pressure wrong-option items; pressured compliance uses a wrong-option-follow proxy on the same local subset; and recovery equals the intervention recovery rate because the reference system has structural zero recovery under this intervention setup. The main text should therefore describe these as objective-local proxy results, not as a leaderboard-equivalent measure that can be directly merged with bridge causal transfer lines.

The evidence levels are fixed throughout this draft. Qwen 7B is the formal mainline result, and Qwen 3B is the formal mainline replication. Qwen 14B is secondary causal confirmation, not a new primary mitigation line. GLM-4-9B is cross-family positive replication with a stronger tradeoff. Llama-3.1-8B is weak replication / limitation: it helps identify the boundary between locating a pressure direction and using it for reliable behavioral control. Mistral-7B is appendix exploratory only. The `identity_profile` line is a weak mechanistic observation / boundary case, not an identity-specific mitigation result.

This hierarchy is central to the paper's interpretation. The paper does not ask whether every model admits the same steering direction. Instead, it asks whether pressure-induced behavior decomposes into mechanistically separable pathways, and whether one pathway, belief-style pressure, has a more linearly steerable late-layer component than identity/profile pressure.

## 4. Mechanistic Map: Pressure Is Not One Mechanism

Figure 1 presents the mechanism map before the numerical results. The left branch corresponds to belief-style pressure. In this branch, pressure is modeled as inducing a late-layer drift away from a baseline answer state. The Qwen mainline tests whether dampening or interpolating this drift can reduce pressure-aligned behavior while preserving recovery. The right branch corresponds to identity/profile pressure. In that branch, localization evidence exists, but the available intervention evidence does not stably separate prefix-specific causal effects from matched controls.

This map establishes the paper's central interpretive frame. The goal is not to find a single pressure vector that explains all pressure-induced sycophancy. The goal is to show that pressure-like behavioral shifts can split into different mechanistic families. Belief-style pressure has a stronger case for a late-layer, partially intervenable drift. Identity/profile pressure remains a boundary case and should not be described as mitigated by the same intervention.

**Figure 1 placeholder: White-box mechanism map.** Left branch: `belief_argument` -> late-layer drift -> baseline-state interpolation / belief-subspace damping -> partial behavioral correction. Right branch: `identity_profile` -> localization observation -> insufficient prefix-specific causal intervention -> boundary/limitation. Bottom guardrail: localization is not equivalent to controllability.

## 5. Qwen Mainline: Belief-pressure Damping

The formal mainline evidence comes from Qwen 7B and Qwen 3B under the default `baseline_state_interpolation` settings. For Qwen 7B, the default setting uses layers 24-26 at strength 0.6. Under the objective-local proxy mapping, the intervention reduces stance drift by -0.1493 with confidence interval [-0.2200, -0.0800], and reduces pressured compliance by -0.1595 with confidence interval [-0.2300, -0.0900]. Recovery increases by 0.7127 with confidence interval [0.5000, 0.9000], while measured baseline damage is 0.

Qwen 3B provides the formal replication under layers 31-35 at strength 0.6. It reduces stance drift by -0.1597 with confidence interval [-0.2402, -0.0800], and pressured compliance by -0.2789 with confidence interval [-0.3700, -0.1900]. Recovery increases by 0.6906 with confidence interval [0.5000, 0.8667], again with zero measured baseline damage.

Together, these results support the main positive claim: in the tested Qwen settings, belief-style pressure admits a late-layer drift that is both detectable and partially intervenable. Figure 2 should show the Qwen 7B and Qwen 3B effects side by side, separating stance drift, pressured compliance, recovery, and baseline damage. The caption should explicitly state that these are objective-local proxy measurements under the statistical closure mapping.

The interpretation remains narrow. These results do not establish universal pressure mitigation, and they should not be presented as directly comparable to every bridge causal transfer setting. They establish a formal Qwen mainline for belief-style pressure under the specified objective-local mapping.

**Figure 2 placeholder: Qwen mainline intervention effect.** Plot Qwen 7B and Qwen 3B effects with confidence intervals for stance drift delta, pressured compliance delta, recovery delta, and baseline damage. Caption guardrail: metrics follow the objective-local proxy mapping specified in the statistical closure notes and are not treated as leaderboard-equivalent to bridge causal lines.

**Table 1 placeholder: Evidence-level matrix.** Include Qwen 7B, Qwen 3B, Qwen 14B, GLM-4-9B, Llama-3.1-8B, Mistral-7B, and `identity_profile`, with evidence level, placement, supported claim, and required boundary.

## 6. Cross-scale And Cross-family Support

Qwen 14B extends the Qwen evidence across scale, but it should be framed as secondary causal confirmation rather than a new main mitigation line. Under matched belief-subspace damping against no intervention, Qwen 14B reduces drift by -0.2068 with confidence interval [-0.4167, 0.0000], and reduces pressured compliance by -0.2484 with confidence interval [-0.4167, -0.0833]. Its recovery delta is 0, and baseline damage is 0.0416. This supports the direction of the belief-pressure intervention while keeping the main claim anchored in the 3B/7B formal line.

GLM-4-9B provides cross-family evidence. In the representative transfer setting, it reduces stance drift by -0.2932 with confidence interval [-0.5417, -0.0417], and pressured compliance by -0.2095 with confidence interval [-0.4167, 0.0000]. It also improves recovery by 0.3769 with confidence interval [0.1667, 0.5833]. However, this support comes with substantial baseline damage of 0.3325 with confidence interval [0.1667, 0.5417]. GLM should therefore be written as cross-family positive replication with stronger tradeoff, not as a clean no-damage replication.

Table 2 should collect Qwen 14B and GLM together. This table should make the asymmetry visible: the belief-pressure damping direction transfers beyond the primary Qwen 3B/7B settings, but the quality of transfer depends on model family and damage profile.

**Table 2 placeholder: Cross-scale and cross-family support.** Rows: Qwen 14B and GLM-4-9B. Columns: comparison, stance drift delta + CI, pressured compliance delta + CI, recovery delta + CI, baseline damage + CI, interpretation. Required interpretation: Qwen 14B is secondary causal confirmation, while GLM is cross-family directionality with stronger tradeoff.

## 7. Limitation Result: Llama Is Locatable But Not Controllable

Llama-3.1-8B is the paper's most important limitation case. The projection-to-logit summary indicates that the pressured-stage belief projection is stronger than baseline, so a belief-pressure direction is locatable under the English bridge prompt. However, this localization does not translate into reliable behavioral control. Projection magnitude is weakly and negatively correlated with stance drift, and the tested belief damping moves logits in the targeted direction while remaining entangled with the baseline manifold.

The behavioral closure makes the limitation concrete. In the representative weak-replication setting, Llama reduces stance drift only by -0.0824 with confidence interval [-0.2917, 0.1250], leaves pressured compliance unchanged, reduces recovery by -0.0847 with confidence interval [-0.2917, 0.1250], and incurs baseline damage of 0.2499 with confidence interval [0.0833, 0.4167]. This is not a positive replication of the Qwen/GLM pattern.

The main text uses Llama to separate two concepts that are often conflated: locating a pressure-related direction and controlling pressure-induced behavior. Llama shows that the former is insufficient for the latter. The limitation panel should therefore be titled "locatable but not controllable" or equivalent, and the surrounding text should state that Llama is a weak replication / limitation.

**Limitation panel placeholder: Llama locatable-but-not-controllable.** Include projection-to-logit summary, weak alignment between projection and stance drift, and behavioral intervention outcome: small drift reduction, compliance unchanged, recovery down, damage up.

## 8. Identity/profile Boundary

The `identity_profile` line further supports the paper's mechanism-separation thesis. It provides weak localization evidence, but the available causal intervention evidence is insufficient to support identity-specific mitigation. The correct interpretation is not that identity/profile pressure is a failed version of belief pressure. Rather, it appears to follow a less linearly steerable pathway under the tested setup.

This result sharpens the paper's claim. If belief-style pressure and identity/profile pressure were merely two surface forms of the same mechanism, then a belief-drift intervention would be expected to transfer more cleanly. The current evidence does not support that. The identity/profile result therefore belongs in the main narrative as a boundary on generalization, with detailed evidence placed in the appendix.

## 9. Limitations

This paper has several important limitations. First, the Qwen 3B and Qwen 7B mainline uses an objective-local proxy mapping defined in the statistical closure notes. These results are the formal mainline for the present evidence package, but they should not be merged into a single leaderboard with bridge causal transfer lines. The paper should therefore report the mapping explicitly and avoid comparing proxy and bridge settings as if they were identical measurements.

Second, the cross-scale and cross-family evidence is supportive but bounded. Qwen 14B provides secondary causal confirmation, but the evidence is not positioned as a new primary mitigation line. Its sample size is smaller, recovery delta is 0, and baseline damage is nonzero. GLM-4-9B reproduces the intervention direction across model families, but with substantially stronger baseline damage. This makes GLM an important cross-family support result, not a no-cost deployment result.

Third, Llama-3.1-8B is a limitation rather than a positive replication. Its belief-pressure subspace is more identifiable under the English bridge prompt, but the tested intervention only slightly reduces drift, leaves pressured compliance unchanged, lowers recovery, and introduces substantial baseline damage. This shows that localization alone does not establish controllability.

Fourth, the identity/profile line remains unresolved. The current evidence supports a boundary claim: identity/profile pressure has localization observations but insufficient causal intervention support. It should not be described as identity-specific mitigation, and the paper should not imply that the belief-pressure damping method solves identity/profile pressure.

Fifth, Mistral-7B is appendix exploratory only. Its directional signals are useful for documenting the search boundary, but they are not stable enough to carry a main-text replication claim. Similarly, late-layer residual subtraction should remain a method-boundary or negative-control family, not the paper's main usable intervention.

Taken together, these limitations are not peripheral caveats; they are part of the paper's core argument. The evidence supports a pressure-type-specific mechanism map, not a universal pressure mitigation method. The strongest supported claim is that belief-style pressure admits a transferable and partially intervenable late-layer drift in the Qwen mainline, with bounded support from Qwen 14B and GLM. The evidence does not support universal transfer, positive Llama replication, or solved identity/profile mitigation.

## 10. Conclusion

This paper argues that pressure-induced sycophancy is mechanistically non-uniform. Belief-style pressure admits a transferable and partially intervenable late-layer drift in the Qwen mainline, with bounded cross-scale and cross-family support. Identity/profile pressure, however, does not follow the same intervention pattern, and Llama shows that a locatable pressure direction need not be a controllable one.

The resulting picture is not universal pressure mitigation, but a more precise mechanistic map. Different pressure types require different causal accounts. Belief-style pressure can be partially controlled in the tested Qwen settings, while identity/profile pressure remains less linearly steerable, and Llama marks the boundary between identifying a pressure-related direction and reliably using it for behavioral intervention.

## Appendix Placement Notes

Mistral-7B should remain appendix exploratory. The available evidence shows directional belief-subspace damping signals under an English bridge prompt, but these signals are unstable across prompt variants and accompanied by high baseline damage. The main text may mention Mistral briefly as exploratory appendix evidence, but it should not be counted as formal cross-family replication and should not be used to strengthen the central causal claim.

Appendix sections should also preserve detailed statistical closure notes, Qwen 7B/3B default-setting justification, Qwen 14B secondary confirmation, GLM tradeoff details, Llama projection-to-logit diagnostics, residual subtraction as method-boundary evidence, and the `identity_profile` localization/intervention boundary.

