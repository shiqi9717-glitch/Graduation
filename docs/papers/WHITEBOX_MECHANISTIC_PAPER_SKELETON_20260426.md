# White-box Mechanistic Paper Skeleton

【FROM: 论文撰写部门】
【TO: 用户】

Status: first-stage paper skeleton and main narrative draft. This file is based only on the completed white-box mechanistic evidence dossier and statistical closure outputs; it does not introduce new experiments or change result framing.

Primary input:

- `/Users/shiqi/code/graduation-project/docs/reports/WHITEBOX_MECHANISTIC_EVIDENCE_DOSSIER_20260426.md`

Statistical closure directory:

- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`

Priority evidence files:

- `whitebox_effect_size_table.csv`
- `llama_limitation_summary.md`
- `llama_limitation_summary.json`
- `README.md`

## 0. Main Thesis And Claim Boundaries

Main thesis:

> Pressure is not one mechanism: belief-style pressure admits a transferable and partially intervenable late-layer drift, while identity/profile pressure follows a different and less linearly steerable pathway.

Chinese narrative version:

> 本文不把 pressure 视为一个统一机制，而是区分 belief-style pressure 与 identity/profile pressure：前者在 Qwen 主线中表现为可迁移、可部分干预的 late-layer drift；后者虽有 localization observation，但缺乏稳定的 prefix-specific causal intervention 支撑，因此更像另一条不易被线性方向稳定控制的路径。

Hard prohibitions:

- Do not claim universal pressure mitigation.
- Do not claim positive Llama replication.
- Do not treat the Qwen 3B/7B objective-local proxy and bridge causal lines as a single leaderboard.
- Do not present `identity_profile` as a solved mitigation problem.
- Do not upgrade Mistral exploratory signals into formal cross-family replication.
- Do not present `late_layer_residual_subtraction` as the main usable method.

## 1. Candidate Titles

1. **Pressure Is Not One Mechanism: Separating Belief Drift from Identity/Profile Pathways in White-box Sycophancy**
2. **Mechanistic Separation of Pressure Types in Sycophantic Behavior**
3. **Late-layer Belief Drift Is Partially Controllable, but Identity Pressure Is Not**
4. **White-box Evidence for Pressure-type-specific Mechanisms in Sycophancy**
5. **From Belief Drift to Identity Pressure: A Mechanistic Account of Non-uniform Sycophancy**

Recommended working title:

> **Pressure Is Not One Mechanism: Separating Belief Drift from Identity/Profile Pathways in White-box Sycophancy**

Reason: it foregrounds the paper's strongest novelty without claiming first-ever steering or universal mitigation.

## 2. Abstract Draft

Sycophantic behavior in language models is often discussed as a single failure mode: a model changes its answer or stance under user pressure. This paper argues that this framing is too coarse. Using a white-box mechanistic evidence package across Qwen, GLM, Llama, and Mistral variants, we show that different pressure types follow separable internal pathways. For belief-style pressure, Qwen 7B and Qwen 3B provide the formal mainline evidence: a late-layer belief-pressure drift can be detected and partially damped under the default `baseline_state_interpolation` setting, reducing stance drift and pressured compliance while preserving recovery and introducing no measured baseline damage under the objective-local proxy mapping. Qwen 14B provides secondary causal confirmation, and GLM-4-9B reproduces the damping direction across model families, albeit with a substantially stronger damage tradeoff. In contrast, identity/profile pressure shows weaker causal intervention support: localization evidence exists, but prefix-specific intervention does not stably separate from matched controls. Llama-3.1-8B further clarifies the boundary of the claim: its belief-pressure subspace is locatable under an English bridge prompt, but intervention transfer remains weak, leaving pressured compliance unchanged while reducing recovery and increasing baseline damage. Together, these results support a pressure-type-specific account of sycophancy: belief-style pressure admits a transferable and partially intervenable late-layer drift, whereas identity/profile pressure is not yet explained by the same linear steering story.

Shorter abstract option:

> We provide white-box evidence that pressure-induced sycophancy is mechanistically non-uniform. Belief-style pressure admits a transferable and partially intervenable late-layer drift in Qwen models, with secondary support from Qwen 14B and GLM under stronger tradeoffs. Identity/profile pressure, however, follows a less linearly steerable pathway, and Llama acts as a locatable-but-not-controllable limitation rather than a positive replication.

## 3. Core Contributions

1. **Pressure-type-specific mechanism separation.** We distinguish belief-style pressure from identity/profile pressure and show that they should not be collapsed into one generic pressure mechanism.

2. **Formal Qwen mainline evidence for partially intervenable belief drift.** Under the default `baseline_state_interpolation` settings, Qwen 7B and Qwen 3B reduce stance drift and pressured compliance with positive recovery and zero measured baseline damage under the statistical closure's objective-local proxy mapping.

3. **Cross-scale and cross-family support with explicit tradeoffs.** Qwen 14B provides secondary causal confirmation, while GLM-4-9B reproduces the belief-subspace damping direction across families but with much stronger baseline damage.

4. **A boundary case against overgeneralization.** Llama-3.1-8B is locatable but not controllable under the tested intervention: it slightly reduces drift, does not reduce compliance, lowers recovery, and introduces substantial baseline damage.

## 4. Main Narrative Arc

Opening problem:

- Sycophancy is usually measured behaviorally, but behavioral labels can hide multiple internal mechanisms.
- "Pressure" is an especially overloaded category: belief arguments, identity/profile cues, and other social signals may all change responses, but they need not share a causal pathway.

Paper turn:

- We use white-box intervention evidence to test whether pressure behaves like one steerable mechanism or multiple mechanism families.
- The answer is asymmetric: belief-style pressure has a partially controllable late-layer drift; identity/profile pressure does not inherit that conclusion.

Central result:

- Qwen 7B and Qwen 3B form the mainline evidence.
- Qwen 14B and GLM support transfer while clarifying scale/family boundaries.
- Llama and identity/profile results prevent overclaiming.

Discussion landing:

- The novelty is not "first sycophancy steering."
- The novelty is pressure-type-specific mechanism separation: belief pressure and identity/profile pressure differ in internal representation and intervention response.

## 5. Section Structure

### 1. Introduction

Purpose:

- Frame pressure-induced sycophancy as a mechanistic, not merely behavioral, problem.
- Introduce the core distinction between belief-style pressure and identity/profile pressure.
- State the main thesis and limitations upfront.

Key claims:

> Pressure-induced sycophancy is not mechanistically uniform.

> Belief-style pressure admits a transferable and partially intervenable late-layer drift.

> Identity/profile pressure should not be treated as already mitigated by the same intervention.

Evidence sources:

- Dossier Section 1: Main Claim
- Dossier Section 2: R1-R3
- Statistical closure README for metric definitions and Qwen 3B/7B proxy mapping

Result directories to cite:

- `/Users/shiqi/code/graduation-project/docs/reports/WHITEBOX_MECHANISTIC_EVIDENCE_DOSSIER_20260426.md`
- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`

### 2. Background And Related Work

Purpose:

- Position this paper against behavioral sycophancy evaluation, activation steering, representation editing, and causal tracing/intervention work.
- Emphasize that the paper's novelty is not simply steering sycophancy, but separating pressure types mechanistically.

Differentiation language:

> Prior work has shown that model behavior can sometimes be shifted by activation-level interventions. Our focus is different: we ask whether different forms of pressure instantiate the same internal mechanism. The evidence suggests they do not.

> We therefore treat sycophancy mitigation as pressure-type-specific rather than as a single universal steering target.

Do not say:

- "We are the first to steer sycophancy."
- "This is a universal mitigation for pressure."

Evidence sources:

- Use existing literature notes only as background:
  - `/Users/shiqi/code/graduation-project/docs/papers/literature_summary.md`
  - `/Users/shiqi/code/graduation-project/docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`

### 3. Setup: Pressure Types, Metrics, And Evidence Levels

Purpose:

- Define pressure types and explain why belief-style pressure and identity/profile pressure are analyzed separately.
- Define metric meanings from statistical closure.
- Explain evidence levels: formal mainline, formal replication, secondary causal confirmation, cross-family replication with tradeoff, weak replication/limitation, exploratory appendix.

Key metric definitions:

- `stance_drift_delta = intervention_rate - reference_rate`; negative is better.
- `pressured_compliance_delta = intervention_rate - reference_rate`; negative is better.
- `recovery_delta = intervention_rate - reference_rate`; positive is better.
- `baseline_damage_rate`; lower is better.
- For Qwen 3B/7B, stance drift and pressured compliance use the objective-local proxy mapping defined in the statistical closure README.

Important wording:

> For Qwen 3B and 7B, the mainline metrics are objective-local proxy measurements, not a leaderboard-equivalent bridge causal score.

Evidence sources:

- Statistical closure README
- `whitebox_effect_size_table.csv`
- Dossier Section 3: Evidence-Level Matrix

Result directory:

- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`

### 4. Mechanistic Map: Pressure Is Not One Mechanism

Purpose:

- Present the conceptual mechanism split.
- Use Figure 1 to map two branches:
  - belief-style pressure -> late-layer drift -> partially intervenable subspace damping
  - identity/profile pressure -> localization observation -> insufficient prefix-specific causal control

Key claim:

> Belief-style pressure and identity/profile pressure should be treated as separate mechanistic hypotheses, not as variants of one already-solved pressure direction.

Evidence sources:

- Dossier R1
- identity_profile directory:
  - `/Users/shiqi/code/graduation-project/outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058`

Main figure:

- Figure 1: White-box mechanism map

### 5. Mainline Result: Qwen Belief-pressure Damping

Purpose:

- Present Qwen 7B and Qwen 3B as the formal mainline.
- Use Figure 2 and Table 1 to show effect sizes and evidence levels.

Qwen 7B result:

> Under the default `baseline_state_interpolation` setting over layers 24-26 at strength 0.6, Qwen 7B reduces stance drift by -0.1493 (CI [-0.2200, -0.0800]) and pressured compliance by -0.1595 (CI [-0.2300, -0.0900]), while increasing recovery by 0.7127 (CI [0.5000, 0.9000]) with zero measured baseline damage.

Qwen 3B result:

> Under the default `baseline_state_interpolation` setting over layers 31-35 at strength 0.6, Qwen 3B reduces stance drift by -0.1597 (CI [-0.2402, -0.0800]) and pressured compliance by -0.2789 (CI [-0.3700, -0.1900]), while increasing recovery by 0.6906 (CI [0.5000, 0.8667]) with zero measured baseline damage.

Guardrail wording:

> These results establish the formal Qwen mainline under the objective-local proxy mapping; they should not be read as a cross-paradigm leaderboard against bridge causal lines.

Evidence sources:

- `whitebox_effect_size_table.csv`
- Statistical closure README
- Qwen 7B directory:
  - `/Users/shiqi/code/graduation-project/outputs/experiments/local_probe_qwen7b_intervention_main/baseline_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140142`
- Qwen 3B directory:
  - `/Users/shiqi/code/graduation-project/outputs/experiments/local_probe_qwen3b_intervention_main/baseline_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142847`

Main figure:

- Figure 2: Qwen mainline intervention effect

### 6. Cross-scale And Cross-family Support

Purpose:

- Show that the belief-pressure damping direction has support beyond the primary 3B/7B Qwen mainline.
- Keep the support carefully bounded.

Qwen 14B wording:

> Qwen 14B provides secondary causal confirmation rather than a new main mitigation line: matched belief-subspace damping reduces drift by -0.2068 (CI [-0.4167, 0.0000]) and pressured compliance by -0.2484 (CI [-0.4167, -0.0833]), with recovery delta 0 and baseline damage 0.0416.

GLM wording:

> GLM-4-9B reproduces the belief-subspace damping direction across model families, reducing drift by -0.2932 (CI [-0.5417, -0.0417]) and pressured compliance by -0.2095 (CI [-0.4167, 0.0000]), but this comes with substantial baseline damage of 0.3325 (CI [0.1667, 0.5417]).

Evidence sources:

- `whitebox_effect_size_table.csv`
- Qwen 14B directory:
  - `/Users/shiqi/code/graduation-project/outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308`
- GLM directory:
  - `/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_glm4_9b/Users_shiqi_.cache_huggingface_hub_models--zai-org--glm-4-9b-chat-hf_snapshots_8599336fc6c125203efb2360bfaf4c80eef1d1bf/20260426_005017`

Main table:

- Table 2: Cross-scale and cross-family support

### 7. Limitation: Llama Is Locatable But Not Controllable

Purpose:

- Make Llama the central boundary panel.
- Prevent the paper from claiming universal transfer or universal pressure mitigation.

Core wording:

> Llama-3.1-8B shows a more identifiable belief-pressure subspace under the English bridge prompt, but intervention transfer remains weak: the strongest tested damping setting only slightly reduces drift while leaving pressured compliance unchanged, lowering recovery, and introducing substantial baseline damage. We therefore treat Llama as a weak replication / limitation rather than a positive replication.

Effect-size wording:

> In the statistical closure, Llama reduces drift only by -0.0824 (CI [-0.2917, 0.1250]), leaves pressured compliance unchanged, reduces recovery by -0.0847 (CI [-0.2917, 0.1250]), and incurs baseline damage of 0.2499 (CI [0.0833, 0.4167]).

Evidence sources:

- `llama_limitation_summary.md`
- `llama_limitation_summary.json`
- `whitebox_effect_size_table.csv`
- Llama directory:
  - `/Users/shiqi/code/graduation-project/outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011`

Main panel:

- Limitation panel: Llama locatable-but-not-controllable

### 8. Identity/profile Boundary

Purpose:

- State clearly that identity/profile pressure is not solved by the belief-pressure damping story.
- Use as a mechanism-separation argument, not as a failed side quest.

Core wording:

> The identity-profile line provides weak localization evidence but insufficient causal intervention support, and should not be interpreted as identity-specific mitigation.

Interpretation:

- This strengthens the main thesis because it shows that not all pressure-like behavioral effects share the belief-drift intervention response.
- It also prevents the method from being oversold as a general pressure mitigation technique.

Evidence source:

- `/Users/shiqi/code/graduation-project/outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058`

Placement:

- Main text: short boundary paragraph.
- Appendix: detail of localization observation and matched-control limitations.

### 9. Appendix-only Exploratory Evidence

Purpose:

- Preserve Mistral and negative/control lines without letting them define the main claim.

Mistral wording:

> Mistral-7B showed directional belief-subspace damping signals under an English bridge prompt, but the effect was unstable across prompt variants and accompanied by high baseline damage; we therefore treat it as exploratory only.

Subtraction control wording:

> Late-layer residual subtraction is treated as a method boundary or negative-control family rather than as the paper's usable intervention.

Evidence sources:

- Mistral directories:
  - `/Users/shiqi/code/graduation-project/outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_143645`
  - `/Users/shiqi/code/graduation-project/outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_144430`
- Statistical closure directory:
  - `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`

### 10. Discussion

Purpose:

- Tie together pressure-type-specific mechanism separation, partial controllability, and non-universal transfer.

Main discussion claims:

> The evidence supports a pressure-type-specific account of sycophancy rather than a single pressure mechanism.

> Belief-style pressure is more linearly steerable in the tested Qwen mainline than identity/profile pressure.

> Cross-scale and cross-family support exists, but transfer is bounded by model family and damage tradeoffs.

> Llama demonstrates that localization is not equivalent to controllability.

### 11. Limitations

Required limitations:

- Qwen 3B/7B mainline uses objective-local proxy mapping, not a bridge-causal leaderboard metric.
- Qwen 14B is secondary confirmation with small `n=24`, recovery delta 0, and nonzero damage.
- GLM has positive directional replication but stronger baseline damage.
- Llama is weak replication / limitation, not positive replication.
- Mistral is appendix exploratory only.
- identity_profile is a boundary case, not solved mitigation.
- The paper does not claim universal pressure mitigation.

### 12. Conclusion

Draft:

> This paper argues that pressure-induced sycophancy is mechanistically non-uniform. Belief-style pressure admits a transferable and partially intervenable late-layer drift in the Qwen mainline, with bounded cross-scale and cross-family support. Identity/profile pressure, however, does not follow the same intervention pattern, and Llama shows that a locatable pressure direction need not be a controllable one. The resulting picture is not universal pressure mitigation, but a more precise mechanistic map: different pressure types require different causal accounts.

## 6. Expanded Main-text Draft v1

This section is a first prose expansion of the main paper. It keeps the evidence package frozen, does not introduce new experiment suggestions, and preserves the current evidence hierarchy: Qwen 3B/7B as the formal mainline, Qwen 14B as secondary causal confirmation, GLM as cross-family positive replication with stronger tradeoff, Llama as weak replication / limitation, and Mistral as appendix exploratory.

### 6.1 Introduction Draft

Language models often change their answers when a user applies pressure. In sycophancy evaluations, this behavior is commonly treated as a single phenomenon: a model appears to follow the user's stated preference, identity, or argument even when doing so conflicts with a more stable answer. This behavioral framing is useful for measurement, but it can obscure a mechanistic question that matters for both interpretation and intervention: are all pressure-induced shifts produced by the same internal mechanism?

This paper argues that the answer is no. We study pressure-induced sycophancy through a white-box evidence package that separates belief-style pressure from identity/profile pressure. Belief-style pressure refers to cases where the user supplies an argument or stance cue that pushes the model toward a pressure-aligned belief. Identity/profile pressure refers to cases where the user profile or identity frame changes the social context of the answer. Both can look like pressure at the behavioral level, but the evidence does not support treating them as one shared internal pathway.

Our main claim is:

> Pressure is not one mechanism: belief-style pressure admits a transferable and partially intervenable late-layer drift, while identity/profile pressure follows a different and less linearly steerable pathway.

The positive part of the claim is anchored in the Qwen mainline. Under the default `baseline_state_interpolation` settings, Qwen 7B and Qwen 3B show consistent reductions in stance drift and pressured compliance while preserving recovery and introducing no measured baseline damage under the objective-local proxy mapping defined in the statistical closure notes. These results support the view that belief-style pressure can induce a late-layer drift that is not merely detectable but partially intervenable.

The broader evidence is deliberately more bounded. Qwen 14B provides secondary causal confirmation, suggesting that the belief-pressure damping direction is not confined to one model size. GLM-4-9B gives cross-family support, but with a substantially stronger baseline-damage tradeoff. Llama-3.1-8B provides the main boundary case: the belief-pressure direction is more identifiable under an English bridge prompt, yet the tested intervention transfers weakly, leaving pressured compliance unchanged, reducing recovery, and introducing substantial baseline damage. Mistral-7B remains appendix exploratory rather than formal replication.

The negative part of the claim is equally important. The identity/profile line provides weak localization evidence but insufficient causal intervention support. We therefore do not claim identity-specific mitigation, and we do not treat identity/profile pressure as solved by the same linear belief-damping story. This distinction is the paper's central contribution: not a claim of universal pressure mitigation, and not a claim of first-ever sycophancy steering, but a pressure-type-specific account of when a white-box pressure direction is detectable, transferable, and behaviorally useful.

The paper proceeds as follows. Section 2 situates the work relative to behavioral sycophancy measurement and activation-level intervention work, emphasizing that our novelty lies in pressure-type separation rather than in generic steering. Section 3 defines the pressure types, metrics, and evidence levels used throughout the paper. Section 4 presents the white-box mechanism map. Section 5 reports the Qwen mainline. Section 6 summarizes cross-scale and cross-family support. Section 7 discusses the Llama limitation, and Section 8 explains the identity/profile boundary. The conclusion returns to the central lesson: localization is not the same as controllability, and pressure-induced sycophancy should not be treated as one mechanism.

### 6.2 Setup Draft: Pressure Types, Metrics, And Evidence Levels

We organize the evidence around pressure type rather than around a single aggregate sycophancy score. The first pressure type, `belief_argument`, captures settings where the user introduces a belief-like argument or stance cue. In this setting, the core mechanistic hypothesis is that pressure induces a late-layer representational drift that can be partially damped by moving the residual stream toward a baseline state or by damping a matched belief-pressure subspace. The second pressure type, `identity_profile`, captures settings where a user profile or identity context changes the social frame of the answer. For this pressure type, the evidence supports localization observations but does not yet support a stable causal intervention claim.

The primary behavioral quantities come from the frozen statistical closure. `stance_drift_delta` is defined as the intervention rate minus the reference rate, so negative values indicate less pressure-induced drift. `pressured_compliance_delta` is also intervention rate minus reference rate, so negative values indicate less pressure-aligned compliance. `recovery_delta` measures intervention recovery relative to the reference setup, where positive values indicate improved recovery. `baseline_damage_rate` measures damage on baseline behavior, where lower is better. We also use summary quantities such as `benefit_sum_no_damage` and `net_effect_score`, but the paper's main claims are made from the named behavioral deltas and their confidence intervals.

For the Qwen 3B and Qwen 7B mainline, the statistical closure uses a special objective-local proxy mapping. In this mapping, stance drift uses an interference-induced-error proxy on strict-positive, high-pressure wrong-option items; pressured compliance uses a wrong-option-follow proxy on the same local subset; and recovery equals the intervention recovery rate because the reference system has structural zero recovery under this intervention setup. The main text should therefore describe these as objective-local proxy results, not as a leaderboard-equivalent measure that can be directly merged with bridge causal transfer lines.

The evidence levels are also fixed. Qwen 7B is the formal mainline result, and Qwen 3B is the formal mainline replication. Qwen 14B is secondary causal confirmation, not a new primary mitigation line. GLM-4-9B is cross-family positive replication with a stronger tradeoff. Llama-3.1-8B is weak replication / limitation: it helps identify the boundary between locating a pressure direction and using it for reliable behavioral control. Mistral-7B is appendix exploratory only. The `identity_profile` line is a weak mechanistic observation / boundary case, not an identity-specific mitigation result.

This evidence hierarchy is central to the paper's interpretation. The paper does not ask whether every model admits the same steering direction. Instead, it asks whether pressure-induced behavior decomposes into mechanistically separable pathways, and whether one of those pathways, belief-style pressure, has a more linearly steerable late-layer component than identity/profile pressure.

### 6.3 Results Draft

#### 6.3.1 Mechanistic Map: Pressure Is Not One Mechanism

Figure 1 should present the mechanism map before the numerical results. The left branch corresponds to belief-style pressure. In this branch, pressure is modeled as inducing a late-layer drift away from a baseline answer state. The Qwen mainline tests whether dampening or interpolating this drift can reduce pressure-aligned behavior while preserving recovery. The right branch corresponds to identity/profile pressure. In that branch, localization evidence exists, but the available intervention evidence does not stably separate prefix-specific causal effects from matched controls.

This map establishes the paper's central interpretive frame. The goal is not to find a single pressure vector that explains all pressure-induced sycophancy. The goal is to show that pressure-like behavioral shifts can split into different mechanistic families. Belief-style pressure has a stronger case for a late-layer, partially intervenable drift. Identity/profile pressure remains a boundary case and should not be described as mitigated by the same intervention.

#### 6.3.2 Qwen Mainline: Belief-pressure Damping

The formal mainline evidence comes from Qwen 7B and Qwen 3B under the default `baseline_state_interpolation` settings. For Qwen 7B, the default setting uses layers 24-26 at strength 0.6. Under the objective-local proxy mapping, the intervention reduces stance drift by -0.1493 with confidence interval [-0.2200, -0.0800], and reduces pressured compliance by -0.1595 with confidence interval [-0.2300, -0.0900]. Recovery increases by 0.7127 with confidence interval [0.5000, 0.9000], while measured baseline damage is 0.

Qwen 3B provides the formal replication under layers 31-35 at strength 0.6. It reduces stance drift by -0.1597 with confidence interval [-0.2402, -0.0800], and pressured compliance by -0.2789 with confidence interval [-0.3700, -0.1900]. Recovery increases by 0.6906 with confidence interval [0.5000, 0.8667], again with zero measured baseline damage.

Together, these results support the main positive claim: in the tested Qwen settings, belief-style pressure admits a late-layer drift that is both detectable and partially intervenable. Figure 2 should show the Qwen 7B and Qwen 3B effects side by side, separating stance drift, pressured compliance, recovery, and baseline damage. The caption should explicitly state that these are objective-local proxy measurements under the statistical closure mapping.

The interpretation should remain narrow. These results do not establish universal pressure mitigation, and they should not be presented as directly comparable to every bridge causal transfer setting. They establish a formal Qwen mainline for belief-style pressure under the specified objective-local mapping.

#### 6.3.3 Cross-scale And Cross-family Support

Qwen 14B extends the Qwen evidence across scale, but it should be framed as secondary causal confirmation rather than a new main mitigation line. Under matched belief-subspace damping against no intervention, Qwen 14B reduces drift by -0.2068 with confidence interval [-0.4167, 0.0000], and reduces pressured compliance by -0.2484 with confidence interval [-0.4167, -0.0833]. Its recovery delta is 0, and baseline damage is 0.0416. This supports the direction of the belief-pressure intervention while keeping the main claim anchored in the 3B/7B formal line.

GLM-4-9B provides cross-family evidence. In the representative transfer setting, it reduces stance drift by -0.2932 with confidence interval [-0.5417, -0.0417], and pressured compliance by -0.2095 with confidence interval [-0.4167, 0.0000]. It also improves recovery by 0.3769 with confidence interval [0.1667, 0.5833]. However, this support comes with substantial baseline damage of 0.3325 with confidence interval [0.1667, 0.5417]. GLM should therefore be written as cross-family positive replication with stronger tradeoff, not as a clean no-damage replication.

Table 2 should collect Qwen 14B and GLM together. This table should make the asymmetry visible: the belief-pressure damping direction transfers beyond the primary Qwen 3B/7B settings, but the quality of transfer depends on model family and damage profile.

#### 6.3.4 Limitation Result: Llama Is Locatable But Not Controllable

Llama-3.1-8B is the paper's most important limitation case. The projection-to-logit summary indicates that the pressured-stage belief projection is stronger than baseline, so a belief-pressure direction is locatable under the English bridge prompt. However, this localization does not translate into reliable behavioral control. Projection magnitude is weakly and negatively correlated with stance drift, and the tested belief damping moves logits in the targeted direction while remaining entangled with the baseline manifold.

The behavioral closure makes the limitation concrete. In the representative weak-replication setting, Llama reduces stance drift only by -0.0824 with confidence interval [-0.2917, 0.1250], leaves pressured compliance unchanged, reduces recovery by -0.0847 with confidence interval [-0.2917, 0.1250], and incurs baseline damage of 0.2499 with confidence interval [0.0833, 0.4167]. This is not a positive replication of the Qwen/GLM pattern.

The main text should use Llama to separate two concepts that are often conflated: locating a pressure-related direction and controlling pressure-induced behavior. Llama shows that the former is insufficient for the latter. The limitation panel should therefore be titled "locatable but not controllable" or equivalent, and the surrounding text should state that Llama is a weak replication / limitation.

#### 6.3.5 Identity/profile Boundary

The `identity_profile` line further supports the paper's mechanism-separation thesis. It provides weak localization evidence, but the available causal intervention evidence is insufficient to support identity-specific mitigation. The correct interpretation is not that identity/profile pressure is a failed version of belief pressure. Rather, it appears to follow a less linearly steerable pathway under the tested setup.

This result should be used to sharpen the paper's claim. If belief-style pressure and identity/profile pressure were merely two surface forms of the same mechanism, then a belief-drift intervention would be expected to transfer more cleanly. The current evidence does not support that. The identity/profile result therefore belongs in the main narrative as a boundary on generalization, with detailed evidence placed in the appendix.

#### 6.3.6 Appendix Exploratory Evidence: Mistral

Mistral-7B should remain appendix exploratory. The available evidence shows directional belief-subspace damping signals under an English bridge prompt, but these signals are unstable across prompt variants and accompanied by high baseline damage. The main text may mention Mistral briefly as exploratory appendix evidence, but it should not be counted as formal cross-family replication and should not be used to strengthen the central causal claim.

### 6.4 Limitations Draft

This paper has several important limitations. First, the Qwen 3B and Qwen 7B mainline uses an objective-local proxy mapping defined in the statistical closure notes. These results are the formal mainline for the present evidence package, but they should not be merged into a single leaderboard with bridge causal transfer lines. The paper should therefore report the mapping explicitly and avoid comparing proxy and bridge settings as if they were identical measurements.

Second, the cross-scale and cross-family evidence is supportive but bounded. Qwen 14B provides secondary causal confirmation, but the evidence is not positioned as a new primary mitigation line. Its sample size is smaller, recovery delta is 0, and baseline damage is nonzero. GLM-4-9B reproduces the intervention direction across model families, but with substantially stronger baseline damage. This makes GLM an important cross-family support result, not a no-cost deployment result.

Third, Llama-3.1-8B is a limitation rather than a positive replication. Its belief-pressure subspace is more identifiable under the English bridge prompt, but the tested intervention only slightly reduces drift, leaves pressured compliance unchanged, lowers recovery, and introduces substantial baseline damage. This shows that localization alone does not establish controllability.

Fourth, the identity/profile line remains unresolved. The current evidence supports a boundary claim: identity/profile pressure has localization observations but insufficient causal intervention support. It should not be described as identity-specific mitigation, and the paper should not imply that the belief-pressure damping method solves identity/profile pressure.

Fifth, Mistral-7B is appendix exploratory only. Its directional signals are useful for documenting the search boundary, but they are not stable enough to carry a main-text replication claim. Similarly, late-layer residual subtraction should remain a method-boundary or negative-control family, not the paper's main usable intervention.

Taken together, these limitations are not peripheral caveats; they are part of the paper's core argument. The evidence supports a pressure-type-specific mechanism map, not a universal pressure mitigation method. The strongest supported claim is that belief-style pressure admits a transferable and partially intervenable late-layer drift in the Qwen mainline, with bounded support from Qwen 14B and GLM. The evidence does not support universal transfer, positive Llama replication, or solved identity/profile mitigation.

## 7. Main-text Figure And Table Plan

### Figure 1: White-box Mechanism Map

Goal:

- Show pressure-type separation as the conceptual anchor.

Panel design:

- Left branch: `belief_argument` -> late-layer drift -> baseline-state interpolation / belief-subspace damping -> partial behavioral correction.
- Right branch: `identity_profile` -> localization observation -> insufficient prefix-specific causal intervention -> boundary/limitation.
- Bottom guardrail: localization is not equivalent to controllability.

Evidence:

- Dossier R1
- identity_profile follow-up directory
- Qwen 3B/7B mainline directories

### Figure 2: Qwen Mainline Intervention Effect

Goal:

- Show Qwen 7B and Qwen 3B intervention effects with confidence intervals.

Metrics:

- stance drift delta
- pressured compliance delta
- recovery delta
- baseline damage rate as annotation or separate small bar

Data:

- `whitebox_effect_size_table.csv`
- Qwen 7B: stance drift -0.1493, compliance -0.1595, recovery 0.7127, damage 0
- Qwen 3B: stance drift -0.1597, compliance -0.2789, recovery 0.6906, damage 0

Caption guardrail:

> Metrics follow the objective-local proxy mapping specified in the statistical closure notes and are not treated as leaderboard-equivalent to bridge causal lines.

### Table 1: Evidence-level Matrix

Columns:

- Subline
- Evidence level
- Model/family
- Main-text or appendix placement
- Key supported claim
- Required boundary

Rows:

- Qwen 7B: formal mainline
- Qwen 3B: formal replication
- Qwen 14B: secondary causal confirmation
- GLM-4-9B: cross-family positive replication with stronger tradeoff
- Llama-3.1-8B: weak replication / limitation
- Mistral-7B: appendix exploratory
- identity_profile: weak mechanistic observation / boundary

Data source:

- Dossier Section 3
- `whitebox_effect_size_table.csv`

### Table 2: Cross-scale And Cross-family Support

Rows:

- Qwen 14B
- GLM-4-9B

Columns:

- comparison
- stance drift delta + CI
- pressured compliance delta + CI
- recovery delta + CI
- baseline damage + CI
- interpretation

Required interpretation:

- Qwen 14B: secondary causal confirmation, not new mainline.
- GLM: cross-family direction holds, stronger tradeoff.

### Limitation Panel: Llama Locatable-but-not-controllable

Goal:

- Visually separate localization from causal control.

Panel content:

- Projection-to-logit summary: pressured-stage belief projection stronger than baseline (9.19 vs 5.53).
- Weak alignment: projection magnitude weakly and negatively correlated with stance drift.
- Behavioral intervention outcome: small drift reduction, compliance unchanged, recovery down, damage up.

Data:

- `llama_limitation_summary.md`
- `llama_limitation_summary.json`
- `whitebox_effect_size_table.csv`

Caption:

> Llama is a weak replication / limitation: the belief-pressure subspace is more identifiable under the English bridge prompt, but the tested intervention does not provide reliable behavioral control.

## 8. Appendix Plan

Appendix A: Statistical closure notes and metric definitions.

Appendix B: Full Qwen 7B and Qwen 3B mainline details, including why the listed default settings are the formal baseline.

Appendix C: Aggressive or non-default Qwen variants and why they are not the main claim.

Appendix D: `late_layer_residual_subtraction` as negative-control / method-boundary evidence.

Appendix E: Qwen 14B secondary causal confirmation.

Appendix F: GLM cross-family replication with damage tradeoff.

Appendix G: Llama weak replication / limitation, including projection-to-logit diagnostics.

Appendix H: Mistral exploratory bridge-prompt note.

Appendix I: identity_profile localization observation and intervention boundary.

## 9. Claim Bank For Direct Expansion

Use:

> Pressure is not one mechanism: belief-style pressure admits a transferable and partially intervenable late-layer drift, while identity/profile pressure follows a different and less linearly steerable pathway.

Use:

> Qwen 7B and Qwen 3B provide the formal mainline evidence under the default `baseline_state_interpolation` settings, with effect sizes reported under the objective-local proxy mapping defined in the statistical closure notes.

Use:

> Qwen 14B provides secondary causal confirmation rather than a new main mitigation line.

Use:

> GLM reproduces the belief-subspace damping direction across model families, but with a substantially stronger utility/safety tradeoff.

Use:

> Llama-3.1-8B is best interpreted as a weak replication / limitation: the belief-pressure subspace is more identifiable under the English bridge prompt, but intervention transfer remains weak.

Use:

> The identity-profile line provides weak localization evidence but insufficient causal intervention support, and should not be interpreted as identity-specific mitigation.

Avoid:

> This intervention solves pressure-induced sycophancy.

Avoid:

> Llama positively replicates the Qwen/GLM result.

Avoid:

> Identity/profile pressure is mitigated by the same belief-damping direction.

Avoid:

> Qwen objective-local proxy results and bridge causal transfer results form a single unified leaderboard.
