# White-box Mechanistic Paper Rewrite Plan

【FROM: 论文撰写部门】
【TO: 用户】

Status: structural rewrite plan after external reviewer feedback and result-analysis availability check.

Primary draft:

- `docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex`

Evidence availability check received from:

- 结果分析部门

## 1. Revised Paper Positioning

The paper should be rewritten as a mechanistic heterogeneity / failure-boundary paper, not a universal mitigation paper.

Old over-strong center:

> Pressure is not one mechanism.

Revised main claim:

> We provide evidence for pressure-type-specific linear steerability: belief-style pressure admits a partially intervenable late-layer drift in the Qwen mainline, while identity/profile pressure and Llama transfer reveal limits of treating pressure as a single steerable mechanism.

Short version:

> Pressure-induced sycophancy is not uniformly linearly steerable.

Core framing changes:

- Say "provide evidence", not "prove".
- Say "under the tested setup".
- Say "partially intervenable" and "linear steerability profile".
- Treat Llama as a scientific boundary: localization does not imply controllability.
- Treat identity/profile as a boundary on generalization, not as proven separate mechanism.

## 2. Main Structural Rewrite

Use this structure for the next LaTeX revision:

1. Introduction
   - Pressure-induced sycophancy is behaviorally similar but mechanistically heterogeneous.
   - Main contribution: pressure-type-specific linear steerability.
   - Summary: Qwen positive mainline, GLM tradeoff, Llama limitation, identity/profile boundary.

2. Related Work
   - Behavioral sycophancy evaluation.
   - Activation steering and representation engineering.
   - Mechanistic localization versus causal control.
   - Persona and identity conditioning.
   - Positioning of this work.

3. Task Setup and Pressure Types
   - Bridge benchmark sources and taxonomy.
   - `belief_argument`.
   - `identity_profile`.
   - Baseline / pressured / recovery prompt pairing.
   - Prompt examples.
   - Objective-local proxy subset and its limits.

4. Methods
   - Models and decoding setup.
   - Activation collection.
   - Intervention methods.
   - Layer / strength selection.
   - Metrics.
   - Statistical inference.

5. Qwen Mainline Results
   - Qwen 7B formal mainline.
   - Qwen 3B formal replication.
   - Figure 2.
   - Objective-local proxy caveat.

6. Controls and Robustness
   - Use existing evidence first: subtraction controls, sanity control, stability sweeps.
   - Put random/shuffled/non-pressure controls only if Code/Analysis later confirms outputs.
   - Layer and strength sweep should be appendix-forward unless Analysis produces clean main-text summary.

7. Cross-model Boundary
   - Qwen 14B secondary causal confirmation.
   - GLM cross-family directionality with stronger tradeoff.
   - Llama locatable-but-not-controllable.
   - Figure 3 and Figure 4.

8. Identity/Profile as a Boundary Case
   - Task definition.
   - Localization observation.
   - Prefix-specific intervention not stable beyond matched controls.
   - No identity-specific mitigation claim.

9. Human / Manual Audit
   - Use existing bridge rationale conformity audit if sufficient.
   - Do not invent a new human audit result.
   - If current audit is not aligned with stance/compliance/recovery labels, present only as limited audit evidence and request a new audit design from Architecture/Code.

10. Limitations

11. Conclusion

Appendix:

- Prompt templates.
- Dataset statistics.
- Model details.
- Full effect-size table.
- Layer / strength sweeps.
- Controls and robustness.
- Audit rubric / available audit results.
- Identity/profile details.
- Llama diagnostics.
- Mistral exploratory results.
- Residual subtraction boundary results.

## 3. Figure Decisions

### Figure 1

Keep as conceptual mechanism map.

Content:

- `belief_argument` -> late-layer drift -> baseline-state interpolation / belief-subspace damping -> partial behavioral correction.
- `identity_profile` -> localization observation -> prefix-specific intervention not stable beyond matched controls -> boundary / unresolved.
- Guardrail: localization is not equivalent to controllability.

Owner:

- 论文撰写部门 can draft figure spec.
- 结果分析部门 or 文件整理部门 should generate final graphic asset.

### Figure 2

Use Qwen 3B / 7B only.

Data:

- `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/whitebox_effect_size_table.csv`

Caption must say:

- Qwen 7B is formal mainline.
- Qwen 3B is formal replication.
- Metrics use objective-local proxy mapping.
- Not bridge-causal leaderboard equivalent.

### Figure 3

Decision: split into panels, not a single unqualified cross-model plot.

Recommended design:

- Panel A: Qwen 3B/7B objective-local proxy summary.
- Panel B: Qwen 14B / GLM / Llama bridge causal or transfer-style summary.
- Optional appendix-only marker for Mistral.

Reason:

- Result-analysis confirmed that combining proxy and bridge causal metrics into one unqualified plot would invite leaderboard-style misreading.

### Figure 4

Use Llama as "localization vs controllability" figure.

Data:

- `outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/llama_projection_alignment_summary.json`
- `outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011/llama_projection_alignment_diagnostic.csv`
- `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/llama_limitation_summary.md`

Message:

> A pressure-related direction can be detectable without being behaviorally useful.

## 4. Controls Decision

Result-analysis found existing material for:

- 3B / 7B subtraction controls.
- 7B sanity control.
- 3B / 7B stability runs.
- 3B / 7B layer / strength sweep appendix material.
- Llama alpha sweep.

Writing decision:

- Do not claim all requested negative controls exist.
- Add a `Controls and Robustness` section using existing controls.
- Put detailed sweeps in appendix unless result-analysis produces clean figure/table summaries.
- If reviewers require random direction / shuffled label / matched non-pressure controls, route to Architecture and Code.

## 5. Method Material Available Now

Usable now:

- Bridge manifest:
  - `outputs/experiments/bridge_benchmark_protocol/formal_bridge_protocol/20260423_160641/bridge_benchmark_manifest.json`
- Prompt examples:
  - `outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308/pressure_pairs_train.jsonl`
- Statistical closure definitions:
  - `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/README.md`
- Effect-size table:
  - `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452/whitebox_effect_size_table.csv`

Still needs confirmation from Architecture/Code if required:

- Exact decoding parameters for all model runs.
- Random seed policy.
- Bootstrap resample count and CI implementation details.
- Exact activation site / token position for each intervention run.
- Whether held-out templates were used for the Qwen mainline.
- Whether random direction / shuffled label controls exist or must be run.

## 6. Immediate Writing Plan

Next paper-writing pass should:

1. Rewrite title/abstract/introduction around "not uniformly linearly steerable".
2. Insert `Task Setup and Pressure Types`.
3. Insert `Methods` with clear "known from current artifacts" wording and no invented parameters.
4. Move current Qwen content into `Qwen Mainline Results`.
5. Add `Controls and Robustness` with existing controls only.
6. Rewrite Llama as central boundary result.
7. Rewrite identity/profile as boundary case.
8. Replace local-path-only appendix with formal appendix sections.

Do not yet:

- Claim random direction / shuffled label controls unless confirmed.
- Claim a full human audit matching stance/compliance/recovery unless confirmed.
- Merge Qwen proxy and bridge causal metrics into one leaderboard plot.

