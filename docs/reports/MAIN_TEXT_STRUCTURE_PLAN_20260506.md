# Main Text Structure Plan

## 主文保留的四条结果线

### 1. Qwen Mainline Positive Case
- Section: Section 6 (`Qwen Mainline: Belief-pressure Damping`)
- 内容: formal mainline closure, Qwen 3B replication, formal controls, held-out support
- 建议:
  - 保留为主文重心，不要降级
  - prompt-family / free-form / intervention-family 只保留一句 boundary-support reference，不展开成并列结果支线
  - 收紧 free-form 和 predictor 语气
  - 明确这是 clean positive case，不要让 Section 6 读成方法论文

### 2. Cross-model Boundary
- Section: Section 7 (`Cross-scale and Cross-family Support`) + Llama limitation section
- 内容: Qwen14B secondary positive confirmation, GLM weak positive with residual damage tradeoff, Llama as the clearest limitation case
- 建议:
  - 保留这一条 narrative arc，但让 Section 7 更轻、更边界化
  - Qwen14B / GLM 的详细数字对照可进一步依赖 appendix tables
  - 让 Llama 成为 main-text cross-model boundary 的最醒目落点

### 3. Free-form Diagnostic Boundary
- Section: Section 10 (`Limitations`) + appendix-linked note
- 内容: `n=50`, five patch conditions, front-loaded effect, intervention-continuity tradeoff
- 建议:
  - 保留在 Limitations 中，不独立成新的 main-text result section
  - 主文作用是说明：effect is visible in open-ended generation, but patched free-form is not clean success
  - 避免任何 deployment-validation 读法

### 4. Predictor Framework
- Section: Section 8 (`What Makes a Pressure Direction Controllable?`)
- 内容: old proposition fails as a stable predictor; specificity can be informative in key comparisons; no single feature generalizes cleanly
- 建议:
  - 让 predictor-boundary paragraph 统领 Section 8
  - 旧的 “necessary structural condition” framing 应被删除或降为 bounded diagnostic hypothesis
  - 本节目标不是提出 predictor，而是解释为什么现有 feature bundle 不足以形成统一 controllability law

## 建议移到 Appendix 的内容

- Qwen formal controls detail（主文只保留一句用途和结果方向）
- Qwen held-out detail（主文保留一句 robustness-side support）
- Prompt-family generalization
- Intervention-family comparison
- Qwen14B / GLM / Llama sub-detail tables and auxiliary diagnostics
- Identity/profile detail
- Human audit and expert adjudication detail
- Clean-protocol boundary note
- Older mechanistic supporting diagnostics already in appendix

## 当前结构 vs 建议结构对照

| Current section / material | Keep in main text? | If kept, narrative role | If moved, appendix destination |
|---|---|---|---|
| Section 6 Qwen mainline | Yes | Core positive case; formal mainline + replication + minimal robustness framing | N/A |
| Qwen formal controls detail | Condense | One-sentence support that mainline is not a trivial artifact | Appendix robustness table |
| Qwen held-out detail | Condense | One-sentence robustness-side support | Appendix robustness materials |
| Prompt-family generalization | Condense | One-sentence wording-family support only | Appendix robustness |
| Intervention-family comparison | Condense | One-sentence “not a single-implementation artifact” support | Appendix intervention-family section |
| Free-form patched diagnostic | Yes, but only in limitations | Boundary evidence showing intervention-continuity tradeoff | Appendix free-form patched diagnostic |
| Section 7 cross-model support | Yes, but lighter | Boundary framing for Qwen14B / GLM before Llama | Appendix cross-model tables |
| Llama limitation section | Yes | Clearest cross-model negative / boundary case | Appendix Llama diagnostics |
| Section 8 predictor / controllability framing | Yes | Mechanism-boundary framing; explain failed predictor law | Appendix mechanistic diagnostics |
| Identity/profile detail | Condense | One clear boundary statement only | Appendix identity/profile evidence |
| Human audit + expert adjudication | No as a main result line | Mention only as weak corroboration in limitations | Appendix human-audit note |

## Implementation notes for the eventual editor

- Do **not** change experimental numbers or evidence levels.
- Do **not** add sections; compression should come from trimming, merging, or demoting emphasis within existing sections.
- Preserve the current evidence hierarchy:
  - Qwen 7B / 3B mainline
  - Qwen14B secondary
  - GLM weak positive with residual damage tradeoff
  - Llama limitation
  - identity/profile boundary
  - appendix-only support materials
- If a wording fix and a structure edit compete, prioritize the audit-driven wording fix and let the structure follow that boundary.

## Assumptions

- This plan uses the current live structure of `WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex`.
- The structure memo is a paper-writing guide, not a code or experiment refactor spec.
- “No new section” means compression by internal reweighting and appendix demotion only, not by creating new headline sections.
