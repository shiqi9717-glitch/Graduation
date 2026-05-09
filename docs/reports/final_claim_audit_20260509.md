# Final Claim Audit (2026-05-09)

审计范围：

- `/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex`
- `/Users/shiqi/code/graduation-project/docs/reports/FINAL_EVIDENCE_LEDGER_20260508.md`

审计原则：

- 只做审计，不改正文
- 不改数字
- 不改证据层级
- 不改结构

---

## 1. 禁用词扫描

总体结论：**大部分为 PASS**。主稿中的高风险词基本都出现在允许的否定或边界语境中，没有发现明确的 `we mitigate ...`、`universal intervention`、`formal predictor success`、`human-validated recovery`、`free-form success` 这类 P0 级残留。

### 扫描结果

| Line | Current wording | Risk level | Recommended fix |
|---|---|---:|---|
| 103 | `...without upgrading the paper into a formal predictor story.` | PASS | 无需修改 |
| 123 | `we are not proposing a universal steering vector` | PASS | 无需修改 |
| 135 | `...rather than as a second positive mitigation line.` | PASS | 无需修改 |
| 137 | `...not as a second positive mitigation line or a standalone mitigation success.` | PASS | 无需修改 |
| 182 | `Qwen 14B is secondary causal confirmation, not a new primary mitigation line.` | PASS | 无需修改 |
| 211 | `...rather than as a proposed deployable mitigation system.` | PASS | 无需修改 |
| 250 | `These results do not establish universal pressure mitigation...` | PASS | 无需修改 |
| 265 | `...a bounded diagnostic framework, not a formal predictor.` | PASS | 无需修改 |
| 303 | `High specificity is useful but not sufficient...` | PASS | 无需修改 |
| 315 | `...serves as a stable predictor across the current model families...` | PASS | 无需修改 |
| 316 | `...a bounded mechanism hypothesis... not ... a universal steering law.` | PASS | 无需修改 |
| 370 | `...not as a complete localization of the underlying circuit.` | PASS | 无需修改 |
| 429 | `...it does not establish free-form mitigation...` | PASS | 无需修改 |
| 485 | `It does not claim a universal mitigation method, a formal predictor, or a circuit-level mechanism.` | PASS | 无需修改 |
| 517 | `...not a universal mitigation method or a formal predictor...` | PASS | 无需修改 |
| 593 | `...it is sufficient to document method boundaries...` | LOW | 这不是 controllability 条件语境，不构成 overclaim；可保留 |
| 715 | `Their role is not to prove a complete causal circuit...` | PASS | 无需修改 |

结论：**禁用词扫描 PASS**，无 P0 级词汇残留。

---

## 2. Metric-family 混用检查

总体结论：**存在 2 处 NEEDS_FIX，均为标注层面，不是数字错误。**

### 2.1 Regime table (`tab:regime`)

- 位置：主稿 77–96 行，尤其 [82–95](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:82)
- 结果：**NEEDS_FIX**
- 原因：
  - 同一表中并列了：
    - objective-local proxy：Qwen 7B / 3B
    - bridge / transfer-style：Qwen 14B / Llama / GLM / Mistral
    - held-out mixed diagnostic + behavioral：authority pressure
    - boundary row：identity/profile
  - caption 只写了 `Evidence levels follow the frozen cross-model evaluations (n=100 each)`，**没有明确提示 metric family mixed / non-leaderboard interpretation**。
- 建议修改：
  - 在 caption 末尾补一句类似：
    - `Rows mix objective-local, bridge-style, and held-out diagnostic summaries and should be read as regime classification rather than as a single metric-family leaderboard.`

### 2.2 Diagnostic matrix (`tab:diagnostic-matrix`)

- 位置：主稿 268–289 行，尤其 [274–288](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:274)
- 结果：**NEEDS_FIX**
- 原因：
  - 该表已做成 qualitative checklist，风险低于 regime table。
  - 但它仍并列了 objective-local、bridge/transfer、authority held-out、free-form row。
  - 当前 caption 说它是 bounded framework / not a formal predictor，**但没有直接说 mixed metric-family synthesis**。
- 建议修改：
  - 在 caption 中再补一句类似：
    - `Rows synthesize objective-local, bridge-style, and free-form diagnostic evidence and are not a unified metric-family scorecard.`

### 2.3 主文段落中的跨 family 数字对比

- Qwen mainline / Figure 2 区域：[220–257](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:220)
- Problem Setup 中 bridge vs objective-local 边界：[152–167](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:152)
- Appendix robustness note：[588](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:588)

结果：**PASS**

原因：

- 主文已经多次明确：
  - objective-local mainline 与 bridge metrics 不应并成单一 leaderboard
  - free-form 使用自己的 rate family
- 没有发现未加注释的“同表直接拼分”式 raw-number leaderboard。

---

## 3. 数字一致性检查

总体结论：**未发现主稿与 evidence ledger 之间的明确数字冲突。**

### 3.1 Qwen 7B mainline

- Ledger：`-0.1493 / -0.1595 / +0.7127 / 0.0000`
- 主稿：
  - [221–222](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:221)
  - [604](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:604)
- 结果：**PASS**

### 3.2 Qwen 3B replication

- Ledger：`-0.1597 / -0.2789 / +0.6906 / 0.0000`
- 主稿：
  - [223](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:223)
  - [608](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:608)
- 结果：**PASS**

### 3.3 Qwen 14B secondary

- Ledger：`-0.06 / -0.07 / +0.03 / 0.01`
- 主稿：
  - [232](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:232)
- 结果：**PASS**

### 3.4 Llama

- Ledger：`+0.03 / +0.01 / -0.03 / 0.08`
- 主稿：
  - 主文主叙事没有直接重报这组 `n=100` behavioral closure 数字
  - 主文保留的是：
    - regime row 中的 `damage 0.08`：[89](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:89)
    - earlier representative `n=24` setting 已明确标注：[335](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:335)
- 结果：**PASS**
- 备注：未发现把旧 `n=24` 数字误写成 `n=100` 的情况。

### 3.5 GLM

- Ledger：`-0.01 / -0.01 / +0.12 / 0.06`
- 主稿：
  - damage `0.06` 一致：[90](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:90), [343](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:343), [390](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:390)
  - 主文没有把完整四元组全部重报
- 结果：**PASS**

### 3.6 Mistral

- Ledger：`-0.12 / -0.05 / +0.38 / 0.39`
- 主稿：
  - behavior closure 的主文显式值主要保留 `damage 0.39`：[91](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:91), [350](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:350), [391](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:391)
  - 主文没有把 `-0.12 / -0.05 / +0.38` 全部重报
- 结果：**PASS**

### 3.7 Authority pressure

- Ledger：`-0.31 / -0.31 / +0.37 / 0.00`
- 主稿：
  - regime row 给出的是 held-out status 与 diagnostic summary，不是完整 behavioral four-tuple：[87](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:87), [297](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:297)
- 结果：**PASS**
- 备注：主稿没有写出与 ledger 冲突的 authority behavioral numbers。

数字一致性总评：**PASS**

---

## 4. 三段论一致性检查

### 4.1 Abstract

- 位置：[32–46](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:32)
- 结果：**PASS**
- 检查：
  - `Clean controllability exists`：有
  - `does not follow from localization or specificity alone`：有
  - `does not guarantee generation stability`：有

### 4.2 Introduction

- 位置：[51–66](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:51)
- 结果：**PASS**
- 检查：
  - activation steering assumption 切入：有
  - 三段论三段：有
  - 最后收束到 “when and why linear intervention succeeds” ：有

### 4.3 Discussion

- 位置：[464–487](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:464)
- 结果：**PASS**
- 检查：
  - `What did we learn?`：有
  - `Which axes separate?`：有
  - `Why does this matter?`：有
  - `What does the framework not claim?`：有

### 4.4 Conclusion

- 位置：[522–524](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:522)
- 结果：**PASS**
- 检查：
  - 是三段论压缩版
  - 与 Abstract / Introduction / Discussion 不矛盾

三段论一致性总评：**PASS**

---

## 5. Artifact Registry 完整性

总体结论：**存在 4 类缺项，属于 Registry completeness NEEDS_FIX，不是正文 claim 错误。**

### 已覆盖条目

| Run / artifact family | Status | Evidence |
|---|---|---|
| Qwen 7B mainline | PASS | [991](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:991) |
| Qwen 3B replication | PASS | [992](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:992) |
| Qwen 14B n=100 | PASS | [993](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:993) |
| Llama n=100 | PASS | [995](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:995) |
| GLM n=100 | PASS | [994](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:994) |
| Mistral n=100 behavioral | PASS | [1012](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1012) |
| Mistral projection diagnostic | PASS | [1013](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1013) |
| Authority pressure diagnostic | PASS | [1008](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1008) |
| Authority pressure behavioral | PASS | [1009](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1009) |
| Free-form 7-condition | PASS | [1014](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1014) |
| Layer-wise Qwen 7B / 3B | PASS | [1003](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1003) |
| Llama per-layer | PASS | [1010](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1010) |
| Prompt-family | PASS | [1006](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1006) |
| Intervention-family | PASS | [1007](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:1007) |
| Identity/profile follow-up | PASS | [996](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:996) |

### 缺失条目

| Run / artifact family | Status | Recommended fix |
|---|---|---|
| Component-level intervention (Qwen / GLM / Llama) | MISSING | 新增 dedicated artifact row，指向 component-level exploratory run summary |
| Dose-response sweep (Qwen / GLM / Llama) | MISSING | 新增 dedicated artifact row，指向 beta sweep summary |
| Damage mechanism analysis (GLM / Mistral) | MISSING | 新增 dedicated artifact row，指向 damage decomposition / subtype summary |
| Human audit + expert adjudication | MISSING | 新增 dedicated artifact rows，分别对应 audit report 和 expert adjudication note |

Artifact Registry 总评：**NEEDS_FIX**

---

## 6. Overclaim 残留扫描

重点扫描区域：

- Abstract 36–46
- Introduction 55–66
- Discussion 464–487
- Conclusion 522–524
- Section 6 damage mechanism 377–381
- Section 7 free-form quantitative interpretation 458–461

### 6.1 Abstract

- 位置：[36–46](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:36)
- 结果：**PASS**
- 说明：
  - 三段论完整
  - 没有 universal / predictor / mitigation success overclaim
  - `This establishes that ... can, in some cases, be cleanly controlled` 在当前 evidential package 下可接受

### 6.2 Introduction

- 位置：[55–66](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:55)
- 结果：**PASS**
- 说明：
  - 切入方式正确
  - 没有再把文章写成 universal mitigation / benchmark / circuit discovery

### 6.3 Discussion

- 位置：[466–487](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:466)
- 结果：**PASS**
- 说明：
  - 现在主要解释意义和边界，不再重复主结果数字
  - `bounded diagnostic framework` / `bounded mechanism hypothesis` 口径统一

### 6.4 Conclusion

- 位置：[522–524](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:522)
- 结果：**PASS**
- 说明：
  - 三段论压缩清楚
  - 没有回退到 stronger-than-evidence framing

### 6.5 Section 6 damage mechanism 升级区

- 位置：[377–381](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:377)
- 结果：**NEEDS_FIX**
- 当前文字：
  - `This damage subtype---off-target ranking instability coexisting with intact target-logit specificity---is the mechanism boundary that separates GLM and Mistral from the clean Qwen case.`
- 风险：
  - `is the mechanism boundary` 语气偏强，略像已经锁定了 definitive separating mechanism
- 建议修改：
  - `This damage subtype ... marks the mechanism-boundary pattern observed in GLM and Mistral relative to the clean Qwen case.`
  - 或：
  - `This damage subtype ... is the clearest mechanism-boundary pattern observed here...`

### 6.6 Section 7 free-form 量化区

- 位置：[458–461](/Users/shiqi/code/graduation-project/docs/papers/WHITEBOX_MECHANISTIC_MAIN_DRAFT_20260426.tex:458)
- 结果：**NEEDS_FIX**
- 当前文字：
  - `This suggests that generation stability failure is not gradual degradation but a threshold effect...`
- 风险：
  - `is not gradual degradation but a threshold effect` 语气略强；当前证据更像 tested sweep 上的 threshold-like pattern
- 建议修改：
  - `This suggests a threshold-like stability failure in the tested sweep rather than smooth gradual degradation...`

Overclaim 总评：**存在 2 处 P1 wording-level NEEDS_FIX；未发现 P0 overclaim。**

---

## Overall Verdict

### 结论

- **是否可以投稿：基本可以。**
- **是否还有 P0 级问题：没有发现。**

### 当前剩余问题级别

- **P0：0 项**
- **P1：4 项**
  - Regime table metric-family 标注不够明确
  - Diagnostic matrix metric-family 标注不够明确
  - Section 6 damage mechanism 句子有一处偏强
  - Section 7 free-form threshold-effect 句子有一处偏强
- **Registry completeness：1 组问题**
  - Component-level / dose-response / damage mechanism / human audit + expert adjudication 缺 dedicated artifact rows

### 最稳妥的最终判断

当前版本已经通过：

- 三段论一致性
- 关键数字一致性
- 主体 overclaim 边界
- 禁用词主扫描

若要进入“最终投稿版”而不是“基本可投版”，建议再补一轮**纯 wording / registry-level polish**，优先顺序如下：

1. 给 `tab:regime` 和 `tab:diagnostic-matrix` 明确补 mixed metric-family non-leaderboard 注释  
2. 收紧 Section 6 和 Section 7 的两句偏强表述  
3. 在 Artifact Registry 补齐 component-level / dose-response / damage-case / human-audit artifacts

