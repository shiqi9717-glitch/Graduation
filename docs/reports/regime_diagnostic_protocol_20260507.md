# Regime Diagnostic Protocol

更新时间：2026-05-07

本协议只做诊断框架设计，不跑新实验，不改代码，不改论文正文。目标是把现有 evidence 组织成一套可复用的 `controllability regime diagnostic framework`，供 Code / Analysis / Paper Writing 三个部门直接接用。

## 0. Regime Vocabulary

本文统一使用以下 regime：

- `clean-controllable`: drift / compliance 明显改善，recovery 不差，damage 很低。
- `secondary-controllable`: 方向正确，但效应较弱或只构成 secondary support。
- `specific-but-not-controllable`: 表征可定位、target-logit 变化强，但行为上不形成 clean control。
- `locatable-but-not-controllable`: 有投影和局部方向，但 target/negative pattern 或行为结果不足。
- `damage-prone / misaligned`: 伴随明显 damage，或 control family 显示错配。
- `generation-unstable`: 行为方向存在，但 free-form 文本质量快速崩塌。

符号约定：

- `drift_delta < 0` 表示改善。
- `compliance_delta < 0` 表示改善。
- `recovery_delta > 0` 表示改善。
- `damage` 越低越好。
- `target-logit Δ < 0` 表示 belief-pressure 目标 logit 被压低。

## 1. Seven Diagnostic Indicators

### D1. Target-Logit Delta

| 项目 | 内容 |
| --- | --- |
| 指标名 | `target_logit_delta` |
| 公式 | `Δ_target = mean_belief_logit_delta_vs_no_alpha_a` |
| 数据来源 | `projection_alignment_summary.json` 中 `mean_belief_logit_delta_vs_no_alpha_*` |
| 阈值 | `strong <= -5.0`; `moderate (-5.0, -2.0]`; `weak > -2.0` |
| 当前值 | Qwen7B mainline `-5.39`; Qwen7B PSD `-5.90`; Qwen3B `-5.10`; Qwen14B `-7.33`; GLM `-2.87`; Llama `-1.83` |

解释：

- 它更像 `specific pressure-axis effect strength`。
- 当前样本支持“强 `Δ_target` 是 clean control 的必要但非充分条件”。

### D2. Negative-Control Logit Delta

| 项目 | 内容 |
| --- | --- |
| 指标名 | `negative_logit_delta` |
| 公式 | `Δ_neg = mean_negative_logit_delta_vs_no_alpha_a` |
| 数据来源 | `projection_alignment_summary.json` 中 `mean_negative_logit_delta_vs_no_alpha_*` |
| 阈值 | `supportive <= -0.10`; `near-zero (-0.10, +0.01)`; `bad > +0.01` |
| 当前值 | Qwen7B mainline `-0.0050`; Qwen7B PSD `-0.0017`; Qwen3B `-0.9303`; Qwen14B `-1.4254`; GLM `-0.0011`; Llama `+0.1716` |

解释：

- 当前 frozen sample 中，`near-zero` 并不自动意味着好。
- `positive Δ_neg` 是明确坏信号；`moderately negative Δ_neg` 比 `near-zero` 更接近 clean control。

### D3. Specificity Ratio

| 项目 | 内容 |
| --- | --- |
| 指标名 | `specificity_ratio` |
| 公式 | `S = |Δ_target| / max(|Δ_neg|, 1e-6)` |
| 数据来源 | 用 D1 与 D2 两个字段计算 |
| 阈值 | `very high > 100`; `moderate 3-50`; `low < 3` |
| 当前值 | Qwen7B mainline `1083.72`; Qwen7B PSD `3387.65`; Qwen3B `5.48`; Qwen14B `5.14`; GLM `2580.02`; Llama `10.65` |

解释：

- `specificity` 是有信息量的，但不是充分条件。
- 当前最关键反例恰好是高 specificity 但不 controllable 的 Qwen7B PSD 与 GLM。

### D4. Projection-Drift Correlation

| 项目 | 内容 |
| --- | --- |
| 指标名 | `proj_drift_corr` |
| 公式 | `r_pd = corr_pressured_projection_with_stance_drift` |
| 数据来源 | `projection_alignment_summary.json` 中 `corr_pressured_projection_with_stance_drift` |
| 阈值 | `aligned <= -0.30`; `weak (-0.30, -0.10)`; `unclear > -0.10` |
| 当前值 | Qwen7B mainline `-0.3899`; Qwen7B PSD `-0.4086`; Qwen3B `-0.5243`; Qwen14B `+0.0777`; GLM `-0.4459`; Llama `-0.2861` |

解释：

- 这是 `localization aligns with behavior` 的 readout，不是 clean control guarantee。
- Qwen7B PSD 与 GLM 都显示良好负相关，但行为上仍不是 clean-control。

### D5. Baseline Damage

| 项目 | 内容 |
| --- | --- |
| 指标名 | `baseline_damage` |
| 公式 | `damage = baseline_damage_rate` 或 closure `baseline_damage` |
| 数据来源 | `FINAL_EVIDENCE_LEDGER_20260506.md`、`cross_model_belief_transfer_boundary_update_20260506.md`、`intervention_family_comparison_20260506.md` |
| 阈值 | `clean <= 0.02`; `residual (0.02, 0.10)`; `high > 0.10` |
| 当前值 | Qwen7B mainline `0.0000`; Qwen3B `0.0000`; Qwen14B `0.01`; GLM `0.06`; Llama `0.08`; Qwen7B PSD `0.04`; subtraction `0.0460`; shuffled `0.5848` |

解释：

- `low damage` 是 clean control 的必要条件。
- 但 `low damage` 本身不保证 controllability；GLM 与 Qwen7B PSD 都是反例。

### D6. Intervention-Family Agreement

| 项目 | 内容 |
| --- | --- |
| 指标名 | `intervention_family_agree` |
| 公式 | 若同模型不同 intervention family 在 `drift/compliance/recovery` 的方向一致，则记为 `yes/partial`; 否则 `no` |
| 数据来源 | `docs/reports/intervention_family_comparison_20260506.md` |
| 阈值 | `yes`: 主线与替代 family 同向且无明显对照崩坏；`partial`: 同向但效应弱或 tradeoff 明显；`no`: 主线外 family 大多失败 |
| 当前值 | Qwen7B = `partial`；其他模型当前 `N/A`，因为尚无第二 family 的冻结对照 |

解释：

- 这是 `implementation robustness` 指标。
- 它不是跨模型必要条件，但对主文 claims 的强弱影响很大。

### D7. Free-form Stability

| 项目 | 内容 |
| --- | --- |
| 指标名 | `freeform_stability` |
| 公式 | 结合 `readable_rate`, `repetition_rate`, `distinct-1` 判定 |
| 数据来源 | `freeform_diagnostic_confirm_boundary_note_20260504.md` 与对应 `run_summary.json` |
| 阈值 | `stable`: `readable_rate >= 0.90` 且 `repetition_rate <= 0.10`; `brittle`: `0.20 <= readable_rate < 0.90`; `unstable`: `readable_rate < 0.20` 或 `repetition_rate >= 0.90` |
| 当前值 | Qwen7B `prefill_only = stable`; `first_token_only = stable`; `first_3_tokens = brittle`; `continuous = unstable`; 其他模型 `N/A` |

解释：

- 这是 `generation stability` 指标，不等价于 option-logit controllability。
- 它让论文可以把 `behavioral controllability` 与 `deployment-level open-ended usability` 明确拆开。

## 2. Diagnostic Matrix

| Model/Setting | target-logit Δ | neg-ctrl Δ | specificity | proj-drift corr | damage | interv-family agree | free-form stability | predicted regime | observed regime |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Qwen 7B `baseline_state_interpolation` mainline | -5.39 | -0.0050 | 1083.72 | -0.3899 | 0.0000 | partial | N/A | clean-controllable | clean-controllable |
| Qwen 3B `baseline_state_interpolation` mainline | -5.10 | -0.9303 | 5.48 | -0.5243 | 0.0000 | N/A | N/A | clean-controllable | clean-controllable |
| Qwen 14B matched damping n=100 | -7.33 | -1.4254 | 5.14 | +0.0777 | 0.01 | N/A | N/A | secondary-controllable | secondary-controllable |
| GLM-4-9B PSD n=100 | -2.87 | -0.0011 | 2580.02 | -0.4459 | 0.06 | N/A | N/A | specific-but-not-controllable | weak / tradeoff-limited |
| Llama-3.1-8B PSD n=100 | -1.83 | +0.1716 | 10.65 | -0.2861 | 0.08 | N/A | N/A | locatable-but-not-controllable | locatable-but-not-controllable |
| Qwen 7B PSD n=100 | -5.90 | -0.0017 | 3387.65 | -0.4086 | 0.04 | partial | N/A | specific-but-not-controllable | weak / tradeoff-limited |
| Qwen 7B subtraction | N/A | N/A | N/A | N/A | 0.0460 | no | N/A | damage-prone / misaligned | method-boundary failure |
| Qwen 7B random control | N/A | N/A | N/A | N/A | 0.0127 | no | N/A | null-like / non-specific | null-like control |
| Qwen 7B shuffled-label control | N/A | N/A | N/A | N/A | 0.5848 | no | N/A | damage-prone / misaligned | pairing-artifact damage |
| Qwen 7B free-form `prefill_only` | N/A | N/A | N/A | N/A | N/A | partial | stable | front-loaded controllable-but-bounded | front-loaded diagnostic effect |
| Qwen 7B free-form `first_3_tokens` | N/A | N/A | N/A | N/A | N/A | partial | brittle | front-loaded but quality-limited | brittle / quality-collapsing |
| Qwen 7B free-form `continuous` | N/A | N/A | N/A | N/A | N/A | partial | unstable | generation-unstable | generation-unstable |

缺失说明：

- subtraction / control rows 没有对应 `projection_alignment_summary.json`，所以前四列必须标 `N/A`。
- free-form rows 不在 projection-diagnostic family 内，也不测 baseline damage，所以 logit / projection / damage 列标 `N/A`。
- 除 Qwen7B 外，当前没有冻结的第二 intervention family 对照，因此 `interv-family agree = N/A`。

## 3. Five Empirical Diagnostic Rules

### Rule 1

`|target-logit Δ| > 5` 是 clean 或 secondary controllability 的必要条件，但不是充分条件。

- Supported by: Qwen7B mainline (`-5.39`), Qwen3B (`-5.10`), Qwen14B (`-7.33`)
- Counterexample: Qwen7B PSD (`-5.90`) 仍非 clean-controllable
- Note: GLM (`-2.87`) 与 Llama (`-1.83`) 都落在弱侧，且都不是 clean control

To validate on a new row:
- Code 部门需要先跑该 setting 的 `projection_alignment_summary.json`
- Analysis 部门取 `mean_belief_logit_delta_vs_no_alpha_*`

### Rule 2

`specificity` 只能说明“不是任意 logit 噪声”，不能单独预测 controllability。

- Supported by false positives: Qwen7B PSD (`3387.65`), GLM (`2580.02`)
- Counterexample to a strong claim: Qwen14B 只有 `5.14`，却比 GLM 更 controllable
- Note: 这条 rule 直接支持 “specificity alone is not enough”

To validate on a new row:
- Code 部门同样只需跑 projection diagnostic
- Analysis 部门按 `|Δ_target| / |Δ_neg|` 统一计算

### Rule 3

`negative-logit Δ > 0` 是强坏信号；`negative-logit Δ` 明显小于 0 比 `near-zero` 更接近 clean control。

- Supported by: Llama `+0.1716` 且 non-controllable
- Supported by positive rows: Qwen3B `-0.9303`, Qwen14B `-1.4254`
- Counterexample to “near-zero is good”: Qwen7B mainline near-zero 但好，Qwen7B PSD / GLM near-zero 但不好

To validate on a new row:
- Code 部门只需追加 projection diagnostic
- Analysis 部门重点检查 `Δ_neg` 的符号和幅度，而不是只看它是否接近 0

### Rule 4

`damage <= 0.02` 是 clean control 的必要条件；`damage > 0.05` 时应优先落入 tradeoff / damage-prone regime。

- Supported by: Qwen7B `0.0000`, Qwen3B `0.0000`, Qwen14B `0.01`
- Counterexample to sufficiency: 低 damage 本身不够，仍需看 logit / behavior；例如 GLM `0.06` 只弱正向
- Note: shuffled-label `0.5848` 是明显的 damage-prone anchor

To validate on a new row:
- Code 部门需要跑对应 behavioral closure
- Analysis 部门直接填 `baseline_damage`

### Rule 5

free-form horizon 主要诊断 `generation stability`，不是额外增强 `behavioral effect`。

- Supported by: `prefill_only` / `first_token_only` 行为已饱和，`first_3_tokens` 与 `continuous` 不再改善 drift / wrong-follow
- Supported by quality collapse: `readable_rate 0.94 -> 0.42 -> 0.00`, `repetition_rate 0.00 -> 0.58 -> 1.00`
- Note: generation stability 与 behavioral controllability 必须分列报告

To validate on a new row:
- Code 部门跑 free-form patch-horizon condition
- Analysis 部门取 `readable_rate`, `repetition_rate`, `distinct-1`, `drift_rate`, `wrong_follow_rate`

## 4. Necessary vs Informative Signals

### Necessary

- 强 `target-logit Δ`：当前所有 clean / secondary controllable rows 都满足。
- 低 `damage`：当前所有 clean / secondary controllable rows 都满足。
- free-form 若要写 deployment-adjacent 口径，还需要至少 `stable` 而不是 `unstable`。

### Informative But Not Sufficient

- `specificity_ratio`
- `proj_drift_corr`
- `intervention-family agreement`
- `baseline_projection_as_fraction_of_pressured_projection`

为什么 `specificity` 不够：

- Qwen7B PSD 和 GLM 都显示出极高 specificity，但行为上只给出 weak / tradeoff-limited profile。
- 这说明 “target axis 很干净” 与 “behavior can be cleanly controlled” 之间还有中间断层。
- 这正是本文要强调的四分离：`localization`, `specificity`, `behavioral controllability`, `generation stability` 不能合并成单一轴。

## 5. Minimal Validation Checklist For New Settings

若 Code 部门要新增一行 setting，最小实验包如下：

1. 跑 `projection_alignment_summary.json`
2. 跑 behavioral closure summary
3. 若是 alternative intervention family，再补 family comparison row
4. 若是 free-form 条件，再补 `run_summary.json`

Analysis 部门据此填写：

- D1-D4: 从 projection summary
- D5: 从 behavioral closure
- D6: 从 family comparison
- D7: 从 free-form summary

## 6. Bottom Line

当前最稳的 diagnostic conclusion 是：

- `target-logit strength` 和 `low damage` 更接近必要条件。
- `specificity`、`projection-drift correlation`、`layer localization` 都是有信息量但非充分条件。
- Qwen7B / Qwen3B / Qwen14B 支持 clean 或 secondary controllability。
- GLM 与 Qwen7B PSD 共同提供“specific but not controllable”的关键反例。
- Llama 提供“locatable but not controllable”的 boundary。
- free-form results 进一步证明 generation stability 必须被单独建模，而不能并入单一 controllability 分数。
