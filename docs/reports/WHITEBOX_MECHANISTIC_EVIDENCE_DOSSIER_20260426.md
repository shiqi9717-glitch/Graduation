# White-box Mechanistic Evidence Dossier

更新时间：2026-04-26

本文档整理 white-box mechanistic 主线的统一证据包，用于主会论文写作、图表规划与附录分层。本文只整合已有结果，不新增实验建议。

统计收口目录：

- `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`

优先引用文件：

- `whitebox_effect_size_table.csv`
- `llama_limitation_summary.md`
- `llama_limitation_summary.json`
- `README.md`

## 1. Main Claim

主命题：

> Pressure is not one mechanism. Belief-style pressure has a transferable and partially intervenable late-layer drift, while identity/profile pressure follows a different, less linearly steerable pathway.

中文写作口径：

> pressure 不是单一机制。`belief_argument` 类型 pressure 存在可迁移、可部分干预的 late-layer drift；而 `identity_profile` 类型 pressure 更像另一条不易被线性方向稳定控制的路径。

禁止 claim：

- 不要 claim universal pressure mitigation。
- 不要 claim Llama positive replication。
- 不要把 Qwen 3B / 7B objective-local proxy 与 bridge causal lines 做完全同义的跨范式 leaderboard。
- 不再建议继续 Llama / Mistral alpha-k-layer sweep。

## 2. Result Structure

### R1: Pressure types separate mechanistically

可说结论：

- `belief_argument` 与 `identity_profile` 不应合并成一个通用 pressure 机制。
- Qwen 3B / 7B 的 `belief_argument` 主线有清晰 intervention 支撑。
- identity/profile 线有 localization observation，但 prefix-specific causal intervention 没有稳定拉开 matched control。

不可说结论：

- 不要说所有 pressure 都共享一个可线性干预方向。
- 不要说 identity/profile 已建立 identity-specific mitigation。

主文/附录：

- 主文可用 R1 作为机制分化的核心论点。
- identity/profile 细节放 limitation 或 appendix。

### R2: Belief pressure admits causal subspace damping

可说结论：

- Qwen 3B / 7B 默认 `baseline_state_interpolation` 是 formal mainline。
- Qwen 14B 提供 secondary causal confirmation。
- GLM 复现了 belief-subspace damping 的方向，但 tradeoff 明显更强。

不可说结论：

- 不要把 Qwen 14B 写成新的 main mitigation line。
- 不要把 GLM 写成无 baseline damage 的强复现。
- 不要把 `late_layer_residual_subtraction` 写成可用主方法。

主文/附录：

- Qwen 3B / 7B 主结果进主文主表。
- Qwen 14B 和 GLM 可进主文小表或短段。
- subtraction controls 只进附录或方法边界。

### R3: Cross-scale and cross-family support, with limitations

可说结论：

- 14B 支持 cross-scale secondary causal confirmation。
- GLM 支持 cross-family positive replication with stronger tradeoff。
- Llama 是 weak replication / limitation：机制可定位，但 intervention transfer weak。
- Mistral 只作为 appendix exploratory note。

不可说结论：

- 不要说 Llama 是 positive replication。
- 不要说 Mistral 已完成正式 cross-family replication。
- 不要把 Llama / Mistral 的探索结果升级为主线成功。

主文/附录：

- 14B / GLM 可进主文。
- Llama / Mistral 进 limitation 或 appendix。

## 3. Evidence-Level Matrix

| 子线 | 证据等级 | 结果目录 | 主文/附录 | 可说结论 | 不可说结论 |
|---|---|---|---|---|---|
| Qwen 7B | formal mainline | `outputs/experiments/local_probe_qwen7b_intervention_main/baseline_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140142`; `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452` | 主文主表 | 默认 `baseline_state_interpolation`, `24-26`, `0.6` 是 7B 主结果；closure: stance drift delta `-0.1493` CI `[-0.2200, -0.0800]`, compliance delta `-0.1595` CI `[-0.2300, -0.0900]`, recovery delta `0.7127` CI `[0.5000, 0.9000]`, damage `0` | 不要把 `24-27, 0.6` 当默认 baseline；不要与 bridge causal lines 做同义 leaderboard |
| Qwen 3B | formal replication | `outputs/experiments/local_probe_qwen3b_intervention_main/baseline_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142847`; `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452` | 主文主表 | 默认 `baseline_state_interpolation`, `31-35`, `0.6` 是 3B formal replication；closure: stance drift delta `-0.1597` CI `[-0.2402, -0.0800]`, compliance delta `-0.2789` CI `[-0.3700, -0.1900]`, recovery delta `0.6906` CI `[0.5000, 0.8667]`, damage `0` | 不要把 subtraction control 写成可用方法；不要做跨范式 leaderboard |
| Qwen 14B | secondary causal confirmation | `outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308`; `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452` | 主文小表/短段 | belief-subspace damping 降低 drift 和 compliance；closure: drift delta `-0.2068` CI `[-0.4167, 0.0000]`, compliance delta `-0.2484` CI `[-0.4167, -0.0833]` | 不要升级成新的 main mitigation line；不要忽略 `n=24`、recovery delta `0`、damage `0.0416` |
| GLM | cross-family positive replication with stronger tradeoff | `outputs/experiments/pressure_subspace_damping_glm4_9b/Users_shiqi_.cache_huggingface_hub_models--zai-org--glm-4-9b-chat-hf_snapshots_8599336fc6c125203efb2360bfaf4c80eef1d1bf/20260426_005017`; `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452` | 主文跨家族小表/短段 | 方向成立；closure: drift delta `-0.2932` CI `[-0.5417, -0.0417]`, compliance delta `-0.2095` CI `[-0.4167, 0.0000]`, recovery delta `0.3769` CI `[0.1667, 0.5833]` | 不要写成无代价复现；必须保留 damage `0.3325` CI `[0.1667, 0.5417]` |
| Llama | weak replication / limitation | `outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011`; `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452` | Limitation / appendix | mechanism locatable, intervention transfer weak；drift 仅小幅下降，compliance 不降，recovery 下降，damage `0.25` 左右 | 不要写成 positive replication；不再建议继续 alpha/k/layer sweep |
| Mistral | appendix exploratory | `outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_143645`; `outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_144430`; `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452` | Appendix exploratory note | English bridge prompt 有方向信号，但 baseline damage 高、跨 prompt 不稳；仅保留一句 exploratory note | 不要升格为 positive replication；不再建议继续 sweep |
| identity_profile | weak mechanistic observation / boundary | `outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058` | Limitation / appendix | localization evidence exists; causal intervention support insufficient | 不要 claim identity-specific mitigation；不要写成正式副线 |

## 4. Main-Text Figure And Table Suggestions

建议主文图表：

- Figure 1: White-box mechanism map. 展示 pressure types 分化：belief-style pressure 的 late-layer drift 与 identity/profile pathway 分开。
- Figure 2: Qwen mainline intervention effect. 展示 Qwen 7B formal mainline 与 Qwen 3B formal replication 的 stance drift / compliance / recovery delta + CI；脚注注明 objective-local proxy。
- Table 1: Evidence-level matrix. 放 Qwen 7B、Qwen 3B、Qwen 14B、GLM、Llama、Mistral、identity_profile 的证据等级、主文/附录定位、关键边界。
- Table 2: Cross-scale and cross-family support. 只放 Qwen 14B 与 GLM 作为主文 secondary / cross-family evidence，同时列 damage tradeoff。
- Figure 3 or small panel: Limitation boundary. 用 Llama 展示 locatable-but-not-controllable：drift 小幅下降、compliance 不降、recovery 下降、damage 上升。

主文图表注意：

- Qwen 3B / 7B 与 bridge causal lines 不做同义 leaderboard。
- Llama 和 Mistral 不放入 positive replication 主图。
- identity_profile 不放入 mitigation success 图。

## 5. Appendix Result List

建议附录结果：

- Appendix A: Statistical closure notes and full `whitebox_effect_size_table.csv` explanation.
- Appendix B: Qwen 7B aggressive secondary `24-27, 0.6` and why it is not default baseline.
- Appendix C: `late_layer_residual_subtraction` negative-control / failure evidence.
- Appendix D: Qwen 14B small-sample causal confirmation details.
- Appendix E: GLM tradeoff details and baseline damage discussion.
- Appendix F: Llama weak replication / limitation, including `llama_limitation_summary.md` and projection-to-logit interpretation.
- Appendix G: Mistral appendix exploratory note.
- Appendix H: identity_profile weak mechanistic observation / boundary.

## 6. Copy-ready Claims

Main claim:

> Pressure is not one mechanism: belief-style pressure admits a transferable and partially intervenable late-layer drift, while identity/profile pressure follows a different and less linearly steerable pathway.

Qwen mainline:

> Qwen 7B and Qwen 3B provide the formal mainline evidence under the default `baseline_state_interpolation` settings, with effect sizes reported under the objective-local proxy mapping defined in the statistical closure notes.

14B:

> Qwen 14B provides a secondary causal confirmation rather than a new main mitigation line.

GLM:

> GLM reproduces the belief-subspace damping direction across model families, but with a substantially stronger utility/safety tradeoff.

Llama:

> Llama-3.1-8B is best interpreted as a weak replication / limitation: the belief-pressure subspace is more identifiable under the English bridge prompt, but intervention transfer remains weak.

Mistral:

> Mistral-7B showed directional belief-subspace damping signals under an English bridge prompt, but the effect was unstable across prompt variants and accompanied by high baseline damage; we therefore treat it as exploratory only.

identity_profile:

> The identity-profile line provides weak localization evidence but insufficient causal intervention support, and should not be interpreted as identity-specific mitigation.
