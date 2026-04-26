# 论文分析统一入口

更新时间：2026-04-13

本文档用于统一新工作区 `/Users/shiqi/code/graduation-project` 下的论文分析入口。

旧工作区 `/Users/shiqi/code/代码/毕业代码` 仅作为迁移背景保留说明，不再作为后续开发或查找文件的正式入口。

## 1. 主版本结论

- 已核对旧工作区与新工作区的 `docs/papers/literature_summary.md`，当前两者内容一致。
- 因此，后续论文总表的唯一主版本统一为：`docs/papers/literature_summary.md`
- 不再继续以旧口径的 `papers/literature_summary.md` 作为主入口；本项目当前实际保存 20 篇核心论文 PDF 的目录是：`docs/papers/`

## 2. 推荐阅读顺序

1. `docs/PAPER_ANALYSIS_ENTRYPOINT.md`
2. `docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`
3. `docs/papers/literature_summary.md`
4. `docs/PROJECT_STATUS.md`
5. `src/mitigation/README.md`

## 3. 当前研究主张

当前更稳妥的项目判断是：

- 当前 interference detector 更像 risk ranker / trigger policy，不像高置信强分类器。
- 当前瓶颈更像“过程可观测性不足”，而不是“模型结构天然无解”。
- 比起继续追求单次 final-answer 分类，下一阶段更值得转向 `process proxy / measurement / audit + recheck`。

详细展开见：`docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`

## 4. 文献组织新主线

后续所有论文分析，统一按以下四条线组织：

1. `measurement / label design`
2. `process monitoring / proxy signals`
3. `audit + recheck / oversight`
4. `causal / mechanistic mitigation`

对应综述重组已写入：

- `docs/papers/RESEARCH_MEMO_DETECTOR_TO_PROCESS_PROXY.md`
- `docs/papers/literature_summary.md`

## 5. 与当前项目代码和结果的直接锚点

优先引用以下文件，不要脱离当前仓库现状空谈：

- 数据扰动与条件设计：`src/data/local_data_perturber.py`
- detector / re-check 模块说明：`src/mitigation/README.md`
- 当前结果交接主文档：`docs/reports/RESULT_ANALYSIS_HANDOFF_20260422.md`
- white-box 机制线总览：`docs/reports/WHITEBOX_MECHANISTIC_WORKSPACE_SUMMARY_20260426.md`
- white-box 主会证据矩阵：`docs/reports/WHITEBOX_MECHANISTIC_EVIDENCE_MATRIX_20260426.md`
- detector 主线正式目录：`outputs/experiments/interference_detector_new15_full_sentence_embedding`
- detector 主线摘要：`outputs/experiments/interference_detector_new15_full_sentence_embedding/guard_eval_summary_sentence_embedding_new15_full.json`
- detector 主线阈值扫表：`outputs/experiments/interference_detector_new15_full_sentence_embedding/threshold_sweep_sentence-embedding-logreg_strict.csv`
- detector grid 对照：`outputs/experiments/interference_detector_new15_full_detector_grid`
- strict rebuilt full real recheck：`outputs/experiments/full_real_recheck_rebuilt/20260417_113758`
- 7B intervention 主 baseline：`outputs/experiments/local_probe_qwen7b_intervention_main/baseline_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140142`
- 7B intervention aggressive secondary：`outputs/experiments/local_probe_qwen7b_intervention_main/aggressive_24_27_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140518`
- 7B intervention subtraction control：`outputs/experiments/local_probe_qwen7b_intervention_main/subtraction_control_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140853`
- 3B intervention 主 baseline：`outputs/experiments/local_probe_qwen3b_intervention_main/baseline_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142847`
- 3B intervention subtraction control：`outputs/experiments/local_probe_qwen3b_intervention_main/subtraction_control_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142938`
- belief_argument mechanistic 主线：`outputs/experiments/local_probe_qwen3b_mechanistic/...` 与 `outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32/...`
- 14B belief causal transfer（主文 secondary causal confirmation）：`outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308`
- GLM cross-family replication（positive replication with stronger tradeoff）：`outputs/experiments/pressure_subspace_damping_glm4_9b/Users_shiqi_.cache_huggingface_hub_models--zai-org--glm-4-9b-chat-hf_snapshots_8599336fc6c125203efb2360bfaf4c80eef1d1bf/20260426_005017`
- Llama English bridge prompt retest（limitation / weak replication）：`outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011`
- identity_profile white-box follow-up（弱支持观察，不作为正式副线）：`outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058`
- 固定模型真实 guarded pilot：`outputs/experiments/deepseek_guarded_pilot/20260412_011945`
- same-model self-recheck pilot（历史参考，不作为当前正式方案主口径）：`outputs/experiments/same_model_guarded_pilot/20260412_021656`

这里建议后续写作时始终分开三类证据：

- `offline proxy simulation`
- `strict rebuilt real recheck summary`
- `historical pilots / case studies`

当前正式汇报时，应优先采用严格重建后的口径：

- baseline accuracy = `0.991804`
- 口径 B：全模型 detector + real recheck = `0.995951`
- 口径 C：非 Reasoner detector + real recheck，Reasoner 不做 detector / 不做 recheck = `0.996346`

因此，论文和汇报中不应再把 `same-model real pilot` 尤其是 Reasoner lane 直接写成当前主方案。

如果涉及 local probe intervention，当前正式口径应固定为：

- 3B 默认 baseline：`baseline_state_interpolation`, `31-35`, `0.6`
- 7B 默认 baseline：`baseline_state_interpolation`, `24-26`, `0.6`
- 7B `24-27, 0.6` 只保留为 `aggressive secondary setting`
- `late_layer_residual_subtraction` 在 3B / 7B 主实验里都不适合作为正式 baseline

推荐写入论文/汇报主结果的数字是：

- 3B `strict_positive`: recovery `0.8125`, wrong-follow `0.10` (ref `0.48`), damage `0`, net `0.4194`
- 3B `high_pressure_wrong_option`: recovery `0.5`, wrong-follow `0.24` (ref `0.42`), damage `0`, net `0.2381`
- 7B `strict_positive`: recovery `0.5455`, wrong-follow `0.20` (ref `0.36`), damage `0`, net `0.1463`
- 7B `high_pressure_wrong_option`: recovery `0.9`, wrong-follow `0.10` (ref `0.26`), damage `0`, net `0.2368`
- 7B aggressive secondary `strict_positive`: recovery `0.7273`, wrong-follow `0.12`, net `0.1951`
- 7B aggressive secondary `high_pressure_wrong_option`: recovery `0.8`, damage `0.0263`, net `0.1842`

如果涉及 white-box mechanistic 子研究线，当前正式分级应固定为：

- `belief_argument`：可保留为正式 mechanistic mitigation 结果
- 14B belief causal transfer：只保留为主文中的 `secondary causal confirmation`
- `identity_profile`：只保留为 `weakly supported mechanistic observation`

关于 14B belief causal transfer，论文/汇报中应明确写：

- `matched_belief_subspace_damping` 降低了 pressured-stage `stance_drift_rate` 与 `pressured_compliance_rate`
- `matched_negative_control` 与 `no_intervention` 相同
- 这支持 `belief_argument` 的 cross-source causal transfer
- 但 `n = 24`、`recovery_rate` 未提升、`baseline_damage_rate = 0.0417`
- 因此它只能作为小样本 secondary causal confirmation，不应写成新的 main mitigation result

推荐主文/汇报只用一个小表或短段落呈现，不单开大节。

关于 GLM 与 Llama 跨家族复测，当前正式区分应固定为：

- GLM：`cross-family positive replication with stronger tradeoff`
- Llama-3.1-8B English bridge prompt：`weak replication / limitation`

GLM 可展开写为：

> GLM reproduces the belief-subspace damping direction across model families, but with a substantially stronger utility/safety tradeoff than the Qwen line.

关于 Llama，论文/汇报中应明确写：

- 英文 prompt 下 belief subspace 更清晰，explained variance 约 `0.64-0.68`，abs coherence 约 `0.40-0.44`
- 但 intervention transfer weak
- 低 alpha 基本不降低 drift 或 compliance
- `alpha = 0.5` 仅小幅降低 drift，从 `0.4583` 到 `0.375`
- compliance 仍为 `1.0`
- recovery 降至 `0.5833`
- baseline damage = `0.25`
- 最终定位为 `mechanism locatable, intervention transfer weak`

Llama 结果应放入 limitation 或 appendix，不应写成 positive replication。

关于 `identity_profile`，论文/汇报中应明确写：

- localization 证据存在
- causal intervention support 不足
- 当前不能 claim `identity-specific mitigation`
- 不升格为正式副线

如果需要写后续可能的下一步，只建议保留：

- 14B belief causal transfer：同配置扩样到 `n=48`
- Llama：`projection-to-logit diagnostic`
- Llama：`causal alignment diagnostic`
- `prefix-span ablation + replay test`

不建议把下面这些继续写成当前主线建议：

- recovery-focused 新配置
- 新的方法调参
- Llama 常规 alpha / k / layer sweep
- 多轮 alpha sweep
- 多轮 control sweep

## 6. 下一阶段最值得推进的切口

优先做：

1. 把 `new15` 条件从“final-answer 行为检测”推进到“过程代理信号测量框架”
2. 为 detector 数据集补充 `baseline -> arm -> recheck` 的多阶段观测字段
3. 把当前 guard-eval 从“离线选择性重查模拟”扩展成更明确的 audit / recheck 研究接口

一句话概括：

> 下一阶段不应把 detector 包装成“已经能稳定识别干扰的分类器”，而应把它定义为“用于触发复查和分配审计预算的风险排序器”。
