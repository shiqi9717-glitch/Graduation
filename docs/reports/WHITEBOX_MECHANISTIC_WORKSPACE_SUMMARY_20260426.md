# White-box Mechanistic Workspace Summary

更新时间：2026-04-26

本文档整理当前工作区 `/Users/shiqi/code/graduation-project` 中 white-box mechanistic 相关流程、结果目录与正式汇报边界。本文只基于已有产物做文件整理与口径归纳；不在 Codex 内运行任何 MPS 依赖实验。

统计收口目录：

- `outputs/experiments/whitebox_mechanistic_statistical_closure/20260426_175452`

优先引用：

- `whitebox_effect_size_table.csv`
- `llama_limitation_summary.md`
- `llama_limitation_summary.json`
- `README.md`

## 1. 总体结论

当前 white-box 机制线应分为四个证据层级：

1. 正式主线：Qwen 3B / 7B 的 `belief_argument` mechanistic mitigation，默认方法为 `baseline_state_interpolation`。
2. 主文 secondary confirmation：Qwen 14B 小样本 cross-source causal transfer，以及 GLM cross-family positive replication with stronger tradeoff。
3. limitation / weak observation：Llama-3.1-8B English bridge prompt 复测、identity_profile white-box follow-up。
4. 探索性支线：Mistral-7B、runtime predictor、sparse monitor、single-head recheck、bridge intervention pilot 等，当前不作为正式主结果；Mistral-7B 仅建议作为 appendix exploratory note 保留一句。

一句话汇报口径：

> `belief_argument` 是当前可进入正式论文/汇报的 white-box mechanistic mitigation 主线；14B 与 GLM 提供有限但有价值的跨规模/跨家族支持；Llama 与 identity_profile 只能作为 limitation 或弱支持观察；Mistral-7B 只作为 appendix exploratory note；subtraction control 与若干探索性脚本不应写成可用主方法。

统计口径边界：

- Qwen 3B / 7B 的 `stance_drift` 与 `recovery` 是 objective-local proxy 口径。
- 不要把 Qwen mainline 与 bridge causal lines 做完全同义的跨范式 leaderboard。

## 2. 核心流程

当前 white-box 机制流程可以按以下顺序理解：

1. 构造 local probe 样本集：比较 baseline、interference / pressured、recovery 等条件下的行为变化。
2. 读取本地模型 hidden states：在普通 macOS Terminal + MPS 中运行 Qwen / GLM / Llama / Mistral 等本地模型，Codex 内只整理已有输出。
3. 做 mechanistic localization：统计层级差异、transition group、patching 或 subspace localization，识别受 pressure 影响的表征区域。
4. 做 causal intervention：用 `baseline_state_interpolation` 或 belief-subspace damping 测试是否能降低 wrong-follow / stance drift / pressured compliance。
5. 做 negative control：用 `late_layer_residual_subtraction`、matched negative control 或 identity-profile matched control 检查干预是否只是泛化扰动。
6. 做跨模型边界检查：Qwen 14B、GLM、Llama、Mistral 只用于支持或限制主线，不自动升格为新主结果。

关键代码入口：

- `scripts/run_local_probe.py`
- `scripts/run_local_probe_mechanistic_analysis.py`
- `scripts/run_local_probe_intervention.py`
- `scripts/run_pressure_subspace_damping.py`
- `scripts/run_belief_causal_transfer.py`
- `scripts/run_identity_profile_whitebox.py`
- `src/open_model_probe/model_runner.py`
- `src/open_model_probe/intervention.py`
- `src/open_model_probe/pressure_subspace.py`
- `src/open_model_probe/identity_profile.py`

MPS 运行规则：

- 任何依赖 MPS / Apple GPU / 本地 transformers 直载模型的命令，都不要在 Codex 内执行。
- 文件整理部门只整理已有输出；如需补跑实验，应给用户一整段可复制的 macOS Terminal 命令，由用户在普通终端执行。

## 3. Qwen 3B / 7B 主线结果

### 3.1 Mechanistic localization

3B 主 mechanistic 目录：

- `outputs/experiments/local_probe_qwen3b_mechanistic/Qwen_Qwen2.5-3B-Instruct/20260419_131943`

3B summary：

- `num_samples = 200`
- `baseline_correct_to_interference_wrong = 48`
- `baseline_correct_to_interference_correct = 48`
- `baseline_wrong_to_interference_correct = 5`
- `baseline_wrong_to_interference_wrong = 99`

7B 主 mechanistic 目录：

- `outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32/Qwen_Qwen2.5-7B-Instruct/20260422_131858`
- `outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32_generalization/Qwen_Qwen2.5-7B-Instruct/20260422_134717`

7B generalization summary：

- `num_samples = 200`
- `baseline_correct_to_interference_wrong = 30`
- `baseline_correct_to_interference_correct = 124`
- `baseline_wrong_to_interference_correct = 5`
- `baseline_wrong_to_interference_wrong = 41`

整理结论：

- Qwen 3B 与 7B 都存在可观察的 pressure-sensitive transition group。
- 3B vulnerable group 更大，7B 更稳但仍存在可定位的错误转移。

### 3.2 Intervention main baseline

主结果表格只应保留下面两个默认 `baseline_state_interpolation` 设置：

- 3B：`baseline_state_interpolation`, layer `31-35`, scale `0.6`
- 7B：`baseline_state_interpolation`, layer `24-26`, scale `0.6`

7B `24-27, 0.6` 只能写成 `aggressive secondary setting`，不要和默认 baseline 放在同等级主结果表里。

3B 默认 baseline：

- 方法：`baseline_state_interpolation`
- layer：`31-35`
- scale：`0.6`
- 目录：`outputs/experiments/local_probe_qwen3b_intervention_main/baseline_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142847`

3B 主结果：

- `strict_positive`: recovery `0.8125`, wrong-follow `0.10` (ref `0.48`), damage `0`, net `0.4194`
- `high_pressure_wrong_option`: recovery `0.5`, wrong-follow `0.24` (ref `0.42`), damage `0`, net `0.2381`
- statistical closure：stance drift delta `-0.1597` CI `[-0.2402, -0.0800]`, compliance delta `-0.2789` CI `[-0.3700, -0.1900]`, recovery delta `0.6906` CI `[0.5000, 0.8667]`, damage `0`

7B 默认 baseline：

- 方法：`baseline_state_interpolation`
- layer：`24-26`
- scale：`0.6`
- 目录：`outputs/experiments/local_probe_qwen7b_intervention_main/baseline_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140142`

7B 主结果：

- `strict_positive`: recovery `0.5455`, wrong-follow `0.20` (ref `0.36`), damage `0`, net `0.1463`
- `high_pressure_wrong_option`: recovery `0.9`, wrong-follow `0.10` (ref `0.26`), damage `0`, net `0.2368`
- statistical closure：stance drift delta `-0.1493` CI `[-0.2200, -0.0800]`, compliance delta `-0.1595` CI `[-0.2300, -0.0900]`, recovery delta `0.7127` CI `[0.5000, 0.9000]`, damage `0`

7B aggressive secondary：

- 设置：`baseline_state_interpolation`, layer `24-27`, scale `0.6`
- 目录：`outputs/experiments/local_probe_qwen7b_intervention_main/aggressive_24_27_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140518`
- `strict_positive`: recovery `0.7273`, wrong-follow `0.12`, net `0.1951`
- `high_pressure_wrong_option`: recovery `0.8`, damage `0.0263`, net `0.1842`

整理结论：

- Qwen 3B / 7B 主结果表格仅保留默认 `baseline_state_interpolation` 设置。
- 7B `24-27, 0.6` 可以作为 aggressive secondary setting。
- 7B 默认不应改成 `24-27`，因为它牺牲 high-pressure 稳健性，并引入少量 baseline damage。

### 3.3 Negative control

不应作为主方法的设置：

- `late_layer_residual_subtraction`

证据目录：

- `outputs/experiments/local_probe_qwen3b_intervention_main/subtraction_control_31_35_s06/Qwen_Qwen2.5-3B-Instruct/20260423_142938`
- `outputs/experiments/local_probe_qwen7b_intervention_main/subtraction_control_24_26_s06/Qwen_Qwen2.5-7B-Instruct/20260423_140853`

整理结论：

- subtraction control 在 3B / 7B 上都不稳定，并出现 baseline damage 或负净收益。
- 它只能写成 failure / control evidence，不能写成可用主方法。

## 4. Belief-subspace causal transfer

### 4.1 Qwen 14B

正式定位：

- 主文中的 `secondary causal confirmation`
- 不是新的 main mitigation result

目录：

- `outputs/experiments/qwen14b_belief_causal_transfer/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/20260424_221308`

关键结果：

- `no_intervention`: stance drift `0.5000`, compliance `0.9583`, recovery `0.7917`, damage `0`
- `matched_belief_subspace_damping`: stance drift `0.2917`, compliance `0.7083`, recovery `0.7917`, damage `0.0417`
- `matched_negative_control`: 与 `no_intervention` 相同
- statistical closure：drift delta `-0.2068` CI `[-0.4167, 0.0000]`, compliance delta `-0.2484` CI `[-0.4167, -0.0833]`, recovery delta `0`, baseline damage `0.0416`

边界：

- `n = 24`
- recovery 未提升
- baseline damage `0.0417`

整理结论：

- 该结果支持 `belief_argument` 的 cross-source causal transfer。
- 只能作为小样本 secondary confirmation，不应写成 14B 强 mitigation 已建立。

### 4.2 GLM

正式定位：

- `cross-family positive replication with stronger tradeoff`

目录：

- `outputs/experiments/pressure_subspace_damping_glm4_9b/Users_shiqi_.cache_huggingface_hub_models--zai-org--glm-4-9b-chat-hf_snapshots_8599336fc6c125203efb2360bfaf4c80eef1d1bf/20260426_005017`

代表性结果：

- 在 `philpapers_belief_argument_to_nlp_survey_belief_argument` transfer 中，alpha `0.75`、k `2` 将 drift 从 `0.4583` 降至 `0.1667`，compliance 从 `0.9583` 降至 `0.7500`，recovery 从 `0.5833` 升至 `0.9583`。
- 同时 baseline damage 达到 `0.3333`，tradeoff 明显强于 Qwen。
- statistical closure：drift delta `-0.2932` CI `[-0.5417, -0.0417]`, compliance delta `-0.2095` CI `[-0.4167, 0.0000]`, recovery delta `0.3769` CI `[0.1667, 0.5833]`, baseline damage `0.3325` CI `[0.1667, 0.5417]`

整理结论：

- GLM 复现了 belief-subspace damping 的方向。
- 但安全/效用 tradeoff 更强，因此不能直接写成无代价跨家族成功。

### 4.3 Llama-3.1-8B

正式定位：

- `weak replication / limitation`
- `mechanism locatable, intervention transfer weak`

目录：

- `outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep/0e9e39f249a16976918f6564b8830bc894c89659/20260426_151011`

关键结果：

- 英文 bridge prompt 下，explained variance 约 `0.64-0.68`，abs coherence 约 `0.40-0.44`。
- alpha `0.5` 仅将 drift 从 `0.4583` 降至 `0.3750`。
- compliance 仍为 `1.0`。
- recovery 从 `0.6667` 降至 `0.5833`。
- baseline damage `0.25`。
- statistical closure：drift delta `-0.0824` CI `[-0.2917, 0.1250]`, compliance delta `0`, recovery delta `-0.0847` CI `[-0.2917, 0.1250]`, baseline damage `0.2499` CI `[0.0833, 0.4167]`

整理结论：

- Llama 的 subspace 更清晰，但可定位性没有稳定转化为可控干预。
- 应放入 limitation 或 appendix，不能写成 positive replication。
- 停止后续 Llama alpha / k / layer sweep。

### 4.4 Mistral-7B

当前定位：

- 探索性 cross-family 支线
- 不进入主文 positive replication
- 建议只作为 appendix exploratory note 保留一句

目录：

- `outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_143645`
- `outputs/experiments/non_chinese_belief_causal_transfer_mistral7b/mistralai_Mistral-7B-Instruct-v0.3/20260426_144430`

英文 prompt 结果：

- drift `0.5833 -> 0.3333`
- compliance `1.0 -> 0.7917`
- recovery `0.4167 -> 1.0`
- baseline damage `0.375`

中文指令 prompt 结果：

- drift `0.5417 -> 0.5000`
- compliance `1.0 -> 0.9583`
- recovery `0.4583 -> 0.5833`
- baseline damage `0.4167`

整理结论：

- Mistral 有方向性信号，但 baseline damage 很高，且跨 prompt variant 不稳定。
- 当前不建议写成 positive replication；建议仅作为 appendix exploratory note。

推荐英文句式：

> Mistral-7B showed directional belief-subspace damping signals under an English bridge prompt, but the effect was unstable across prompt variants and accompanied by high baseline damage; we therefore treat it as exploratory only.

statistical closure：

- English prompt drift delta `-0.2491`, compliance delta `-0.2094`, recovery delta `0.5836`, baseline damage `0.3769`

## 5. Identity Profile 子研究线

正式定位：

- `weakly supported mechanistic observation`
- 不是正式副线
- 不是 identity-specific mitigation

目录：

- `outputs/experiments/identity_profile_whitebox_qwen3b/Qwen_Qwen2.5-3B-Instruct/20260424_104752`
- `outputs/experiments/identity_profile_whitebox_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_103515`
- `outputs/experiments/identity_profile_whitebox_followup_qwen7b/Qwen_Qwen2.5-7B-Instruct/20260424_120058`

follow-up 结论：

- localization 证据存在。
- `profile_prefix_gating` 没有稳定优于 `matched_early_mid_negative_control`。
- causal intervention support 不足。

整理结论：

- 当前不能 claim identity-specific mitigation。
- 如果后续要补实验，只保留 `prefix-span ablation + replay test` 作为最小诊断方向。
- 不建议继续写常规多轮 alpha / control sweep。

## 6. 辅助与历史探索目录

以下目录说明工作区内还存在较多探索性白盒或半白盒产物，但当前不应进入正式主结果：

- `outputs/experiments/local_probe_internal_signal_predictor*`
- `outputs/experiments/local_probe_runtime_safe_predictor*`
- `outputs/experiments/local_probe_runtime_safe_signal*`
- `outputs/experiments/local_probe_sparse_runtime_monitor*`
- `outputs/experiments/local_probe_single_head_recheck*`
- `outputs/experiments/local_probe_qwen3b_late_layer*`
- `outputs/experiments/bridge_intervention_pilot_qwen7b*`
- `outputs/experiments/bridge_benchmark*`

整理结论：

- 这些支线可作为后续创新或失败模式材料。
- 当前论文/汇报主线应优先引用 Qwen intervention、Qwen14B secondary confirmation、GLM/Llama boundary、identity weak observation。

## 7. 当前推荐写作分层

主文可写：

- Qwen 3B / 7B 默认 `baseline_state_interpolation` 是当前 white-box mitigation 主结果，其中主结果表格只保留 3B `31-35, 0.6` 与 7B `24-26, 0.6`。
- Qwen 3B / 7B 主文 effect size 与 CI 优先引用 statistical closure 表，但需注明 objective-local proxy 口径。
- 7B `24-27, 0.6` 只作为 aggressive secondary setting，不与默认 baseline 同级呈现。
- Qwen 14B 是小样本 `secondary causal confirmation`。
- GLM 是 cross-family positive replication with stronger tradeoff。

Limitation / appendix 可写：

- Llama-3.1-8B English bridge prompt：mechanism locatable, intervention transfer weak。
- identity_profile：localization evidence exists, causal intervention support insufficient。
- Mistral：appendix exploratory note，存在方向信号但跨 prompt variant 不稳定且 baseline damage 过高。

不要写：

- `late_layer_residual_subtraction` 是可用主方法。
- 7B `24-27, 0.6` 是默认 baseline。
- 7B `24-27, 0.6` 和默认 baseline 放在同等级主结果表里。
- 14B 已建立强 mitigation。
- Llama 是 positive replication。
- identity_profile 已建立 identity-specific mitigation。
- Mistral 当前已完成正式 cross-family replication。

## 8. 后续最小建议

只建议保留以下最小后续项：

- Qwen 14B：同配置扩样到 `n=48`。
- Llama：`projection-to-logit diagnostic` 与 `causal alignment diagnostic`。
- identity_profile：`prefix-span ablation + replay test`。
- Mistral：如需写入，只保留 appendix exploratory note，不建议补常规 sweep。

不建议继续把常规 alpha / k / layer sweep 写成主线后续工作。
