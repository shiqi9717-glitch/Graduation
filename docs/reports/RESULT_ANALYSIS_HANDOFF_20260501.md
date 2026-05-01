# Result Analysis Handoff

更新时间：2026-05-01

本说明只整理当前已经冻结或已明确采用的 white-box / mechanistic intervention 结果口径，供结果分析部门与论文撰写部门继续消化使用。

边界：
- 不新增实验
- 不覆盖 frozen outputs
- 不把 exploratory pilot 升格为正式 benchmark
- 不把 human audit 写成 full human validation

## 1. Main Claim Routing

当前主线已经明确转为 `white-box mechanism-informed intervention`。

主 claim 应优先建立在以下结构上：
- `baseline_state_interpolation` 作为主 baseline intervention
- `pressure_subspace_damping` 作为 less-oracle 副主线
- Qwen 3B / 7B projection-to-logit diagnostic 作为 mechanistic support
- Qwen 3B / 7B layer-wise clean subspace diagnostic 作为 localization support

推荐分桶：
- main claim: Qwen objective-local intervention mainline
- robustness: formal controls, held-out validation
- boundary evidence: pressure_subspace_damping clean protocol, Llama localization-vs-controllability boundary, 14B cautionary result
- appendix sanity check: human audit sanity package, targeted expert adjudication

## 2. Priority Result Directories

结果分析部门本轮最优先消化的四个目录：
- `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_projection_logit_diagnostic/Qwen_Qwen2.5-7B-Instruct/20260501_183852`
- `/Users/shiqi/code/graduation-project/outputs/experiments/qwen3b_projection_logit_diagnostic/Qwen_Qwen2.5-3B-Instruct/20260501_185401`
- `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_layer_wise_subspace/Qwen_Qwen2.5-7B-Instruct/20260501_185620`
- `/Users/shiqi/code/graduation-project/outputs/experiments/qwen3b_layer_wise_subspace/Qwen_Qwen2.5-3B-Instruct/20260501_184906`

配套应一起引用的稳健性目录：
- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_formal_controls_qwen7b/20260427_121722`
- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_formal_controls_qwen3b/20260427_125304`
- `/Users/shiqi/code/graduation-project/outputs/experiments/whitebox_qwen7b_heldout_eval/qwen7b_heldout_mainline/20260427_135639`
- `/Users/shiqi/code/graduation-project/outputs/experiments/human_audit_freeform_sanity/20260427_184953`

pressure-subspace clean protocol 结果目录：
- `/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen7b_clean/Qwen_Qwen2.5-7B-Instruct/20260430_161102`
- `/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen3b_clean/Qwen_Qwen2.5-3B-Instruct/20260430_162151`
- `/Users/shiqi/code/graduation-project/outputs/experiments/qwen14b_belief_causal_transfer_n48/Qwen_Qwen2.5-14B-Instruct/20260430_162437`

## 3. Stable Readout

### 3.1 Projection-to-logit diagnostic

Qwen 7B:
- pressured projection 并未高于 baseline projection
- 但 belief-subspace damping 在 logit margin 上仍呈强负向移动
- 更适合写成 `causal alignment is weak/mixed rather than absent`

Qwen 3B:
- pressured projection 显著高于 baseline projection
- projection increase 与 stance drift 呈负相关
- belief-subspace damping 对 belief logit 呈明显负向干预
- 适合写成 `best positive mechanistic confirmation in the clean protocol family`

### 3.2 Layer-wise clean subspace

Qwen 7B strongest band：
- layer 24: `explained_variance_sum = 0.8941`
- layer 25: `0.8510`
- layer 23: `0.8135`
- layer 22: `0.8045`
- layer 26: `0.7696`

Qwen 3B strongest band：
- layer 31: `0.8468`
- layer 32: `0.8205`
- layer 33: `0.8111`
- layer 27: `0.7985`
- layer 34: `0.7647`

支持的稳健口径：
- 7B strongest band 集中在 `24-26` 左右，与主 baseline 选层一致
- 3B strongest band 集中在 `31-33`，支持 `31-35` baseline 区间

### 3.3 Pressure-subspace clean protocol

当前最稳写法：
- 3B: strongest positive confirmation
- 7B: weak / mixed support
- 14B: harmful or over-strong; not recommended as positive replication

## 4. Boundary / Do Not Overclaim

不要写：
- `human audit provides full human validation`
- `Qwen 14B cleanly replicates the pressure-subspace intervention`
- `Llama is a strong positive replication`
- `Qwen 7B pressure_subspace_damping is a strong success`
- `Table 4 exploratory pilot is a full intervention benchmark`

推荐替代表述：
- `human audit provides a sanity-check style human corroboration`
- `Qwen 14B currently serves as a cautionary or boundary result`
- `Llama currently supports a localization-vs-controllability boundary`
- `Qwen 7B pressure_subspace_damping remains weak or mixed`
- `the exploratory pilot should remain clearly labeled as exploratory`

## 5. Deprecated Artifact Routing

以下目录或口径不要再作为正式结果引用：
- `/Users/shiqi/code/graduation-project/outputs/experiments/qwen7b_layer_wise_subspace/Qwen_Qwen2.5-7B-Instruct/20260501_184826`

原因：
- 该目录对应 7B layer range bug 的旧失败结果
- 正式结果应改引 `20260501_185620`

职责边界也不要再混用：
- `scripts/run_belief_causal_transfer.py`: clean source-specified protocol / projection diagnostic / train-only layer-wise export
- `scripts/run_pressure_subspace_damping.py`: exploratory mixed-strata sweep only

## 6. Recommended Writing Order

如果要给论文撰写部门供料，推荐先按以下顺序组织：
1. objective-local mainline baseline state interpolation
2. formal controls + held-out validation
3. pressure_subspace_damping clean protocol
4. Qwen 3B / 7B projection and layer-wise mechanistic diagnostics
5. human audit sanity package and targeted adjudication as appendix-side support

这样可以保持：
- 主效果先行
- 机制支持跟随
- boundary evidence 单独说明
- appendix sanity check 不误升格
