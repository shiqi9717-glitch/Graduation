# Qwen Clean Protocol Appendix Note

更新时间：2026-05-01

本说明用于把 2026-04-30 新增的 Qwen clean protocol belief causal transfer 结果接入 evidence package / appendix。它只整理正式统计分析后的收束口径，不修改实验结果，不改代码，也不上调现有主文结论强度。

## Artifact Paths

- Qwen 7B clean
  - `/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen7b_clean/Qwen_Qwen2.5-7B-Instruct/20260430_161102`
- Qwen 3B clean
  - `/Users/shiqi/code/graduation-project/outputs/experiments/pressure_subspace_damping_qwen3b_clean/Qwen_Qwen2.5-3B-Instruct/20260430_162151`
- Qwen 14B n=48
  - `/Users/shiqi/code/graduation-project/outputs/experiments/qwen14b_belief_causal_transfer_n48/Qwen_Qwen2.5-14B-Instruct/20260430_162437`

## Conservative Readout

### Qwen 3B clean protocol

定位：

- `positive confirmation`
- 可作为 clean protocol 下的副主线支持

delta vs `no_intervention`：

- drift `-0.3750`, CI `[-0.5208, -0.2292]`
- compliance `-0.4167`, CI `[-0.5625, -0.2708]`
- recovery `+0.0833`, CI `[0.0000, 0.1875]`
- damage `+0.0417`, CI `[0.0000, 0.1042]`

边界：

- matched negative control 未复现收益
- negative-control damage `+0.1458`
- 可支持 `clean protocol positive confirmation`
- 不应被写成“所有 Qwen clean protocol 都正向确认”

### Qwen 7B clean protocol

定位：

- `weak / mixed`
- 只适合作为 `appendix / boundary evidence`

delta vs `no_intervention`：

- drift `-0.0417`, CI `[-0.1042, 0.0000]`
- compliance `0.0000`
- recovery `+0.0417`, CI `[0.0000, 0.1042]`
- damage `+0.0417`, CI `[0.0000, 0.1042]`

边界：

- matched negative control 为严格零效应
- 当前只支持 `specific but behaviorally weak`
- 不要写成 `positive confirmation`
- 不要把它升级成 clean protocol 的稳定主结果

### Qwen 14B n=48

定位：

- `harmful / not recommended`
- 只适合作为 `cautionary / boundary evidence`

delta vs `no_intervention`：

- drift `+0.0417`, CI `[-0.1042, 0.1875]`
- compliance `-0.1042`, CI `[-0.2083, 0.0000]`
- recovery `-0.1042`, CI `[-0.2500, 0.0625]`
- damage `+0.1042`, CI `[0.0208, 0.1875]`

边界：

- matched negative control 为零效应
- 不支持 `clean positive secondary confirmation`
- 不要再写 `Qwen 14B n=48 strengthens the 14B line`
- 不应覆盖旧的 14B `n=24` secondary causal confirmation 的历史定位；它只说明 clean protocol 扩样后并未形成更强支持

## Stable Overall Closing

当前最稳的收束是：

- Qwen 3B clean protocol 支持副主线
- Qwen 7B clean protocol 只能做 appendix / boundary evidence
- Qwen 14B `n=48` 只能做 cautionary / boundary evidence

## Deprecated Wording

以下口径应视为废弃，不应继续写入 package、appendix 或论文草稿：

- `all Qwen clean protocol runs confirm belief causal transfer mitigation`
- `Qwen 7B clean is positive confirmation`
- `Qwen 14B n=48 strengthens the 14B line`

## Use in Package

这份说明的用途是：

- 为论文撰写部门提供 appendix-ready 边界说明
- 为结果分析部门和文件整理部门提供统一索引口径
- 避免把 clean protocol 结果误并入旧的主文主线强结论
