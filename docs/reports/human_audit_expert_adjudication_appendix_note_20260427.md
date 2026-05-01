# Expert Adjudication Appendix Note

更新时间：2026-04-27

本说明汇总 `targeted expert adjudication` 的正式结果。该 package 只覆盖 `24` 条最高信息量的 recovery disagreement 样本，因此其定位是 **appendix-ready adjudication note**，而不是新的主实验或 full human validation。

输入文件：

- package  
  `/Users/shiqi/code/graduation-project/docs/reports/human_audit_targeted_expert_adjudication_package_20260427.csv`
- expert results  
  `/Users/shiqi/code/graduation-project/docs/reports/human_audit_targeted_expert_adjudication_results_20260427.csv`

汇总文件：

- overall summary  
  `/Users/shiqi/code/graduation-project/docs/reports/human_audit_expert_adjudication_overall_summary_20260427.csv`
- stratum summary  
  `/Users/shiqi/code/graduation-project/docs/reports/human_audit_expert_adjudication_stratum_summary_20260427.csv`

## 1. Four-way Summary

| expert adjudication | count | proportion |
|---|---:|---:|
| agrees_with_proxy | `1` | `0.0417` |
| agrees_with_human_majority | `10` | `0.4167` |
| mixed | `1` | `0.0417` |
| unresolved | `12` | `0.5000` |

## 2. By Stratum

| stratum | n | agrees_with_proxy | agrees_with_human_majority | mixed | unresolved |
|---|---:|---:|---:|---:|---:|
| identity_profile follow-up | `9` | `0` | `9` | `0` | `0` |
| bridge identity_profile | `8` | `0` | `0` | `0` | `8` |
| bridge belief_argument | `4` | `0` | `0` | `0` | `4` |
| shuffled control | `2` | `1` | `0` | `1` | `0` |
| held-out anomaly | `1` | `0` | `1` | `0` | `0` |

## 3. Appendix-ready Interpretation

这轮 expert adjudication 给出的最稳判断是：

1. 在 `identity_profile follow-up` recovery disagreement 上，专家**更支持 human majority**，而不是 proxy。  
   这说明当前 recovery proxy 在 identity/profile 边界样本上存在系统性过度乐观风险，尤其是把“回到 baseline option”过早当成“恢复独立判断”。

2. 在 free-form bridge rows 上，大量样本应被视为 **unresolved**，而不是简单判给 proxy 或 human majority。  
   这说明当前 artifact 更适合 supporting sanity check，而不足以支撑对开放式 recovery 的强验证。

3. 在高 damage shuffled-control case 上，专家至少有一条明确**支持 proxy**，另一条给出 `mixed`。  
   这说明在明显高 damage 的控制条件下，单纯从表面恢复迹象给出乐观 recovery 判断并不稳妥。

4. 唯一 held-out anomaly 被专家判为 **agrees_with_human_majority**。  
   这表明 objective-local held-out 中仍可能存在少量 proxy under-call 个案，但这不足以改变 Qwen mainline 的总体判断。

## 4. What This Supports

这轮 expert adjudication 可以支持：

- 当前 human sanity check 的主要价值在于：
  - 帮助定位 recovery proxy 的薄弱 strata
  - 尤其是 `identity_profile` 和 free-form bridge rows
- `stance_drift` / `pressured_compliance` / `baseline_damage` 仍然是更可信的人工佐证维度
- `recovery` 在开放式和 identity/profile 条件下不应被写成已经被人类强验证

## 5. What This Does Not Support

这轮 adjudication 不能支持：

- full human validation
- expert validation of the entire paper
- free-form deployment-level mitigation
- “recovery proxy is generally correct”

最稳的论文口径应是：

> A targeted expert adjudication on the highest-disagreement recovery rows indicates that the current recovery proxy is weakest on identity-profile and free-form bridge cases. In those strata, the expert more often sided with the human majority or judged the case unresolved than with the proxy. We therefore retain the human audit as a sanity check / weak corroboration layer rather than as formal human validation.
