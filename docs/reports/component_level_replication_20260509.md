# Component-level Replication (Qwen 7B, n=50)

结果表：`full_residual` `drift -0.44 / compliance -0.44 / recovery +0.42 / damage 0.00 / score 1.30`；`attention_only` `0.00 / 0.00 / +0.02 / 0.00 / 0.02`；`mlp_only` `0.00 / 0.00 / +0.02 / 0.00 / 0.02`。

1. `full_residual` pathway 在 `n=50` 下仍然显著为正，且无 baseline damage，说明 clean-control effect 继续可见。  
2. `attention_only` 与 `mlp_only` 都接近零效应；拆分后，主效应基本消失。  
3. 这支持更窄的结论：在当前 hook interface 与 tested window 下，干预作用依赖完整 `full_residual` pathway，而不是单一组件即可复现。  
4. 与此前 `n=24` exploratory 结果一致，属于 replication success；更准确的分级是 `lightweight but replicated exploratory`，不是 pathway-final proof。  

建议：可写入论文正文，但更适合放在 Section 8 的 mechanism-boundary / component-level paragraph，作为对主机制命题的补强，而非新的主结果线。
