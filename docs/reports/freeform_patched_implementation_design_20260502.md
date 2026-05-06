# Free-form Patched Implementation Design

更新时间：2026-05-02

本说明只做实现设计确认，不改代码，不跑实验，不改论文正文。

## 1. 函数签名

建议在 `LocalProbeRunner` 新增：
`generate_with_final_token_residual_patches(self, *, prompt: str, layer_patch_map: Dict[int, Any], max_new_tokens: int = 150, temperature: float = 0.0, do_sample: bool = False, stop_token_ids: Sequence[int] | None = None, use_cache: bool = False) -> Dict[str, Any]`。
返回值建议包含：`generated_text`、`generated_token_ids`、`num_new_tokens`、`finish_reason`、`patched_layers`、`per_step_top_logits`。
风格上保持和 `patch_final_token_residuals_multi()` 一致：输入显式、输出为单个 `Dict[str, Any]`。

## 2. Per-step hook 集成

每个 decode step 都应在目标层把当前 step 的 `hidden_states[:, -1, :]` 替换为固定 patch tensor，然后再读该步 logits。
hook 可以在整个 generation 期间注册一次并复用，不必每步重新注册；这样更简洁，也更少 Python 开销。
每步只变化 `model(**kwargs)` 的输入，不变化 hook 函数闭包里的 tensor。
移除时机应是整个 generation `try/finally` 结束后统一 `handle.remove()`，避免异常路径泄漏。

## 3. Patch tensor 复用

对当前 `baseline_state_interpolation` 定义，单个 item + scenario 在整段 free-form decode 中复用同一组 patch tensor 是正确语义。
原因是现有 `build_layer_patch_map()` 产出的就是静态 anchor：baseline 用 `baseline_tensor`，pressured/recovery 用 `(1-β)*interference + β*baseline`。
如果改成每步基于动态 hidden state 重新插值，那就不再是当前 mainline 的同一干预，而是新方法。
因此本轮应坚持“固定 patch tensor，逐步重复施加”。

## 4. KV cache 处理

 correctness-first 建议：首版先 `use_cache=False`，用 naive autoregressive loop 保证 patch 后状态不会与旧 cache 语义冲突。
原因是 patch 发生在 block output，若继续复用前一步未按 patch 重算的 layer cache，严格语义上会出现不完全一致。
等首版跑通后，如需优化，再单独验证“开启 cache 的近似实现”和无 cache 结果是否一致。
所以代码部门首版不应把 cache 作为默认路径。

## 5. MPS 内存管理

7B + MPS + fp32 + 每步 hook 的主要风险更像耗时和碎片化，不像立即的巨型 batch OOM；因为这里本质应是 batch size 1、逐条 decode。
建议严格逐 item、逐 scenario、逐 generation 执行，不做 batch generation，不把多个 item 拼成同一 forward。
每轮 generation 结束后立刻释放 step 级列表与 outputs 引用；必要时做 `gc.collect()`，并在有 torch 接口时清理 MPS cache。
正式 500 次 generation 前，先用 10 条样本检查内存是否单调上涨。

## 6. 试点策略

建议先跑 10 条样本试点，参数与正式完全相同，只缩小 item 数，不缩 prompt、不缩 `max_new_tokens`、不改 patch 层。
试点检查三件事：文本是否明显退化，首字母/答案抽取率是否足够高，长时间 decode 后内存是否稳定。
若出现大量重复、空泛自纠、截断或抽取率过低，再回头调整 prompt 末尾 wording，而不是先动 patch 数学定义。
试点通过后再扩到 100 条正式集。

## 7. Patch tensor 来源

如果 free-form prompt 末尾改成“给出选项字母并用 1-2 句话说明理由”，那它已不再等同于 mainline 旧 prompt，所以旧 hidden states 不能视为严格同构来源。
因此，若追求语义严格一致，patch tensor 应在新的 free-form baseline / pressured prompt 下重新提取，再交给 `build_layer_patch_map()`。
只有在 prompt 文本完全不变时，才适合直接复用现有 `run_local_probe_intervention.py` 相关中间 hidden-state 输出。
本轮设计建议把“重新提取 patch 来源 hidden states”视为正确默认方案。

## Go / No-Go

`Go`，但前提是采用 correctness-first 版本：
固定 patch tensor、per-step hook、首版 `use_cache=False`、batch size 1、先做 10 条试点。
当前没有新的结构性技术阻塞点；最大的真实风险是速度慢和文本退化，而不是实现路径不成立。
不建议再沿用 2026-05-01 note 中对 full patched free-form 的总体 `No-Go` 口径。
