# Held-out Prediction Protocol

更新时间：2026-05-07

本协议用于验证 `regime_diagnostic_protocol_20260507.md` 里的 rules 不是纯事后拟合。它不跑新实验，只规定 train / held-out split、预测伪代码、expected outcome table 和 mismatch 处理办法。

## 1. Split Design

### Formal train rows

用以下 4 行建 rule：

- Qwen7B mainline
- Qwen7B PSD n=100
- GLM-4-9B n=100
- Llama-3.1-8B n=100

理由：

- 这 4 行已经覆盖了 4 类 regime：`clean-controllable`、`specific-but-not-controllable`、`weak/tradeoff-limited`、`locatable-but-not-controllable`。
- 它们也同时覆盖了 `near-zero negative delta`、`negative delta strongly negative`、`positive negative delta` 三种关键诊断形态。

### Primary held-out rows

用以下 3 行做 primary held-out：

- Qwen3B mainline
- Qwen14B n=100
- Qwen7B free-form `prefill_only`

理由：

- Qwen3B 与 Qwen14B 都不是 train rows，但各自代表 replication / cross-scale support。
- free-form `prefill_only` 用来检查 regime rules 能否迁移到 `generation stability` 这一维。

### Secondary held-out rows

作为 boundary-style held-out，再单独看：

- Qwen7B free-form `first_3_tokens`
- Qwen7B free-form `continuous`
- identity/profile follow-up

理由：

- 它们不是完整同构 feature rows，但最能检验 framework 是否真的把 `quality collapse` 与 `family boundary` 分开。
- `identity/profile` 应被视为 `partial-feature held-out`，不是与 belief rows 完全同协议的 test row。

### Exclusions

- Qwen3B PSD 暂不作为 formal held-out。
- 原因：当前 frozen analysis 里它与 Qwen3B mainline 共享同一套 projection diagnostic features，容易形成 feature leakage。

## 2. Prediction Rule Pseudocode

```python
def predict_regime(row):
    if row.has_freeform_metrics:
        if row.readable_rate < 0.20 or row.repetition_rate >= 0.90:
            return "generation-unstable"
        if row.readable_rate >= 0.90 and row.drift_rate <= 0.10 and row.wrong_follow_rate <= 0.16:
            return "front-loaded controllable-but-bounded"
        return "quality-limited free-form boundary"

    if row.identity_or_profile_family:
        return "boundary / insufficient-transfer-evidence"

    if row.target_logit_delta <= -5.0 and row.negative_logit_delta <= -0.10 and 3.0 <= row.specificity_ratio <= 50.0:
        return "clean-or-secondary controllable"

    if row.target_logit_delta <= -5.0 and abs(row.negative_logit_delta) < 0.01 and row.specificity_ratio > 100:
        return "specific-but-not-controllable"

    if row.target_logit_delta > -2.0 or row.negative_logit_delta > 0.0:
        return "locatable-but-not-controllable"

    return "weak / tradeoff-limited"
```

解释：

- 该规则故意把 `high specificity + near-zero negative delta` 切成单独分支。
- 这正是为了防止把 Qwen7B PSD / GLM 误判成 clean control。

## 3. Predicted Regimes For Held-out Rows

| Held-out setting | Diagnostic signals | Predicted regime | Expected observed regime | Predicted match? |
| --- | --- | --- | --- | --- |
| Qwen3B mainline | `Δ_target=-5.10`, `Δ_neg=-0.93`, `S=5.48`, `corr=-0.52`, `damage=0.00` | clean-or-secondary controllable | clean-controllable | Yes |
| Qwen14B n=100 | `Δ_target=-7.33`, `Δ_neg=-1.43`, `S=5.14`, `corr=+0.08`, `damage=0.01` | clean-or-secondary controllable | secondary-controllable | Yes |
| Qwen7B free-form `prefill_only` | `readable=0.94`, `repetition=0.00`, `drift=0.10`, `wrong_follow=0.16` | front-loaded controllable-but-bounded | front-loaded diagnostic effect | Yes |
| Qwen7B free-form `first_3_tokens` | `readable=0.42`, `repetition=0.58`, behavior unchanged vs short horizon | quality-limited free-form boundary | brittle / quality-collapsing | Yes |
| Qwen7B free-form `continuous` | `readable=0.00`, `repetition=1.00` | generation-unstable | generation-unstable | Yes |
| identity/profile follow-up | no matched belief-style projection suite; causal support insufficient | boundary / insufficient-transfer-evidence | boundary / weak mechanistic observation | Yes |

## 4. Optional Secondary Check

若 Analysis 部门希望再做一个 non-formal held-out row，可加入：

| Setting | Why secondary only |
| --- | --- |
| Qwen7B PSD n=100 | 它在 train 规则形成时已经作为关键反例参与阈值确定，不宜再算 formal held-out |

## 5. Mismatch Plan

### Mismatch that weakens the framework

- 若 Qwen3B mainline 被预测成 non-controllable，而观察上仍是 clean mainline replication。
- 若 Qwen14B 被预测成 `specific-but-not-controllable`，而观察上继续保持当前 secondary positive profile。
- 若 free-form `continuous` 被预测为 stable，但实际仍是 unreadable collapse。

这些 mismatch 说明阈值设计本身不稳，尤其是 `negative delta` 与 `specificity` 的切分规则可能过拟合。

### Mismatch that may still support the boundary claim

- 若 identity/profile 出现部分 localization signal，但仍无法进入 clean-controllable。
- 若 `first_3_tokens` 比预期更稳定，但仍不优于 `prefill_only` / `first_token_only` 的行为方向。

这些 mismatch 不一定推翻框架，反而可能支持：

- `boundary is real, but threshold is noisy`
- `generation stability is a separate axis with softer horizon boundary than first estimated`

## 6. How Analysis Should Score Match Rate

建议两级评分：

- `strict match`: predicted regime 与 observed regime 文本同类
- `collapsed-family match`: predicted / observed 落在同一大类

Collapsed families:

- `clean-or-secondary controllable`
- `specific / weak / tradeoff-limited`
- `locatable-but-not-controllable`
- `free-form quality-limited or unstable`
- `boundary / insufficient-transfer-evidence`

## 7. Bottom Line

这套 held-out 协议的核心不是证明“单一 predictor 完美泛化”，而是测试下面这条更稳的命题：

> High target-logit movement plus low damage is closer to a necessary profile for controllability, while high specificity alone can still map to weak, tradeoff-heavy, or non-controllable regimes; free-form quality collapse should be predicted on a separate axis.

Analysis 部门拿到这份协议后，只需为 held-out rows 填 observed regime 并算 `strict match` / `collapsed-family match` 即可。
