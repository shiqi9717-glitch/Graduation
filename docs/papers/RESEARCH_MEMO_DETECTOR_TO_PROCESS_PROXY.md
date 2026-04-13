# 研究 Memo：为什么当前 detector 更像 risk ranker，以及后续应如何转向 process proxy / measurement

更新时间：2026-04-13

## 1. 执行摘要

当前 interference detector 更适合被定义为：

- 一个 `risk ranker`
- 一个 `trigger policy`
- 一个 `offline selective re-check` 的预算分配器

而不适合被定义为：

- 一个高精度强分类器
- 一个可直接替代人工审计或过程监督的终局方案

原因不是“这个方向完全没用”，而是当前数据和标签主要只看单次回答末端结果，缺少足够的过程可观测性，因此 detector 学到的更多是“哪些样本更值得复查”的弱排序信号，而不是“是否发生了真实干扰”的稳定机制判别。

## 2. 为什么说当前 detector 更像 risk ranker

### 2.1 从结果上看：有一点排序信号，但远不是强分类

当前结果最能支撑的判断是“弱排序、弱触发”，不是“强识别”：

- `outputs/experiments/interference_detector_new15_full_detector_grid/detector_model_comparison.csv`
  - `hybrid-sentence-structured-logreg` 在 relaxed 设定下最佳 `test_f1=0.05696`
  - 同表最佳 `test_roc_auc=0.54525`
  - 对应 `trigger_rate` 大约在 `0.061-0.072`
- `outputs/experiments/interference_detector_new15_10k_embedding/detector_model_comparison_embedding_new15_10k.csv`
  - `embedding-logreg` 在 strict 设定下 `test_f1=0.17391`
  - `test_roc_auc=0.54837`
  - `trigger_rate=0.058`

这些数字说明：

- 分数并非纯噪声，因为 AUC 稍高于 0.5
- 但信号强度仍很有限，远不足以支撑“是否受干扰”的强断言
- 更合理的解释是：模型只能把少量高风险样本排到前面，适合做 selective re-check，而不是一次性最终裁决

### 2.2 从数据上看：标签极稀疏，天然更像风险检索任务

`outputs/experiments/interference_detector_mvp/interference_dataset_summary.json` 显示：

- strict 标签：`951 / 43340`，正例占比约 `2.19%`
- relaxed 标签：`1250 / 44845`，正例占比约 `2.79%`

这种分布意味着：

- 即使 detector 学到真实信号，也更像在极度不平衡空间里做“找可疑样本”
- 在这类问题上，排序、触发率、预算约束下的收益，通常比单点 precision / recall 更适合作为主评价逻辑

### 2.3 从实现上看：系统本来就把 detector 放在 trigger / re-check 位置

`src/mitigation/README.md` 已经明确把 guard eval 定义为：

- `offline selective re-check simulation`

`scripts/run_interference_detector.py` 中的 guard 逻辑也很直接：

- 先读取 `predicted_answer`
- 再根据 `interference_score >= threshold` 决定是否 `trigger_recheck`
- 若触发，则把 `recheck_answer` 覆盖成 `final_answer`

这套实现天然对应的就是：

- 风险打分
- 复查触发
- 离线预算模拟

而不是：

- 细粒度过程解释
- 实时在线机制监督
- 高置信分类判决

### 2.4 从标签构造上看：当前标签更接近“后验行为标签”，不是机制标签

`src/mitigation/interference_dataset.py` 里的 `strict_label` / `relaxed_label` 依赖的是：

- `baseline_accuracy_prob`
- `control_reference_accuracy_prob`
- `answer_equals_wrong_option`
- `arm_correct_prob`
- `arm_sycophancy_prob`
- `explicit_wrong_option`

也就是说，当前标签主要表达的是：

- 在 baseline / control 足够稳定时
- 当前 arm 是否跟随了错误选项
- 当前 arm 是否表现出较高的错误跟随 / sycophancy 概率

这更像“结果层面的 operational label”，而不是“过程层面的 causal label”。因此 detector 学到的其实是：

- 哪些末端行为组合看起来像干扰

而不是：

- 干扰是在哪一步形成的
- 模型是否先想到正确答案、后被用户意见拉偏
- 模型是否在 re-ask 时可恢复

## 3. 真正的瓶颈：不是纯模型结构问题，而是过程不可观测

当前项目最关键的限制，不在于分类器形式太简单，而在于可观测变量太少。

### 3.1 当前可观测到的主要还是末端字段

当前 detector 样本里能稳定用到的主要字段包括：

- `prompt_prefix`
- `question_text`
- `answer_text`
- `predicted_answer`
- `baseline_answer`
- `ground_truth`
- `wrong_option`
- `authority_level / confidence_level / explicit_wrong_option`

这些字段足以支持：

- 条件强度分层
- 最终答案偏移分析
- 离线重查收益模拟

但还不足以支持真正的过程判断。

### 3.2 当前最缺的过程信号

最值得补的过程信号包括：

- reasoning trace
- 中间候选答案变化
- 第一次判断与最终判断是否分离
- 对错误选项的显式引用和引用位置
- neutralized re-ask 后是否回到 baseline / ground truth
- 同题多次采样下答案漂移轨迹
- answer extraction 之前的原始结构化思路片段

如果没有这些信号，detector 只能在“看起来危险”的样本上打高分，但很难解释：

- 是真的被用户锚定了
- 还是本来就不会这道题
- 还是 recency / wording / invalid extraction 在作祟

## 4. 文献重组：四条线怎么对应到当前项目

## 4.1 measurement / label design

这一条线关注“我们到底在测什么”。

核心文献：

- `Towards Understanding Sycophancy in Language Models`
- `Measuring Sycophancy of Language Models in Multi-turn Dialogues`
- `Sycophancy Claims about Language Models: The Missing Human-in-the-Loop`
- `Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models`
- `Value Alignment Tax: Measuring Value Trade-offs in LLM Alignment`

对当前项目的启发：

- 先把 detector 标签写清楚，它测的是 `wrong-option-follow risk`，不是“所有 sycophancy”
- 区分 `behavioral proxy`、`human-perceived sycophancy`、`capability failure`
- 把主评价从单点分类分数，转向预算约束下的 re-check utility

与当前代码最相关的锚点：

- `src/mitigation/interference_dataset.py`
- `outputs/experiments/interference_detector_mvp/interference_dataset_summary.json`

## 4.2 process monitoring / proxy signals

这一条线关注“过程里有没有可监控的代理信号”。

核心文献：

- `Sycophantic Anchors: Localizing and Quantifying User Agreement in Reasoning Models`
- `Internal Reasoning vs. External Control: A Thermodynamic Analysis of Sycophancy in Large Language Models`
- `Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs`

对当前项目的启发：

- 不要只看 final answer，要看用户错误意见在何处进入并固定
- 可把 `prompt_prefix` 中的 authority / confidence / explicit wrong option 作为已知外源因素
- 下一步要补“模型内部或半内部过程代理”，例如 first-pass answer、re-ask drift、anchor span、self-consistency

与当前代码最相关的锚点：

- `src/data/local_data_perturber.py`
  - 已经把 `authority_level / confidence_level / explicit_wrong_option` 离散化成 `new15` 条件
- `scripts/run_interference_detector.py`
  - 已经具备 thresholded trigger 接口，适合接 process proxy

## 4.3 audit + recheck / oversight

这一条线关注“不是一次判死，而是把高风险样本送去复查”。

核心文献：

- `Sycophancy under Pressure: Evaluating and Mitigating Sycophantic Bias via Adversarial Dialogues in Scientific QA`
- `Measuring Sycophancy of Language Models in Multi-turn Dialogues`
- `Sycophancy Claims about Language Models: The Missing Human-in-the-Loop`

对当前项目的启发：

- 当前 guard-eval 已经是一个很自然的 audit 原型
- 后续重点不该是“detector 能否完美识别”，而是“在固定 extra-call 预算下，能否显著降低 wrong-option follow”
- 最重要的研究单位应从“单样本分类正确率”转向“触发策略的收益曲线”

与当前代码最相关的锚点：

- `src/mitigation/README.md`
- `scripts/run_interference_detector.py`
  - `_guard_metrics_for_threshold(...)`
  - `offline_selective_recheck_simulation`

## 4.4 causal / mechanistic mitigation

这一条线关注“为什么会发生，以及能否针对机制做干预”。

核心文献：

- `From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning`
- `Simple synthetic data reduces sycophancy in large language models`
- `Mitigating the Alignment Tax of RLHF`
- `Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization`
- `FLAME: Factuality-Aware Alignment for Large Language Models`
- `Large language models and causal inference in collaboration: A comprehensive survey`

对当前项目的启发：

- 机制级 mitigation 仍然重要，但不应在过程观测极弱时过早重押
- 更合理的顺序是：先把 measurement / proxy / audit 做扎实，再决定是否进入 representation-level 或 tuning-level 干预

与当前项目最相关的现实判断：

- 现在直接跳到“强 mitigation”会比较冒进
- 当前更适合做 `measurement-first` 的中间层研究

## 5. 下一阶段最值得推进的研究切口

我建议优先推进这三个切口，其中第一个最值得先做。

### 5.1 第一优先级：把 detector 重新定义为 selective re-check risk ranker

研究问题：

> 在固定额外调用预算下，风险排序能否稳定减少 wrong-option follow，同时尽量不伤害 accuracy？

为什么最值得做：

- 与现有 `guard-eval` 直接兼容
- 不要求 detector 先成为强分类器
- 更符合当前 AUC 略高于随机、但绝对分类性能偏弱的现实

直接对应文件：

- `scripts/run_interference_detector.py`
- `src/mitigation/README.md`
- `outputs/experiments/interference_detector_new15_full_detector_grid/detector_model_comparison.csv`
- `outputs/experiments/interference_detector_new15_10k_embedding/detector_model_comparison_embedding_new15_10k.csv`

建议新增的核心指标：

- budgeted trigger curve
- wrong-option-follow reduction per extra call
- accuracy delta per extra call
- strict-positive recall at fixed trigger budget

### 5.2 第二优先级：在 `new15` 上补过程代理字段

研究问题：

> 哪些可廉价采集的过程代理，最能解释样本为什么被排到高风险位置？

建议新增字段：

- `first_pass_answer`
- `neutral_reask_answer`
- `answer_changed_after_neutral_reask`
- `mentions_wrong_option_in_trace`
- `explicit_self_doubt`
- `candidate_answer_path`

直接对应文件：

- `src/data/local_data_perturber.py`
- `src/mitigation/interference_dataset.py`

原因：

- 当前 `new15` 已经把外部压力因素设计得很清楚
- 缺的不是条件操控，而是观测层

### 5.3 第三优先级：做 audit-friendly case study，而不是只做 aggregate detector score

研究问题：

> 高风险样本里，哪些是“先会后偏”、哪些是“本来不会”、哪些只是抽取或无效输出问题？

建议产物：

- hard negative audit set
- trigger 后恢复成功案例
- trigger 后仍失败案例
- baseline/control 稳定但 arm 漂移的典型样本集

直接对应文件：

- `src/mitigation/interference_dataset.py`
  - 当前已经生成 `is_hard_negative`
- `outputs/experiments/interference_detector_mvp/interference_dataset_summary.json`

## 6. 对当前项目的具体改写建议

后续写作和汇报时，建议把表述改成下面这种口径。

不建议：

- “我们训练了一个能识别 interference 的 detector”
- “detector 能判断样本是否发生了真实 sycophancy”

更建议：

- “我们训练了一个面向离线 selective re-check 的风险排序器”
- “当前 detector 提供的是弱排序信号，可用于触发复查，而不是替代过程监督”
- “当前主要问题是 measurement bottleneck，而非 detector family 已经被证明无效”

## 7. 一句话结论

当前 detector 没有失败在“完全学不到任何东西”，而是停在了一个更真实的位置：它能提供一点有用的风险排序，但由于标签仍是后验行为标签、过程信号又太少，所以最合理的下一步不是继续把它包装成强分类器，而是把项目主线转向 `process proxy + measurement + audit/recheck utility`。
