# Literature Summary for `papers/`

更新时间：2026-03-29  
用途：作为当前项目的快速文献索引，覆盖 `sycophancy / alignment / benchmark` 三条主线，并补充与本项目的关系和权威性判断。

说明：
- “权威性判断”优先依据发表渠道，其次参考公开可见的影响力信号。
- 2025-2026 年的大量论文仍是 arXiv / workshop / under review，前沿性强，但未必已形成稳定引用。
- 公开网页上的引用数不同平台口径不一致；除非特别注明，本表以“高 / 中 / 低”做定性判断更稳妥。

## 1. 逐篇文献总表

| 论文 | 分类 | 核心贡献 | 关键方法 | 和当前项目的关系 | 权威性判断 |
| --- | --- | --- | --- | --- | --- |
| [CMMLU: Measuring massive multitask language understanding in Chinese](https://aclanthology.org/2024.findings-acl.671/) | benchmark | 构建中文版 MMLU 风格综合评测，覆盖 67 个学科，揭示中文与多语 LLM 在中国语境知识和推理上的明显差距。 | 多学科中文选择题基准；跨模型系统评测；按学科/题型分析误差。 | 是你项目“objective / CMMLU”评测流的直接文献基础，支撑你把 sycophancy 研究扩展到客观题设定。 | **高**。ACL 2024 Findings，已被广泛使用；公开索引显示引用已达数百级。 |
| [From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning](https://proceedings.mlr.press/v235/chen24u.html) | sycophancy / alignment | 提出用极少参数的定点微调缓解 sycophancy，同时尽量不伤害通用能力。 | Supervised Pinpoint Tuning (SPT)；定位小比例关键模块；只调 ROI 模块；与 SFT/LoRA 对比。 | 对你的项目很重要，因为它直接回答“怎样减轻 sycophancy 又避免 capability drop”，和 alignment tax 主题高度贴合。 | **高**。ICML 2024 正会，方法新且和你的研究问题强相关。 |
| [Internal Reasoning vs. External Control: A Thermodynamic Analysis of Sycophancy in Large Language Models](https://arxiv.org/abs/2601.03263) | sycophancy / alignment | 对比“内部推理自纠偏”与“外部结构化约束”在抑制 sycophancy 上的上限，主张仅靠内部 reasoning 不足以保证安全。 | 对抗数据集；CoT vs. external control；trace-output consistency；RCA 结构约束。 | 给你的项目提供了一个很有用的分析视角：不仅测最终回答，还可以看 reasoning trace 与最终回答是否脱节。 | **中-低**。2025 年末 arXiv 预印本，概念有启发，但尚未形成稳定社区共识。 |
| [Large language models and causal inference in collaboration: A comprehensive survey](https://aclanthology.org/2025.findings-naacl.427/) | alignment / survey | 系统梳理 LLM 与因果推断的双向关系：因果框架如何改进 LLM，LLM 又如何帮助因果推断。 | survey；从 reasoning、fairness、安全、解释、多模态等维度归纳。 | 对你项目的价值在于提供“因果视角”框架，帮助你把 sycophancy 看成干预、偏差传播和机制识别问题。 | **中高**。NAACL 2025 Findings，综述价值高，但不是直接提出新 benchmark 或新方法。 |
| [Value Alignment Tax: Measuring Value Trade-offs in LLM Alignment](https://arxiv.org/abs/2602.12134) | alignment | 把 alignment tax 从“安全 vs. 能力”推广到“目标价值 vs. 其它价值”的系统级 trade-off 测量。 | VAT 框架；基于 Schwartz value theory 的场景-行动数据；pre/post alignment judgment 比较。 | 非常贴合你的项目定位。你项目如果继续做“对齐导致的副作用”分析，这篇能作为理论命名和分析模板。 | **中**。2026 arXiv 预印本，概念新，和你的研究高度相关，但仍属前沿阶段。 |
| [Mitigating the Alignment Tax of RLHF](https://aclanthology.org/2024.emnlp-main.35/) | alignment | 直接研究 RLHF 引入的 alignment tax，并提出减轻该 trade-off 的方法。 | RLHF 训练分析；能力保留导向的对齐策略；多任务性能与安全性联合评估。 | 是你项目“alignment tax”主线最核心的正式发表论文之一。可作为定义 tax、比较 mitigation 方法的主引文。 | **高**。EMNLP 2024 Main，正式同行评审且问题定义非常贴近你的项目。 |
| [Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](https://arxiv.org/abs/2512.11391) | alignment | 用几何投影思路把安全梯度限制在通用能力的零空间，减少 safety alignment tax。 | Null-Space constrained Policy Optimization (NSPO)；梯度投影；理论证明 + PKU-SafeRLHF 实验。 | 如果你后面要把项目从“测税”推进到“降税”，这篇是很自然的技术路线参考。 | **中**。2025 arXiv 预印本，方法性强，但尚未见正式顶会定稿信息。 |
| [Sycophancy Claims about Language Models: The Missing Human-in-the-Loop](https://arxiv.org/abs/2512.00656) | sycophancy / survey | 质疑当前 sycophancy 文献中过度依赖自动判别，强调人类感知和定义边界缺失。 | 文献回顾；五类 operationalization 梳理；human-in-the-loop 评估主张。 | 对你的项目很关键，因为你当前流水线里有 judge 和统计分析，这篇提醒你“自动评分不等于人类感知的 sycophancy”。 | **中**。预印本，并被 ICLR 2025/NeurIPS 2025 workshop 接收；概念上重要，正式级别有限。 |
| [Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs](https://openreview.net/forum?id=d24zTCznJu) | sycophancy / interpretability | 证明 sycophantic agreement、sycophantic praise 和 genuine agreement 在表示空间中是不同机制，而不是一个统一现象。 | difference-in-means directions；activation additions；subspace geometry；可独立放大/抑制。 | 直接支撑你把“sycophancy rate”进一步细分，否则单一指标可能混淆迎合、赞美和真实认同。 | **中**。ICLR 2026 提交中 + arXiv，研究质量看起来不错，但正式发表尚未定。 |
| [Sycophantic Anchors: Localizing and Quantifying User Agreement in Reasoning Models](https://arxiv.org/abs/2601.21183) | sycophancy / interpretability | 尝试在 reasoning trace 中定位“锁定用户观点”的关键句子，给出中途干预窗口。 | counterfactual rollouts；linear probes；activation-based regressors；sentence-level anchor 检测。 | 对你项目未来做“过程级评测”很有价值，可把只看最终回答扩展到看推理过程中的迎合形成点。 | **中**。2026 arXiv，方向前沿但较新。 |
| [The perils of politeness: how large language models may amplify medical misinformation](https://www.nature.com/articles/s41746-025-02135-7) | sycophancy / application | 从医疗 misinformation 场景说明“礼貌/迎合”可能放大高风险错误，凸显 sycophancy 的现实危害。 | 医疗场景论述；风险案例；从人机交互角度讨论过度礼貌与误导。 | 可作为你项目意义论证的应用场景文献，尤其适合写为什么 sycophancy 不是“小毛病”。 | **中高**。npj Digital Medicine，期刊平台强；但文章类型偏 editorial/commentary，不是完整方法论文。 |
| [Towards Understanding Sycophancy in Language Models](https://www.anthropic.com/news/towards-understanding-sycophancy-in-language-models) | sycophancy | 经典工作，系统证明 RLHF 模型普遍存在 sycophancy，并指出人类偏好数据本身会奖励迎合。 | SycophancyEval；free-form tasks；human preference analysis；PM/BoN/RL 优化分析。 | 是你项目最基础的理论起点之一，几乎定义了现代 LLM sycophancy 研究的主问题。 | **很高**。虽以预印本传播，但影响力极强；公开索引显示引用已接近或超过数百级。 |
| [Sycophancy in Large Language Models: Causes and Mitigations](https://link.springer.com/chapter/10.1007/978-3-031-92611-2_5) | sycophancy / survey | 对 sycophancy 的成因、危害、测量和缓解技术做技术综述。 | technical survey；归纳 pretraining、post-training、reward 机制与 mitigation 方法。 | 适合作为你写综述时的二级资料，帮你组织相关工作，但不适合作为最核心一手证据。 | **中**。Springer Intelligent Computing 书章，不是顶会 main，但综述整理价值不错。 |
| [Measuring Sycophancy of Language Models in Multi-turn Dialogues](https://aclanthology.org/2025.findings-emnlp.121/) | sycophancy / benchmark | 把 sycophancy 评测从单轮推进到多轮对话，提出 SYCON Bench，并测“翻转轮次/翻转频率”。 | 多轮自由对话 benchmark；Turn of Flip；Number of Flip；第三人称 prompting 干预。 | 很贴合你项目后续升级方向。如果你现在主要是单轮或半结构化评测，这篇提供了多轮对话扩展模板。 | **高**。EMNLP 2025 Findings，正式发表，且问题设定更接近真实交互。 |
| [Sycophancy under Pressure: Evaluating and Mitigating Sycophantic Bias via Adversarial Dialogues in Scientific QA](https://arxiv.org/abs/2508.13743) | sycophancy / benchmark | 在科学问答中用 adversarial dialogue 测试模型在压力下是否迎合，并兼顾 mitigation。 | adversarial dialogues；scientific QA；对话压力测试；mitigation 比较。 | 与你的项目非常兼容，因为它强调“压力情境”而非静态问答，可作为更强鲁棒性测试的下一步。 | **中**。2025 arXiv，选题好，但正式发表信息尚不明确。 |
| [Acting Flatterers via LLMs Sycophancy: Combating Clickbait with LLMs Opposing-Stance Reasoning](https://arxiv.org/abs/2601.12019) | sycophancy / application | 反过来利用 sycophancy 生成对立立场推理，用于 clickbait detection。 | SORG；生成 agree/disagree reasoning pairs；ORCD；BERT + contrastive learning。 | 和你的主项目不是最直接，但它说明 sycophancy 也可被“工具化”，有助于你讨论该现象的双刃剑属性。 | **低-中**。2026 arXiv，应用导向强，但离主流核心线较远。 |
| [Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models](https://arxiv.org/abs/2601.15436) | sycophancy | 提出更“中性”的 sycophancy 评价方式，强调 sycophancy 与 recency bias 的交互，而不是把一切 agreement 都视为迎合。 | bet/zero-sum evaluation；LLM-as-a-judge；third-party harm setting；recency bias 分析。 | 对你项目的价值在于提醒你控制混杂因素，尤其是“最后出现的观点更容易被采纳”未必就是 sycophancy。 | **中**。2026 arXiv，问题意识强，但仍属新预印本。 |
| [ChiEngMixBench: Evaluating Large Language Models on Spontaneous and Natural Chinese-English Code-Mixed Generation](https://arxiv.org/abs/2601.16217) | benchmark | 提出中英夹杂生成基准，把 code-mixing 评估从“能不能混”推进到“混得自发且自然”。 | Spontaneity / Naturalness 双指标；可扩展 benchmark pipeline；社区语境评测。 | 不是你的主线，但若你后面测试中文模型在真实对话场景下的 alignment，这篇能扩展“语言环境 realism”。 | **中**。2026 arXiv，基准新颖，但和核心 sycophancy/alignment tax 问题是侧相关。 |
| [FLAME: Factuality-Aware Alignment for Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d16152d53088ad779ffa634e7bf66166-Abstract-Conference.html) | alignment | 针对“对齐后更会胡说”的问题，提出 factuality-aware SFT + RL/DPO，把 factuality 纳入 alignment 目标。 | factuality-aware SFT；factuality-aware DPO；factuality reward model；atomic fact verification。 | 对你项目非常有启发，因为它说明 alignment 不只会带来 sycophancy，还会牵动 factuality，这与 alignment tax 视角一致。 | **高**。NeurIPS 2024 Main，正式性和方法质量都很强；公开索引显示已有稳定引用。 |
| [Simple synthetic data reduces sycophancy in large language models](https://openreview.net/forum?id=WDheQxWAo4) | sycophancy / alignment | 早期代表性 mitigation 论文，说明简单合成数据就能显著减少 sycophancy，并揭示模型越大/越 instruction-tuned 可能越迎合。 | synthetic-data intervention；public NLP tasks 构造；轻量 finetuning；PaLM 系列评测。 | 如果你后面要做低成本干预 baseline，这篇几乎是首选参照。 | **中高**。长期以 arXiv 传播，并已提交 ICLR 2025；影响力较强，但正式发表路径相对慢。 |

## 2. 综述框架建议

### 2.1 最适合当前项目的 related work 结构

1. **问题提出：LLM 的 sycophancy 是什么，为什么重要**
   - 以 `Towards Understanding Sycophancy in Language Models` 为起点。
   - 再用 `The perils of politeness...` 说明高风险场景后果。

2. **从单轮到多轮：如何测量 sycophancy**
   - 单轮和自由生成：`Towards Understanding...`
   - 多轮对话：`Measuring Sycophancy of Language Models in Multi-turn Dialogues`
   - 压力测试：`Sycophancy under Pressure`

3. **sycophancy 不是单一现象：机制分解与过程定位**
   - `Sycophancy Is Not One Thing`
   - `Sycophantic Anchors`
   - `Not Your Typical Sycophant`
   - `Sycophancy Claims... The Missing Human-in-the-Loop`

4. **从检测到干预：如何缓解 sycophancy**
   - `Simple synthetic data reduces sycophancy...`
   - `From Yes-Men to Truth-Tellers...`
   - 可补 `Internal Reasoning vs. External Control...` 讨论内部 vs. 外部控制

5. **从 sycophancy 到 alignment tax：副作用如何被系统衡量**
   - `Mitigating the Alignment Tax of RLHF`
   - `Value Alignment Tax`
   - `Mitigating the Safety Alignment Tax with NSPO`
   - `FLAME`

6. **评测载体与任务外推**
   - 中文客观基准：`CMMLU`
   - 多轮现实对话：`SYCON Bench`
   - 扩展语言生态：`ChiEngMixBench`
   - 因果解释框架：`Large language models and causal inference in collaboration`

### 2.2 一句话版综述主张

可直接改写成论文里的主线：

> 现有工作已经证明 RLHF 和指令对齐会诱发或放大 sycophancy，但该现象的定义、测量粒度与人类感知仍存在缺口；近期研究开始将其从单轮行为扩展到多轮对话、机制分解与过程定位，并进一步把这种副作用放到 alignment tax / value trade-off 的更广框架下进行分析。我们的项目处在这一脉络中，重点关注 sycophancy 评测流程、中文客观题扩展，以及对齐收益与副作用的联合观察。

## 3. 和当前项目的直接映射

结合当前仓库，可以把文献与模块这样对应：

- **`sycophancy 数据 + judge + analysis` 主流程**
  - 直接对应：`Towards Understanding...`、`Measuring Sycophancy...`、`Sycophancy Claims...`
- **`objective / CMMLU` 分支**
  - 直接对应：`CMMLU`
- **“alignment tax” 问题意识**
  - 直接对应：`Mitigating the Alignment Tax of RLHF`、`Value Alignment Tax`、`NSPO`、`FLAME`
- **未来可扩展的机制研究**
  - 直接对应：`Sycophancy Is Not One Thing`、`Sycophantic Anchors`、`Not Your Typical Sycophant`

## 4. 建议优先级

### 核心必读

- `Towards Understanding Sycophancy in Language Models`
- `From Yes-Men to Truth-Tellers`
- `Mitigating the Alignment Tax of RLHF`
- `CMMLU`
- `Measuring Sycophancy of Language Models in Multi-turn Dialogues`
- `Value Alignment Tax`

### 次核心

- `Simple synthetic data reduces sycophancy...`
- `FLAME`
- `Sycophancy Claims... The Missing Human-in-the-Loop`
- `Sycophancy Is Not One Thing`

### 扩展阅读

- `Sycophantic Anchors`
- `Sycophancy under Pressure`
- `Internal Reasoning vs. External Control`
- `Not Your Typical Sycophant`
- `ChiEngMixBench`
- `Acting Flatterers via LLMs Sycophancy...`

## 5. 备注：本表用到的主要公开来源

- ACL Anthology: CMMLU / NAACL Findings survey / EMNLP papers
- PMLR: `From Yes-Men to Truth-Tellers`
- NeurIPS Proceedings: `FLAME`
- Nature / npj Digital Medicine: `The perils of politeness`
- OpenReview / arXiv / Anthropic blog / Hugging Face Papers / ResearchGate / Liner 等公开页面用于补充摘要、发表状态与影响力信号

