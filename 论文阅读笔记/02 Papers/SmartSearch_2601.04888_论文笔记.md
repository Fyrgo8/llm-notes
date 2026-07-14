---
type: paper-note
title: "SmartSearch: Process Reward-Guided Query Refinement for Search Agents"
authors:
  - "Wen, Tongyu"
  - "Dong, Guanting"
  - "Dou, Zhicheng"
year: 2026
venue: "SIGIR 2026"
status: deeply-read
topics:
  - Deep Search Agent
  - Process Reward
  - Query Refinement
url: "https://arxiv.org/abs/2601.04888"
---

**批注：DPO 与 GRPO 的粒度差异。** DPO 是单 query 粒度的——偏好对来自局部反事实轨迹，两条轨迹只有一步 query 不同，模型学到的是精确的"这一步 query 写法更好"。GRPO 更关注轨迹粒度——reward 来自整条轨迹所有 $S_t$ 的聚合分数，信用分配更模糊，模型只知道整条轨迹总体更好，不确定具体哪步贡献了 reward。GRPO 中的稠密奖励和信用分离问题靠 refinement 扩展 rollout（更多样化的轨迹）和 process reward 的逐轮信号来缓解，但并未彻底解决。

**批注：DPO 阶段挑选 chosen/rejected 的规则是按终局正确性和 $S_t$ 数量排序。** 对于原轨迹和修复轨迹组成的候选集，先比终局是否正确——答对优于答错；终局结果相同时，比坏 query 数量——坏 query 少（即好 query 多）的轨迹更优。具体规则：
- 一条答对、一条答错 → 答对那条为 chosen
- 两条都答对 → 坏 query 更少的那条为 chosen
- 两条都答错 → 好 query 更多的那条为 chosen
这个排序逻辑确保了模型优先学"改好 query 导致答对"的行为，而不只是"两条轨迹都答对时随便选一条"。

Tongyu Wen, Guanting Dong, Zhicheng Dou | SIGIR 2026 / arXiv v1 | [PDF](https://arxiv.org/pdf/2601.04888v1) | `Deep Search Agent` `Process Reward` `Query Refinement`

## 问题与核心判断

SmartSearch 关注的不是 search agent 能不能调用搜索工具，而是每一次调用时生成的 **query 是否真的有效**。现有搜索 Agent 训练通常把最终答案正确性当作主目标，或者把注意力放在整体 reasoning paradigm 上，这会掩盖三种不会被 final accuracy 充分诊断的失败：① query 包含实体歧义，检索器返回了错误对象；② query 与此前轮次重复，没有带来新信息；③ 搜索意图本身不必要，或者返回结果没有回答该意图。一次低质量 query 会改变 Agent 后续看到的证据，因而可能把整条推理轨迹带偏。

论文的例子很直接：问题要求寻找出生于 1914 年的演员 Kevin McCarthy，但 Agent 搜索 `birthdate of Kevin McCarthy`，检索器返回了同名政治人物。模型随后不是“不会推理”，而是忠实地在错误证据上继续推理。把 query 改为 `birthdate of Actor Kevin McCarthy` 后，搜索结果与答案都恢复正确。这说明一部分 search-agent 失败并非知识利用能力不足，而是**证据入口在中间轮次发生了偏航**。

论文的判断是：search agent 的训练目标不能只问“最后答对了吗”，还应问“中间 query 是否新颖、必要，并且确实取回了预期信息”。这一判断本身不复杂，SmartSearch 的实质增量在于把它落实成一条贯穿 SFT、DPO 和 RL 的训练闭环：先给每个 query 生成可计算的分数和可执行的文本诊断，再从坏 query 处分叉出修复轨迹，最后让三个训练阶段以不同方式消费这些信号。

因此，SmartSearch 不是新的通用 RL objective(**强化学习阶段用来优化的目标函数**)，也不只是一个 query rewriting 模块。它想解决的关键矛盾是：**终局 reward 太粗，无法定位坏 query；但只定位、不重造后续轨迹，又无法给模型提供可学习的替代行为。**

只有评分没有改写，模型只知道某步不好，却看不到正确做法；只有改写没有评分，系统又不知道该改哪一步、为什么改。SmartSearch 因而必须把 process reward 和 query refinement 绑在一起：前者负责“找错并解释”，后者负责“从错误处生成对照轨迹”。

所以，**先把 query 变成可判对象，再把坏 query 变成可比较轨迹，最后分阶段内化**，是整篇论文唯一真正承重的叙事。SFT、DPO、GRPO 都是围绕这条叙事服务的优化工具。

## 机制与训练闭环

SmartSearch 的方法分成两层。底层的 process reward 与 query refinement 负责把原始 rollout 变成“可诊断、可修复、可比较”的轨迹；上层三阶段课程再分别用这些轨迹做筛选式 SFT、query 定向 DPO 和过程感知 GRPO。

```text
底层机制：原始轨迹 -> query 逐轮评分与诊断 -> 坏 query 定向改写 -> 重生成轨迹后缀

训练闭环：筛选高质量轨迹做 SFT
          -> 用原轨迹/修复轨迹做 DPO
          -> 用 refinement 扩展 rollout，并用 process reward 做 GRPO
```

三阶段各自解决不同的学习问题。SFT 通过筛选全 $S_t=1$ 的干净轨迹让模型先学会基本搜索格式和好 query 模式，约 60% 的数据通过筛选，避免模型模仿"碰巧答对的坏搜索过程"。DPO 通过原轨迹与 refinement 修复轨迹构造局部反事实偏好对——两条轨迹共享大部分前缀，只有一步 query 不同，因此模型学到的是精确的"这一步 query 写法更好"，而不是模糊的"整条轨迹更好"。GRPO 让模型在在线 rollout 中反复实践，通过评估器实时打分 $S_t$ 作为 reward 来内化和泛化搜索策略。

DPO 与 GRPO 都能让模型"往好轨迹靠"，但方式不同：DPO 的偏好对来自一次性离线构造的局部反事实，对比粒度精确到单步 query，学的是"好 query 长什么样"；GRPO 的 reward 来自整条轨迹的聚合分数，信用分配更模糊——模型只知道轨迹总体更好，不确定具体哪一步贡献了 reward。因此两个阶段不可相互替代：**DPO 先把好 query 的模板精确地教给模型，GRPO 再让模型在真实搜索环境中巩固和泛化这个能力。**

---

### 底层机制一：Dual-Level Credit Assessment ⭐ 核心创新

**核心思想**：一个 query 必须同时做到“带来新信息”和“解决当前必要的搜索意图”，才算高质量。SmartSearch 因此不让单一模型笼统打分，而是把 query quality 拆成规则可判的新颖性和模型可判的有用性。

| 评估维度                 | 实现方式 | 实际检查什么               | 产物                                                        |
| :------------------- | :--- | :------------------- | :-------------------------------------------------------- |
| **Query Novelty**    | 规则检查 | 当前检索结果是否与此前轮次过度重合    | 二值分数 $S_t^{\mathrm{novel}}$ + 解释 $T_t^{\mathrm{novel}}$   |
| **Query Usefulness** | 模型评估 | 搜索意图是否必要，结果是否真正回答该意图 | 二值分数 $S_t^{\mathrm{useful}}$ + 解释 $T_t^{\mathrm{useful}}$ |
|                      |      |                      |                                                           |

#### 1. 新颖性：检查“这一搜有没有看到新东西”

规则不比较 query 字面，而是比较返回文档（通过文档 ID 或内容哈希判等，而非 URL 或标题字面，因此同一页面换了 URL 也能识别）。若第 $t$ 轮的 Top-$n$ 文档与此前任一轮结果的重合数 $O_t$ 超过阈值 $K$，则该 query 被视为冗余：

> 注：并集 $\bigcup$ 保证同一篇文档在多个历史轮次中出现过也只算一次；$s=0$ 通常为空（初始轮尚未搜索），公式保留该索引以便推广。

$$
O_t=\sum_{i=1}^{n}\mathbf{1}\left[D_i^t\in\bigcup_{s=0}^{t-1}\bigcup_{j=1}^{n}D_j^s\right].
$$

这一步专门处理“换一种说法重复搜索”的问题。它判断的不是语言多样性，而是 query 是否真正扩展了 Agent 的可见证据。

#### 2. 有用性：检查“搜索意图和返回结果是否都对”

模型评估器读取用户问题 $q$、golden answer $a$ 和截至当前轮的轨迹 $H_t$：

$$
S_t^{\mathrm{useful}},T_t^{\mathrm{useful}}
=\mathrm{LLM}_{\mathrm{eval}}(q,a,H_t).
$$

只有当当前意图对解题是必要的，并且检索结果确实包含该意图要求的信息时，分数才为 1。意图正确但命中同名错误实体仍然为 0；返回了相关实体却没回答当前问题，也为 0。

论文先用 Qwen3-32B 生成 scoring/refinement 标注，再蒸馏到 Qwen2.5-3B-Instruct。训练后的 3B 模型同时担任 `LLM_eval` 与后面的 `LLM_refine`，避免每条轨迹都调用大模型。

#### 3. 合并：分数负责筛选，解释负责修改

$$
S_t=\mathbf{1}\left[S_t^{\mathrm{novel}}=1\land S_t^{\mathrm{useful}}=1\right],
\qquad
T_t=T_t^{\mathrm{novel}}\Vert T_t^{\mathrm{useful}}.
$$

- $S_t$ 是硬判据：用于筛 SFT 数据、排 DPO 偏好、计算 RL reward。
- $T_t$ 是修改说明：告诉 refinement model 问题来自重复、意图不必要，还是结果未覆盖预期信息。

> **为什么必须同时输出 score 和 explanation？** 只有 score，系统知道“这步错了”却不知道如何修；只有 explanation，又无法稳定地筛数据、排偏好和算 reward。数值信号负责决策，文本信号负责执行。

> SmartSearch 的评估器同时输出分数 （作为 RL reward）和文本解释 （指导 refinement），两者来自同一个蒸馏的 Qwen2.5-3B 模型。它不是传统意义上独立训练的 reward model——它不是一个只打分的黑盒，而是"能打分的通用 LLM"：系统会解析它产出的文本解释来决定如何改写 query，所以评估器是白盒/可解释的。但从 RL 训练的角度看，分数  确实作为 reward signal 进入 GRPO，功能上就是 RL 的 reward 来源。因此准确的说法是：**评估器提供 RL reward signal，但不是传统 learned reward model——它是一个蒸馏的两用 LLM，同时承担评估（打分+解释）和改写两个角色。**

>发现这步 query 不行 → 把上下文打包发给小模型 → 小模型吐一个新 query 回来。
>
>**批注：SmartSearch 本质上解决的是三层信用分离问题：**
> 1. **时间上的信用分离** — 终局 reward 太粗，看不出哪步 query 导致失败。→ 逐轮打分 $S_t$，不依赖终局结果。
> 2. **行为上的信用分离** — 即使定位到坏 query，也分不清是意图选错了还是措辞有歧义。→ 拆成 Novelty + Usefulness 两个维度，附带文本解释 $T_t$ 精确说明原因。
> 3. **轨迹间的信用分离** — 光知道错了不够，模型不知道"换成什么才是好的"。→ refinement 制造局部反事实轨迹（保持前缀不变，只换一个 query），让模型看到好/坏 query 分别导致什么后果。
>
>**批注：三阶段各自的角色 — SFT 教格式和好 query 模式，DPO 学 query 偏好，GRPO 在线练搜索策略。**
> 1. **Stage 1（SFT）** — 筛选最干净的轨迹（所有 $S_t=1$），让模型先学会基本搜索格式和高质量 query 模式。只保留约 60% 数据，避免模型模仿"碰巧答对的坏搜索过程"。
> 2. **Stage 2（DPO）** — 离线偏好学习。用原轨迹 + refinement 修复轨迹构造偏好对，用 $S_t$ 排序选出 chosen/rejected，让模型从局部反事实对比中学"哪一种 query 写法更好"（措辞、消歧、意图切换）。数据一次性离线构造。
> 3. **Stage 3（GRPO）** — 在线强化学习。当前 policy 多次 rollout，评估器实时打分 $S_t$，用 reward 更新模型。让模型在实战中学"如何动态调整搜索策略"，而非只看静态偏好对比。**DPO 先教、GRPO 再练。**

## 证据、定位与边界

### 主实验结果

| 设置 | SmartSearch | 最强对照 | 结论 |
|:--|:--|:--|:--|
| 4 个知识密集 benchmark 平均 EM | **37.5** | StepSearch：30.1 | 相对提升约 25% |
| 4 个知识密集 benchmark 平均 F1 | **47.2** | StepSearch：39.5 | 相对提升约 19% |
| GAIA + WebWalker 平均 EM | **12.5** | StepSearch：9.3 | 本地 Wikipedia 训练后可迁移到 web search |
| GAIA + WebWalker 平均 F1 | **23.9** | StepSearch：19.2 | 开放网页设置仍有增益 |

- **知识任务**：2WikiMultihopQA、HotpotQA、Bamboogle、MuSiQue。
- **网页任务**：GAIA、WebWalker；训练只使用本地 Wikipedia，测试用 Serper API。
- **额外指标**：Search Quality 和“F1/搜索次数”定义的 Search Efficiency 均领先。
- **谨慎看待**：网页任务绝对分数仍低，证明的是局部 query 训练可以迁移，不是已经解决开放网页搜索。

### 消融实验：两个底层机制如何进入三个阶段

| 训练阶段 | 配置 | 平均 F1 | 解读 |
|:--|:--|:--|:--|
| Stage 1 | 完整筛选 | **40.9** | process reward 用于过滤 SFT 数据 |
| | 不用 process reward | 38.2 | 只按答案筛选会保留坏 query 轨迹 |
| Stage 2 | 完整方法 | **43.2** | 评分与 refinement 联合构造偏好 |
| | 不用 process reward | 41.8 | 偏好排序失去 query-quality 信号 |
| | 不用 query refinement | 41.3 | 随机完整轨迹缺少定向 query 对照 |
| Stage 3 | 完整方法 | **47.2** | refinement rollout + process reward |
| | 不用 process reward | 45.1 | rollout 有定向扩展，但 reward 仍过粗 |
| | 不用 query refinement | 45.9 | 有过程 reward，但探索分布没有被修复 |
| | 标准 GRPO | 44.7 | 两项机制都缺失 |

最有解释力的不是“每个组件都掉点”，而是 Stage 2、3 中只保留其中一项都不如完整方法：**评分负责指出差异，refinement 负责制造差异，缺一项都无法形成完整学习闭环。**

**局限**：消融仍不能严格区分收益来自更准确的 query 监督、更多 refinement 计算，还是修复轨迹带来的候选多样性；它证明的是组合 pipeline 有效，不是每个机制的因果贡献已经被完全分离。

### 评估器是否可信、是否值得用小模型

| 比较                            | 分数重合率  |
| :---------------------------- | :----- |
| Qwen3-32B teacher vs 人工       | 接近 90% |
| Qwen2.5-3B student vs teacher | 超过 85% |
| Qwen2.5-3B student vs 人工      | 超过 80% |

将 3B student 换成 32B teacher 后，平均 F1 增益不足 1%，但每个样本的评分与 refinement 时间接近五倍。这支持“小模型承担高频 evaluator/refiner”这一工程选择。不过人工分析只抽取了 100 条轨迹，且论文的 Search Quality 指标也依赖同一评估体系，仍存在训练裁判与评测裁判部分同源的问题。

### 方法定位

SmartSearch = **`query-level process reward + counterfactual trajectory refinement + staged SFT/DPO/GRPO`**

它与一般 process-reward search agent 的区别不只是“多一个 query 分数”：

- **一般过程奖励方法**：主要用中间分数改善 reasoning 或 RL reward。
- **SmartSearch**：先用分数和解释定位坏 query，再从该点重生成轨迹；同一诊断随后贯穿数据筛选、偏好构造和 RL rollout。

最值得借鉴的是监督信号的复用方式：同一个 $S_t/T_t$ 在 Stage 1 是数据过滤器，在 Stage 2 是偏好排序器和修改指令，在 Stage 3 又成为 rollout 路由与 reward 来源。三阶段不是三个孤立技巧，而是逐步把外部 query 诊断内化进 policy。

### 代价与前提

| 局限 | 原因 |
|:--|:--|
| 训练期评估依赖 golden answer | usefulness evaluator 读取标准答案，不能原样部署到未知答案的线上推理 |
| 新颖性只是代理指标 | 新文档不等于新信息；交叉验证时，重复检索也可能是必要行为 |
| query-quality 指标部分同源 | 训练与评测都依赖相同 scoring 逻辑，人工一致性实验只能部分缓解 |
| refinement 增加训练成本 | 每个坏 query 都可能触发一次重写、重新检索和后缀生成 |
| reward 仍是轨迹级 | query 分数最终聚合为单一 reward，没有真正做 step-level advantage |
| 泛化证据仍有限 | 只训练和主测 Qwen2.5-3B；网页 benchmark 有提升，但绝对性能仍低 |

## 研究卡片

- **问题**：只按最终答案训练 search agent，会漏掉歧义、重复、无用的中间 query；一次坏 query 会把错误证据传给整条后续推理。
- **做法**：双层评估（文档重合判断新颖性 + 小模型判断意图/结果有用性）-> 文本诊断指导 query refinement -> 从修改点重生成后缀 -> SFT 筛数据 -> DPO 学局部对照 -> GRPO 做定向 rollout 与过程感知强化。
- **证据**：四个知识任务平均 EM/F1 达到 `37.5/47.2`，显著高于 StepSearch 的 `30.1/39.5`；Stage 2、3 中只去掉评分或 refinement 都会下降，说明两者需要联合工作。
- **前提**：训练期有可靠 golden answer；“文档重合少 = query 新颖”与“评估器判定有用 = 真有信息增益”这两个代理假设基本成立。
- **教训**：不要把 process reward 只当成 RL 里的额外分数。更完整的做法是让它既定位错误，又驱动局部轨迹重造，再把同一监督贯穿数据、偏好和 rollout 三层。
- **遗留**：没有 golden answer 时如何在线判断 query usefulness？如何区分必要复核与冗余搜索？能否用信息增益、证据覆盖和来源可信度替代二值 query 分数，并进一步形成真正的 step-level credit assignment？
- **讨论**：SmartSearch 的评估器同时输出分数 $S_t$（作为 RL reward）和文本解释 $T_t$（指导 refinement），两者来自同一个蒸馏的 Qwen2.5-3B 模型。它不是传统意义上独立训练的 reward model——它不是一个只打分的黑盒，而是"能打分的通用 LLM"：系统会解析它产出的文本解释来决定如何改写 query，所以评估器是白盒/可解释的。但从 RL 训练的角度看，分数 $S_t$ 确实作为 reward signal 进入 GRPO，功能上就是 RL 的 reward 来源。因此准确的说法是：**评估器提供 RL reward signal，但不是传统 learned reward model——它是一个蒸馏的两用 LLM，同时承担评估（打分+解释）和改写两个角色。**
- **讨论**：评估结果 $(S_t, T_t)$ 在各阶段的用途不同 — SFT 用 $S_t$ 做**筛选条件**（只留全部 $S_t=1$ 的轨迹，约 60%）；DPO 用 $S_t$ 做**排序依据**（选 chosen/rejected）；GRPO 用 $S_t$ 做 **reward signal**，但需要当前 policy 的新 rollout + 评估器实时打分，而非复用旧数据。
