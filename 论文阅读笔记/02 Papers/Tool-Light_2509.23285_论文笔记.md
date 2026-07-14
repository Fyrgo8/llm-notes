---
type: paper-note
title: "Toward Effective Tool-Integrated Reasoning via Self-Evolved Preference Learning"
authors:
  - "Chen, Yifei"
  - "Dong, Guanting"
  - "Dou, Zhicheng"
year: 2026
venue: "ICLR 2026（按作者主页口径）"
status: deeply-read
topics:
  - Agentic RL
  - Tool Use
  - Preference Learning
url: "https://arxiv.org/abs/2509.23285"
---

# Tool-Light（arXiv:2509.23285）论文笔记

Yifei Chen, Guanting Dong, Zhicheng Dou | ICLR 2026（作者主页口径）/ arXiv v2 | [PDF](https://arxiv.org/pdf/2509.23285v2) | `Tool-Integrated Reasoning` `Preference Learning` `Entropy-Guided Sampling`

## 问题与核心判断

Tool-Light 关注的不是模型能不能调用工具，而是它能否以**必要、充分且不过度**的方式使用工具。现有 Tool-Integrated Reasoning（TIR）模型通常存在三类相反但相关的失败：① 已经可以直接作答或已有充分证据，却继续重复调用工具；② 面对超出参数知识或计算能力的问题时，没有调用必要工具；③ 工具返回结果后仍长时间反复推理，甚至进入 analysis paralysis。只优化答案正确性会掩盖这些行为，因为“多调用几次后答对”和“用一次必要调用答对”会得到相同终局标签。

论文不只把这些现象归结为 tool-use policy 不好，而是先从信息熵观察工具调用前后的推理变化。作者发现，模型收到 tool result 后，后续 token entropy 通常先上升、随后波动，并在下一次工具调用前明显下降；对于同一道题的多条正确路径，工具调用更少的轨迹往往具有更低的整体 entropy。这个结果提供了一个可观测信号：高熵位置通常意味着模型接下来的推理方向仍有较大分歧，适合从这里分叉采样；低熵且正确的轨迹则更可能对应较紧凑的工具行为。

但 entropy 本身既不是行为好坏的 ground truth，也不是 Tool-Light 的训练 reward。高 entropy 可能意味着有价值的探索，也可能只是模型混乱；低 entropy 可能意味着稳定收敛，也可能是过早自信。论文真正使用 entropy 的方式更克制：一是在采样时寻找最值得分叉的位置，二是在多个正确候选中辅助选择更紧凑的正样本。最终偏好仍由答案正确性、工具次数、轨迹长度和题目难度共同决定。

论文的核心判断是：很多 TIR 行为问题不一定要直接用 full RL 解决，关键是先构造出能够区分“少而够用”和“少但不够”的偏好数据。只惩罚工具次数会让模型不敢探索；只奖励正确答案又会保留过度调用。Tool-Light 因此把 DPO 训练拆成两个方向：Pre-Aligned DPO 先压缩冗余调用和过度推理，Self-Evolved DPO 再针对当前模型仍不会的 hard samples 奖励更长、能答对的探索轨迹，补回必要工具调用。

所以，**先用 entropy 定向扩展候选，再用 easy/hard 非对称标准构造偏好，最后让更新后的模型持续重采样**，是整篇论文真正承重的叙事。标准 SFT 和 DPO objective 只是消费这些自演化偏好数据的优化器。

## 机制与训练闭环

Tool-Light 的数据流和训练流交替运行。SFT 先赋予模型基本 TIR 能力，并筛出“无工具时答不对”的源问题；混合采样再为这些问题制造多条工具轨迹；第一轮 DPO 主要去冗余，后续 self-evolved DPO 则根据更新后模型的能力重新判断 easy/hard，并重新构造偏好对。

```text
Tool-Star SFT 数据 D_sft
  -> SFT 得到 M_sft
  -> 不提供工具做 direct inference，只保留答错题 -> D_source
  -> vanilla sampling + entropy-guided branch sampling
  -> Criteria 1 构造偏好对 D_dpo1
  -> Pre-Aligned DPO 得到 M_dpo1
  -> 用 M_dpo1 重新采样，并按 Criteria 2 重分 easy/hard
  -> 构造 D_dpo2 -> DPO 更新 -> 再采样
  -> 循环至性能收敛，得到最终模型
```

---

### 环节一：SFT 冷启动与困难源数据构造

**核心思想**：偏好学习不能从一个完全不会 TIR 的模型开始。先让模型掌握基本工具格式和调用能力，再把训练预算放到确实需要外部工具的问题上。

作者沿用 Tool-Star 的 SFT 数据 $D_{\mathrm{sft}}$ 训练 instruct model，得到 $M_{\mathrm{sft}}$。随后让原始 instruct model 在同一批题上进行**无工具 direct inference**，只保留直接推理答案错误的题：

$$
D_{\mathrm{source}}=D_{\mathrm{sft}}[y_{\mathrm{label}}\neq y_{\mathrm{inference}}].
$$

$D_{\mathrm{source}}$ 的作用不是提供答案标签，而是筛出更可能需要搜索或代码工具的困难问题。若一道题不用工具已经能答对，再围绕它采样大量 TIR 轨迹，容易制造“为了用工具而用工具”的训练信号。这个筛选因此先在问题层面控制必要性。

> **这里的 SFT 不是核心创新。** 它负责建立可用的初始 policy；真正的增量从 $M_{\mathrm{sft}}$ 如何采样轨迹、如何把轨迹筛成偏好对开始。

---

### 环节二：Vanilla + Entropy-Guided 混合采样 ⭐ 核心创新

**核心思想**：vanilla sampling 保证完整轨迹分布和工具行为覆盖，entropy-guided sampling 则把额外计算集中到最可能改变推理方向的位置。两者不是替代关系，而是以不同方式补充偏好候选。

#### 1. Vanilla TIR Sampling：从头独立采多条完整轨迹

对 $D_{\mathrm{source}}$ 中每个问题，$M_{\mathrm{sft}}$ 从头生成多条 TIR 轨迹。优势是样本彼此独立、行为覆盖自然；代价是每条轨迹都要重新生成完整长链，推理成本高，而且随机采样未必能在关键决策处产生足够差异。

#### 2. Entropy-Guided Sampling：先生成主链，再从高熵位置长出分支

作者先生成一条主推理链 $C_{\mathrm{main}}$，按工具调用划分 reasoning steps。对每个 step，分别计算开头 10、20、30、40、50 个 token 的平均 entropy：

$$
H_{\mathrm{avg}}(i)=\frac{1}{i}\sum_{j=1}^{i}H(j).
$$

每个 step 保留最大的 $H_{\mathrm{avg}}(i)$ 及其对应 token 位置，再从所有 steps 中选出 entropy 最高的 top-$k$ 位置。系统复用该位置之前的前缀，只从这些分叉点继续生成多个后缀：

```text
主链前缀 -----------------> 原后缀
       |-------------------> 分支后缀 1
       |-------------------> 分支后缀 2
       +-------------------> 分支后缀 3
```

这里 entropy 的作用是**branch router**：它不直接判断哪条轨迹更好，只决定“把有限的 rollout 预算花在哪些位置更可能得到有差异的后续行为”。在理想假设下，作者声称其复杂度可从独立采样的 $O(mn)$ 降到 $O(n\log m)$；但这一估计依赖前缀复用、树状生成和分支长度等工程假设，不能简单理解为真实端到端成本必然按该比例下降。

#### 3. 为什么要混合两种采样

纯 entropy-guided sampling 容易围绕单条主链形成局部候选；纯 vanilla sampling 又昂贵且方向性不足。论文最终混合 $D_{\mathrm{dpo}}^1$ 与 $D_{\mathrm{dpo}}^2$，实验设置中的 vanilla : entropy-guided 比例为 `13:7`。

附录的数据比例分析显示：提高 vanilla 数据占比会同时提升总体性能和工具调用次数，但 Efficiency 不会持续上升，因为更多调用最终会抵消正确率收益。说明采样混合的目标不是单向追求低 entropy 或少调用，而是在“愿意使用工具”和“避免无效调用”之间保持覆盖。

---

### 环节三：Pre-Aligned DPO Training——先压缩冗余行为

**核心思想**：第一轮 DPO 先建立一个保守的效率偏好——在已经能够答对的候选中，优先选择调用更少、entropy 更低、轨迹更短的版本；负样本必须答错且比正样本调用更多。

采样后，作者先按每道题候选轨迹的正确率划分 easy/hard，再构造偏好对。entropy-guided 与 vanilla 数据使用略有不同的阈值与选择规则：

| 数据来源 | Easy / Hard 划分 | 正样本 | 负样本 |
|:--|:--|:--|:--|
| Entropy-guided | 正确轨迹比例 `>=50%` / `<=50%` | 正确、工具调用最少，并在候选中 entropy 最低；若没有正确候选，回退到原 SFT 轨迹 | 错误，且工具调用次数多于正样本 |
| Vanilla | 正确轨迹比例 `>=70%` / `<=40%` | 最短的正确轨迹 | 错误，且长度或调用多于正样本 |

训练时 hard : easy 的数据比例设为 `2:1`。这一步得到 $D_{\mathrm{dpo1}}$，再用标准 DPO 将 $M_{\mathrm{sft}}$ 更新为 $M_{\mathrm{dpo1}}$。

> **Criteria 1 的方向是“答对的前提下更省”。** 它主要处理工具过用和 overthinking，不足以解决工具少用：如果模型对困难题始终过早停止，单纯偏好短轨迹只会让这一问题更严重。

---

### 环节四：Self-Evolved DPO Alignment——再补回必要探索 ⭐ 核心创新

**核心思想**：同一套“越短越好”标准不能用于所有题。更新后的模型会把一部分问题学成 easy，但仍有一批 hard problems 需要更长推理和更多必要工具调用；第二阶段必须根据当前 policy 的能力动态改变偏好方向。

$M_{\mathrm{dpo1}}$ 重新采样训练问题，并依据新候选的正确/错误数量重新划分难度：正确轨迹数少于错误轨迹数一半的样本进入 hard set，easy set 继续使用原规则。随后两类题采用非对称偏好：

| 当前难度 | 正样本 | 负样本 | 想教给模型什么 |
|:--|:--|:--|:--|
| **Easy** | 正确、工具调用较少、entropy 最低 | 错误且工具调用最多 | 已经会做的题要更紧凑，继续压缩冗余 |
| **Hard** | 正确且 reasoning chain 最长 | 错误且 reasoning chain 最短 | 不会做的题不要过早收手，允许必要探索 |

hard set 的选择看似违反“更短更好”，其实正是 Tool-Light 为解决 underuse 加入的反向校准。对当前模型尚未掌握的问题，短错误轨迹通常意味着模型没有充分调用工具或过早结束；长而正确的轨迹虽然效率未必最高，却先证明了“继续推理和使用工具可以完成任务”。模型学会完成后，这些题在下一轮可能转为 easy，再接受效率压缩。

```text
当前模型采样
  -> 重新判断每道题是 easy 还是 hard
  -> easy：偏好少调用、低 entropy 的正确轨迹
  -> hard：偏好更充分探索后答对的轨迹
  -> DPO 更新模型
  -> 用新模型再次采样与重分难度
```

这就是 `Self-Evolved` 的准确含义：不是固定数据反复训练，而是 policy 更新后重新生成偏好数据，数据难度和偏好方向都随当前能力变化。论文执行两轮 self-evolved alignment 后取得最佳结果；继续增加轮数反而下降，说明这一闭环并不保证单调自我提升。

推理时不再需要 entropy-guided branch sampling 或偏好对构造，只部署最终 policy 并允许其调用搜索与代码工具。entropy、easy/hard criteria 和循环采样都是训练期数据供给机制。

## 证据、定位与边界

### 主实验结果

| 方法/阶段 | 10 个任务平均性能 | 说明 |
|:--|:--|:--|
| Prompting-Based Multi-TIR | 35.3 | 未训练的模型难以稳定使用多工具 |
| Tool-Star | 56.6 | RL 训练的强 multi-TIR baseline |
| Tool-Light：SFT | 56.6 | 冷启动已建立较强基础能力 |
| + Pre-Aligned DPO | 57.4 | 去冗余后继续提升 |
| + Self-Evolved DPO，1 loop | 57.9 | 动态补探索与效率校准有效 |
| + Self-Evolved DPO，2 loops | **58.0** | 论文最佳配置 |

- **数学任务**：AIME24、AIME25、AMC23、MATH、MATH500、GSM8K。
- **知识任务**：HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle。
- **工具**：代码解释器与搜索；知识任务使用本地 Wikipedia，数学任务使用 Bing Web Search。
- **行为结果**：Tool-Light 的 Efficiency、Necessity 最优，输出序列短于 Tool-Star，支持它同时减少过度调用与不足调用。
- **谨慎看待**：平均性能混合了数学任务的 Qwen2.5-72B LLM-as-Judge 与知识任务 F1；任务尺度和评估方式不同，`58.0` 更适合作为论文内部比较，不宜视为统一能力分数。

主结果支持“偏好学习路线可以达到并略超 Tool-Star”这一判断，但不能证明 DPO 普遍优于 RL。两者的数据生成预算、训练设置和基座并非严格算力匹配；Tool-Light 的提升也大部分来自 SFT 基线，Pre-Aligned 与两轮 self-evolution 合计只把平均性能从 `56.6` 提到 `58.0`。它更有说服力的贡献是行为效率与数据构造，而不是绝对准确率的大幅跃升。

### 消融实验：什么真正决定偏好学习质量

| 配置 | Performance | Efficiency | Necessity | 解读 |
|:--|:--|:--|:--|:--|
| 完整方法，2 loops | **58.0** | **0.44** | 0.75 | 最佳综合点 |
| 1 loop | 57.9 | 0.42 | 0.71 | 已接近最佳 |
| 3 loops | 56.1 | 0.39 | 0.73 | 继续自演化开始退化 |
| 5 loops | 54.1 | 0.36 | 0.72 | 过拟合或有益 pair 枯竭明显 |
| 两种采样改为 1:1 | 56.9 | 0.44 | **0.76** | 混合比例影响存在，但不是最大因素 |
| 正样本随机选 | 53.6 | 0.42 | 0.63 | 正样本标准最关键 |
| 负样本随机选 | 53.9 | 0.41 | 0.74 | 失去有区分度的负样本也明显下降 |

最有解释力的结果不是 entropy 分布变低，而是随机选择正负样本会带来约 4 点性能下降：**DPO 的上限主要由 pair 是否表达了正确的行为差异决定。** 采样策略负责扩大候选空间，真正把行为灌进模型的是 easy/hard 条件下的非对称 pair selection。

多轮自演化的退化也很重要。作者推测，随着模型更新，能提供新学习信号的 preference pairs 逐渐减少，继续训练会过拟合当前训练分布。因而“模型自己采数据”不等于数据质量会自动单调提高；每轮都需要重新评估边际信息增益和停止条件。

### Entropy 证据到底支持什么

预实验和 Figure 5 支持两个相关性判断：工具结果会改变后续 entropy 形态，调用较少的正确路径往往具有较低 entropy；Tool-Light 训练后的输出 entropy 也低于 Search-R1 与 ReCall。但这些图不能证明低 entropy 导致了高效率，更不能证明 entropy 是工具行为的充分指标。性能提升同时受到 SFT 数据、混合采样、pair criteria 和多轮 DPO 影响，论文没有用严格消融单独隔离 entropy-guided branching 的因果贡献。

因此，entropy 在这篇论文里应被理解为**低成本的候选生成与排序 proxy**，不是 verifier，也不是 reward。其价值在于让采样更有方向，而其风险是把模型不确定性误当成任务价值。

### 方法定位

Tool-Light = **`hard-question mining + hybrid trajectory sampling + asymmetric preference construction + self-evolved DPO`**

它与 Tool-Star / ET-Agent 的本质区别是：

- **Tool-Star**：通过 multi-tool RL 优化在线 rollout 行为。
- **Tool-Light**：不做 on-policy RL，把行为校准下沉到偏好数据构造与 DPO。
- **ET-Agent**：先用飞轮扩大行为分布，再以组内相对 RL 收敛；Tool-Light 则把扩展后的分布直接压成 chosen/rejected pairs。

Tool-Light 证明了一个有价值的工程顺序：在上 full RL 之前，先检查问题能否被改写为“如何构造足够有区分度的行为偏好”。但 DPO 的二元压缩也会丢失候选集合中的丰富信息，同一问题多种合法工具路径最终只通过一对 chosen/rejected 进入更新，这正是后续 ET-Agent 转向分布扩展与组内相对强化的背景之一。

### 代价与前提

| 局限 | 原因 |
|:--|:--|
| entropy 是相关性 proxy | 高 entropy 可能是有效探索，也可能是混乱；低 entropy 也可能是过早自信 |
| 偏好标准依赖可靠答案评估 | 数学用 LLM-as-Judge、知识任务用 F1；错误标签会直接污染 pair 排序 |
| 自演化不保证单调改进 | 两轮后继续训练明显下降，需要显式停止条件 |
| 计算成本被转移到离线采样 | 多轨迹、树状分支、工具执行和多轮重采样仍然昂贵，只是避免了 RL 优化成本 |
| 效率与必要性都是代理指标 | Efficiency 用性能/调用次数，Necessity 依赖其他方法的相对比较，不等于真实工具价格或延迟 |
| 泛化受 backbone 影响 | Qwen2.5-7B 平均 58.0，而 Llama3.1-8B 版本为 36.8，说明配方效果依赖初始能力与数据匹配 |
| 工具环境仍受控 | 知识任务主要是本地 Wikipedia；真实网页时效性、异步调用和不同工具成本未覆盖 |

## 研究卡片

- **问题**：TIR 模型会过度调用、调用不足，并在 tool result 后继续 overthinking；最终答案正确性无法区分这些行为。
- **做法**：SFT 冷启动与困难题筛选 -> vanilla/entropy-guided 混合采样 -> Criteria 1 构造“答对且更省”的偏好对 -> Pre-Aligned DPO -> 当前模型重采样并重分 easy/hard -> easy 继续压缩、hard 奖励必要探索 -> 多轮 Self-Evolved DPO。
- **证据**：10 个任务平均性能从 SFT 的 `56.6` 提升到两轮 self-evolved DPO 的 `58.0`，略高于 Tool-Star；随机选择正样本/负样本分别降至 `53.6/53.9`，说明严格 pair construction 是最有解释力的组件。
- **前提**：entropy 与有效探索、工具次数与真实效率、轨迹长短与必要探索之间具有足够稳定的相关性；答案评估能够可靠地区分正负轨迹。
- **教训**：偏好优化的关键不在 DPO 公式，而在候选分布与 pair 语义。对于已掌握问题应偏好紧凑正确轨迹，对于未掌握问题则必须先奖励能够完成任务的充分探索，不能用统一的“越短越好”规则。
- **遗留**：如何用信息增益、实际工具成本或 step-level verifier 替代 entropy/长度代理？如何自动判断 self-evolution 何时停止？能否保留整组候选的相对结构，而不是压缩成单一偏好对？
