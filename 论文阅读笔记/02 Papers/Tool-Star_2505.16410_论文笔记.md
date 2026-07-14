---
type: paper-note
title: "Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning"
authors:
  - "Dong, Guanting"
  - "Chen, Yifei"
  - "Li, Xiaoxi"
  - "Jin, Jiajie"
  - "Qian, Hongjin"
  - "Zhu, Yutao"
  - "Mao, Hangyu"
  - "Zhou, Guorui"
  - "Dou, Zhicheng"
  - "Wen, Ji-Rong"
year: 2025
venue: "arXiv preprint"
status: deeply-read
topics:
  - Multi-Tool Reasoning
  - Data Synthesis
  - Agentic RL
  - Self-Critic DPO
doi: "10.48550/arXiv.2505.16410"
url: "https://arxiv.org/abs/2505.16410"
---

# Tool-Star（arXiv:2505.16410）论文笔记

Guanting Dong et al. | arXiv v1, 2025 | [PDF](https://arxiv.org/pdf/2505.16410v1) | `Multi-Tool Reasoning` `Data Synthesis` `GRPO` `Self-Critic DPO`

## 问题与核心判断

Tool-Star 要解决的不是让模型学会某一种工具调用格式，而是**让同一个小模型在推理过程中根据任务性质协调搜索、网页浏览和代码执行，并避免多工具训练退化成“偏科”或“滥用工具”**。此前 tool-integrated reasoning 大致有两条路线：SFT 从强模型蒸馏工具轨迹，容易受 demonstrations 覆盖范围限制；outcome-RL 允许模型自主探索，但多聚焦单一搜索或代码工具，学到的策略在知识任务和计算任务之间迁移很差。

多工具训练进一步放大三个失效模式。第一，高质量多工具轨迹稀缺，直接从约 1% 的现成 TIR 数据启动，模型很难形成稳定格式和调用习惯。第二，若把所有样本混在一起训练，模型可能在本可直接作答的简单题上也调用工具，或在冷启动尚未掌握工具协议时就被迫探索最难任务。第三，简单的终局正确性 reward 不区分格式合法、单工具成功和跨工具协作；但直接加入复合规则后，policy 又未必能从稀疏组相对信号中稳定理解这些行为原则。

Tool-Star 的真正增量因此不在 GRPO 或 DPO 公式，而在**训练供给与课程如何围绕多工具行为组织**：先用普通 reasoning query 合成工具轨迹，再通过调用频率、重复调用和格式规则清洗；随后比较 direct reasoning 与 TIR 的正确性，把“无需工具/工具有益/两者都难”分流到不同训练阶段；最后用 cold-start SFT 建立基本行为，用 hierarchical reward 的 GRPO 探索困难样本，再把同一规则产生的好坏 rollout 转成 DPO preference pairs，周期性强化 reward principles。

这里的关键矛盾是：只做 SFT，模型会模仿已有工具路径而缺少困难任务探索；只做 RL，初始 policy 又缺乏工具格式与反馈理解，探索成本和策略投机都很高；只给 multi-tool bonus，则可能鼓励无必要的搜索加代码。Tool-Star 用数据分流、冷启动、on-policy 探索和偏好刷新依次缓解这些问题。其承重叙事是：

> **先扩增并清洗工具轨迹，再按“工具是否真正带来收益”分配 SFT 与 RL 样本，最后让规则 reward 既作为在线优化信号，又转化为自生成偏好监督，使多工具行为从可模仿逐步变成可探索、可内化的策略。**

这一定义也帮助区分后续工作。Tool-Star 主要解决多工具数据、课程与 reward 供给；ARPO 才进一步改变工具节点的 rollout 分支与共享前缀归因。Tool-Star 的 reward 仍是 trajectory-level，不能因为包含 multi-tool bonus 就称为 step-level credit assignment。

## 机制与训练闭环

Tool-Star 包含三个不同层次：离线数据构造决定 policy 先看到什么；交替 GRPO/DPO 决定困难任务上怎样更新参数；inference-time debugger、backtracer、refiner 则在部署时修复训练没有消除的执行失败。把三层混成“六个工具”会掩盖真正的数据流。

```text
约 90K language-only query + 约 1K 现有 TIR query
  -> TIR prompting 直接生成工具轨迹
  -> 在普通 reasoning chain 的不确定/反思位置插入 hint，截断并重生成工具后缀
  -> 正确性过滤 + 调用频率控制 + 重复调用删除 + 格式归一
  -> 对同一 query 分别执行 direct reasoning 与 TIR
  -> 按二者正确性分成“无需工具 / 工具有益 / 两者都难”等类别
  -> 54K 较易或工具有益样本 -> cold-start SFT policy
  -> 10K hard samples -> 带搜索、浏览、Python 的 GRPO rollout
  -> correctness + format + multi-tool bonus -> hierarchical reward
  -> 规则 reward 同时筛 chosen/rejected -> 周期性 self-critic DPO
  -> 返回 GRPO 继续在线探索，交替更新最终 policy

部署时：最终 policy + 搜索/浏览/Python
  + 可选 code debugger / rule-based backtracer / chain refiner
```

### 数据环：从普通 query 制造并筛选 TIR 轨迹

Tool-Star 的原始供给约为 90K language-only reasoning 数据与 1K 现有 TIR 数据。作者没有直接蒸馏原数据集中的强模型 reasoning chain，而是只取 query，用与最终 backbone 同规模的 Qwen2.5-3B-Instruct 自行采样，每题三次。这一点很重要：收益不是简单复制更大 teacher 的完整解题过程，而是通过工具环境把当前模型的 query 转化成新轨迹。

第一条生成路径是 TIR prompting。模型在生成中输出 `<search>`、`<python>` 等特殊 token，执行器截获请求、调用搜索/浏览/代码工具，再把 `<result>` observation 拼回 reasoning chain，直到生成 `<answer>`、达到调用次数或长度上限。只保留答案正确的轨迹，形成直接工具采样集。

第二条路径是 hint-based sampling，目的不是增加更多同质完整轨迹，而是强制探索普通推理容易错过的工具分支。系统先生成 language-only chain，再在“不确定表达”附近插入 Logical Verification hint，或在答案后插入 Answer Reflection hint；随后在 $t_H$ 处截断旧 chain，从“此前语言推理前缀 + hint”开始重新生成工具增强后缀：

$$
P_\theta(R^c_{>t_H},y\mid I_T,q,R^c_{\le t_H},\mathcal T).
$$

为什么必须重生成后缀？因为插入工具 action 后，环境 feedback 已经改变了 observation；原来的 language-only 后缀不再条件于这个新证据，直接复用会制造前后不一致轨迹。hint 的作用是提供分叉位置，工具执行与后缀重采才真正产生新的行为数据。

正确并不等于工具使用合理，所以两路样本还要经过 quality normalization：调用次数超过阈值 $\beta=5$ 的样本被删除，同一响应中重复的查询或代码调用被移除，工具调用、反馈与答案标签被统一并检查配对。其输出是格式可解析、调用不明显冗余的 TIR 数据。这里的规则是廉价代理：频率低不一定高效，重复查询也可能是必要验证，但它们能先清除最明显的坏习惯。

### 难度分流：不是按题目标签，而是按工具的边际价值分课程

Tool-Star 对每个清洗后的 query 再执行一次 direct reasoning，并与 TIR 结果的正确性形成四象限。若 direct reasoning 已正确，工具调用被视为非必要，训练样本采用 direct response；若 direct 错而 TIR 对，说明工具对该题有明确增益，采用 TIR response；这两类及相邻较易类别组成约 54K cold-start 数据。若 direct 与 TIR 都失败，说明现有 policy 仍无法靠一次工具增强解决，约 10K 此类 hard sample 被保留给 RL。

这个分流比笼统的 easy/hard curriculum 更有信息量：它同时估计题目难度和工具必要性，防止 SFT 把“凡题必调工具”写进 policy，也把 RL 预算集中到模仿数据无法解决的区域。它依赖一次或少量采样的正确性作为分类依据，因此会受到采样方差影响；“direct 正确”也不代表在其他条件下工具永远无益。

### Stage 1：Cold-start SFT 先建立可探索的工具 policy

Cold-start 阶段对 54K 分流后的 response 做标准 teacher forcing，训练三轮，得到能够识别特殊 token、读取工具 feedback、在必要时组合搜索与代码的初始 policy。SFT objective 本身没有创新；它在闭环中的作用是把 raw model 变成 RL 可用的行为先验。

论文训练曲线显示，1.5B 与 3B 模型在 RL step 0 已有明显高于原 instruct model 的 validation score，后续 response length 也没有先暴涨再坍缩。这支持 cold start 降低早期黑箱探索负担。消融中去掉 Cold-Start 后，HotpotQA、Bamboogle、GSM8K、MATH 分别下降 8.4、11.7、7.2、8.4，说明整个 pipeline 对初始行为供给高度依赖。

### Stage 2：Hierarchical reward 的 GRPO 探索困难样本

在 10K hard samples 上，policy 每题生成 8 条带工具轨迹，工具 request 与输出用 cache 映射，重复请求直接复用结果以降低延迟。训练 reward 同时考虑答案正确性、格式与多工具协作：

$$
R=
\begin{cases}
\operatorname{Acc}+r_M, & \text{格式正确且 }\operatorname{Acc}>0,\\
0, & \text{格式正确且 }\operatorname{Acc}=0,\\
-1, & \text{格式错误},
\end{cases}
\qquad
r_M=
\begin{cases}
0.1, & \text{同时使用 search 与 python},\\
0, & \text{其他}.
\end{cases}
$$

GRPO 用同题 rollout 的标准化相对 reward 作为 advantage，经 clipping 更新 policy；训练中 KL coefficient 设为 0。标准 GRPO 不是贡献，真正变化在 reward 语义与 rollout 工具环境。multi-tool bonus 试图防止策略退化成单工具，但它只检查 search 与 Python 是否同时出现，不检查第二个工具是否对答案产生因果贡献，所以可能奖励装饰性调用。正确性约束和质量课程只能减轻，不能消除这种 reward hacking。

工具返回 token 在 loss 中全部 mask，只训练模型自己生成的 reasoning 与 tool request。原因是 observation 来自外部环境，不是 policy action；若把它们也纳入 likelihood ratio，模型会被迫“预测”不可控的工具输出。附录显示去掉 masking 后 reward 很快跌到 -1 并持续出现策略投机，这是这套多工具 RL 最强的稳定性证据之一。

### Stage 3：Self-critic DPO 把同一规则从 reward 变成偏好监督

复合 reward 对在线 RL 来说仍较稀疏。Tool-Star 每执行若干 GRPO step，就用当前 policy 对随机 hard query 自采样多条响应，并运行同一个 hierarchical reward 程序打分；$r\ge1$ 的响应作为 chosen，$r<1$ 的响应作为 rejected，形成 on-policy preference pairs。随后固定当前 RL policy 的副本作为 reference，用标准 DPO 提高 chosen 相对 rejected 的概率。

这里的 “self-critic” 容易误读：模型没有生成自然语言批评，也没有独立学习 critic；**critic 是可执行的规则 reward，self 指正负样本由当前 policy 自己生成**。同一信号在两阶段扮演不同角色：

| 阶段 | Hierarchical reward 的角色 | 留给下一阶段的产物 |
|---|---|---|
| GRPO | 组内 advantage 来源，驱动在线探索 | 更新后的 policy 与新 rollout 分布 |
| Self-critic DPO | chosen/rejected 排序器 | 更明确偏好高奖励行为的 policy |
| 下一轮 GRPO | 再次评价新 policy 的工具轨迹 | 循环迭代的最终 policy |

DPO 的作用不是获得新事实监督，而是把规则已经表达的行为边界以 pairwise loss 再强化一次。它可能提高稳定性，也构成同源闭环：训练 reward、偏好标签和方法评价中的格式/协作行为共享规则，无法证明模型真正理解了多工具协作的因果价值。

### 推理恢复层：训练能力与系统可靠性分开补齐

最终 policy 推理时自主使用 search、browser 和 Python；训练后的模型之外，系统还可启用三个恢复工具。Code Debugger 读取错误代码与 compiler message 后重写；Tool-Use Backtracer 在调用失败时回滚到错误 tool token 前一行并重新生成；Chain Refiner 在长度溢出时压缩旧 reasoning chain，再从压缩状态继续。作者最终采用 rule-based backtrace，因为模型自身难以可靠定位错误点。

这些组件不是训练闭环的一部分，却参与主系统结果。论文让同参数规模 instruct model 充当 debugger、browser refiner 和 chain refiner，称为 self-play inference-time scaling。它们能降低工具失败，但也增加推理调用与系统差异。因此必须分别回答“Tool-Star policy 学到了什么”和“完整部署栈修复了什么”。

## 证据、定位与边界

### 主结果证明跨工具域的整体 pipeline 有效

在统一 Qwen2.5-3B-Instruct backbone 上，Tool-Star 在十项主 benchmark 的平均分为 46.1，高于 ReCall 39.1、ToRL 37.8 和各搜索增强基线。代表性结果如下：

| 方法 | AIME24 | MATH | WebWalker | HotpotQA | Bamboogle | 十项平均 |
|---|---:|---:|---:|---:|---:|---:|
| ToRL | 20.0 | 81.0 | 12.0 | 37.9 | 25.4 | 37.8 |
| Search-o1 | 16.7 | 63.0 | 13.0 | 34.9 | 35.1 | 30.2 |
| ReCall | 16.6 | 74.2 | 13.0 | 43.5 | 40.8 | 39.1 |
| Tool-Star-3B | **20.0** | **82.6** | **20.8** | **51.9** | **52.5** | **46.1** |

结果最有价值的不是某个单点 SOTA，而是代码方法在知识任务上掉点、搜索方法在计算任务上掉点时，Tool-Star 同时维持两域性能。它支持“多工具统一训练减少单工具偏科”。但 Tool-Star 同时改变了合成数据、课程、reward、GRPO/DPO 训练和推理恢复层，主表只能证明完整 pipeline 有效，不能把 7.0 的平均优势全部归因于 multi-tool reward。

GAIA 上 Tool-Star 得分 15.5，高于 Search-R1 9.7 和 ToRL/RAG 8.7；HLE 为 8.0，仅略高于 Search-R1 7.8。深度网页任务的优势并不均匀，且搜索方式本身影响很大：从本地 Wikipedia/E5 换成 Bing web search + browser 后，2Wiki 与 Bamboogle 分别约提高 13 和 8 F1。这说明检索语料和 browser 质量是重要系统变量。

### 消融支持“先冷启动、再 RL、再偏好刷新”的课程

| 设置 | HotpotQA | Bamboogle | GSM8K | MATH |
|---|---:|---:|---:|---:|
| 完整 Tool-Star | **51.9** | **52.5** | **85.0** | **82.6** |
| 去掉 Cold-Start | 43.5 | 40.8 | 77.8 | 74.2 |
| 去掉 RL stage | 47.5 | 43.9 | 80.2 | 78.4 |
| 去掉 hierarchical reward | 50.4 | 50.3 | 83.1 | 80.2 |
| 去掉 Self-Critic | 49.8 | 48.3 | 82.8 | 77.8 |

Cold-start 是最大承重组件，说明高质量轨迹供给比优化器选择更关键；RL 带来进一步泛化；reward 与 self-critic 提供较小但一致的增量。GRPO 与 REINFORCE++ 的差异总体在 3% 内，也进一步证明 Tool-Star 不是某个 policy-gradient objective 的胜利。

不过消融没有分别移除 hint-based sampling、quality normalization 和 difficulty classification，也没有对 multi-tool bonus 与 format/correctness reward 做完全拆分。因此可以说“两阶段数据与训练课程必要”，不能确定具体是哪种合成策略或 bonus 产生主要增益。Self-critic 消融支持额外 preference phase 有用，但没有与相同额外训练 token 的普通 DPO/SFT 对照，收益也可能部分来自更多更新步数。

### 效率与可靠性证据需要区分代理指标和真实成本

论文定义 $TE$ 为使用工具得到正确答案的数量除以使用工具的样本数，衡量的是“调用工具的样本中有多少答对”，而不是每次工具调用的收益、平均调用数、延迟或费用。Tool-Star 在知识与计算两域的 TE 都较高，支持它不像单工具方法那样严重偏科；但把 TE 称为 tool-use efficiency 容易过度，因为一个样本调用 1 次和 5 次工具在该指标中权重相同。

推理恢复工具显著降低 tool error，并对较弱的 DotaMath 带来更大提升，说明 debugger/backtracer/refiner 是有效工程补丁。它也意味着带恢复工具与不带恢复工具的主结果必须分开解释。训练阶段的 request-output cache、推理阶段的辅助模型调用，以及 web API/browser 成本都没有统一折算成 wall-clock 或货币成本。

### 方法定位与成立边界

Tool-Star 可以概括为 `synthetic TIR trajectories + tool-necessity curriculum + cold-start SFT + hierarchical-reward GRPO / self-generated DPO + inference recovery`。它比只改 reward 的 ToolRL 更系统，比 ARPO 更偏上游数据与课程，比 DeepAgent 更聚焦固定的搜索/浏览/代码组合而非开放工具发现。

| 前提或代价 | 证据边界与风险 |
|---|---|
| 正确答案可被规则或 judge 可靠评价 | 开放任务 reward 噪声会直接污染 GRPO 与 DPO 两个阶段 |
| search + Python 共现近似协作 | 共现不等于必要贡献，可能诱发装饰性调用 |
| 三次采样足以判断工具必要性 | 难度分流可能把采样偶然错误当成稳定属性 |
| 合成轨迹覆盖真实策略 | 3B 自采样减少 teacher mismatch，也限制轨迹上界 |
| 工具 observation 不参与 loss | 避免学习环境 token，但模型仍可能受错误 feedback 影响后续 action |
| 推理辅助模型可用 | debugger/browser/refiner 增加成本，纯 policy 能力低于完整系统能力 |
| 固定工具组合代表多工具泛化 | 训练核心是搜索、浏览、Python，不能直接推出对任意 API/toolset 泛化 |

## 研究卡片

- **问题**：单工具 SFT/RL 容易形成领域偏科；多工具训练又缺少高质量轨迹、工具必要性课程和可被 policy 稳定理解的复合行为信号。
- **做法**：TIR prompting 与 hint 分叉生成工具轨迹 -> 正确性与调用规则清洗 -> 通过 direct/TIR 成败比较分流 SFT 与 hard RL 数据 -> cold-start SFT -> hierarchical-reward GRPO -> 同一规则筛选自生成 chosen/rejected -> 周期性 DPO -> 推理时用 debugger/backtracer/refiner 修复残余失败。
- **证据**：Tool-Star-3B 在十项主 benchmark 平均 46.1，高于 ReCall 39.1 与 ToRL 37.8；去掉 cold-start、RL、reward、self-critic 均下降，其中 cold-start 降幅最大；去掉 tool-result masking 会使 reward 快速坍塌到 -1。
- **前提**：答案 reward 与工具格式规则必须可靠，search/Python 共现必须大体对应真实协作，少量采样的 direct/TIR 对比必须足以估计工具必要性，部署还需承担 browser 和恢复工具成本。
- **教训**：多工具能力不是把多个 schema 同时塞给模型，也不是给 outcome reward 加一个 bonus 就会自然出现。有效训练需要先制造可执行轨迹，再用工具边际价值组织课程，最后让同一行为原则以在线 reward 和离线偏好两种方式被 policy 消费；环境 observation 必须与 policy action 在 loss 中严格分开。
- **遗留**：如何用逐工具边际贡献替代简单共现 bonus？如何减少 reward 与 preference 的同源闭环？如果去掉推理恢复工具，最终能力还剩多少？当工具从搜索/Python 扩展到数千开放 API 时，现有数据合成、难度分流和固定格式还能否维持？
