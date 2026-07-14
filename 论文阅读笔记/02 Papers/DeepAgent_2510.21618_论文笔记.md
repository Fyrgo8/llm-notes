---
type: paper-note
title: "DeepAgent: A General Reasoning Agent with Scalable Toolsets"
authors:
  - "Li, Xiaoxi"
  - "Jiao, Wenxiang"
  - "Jin, Jiarui"
  - "Dong, Guanting"
  - "Jin, Jiajie"
  - "Wang, Yinuo"
  - "Wang, Hao"
  - "Zhu, Yutao"
  - "Wen, Ji-Rong"
  - "Lu, Yuan"
  - "Dou, Zhicheng"
year: 2026
venue: "WWW 2026"
status: deeply-read
topics:
  - General Agent
  - Open-Set Tool Retrieval
  - Memory Folding
  - Agentic RL
doi: "10.1145/3774904.3792460"
url: "https://arxiv.org/abs/2510.21618"
---

# DeepAgent（arXiv:2510.21618）论文笔记

Xiaoxi Li et al. | WWW 2026 / arXiv v3 | [PDF](https://arxiv.org/pdf/2510.21618v3) | `General Agent` `Scalable Toolsets` `Memory Folding` `ToolPO`

## 问题与核心判断

DeepAgent 要解决的不是固定工具集上的调用准确率，而是**reasoning model 如何在不知道预先需要哪些工具的情况下，跨越大工具集与长交互链自主完成任务**。传统 ReAct、Plan-and-Solve 等 agent 把执行组织成预定义循环：先规划，再调用已提供的工具，再读取 observation。这个框架在短任务上可控，但放大到开放工具环境时会暴露四个失效模式。

第一，工具通常在任务开始前一次性给定或检索，后续步骤无法根据新状态重新发现工具；工具集合从几十扩展到数千、上万时，把所有 schema 塞入上下文也不可行。第二，workflow 把 reasoning 切成固定步骤，模型只能在局部节点作答，难以在同一条推理流里同时决定“继续思考、搜索工具、执行动作还是重整任务状态”。第三，长程交互不断累积工具文档、调用结果和错误路径，原始历史既消耗上下文，又让早期错误持续支配后续决策。第四，若训练仍只用终局 reward，模型知道任务是否完成，却不知道中间工具调用是否正确，也不知道主动折叠记忆是否真的提高了效率。

DeepAgent 的贡献因此横跨**推理协议、状态管理和训练信号**三层，而不是单独一个 ToolPO loss。推理协议把 internal thought、tool search、tool call、memory fold 都变成 reasoning model 在连续生成中自主选择的 action；状态管理用辅助 LLM 把历史压缩成 episodic、working、tool 三类结构化记忆；训练层用模拟 API 降低在线 rollout 的成本与波动，再将全局任务 reward 和局部 action reward分别写回整条序列与工具/折叠 token。

这三层必须共同工作。只有动态工具检索而没有 memory folding，长任务仍会被不断增长的工具结果拖垮；只有 memory folding 而没有结构化 schema，摘要可能丢失当前目标、失败工具和参数经验；只有终局 RL，又无法直接约束工具调用与折叠动作。整篇论文唯一承重的叙事是：

> **让主模型在统一推理流中按需发现和调用工具；当历史阻碍继续探索时主动折叠为可执行状态；再用全局成功信号与局部动作信号共同训练这套自主决策。**

这个定位也需要降噪。DeepAgent 不是“完全由单一模型自主运行”的纯端到端系统：工具检索依赖 embedding model，工具文档过滤、结果去噪、API 模拟和 memory folding 都依赖 Qwen2.5-32B-Instruct 辅助模型。它真正取消的是手写的高层 workflow 与预先固定的工具选择，不是取消系统编排和外部模型。

## 机制与训练闭环

DeepAgent 的闭环分成在线执行环和离线 ToolPO 训练环。在线环制造带动态检索、真实或模拟工具反馈、可选 memory fold 的完整轨迹；训练环把叶节点任务结果与中间动作质量变成两种 advantage，再按 token 类型写回 policy。

```text
用户任务 Q + 指令 I + 任意规模工具索引
  -> 主 reasoning model 在连续生成中选择 think / tool_search / tool_call / memory_fold
  -> retriever 返回候选工具；辅助 LLM 压缩过长文档和工具结果
  -> 工具环境执行 action 并返回 observation
  -> 必要时辅助 LLM 将全部历史折叠为 episodic / working / tool memory
  -> 以折叠后的结构化状态开启新一轮推理，直至任务完成

训练时：真实任务混合数据 + LLM-simulated APIs
  -> 每题采样 K 条上述交互轨迹
  -> task-success reward -> 全序列 advantage
  -> 正确工具调用 + 折叠效率 reward -> action advantage
  -> action mask 只把局部 advantage 加到 tool-call / memory-fold token
  -> clipped policy optimization 更新 DeepAgent

部署时：保留主模型、retriever、工具执行器、辅助 LLM 与 memory folding
```

### 连续推理中的动态工具发现

DeepAgent 把 agent action 写入统一的自回归序列。模型可以输出普通 internal thought，也可以用 `<tool_search>` 生成自然语言工具需求、用 `<tool_call>` 生成工具名和参数，或用 `<fold_thought>` 请求重整历史。系统只负责截获这些结构化 action、执行并把 observation 写回上下文，下一步选择仍由主模型决定。

工具搜索不是生成工具名，而是从离线工具文档索引中做 dense retrieval。对查询 $q_s$ 和每个工具文档 $d_i$，系统按 embedding cosine similarity 取 top-$k$：

$$
\mathcal T_{\mathrm{retrieved}}
=\operatorname{top-k}_{\tau_i\in\mathcal T}
\operatorname{sim}\bigl(E(q_s),E(d_i)\bigr).
$$

检索结果过长时，由辅助 LLM 先筛选和摘要，再交给主模型；工具执行结果同样可以先去噪压缩。这个设计把大工具集问题转化成“模型何时提出何种工具需求 + retriever 能否召回正确 schema”。它避免一次性注入上万工具，却引入新的瓶颈：若查询表达错误或 embedding 召回失败，主模型连正确工具都看不到；辅助模型压缩错误也会改变后续 observation。

### Memory folding：不是保存全文，而是重建可继续执行的状态

当模型完成子任务、发现当前路线错误或认为历史过长时，可以主动输出 `<fold_thought>`。辅助模型读取此前完整状态 $s_t$，并行生成三种 JSON memory：

$$
(M_E,M_W,M_T)=f_{\mathrm{compress}}(s_t;\theta_{\mathrm{aux}}).
$$

$M_E$（episodic memory）记录任务目标、关键事件、决策与阶段结果，维持长程任务进度；$M_W$（working memory）记录当前子目标、障碍和下一步动作，使折叠前后能够连续执行；$M_T$（tool memory）记录已用工具、有效参数、常见错误、响应模式和经验规则，防止新一轮重复踩错。三者不是并列装饰，而是分别回答“发生过什么”“现在要做什么”“工具怎样用”。

折叠完成后，原始 interaction history 被结构化 memory **替换**，主模型从新状态继续推理。这里不能把 fold 理解成普通摘要：它同时改变了下一步 policy 的 state。如果关键信息被压掉，旧后缀不能无损续接；系统必须基于新状态重新生成后续 action。固定 JSON schema 只能提高可解析性和字段覆盖，不能保证事实忠实或信息充分。

Memory folding 的另一个功能是策略重启。模型在错误路径上越走越远时，折叠会迫使其显式总结当前进度、障碍和下一步，从而获得“重新规划”的机会。但其触发仍由同一个 policy 决定，action reward 又偏好更短的折叠轨迹，所以模型可能过早折叠、频繁折叠，或为了长度奖励牺牲信息。这是训练信号必须处理的核心权衡。

### Tool Simulator：把大规模 API 从不稳定环境变成训练供给

ToolPO 的训练数据来自四类任务：ToolBench 提供通用工具使用，ALFWorld/WebShop 提供环境交互，WebDancer/WebShaperQA 提供 deep research，DeepMath 提供代码推理，总量约 4.4K query。直接在数千真实 API 上反复 rollout 会遇到接口失效、延迟、费用与不可复现问题，所以 RapidAPI 一类调用由 Qwen2.5-32B-Instruct 模拟；搜索、浏览、代码、VQA、ALFWorld 和 WebShop 等仍由相应真实或任务环境执行。

模拟器的产物是稳定、低延迟的 tool observation，消费者是同轮主模型以及最终 reward 计算。它解决的是训练环境供给，而不是 policy credit assignment。代价是 synthetic gap：模拟器可能返回语义合理却与真实 API schema、错误码、状态变化不一致的结果。论文没有给出模拟响应与真实 API 的人工一致性或迁移误差评估，因此“稳定训练”有实验支持，“忠实模拟真实工具”尚未被单独证明。

### ToolPO：全局成功与局部动作使用不同写回范围

对每个 prompt，ToolPO 采样 $K$ 条轨迹。第一类 reward 是 $R_{\mathrm{succ}}(\tau)$，表示最终答案或环境任务是否成功；它计算组相对 advantage 后作用于该轨迹的全部 policy token：

$$
A_{\mathrm{succ}}(\tau_k)
=R_{\mathrm{succ}}(\tau_k)
-\frac{1}{K}\sum_{j=1}^{K}R_{\mathrm{succ}}(\tau_j).
$$

第二类 reward 评价中间 action：

$$
R_{\mathrm{action}}(\tau)
=\lambda_1\sum_{t=1}^{T}C(a_t^{\mathrm{call}})
+\lambda_2S_{\mathrm{pref}}(\tau),
$$

其中 $C(a_t^{\mathrm{call}})=1$ 表示工具调用正确；折叠偏好分数比较同题折叠轨迹与直接轨迹长度：

$$
S_{\mathrm{pref}}
=\frac{L(\tau_{\mathrm{direct}})-L(\tau_{\mathrm{fold}})}
{L(\tau_{\mathrm{direct}})+L(\tau_{\mathrm{fold}})}.
$$

由 $R_{\mathrm{action}}$ 同样减去组均值得到 $A_{\mathrm{action}}$，但它不广播到普通 reasoning token，只通过 mask 加在 tool-call 与 memory-fold token 上：

$$
A(y_i)=A_{\mathrm{succ}}(\tau_k)+M(y_i)A_{\mathrm{action}}(\tau_k).
$$

这是 ToolPO 真正的优化增量。终局成功继续约束整条推理，局部信号只提高或降低产生结构化 action 的 token 概率，避免“正确工具调用”奖励连普通思考文本一起。随后的 clipped surrogate objective 本身是常规 PPO/GRPO 式更新，不是论文的主要创新。

不过所谓 fine-grained attribution 仍有边界。$R_{\mathrm{action}}$ 是整条轨迹所有正确调用与折叠效率的总和，论文再把同一个组相对 action advantage 加到该轨迹所有 action token；它没有为第 1 次和第 5 次调用分别估计边际贡献。因此这是**action-token masked trajectory advantage**，比全序列广播更细，但不是逐 action 的因果 credit。

训练使用 QwQ-32B 主模型，100 个 ToolPO step，batch size 64，$K=8$，最大序列 32,768，$\lambda_1=\lambda_2=1$，共使用 64 张 H20-141GB GPU。推理最大长度升至 81,920 token、最多 50 个 action。部署时训练用 tool simulator 不必保留，但动态 retriever、真实工具、辅助 LLM 和 memory folding 都是系统的一部分，因此 DeepAgent 的推理成本并未退化成单模型调用。

## 证据、定位与边界

### 主结果支持完整系统有效，尤其支持开放工具检索

在 open-set tool retrieval 场景中，DeepAgent-32B-RL 相对最强 32B workflow baseline 的成功率提升最清晰：

| 任务（工具规模） | 最强 32B workflow | DeepAgent-32B-RL | 绝对差值 |
|---|---:|---:|---:|
| ToolBench（16K+） | 54.0 | **64.0** | +10.0 |
| API-Bank（73） | 22.0 | **24.0** | +2.0 |
| TMDB（54） | 31.0 | **55.0** | +24.0 |
| Spotify（40） | 24.6 | **50.9** | +26.3 |
| ToolHop（3,912） | 29.0 | **40.6** | +11.6 |

在下游任务上，DeepAgent-32B-RL 达到 ALFWorld 91.8、WebShop 34.4、GAIA 53.3、HLE 20.2；相对同 backbone 的 DeepAgent-Base 分别提高 3.7、2.4、6.6、2.4。前者证明“统一 reasoning + 动态检索 + memory folding + 辅助系统”的整体优势，后者才较直接支持 ToolPO 在既定架构上的增量。

论文摘要写“eight benchmarks”，但正文与表格实际枚举 ToolBench、API-Bank、TMDB、Spotify、ToolHop、ALFWorld、WebShop、GAIA、HLE 共 9 个 benchmark。这不影响结果本身，却说明论文的覆盖面表述存在计数不一致。

### 消融支持三个承重组件必要，但没有完全隔离其机制

| 设置 | ToolBench | ToolHop | WebShop | GAIA | 平均 |
|---|---:|---:|---:|---:|---:|
| 完整 DeepAgent-RL | **64.0** | **40.6** | **34.4** | **53.3** | **48.1** |
| 去掉训练（Base） | 60.0 | 38.4 | 32.0 | 46.7 | 44.3 |
| 去掉 Memory Folding | 63.0 | 36.6 | 32.4 | 44.7 | 44.2 |
| 去掉 Tool Simulation | 62.0 | 35.2 | 33.6 | 48.5 | 44.8 |
| 去掉 Tool Advantage | 62.0 | 39.6 | 33.2 | 49.5 | 46.1 |

Memory folding 在 GAIA 上下降 8.6，符合它主要服务长程状态管理的机制预期；去掉模拟器和局部 advantage 也都下降，支持 ToolPO 的两部分不是纯装饰。但“去掉 Tool Simulation”会同时改变 observation 来源、延迟与训练分布，不是只改变稳定性；“去掉 Memory Folding”也同时减少一种 action 和辅助 LLM 调用。消融能支持组件必要性，不能独立证明作者给出的具体因果解释。

动态检索对照更直接：DeepAgent 从预检索工具的平均 42.0 提升到按需检索的 52.6，在 ToolBench 上由 53.0 提升到 64.0。ReAct 和 Plan-and-Solve 使用动态检索也有改善，说明收益一部分来自检索策略本身，而不是 DeepAgent 独占；DeepAgent 提升更大则表明统一推理协议更能消费按需工具供给。

### 方法定位与系统边界

DeepAgent 可以概括为 `continuous agentic reasoning + dynamic tool retrieval + structured context folding + simulated-environment on-policy RL`。它与 ARPO 的区别是：ARPO 改变同一工具节点下的 rollout topology 与共享前缀归因；DeepAgent 不做树状分叉，而是为开放工具发现、记忆 action 和工具 token 增加训练信号。它与 EnvScaler 的关系是互补：EnvScaler自动构造可验证环境，DeepAgent 主要用 LLM 模拟既有 API observation。它与固定 workflow agent 的差别则在 action policy 层，而不只是换更强 backbone。

基线公平性仍需谨慎。主结果使用 QwQ-32B 作为 reasoning model，并额外使用 Qwen2.5-32B-Instruct 进行文档过滤、结果去噪和 memory folding；论文称过滤长结果也应用于所有 baseline，但 baseline 并不拥有相同的 memory action 和统一推理协议。另一方面，闭源或更大模型只作为灰色参考，不应与 32B 主结论混为一谈。

| 前提或代价 | 证据边界与风险 |
|---|---|
| Dense retrieval 能召回正确工具 | 大工具集上的提升支持可用性，但未报告 retrieval recall 与调用失败分解 |
| 辅助 LLM 能忠实压缩 history | 消融支持 folding 有用，未测 memory 事实保持率和关键状态丢失 |
| LLM simulator 近似真实 API | 支持训练更稳，不证明真实响应、异常和状态转移忠实 |
| action reward 能代表工具质量 | “调用正确”判据与折叠长度偏好可能诱发 reward hacking |
| 长 context 与多模型成本可接受 | 81,920 token、辅助 32B 模型和多个外部工具使推理并不轻量 |
| benchmark 能代表开放世界 | 多数工具环境仍是固定索引或模拟环境，线上 API 漂移尚未覆盖 |

## 研究卡片

- **问题**：预定义 workflow、预先给定工具和无限堆叠历史无法同时支持大工具集、动态任务状态与长程交互；终局 reward 又不能直接训练工具调用和 memory fold 这类中间 action。
- **做法**：连续推理流自主选择 think/search/call/fold -> dense retriever 与工具环境返回 observation -> 辅助 LLM 压缩文档、结果和三类 memory -> Tool Simulator 供给稳定 rollout -> 全局 success advantage 写回全序列、action advantage 只写回 tool/fold token -> clipped on-policy 更新。
- **证据**：开放工具检索下，DeepAgent-32B-RL 在 ToolBench、TMDB、Spotify、ToolHop 分别领先最强 32B workflow baseline 10.0、24.0、26.3、11.6；消融中去掉 memory folding、tool simulation、tool advantage 后平均分分别从 48.1 降到 44.2、44.8、46.1。
- **前提**：工具检索、辅助压缩和模拟 API 必须足够忠实；“正确调用数 + 折叠长度效率”必须与真实任务价值一致；部署能够承担长上下文、辅助模型和外部工具的联合成本。
- **教训**：通用 agent 的可扩展性不能只靠扩大 context 或增加工具 schema。工具发现必须成为 policy action，历史压缩必须产出可继续执行的状态，局部 action reward 还必须只作用于相应 token；协议、状态和训练信号需要一致设计。
- **遗留**：如何测量 memory folding 的信息保持与错误传播？如何用真实 API 异常而非 LLM 合理化响应训练恢复能力？如何把轨迹级 action reward进一步拆成逐调用的边际 credit？在减少辅助模型与 80K context 后，DeepAgent 的优势还能保留多少？
