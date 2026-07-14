---
type: paper-note
title: "EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis"
authors:
  - "Song, Xiaoshuai"
  - "Chang, Haofei"
  - "Dong, Guanting"
  - "Zhu, Yutao"
  - "Wen, Ji-Rong"
  - "Dou, Zhicheng"
year: 2026
venue: "arXiv"
status: deeply-read
topics:
  - Agentic RL
  - Tool Use
  - Environment Synthesis
url: "https://arxiv.org/abs/2601.05808"
---

# EnvScaler（arXiv:2601.05808）论文笔记

Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou | arXiv v2, 2026 | [PDF](https://arxiv.org/pdf/2601.05808v2) | `Agentic RL` `Tool Use` `Environment Synthesis`

## 问题与核心判断

EnvScaler 处理的是 agent post-training 中常被算法讨论遮住的供给问题：模型若要学习多轮工具调用和状态修改，需要大量可执行、可重置、可自动判定的环境；但真实系统难以开放和复现，LLM 直接模拟环境会产生状态与规则不一致，手工 sandbox 又很难覆盖足够多的领域。没有环境状态、真实状态转移和可验证终局，所谓 tool-use 训练往往退化为带调用格式的文本轨迹模仿，无法让模型在交互反馈中探索。

这篇论文的核心判断是，环境本身应当成为可规模化合成的训练数据，而不是少量人工构建的固定前提。它不是新的 RL objective，也不主要改 policy update；真正的增量是把“环境逻辑、任务场景、终局验证”接成一条程序化流水线。对 agent 训练而言，这相当于把原先最稀缺的三件事同时供给出来：可修改的状态、可调用的工具，以及由状态而非参考轨迹决定的可验证 reward。

> 💡 **批注——本质**：用一句话说，EnvScaler 就是 **"造 reward 函数"这个步骤的自动化**。它本质上是把"任务完成了吗"这个主观判断，拆成一系列"最终状态满不满足某个具体条件"的客观检查——用外部 LLM（GPT-4.1）拆检查项，用 Python 代码来验证。你不需要再为每个场景手写 reward 函数了，但产出的信号仍然是粗粒度的。它不是一种新的奖励函数形式，而是**造奖励函数的方式从手工变成了流水线**。

这一定位也限定了它的价值。EnvScaler 能证明的是：在一类领域化、状态化的工具环境中，自动合成的 sandbox 可以成为有效的 SFT/RL 数据来源；它不能证明合成环境已经等价于真实系统，更不能解决开放网页、异步服务或不可程序化任务中的环境建模问题。把论文理解成“用一套通用 RL 配方消费自动生成环境”的系统/数据论文，比理解成“提出新的 agent RL 算法”更准确。

## 机制与训练闭环

EnvScaler 的流程不是“让 LLM 写几个工具”，而是先造一个能运行的世界，再为这个世界造任务和判卷器，最后让 agent 在里面练习：

```text
已有任务集
  -> SkelBuilder：发现主题、生成并筛选可执行环境
  -> ScenGenerator：为环境补初始状态、任务和终局检查函数
  -> SFT：模仿教师在环境中的成功轨迹
  -> RL：按最终状态是否满足任务继续探索和更新
```

### 1. SkelBuilder：把文本任务还原成“有状态的世界”

输入是 API-Bank、ToolACE 一类已有任务，而非现成环境。它先筛掉不需要领域状态的题，再让 LLM 从保留下来的任务反推环境主题，例如“移动消息应用”或“订单系统”；相近主题通过 embedding 去重。接下来，LLM 为每个主题分别设计三样东西：状态和数据表长什么样、哪些业务规则必须遵守、用户可调用哪些操作。再把它们写成 Python 类的属性和方法，自动组装成完整环境。

产物不是一份工具清单，而是一个能执行的 sandbox：`F_exec` 负责真实保存与改变状态，`E_doc` 向 agent 说明环境规则，`Sigma_tool` 暴露工具名称、参数和描述。也就是说，agent 调 `cancel_order` 后，订单状态会被程序真正改变；后续工具看到的是改变后的状态。这是 EnvScaler 与仅合成 function-calling 文本样本的根本区别。

### 2. Dual-Agent Assessment：先验环境能不能用，再交给模型训练

LLM 写出的 Python 能通过语法检查，不代表工具语义正确。作者因此加入两类 agent：testing agent 在当前状态下随机提出正常或异常调用，例如删除不存在的文件；checking agent 查看工具源码、返回结果和调用前后的状态变化，判断环境行为是否符合工具声明和业务规则。这个循环跑 `N=100` 轮，平均通过率低于 `0.85` 的环境被丢弃。

这一步的作用不是把合成环境认证为“真实世界”，而是把最低质量门槛从“代码能跑”提高到“常见调用下状态转移基本自洽”。最终留下 191 个环境，平均每个有 18.58 个工具、21.38 个状态类别；这些环境才进入下一环节。环境逻辑是否足够像真实业务，仍是后文需要保留的泛化风险。

### 3. ScenGenerator：为每个世界补齐起点、任务和自动判卷器

只有环境还不能训练，因为 agent 还不知道从什么状态开始、要完成什么任务、完成后由谁判分。ScenGenerator 先根据环境程序和 state schema 生成初始数据库 `S_init`，再结合环境规则与工具生成一项可解但有挑战的任务。它不从某条预设工具序列反推任务，原因是同一任务可能有多条合法路径；若把一条路径当作 gold trajectory，训练和评测都会错误惩罚等价解法。

接着，模型把 task 拆成 `K` 个可以检查的终局条件，并为每项生成读取 `S_final` 的 Python 验证函数 `f_ck`。例如“取消两件已送达商品”可以拆成订单状态、退款状态、相关消息状态等条件。完整轨迹结束后，reward 是通过检查项的比例：

$$
\{c_k\}_{k=1}^{K}=M(P_{check}\Vert task),\qquad
f_{c_k}=M(P_{check}\Vert c_k),\qquad
r(\tau)=\frac{1}{K}\sum_{k=1}^{K}\mathbf{1}[f_{c_k}(S_{final})=\mathrm{True}].
$$

> 💡 **批注——verifiable RL**：这个 reward 形式决定了 EnvScaler 属于 **verifiable RL**（可验证强化学习）——reward 不是来自人类偏好打分，也不是来自训练出来的 reward model，而是来自**可程序化验证的规则**。每个 $f_{c_k}$ 是一个可执行的 Python 函数，检查最终状态是否满足某个条件——是就是 1，否就是 0。这个信号是确定性、可重复、无需人类介入的。因此不要拿 DPO/偏好优化的 lens 去理解它。

这个 reward 的关键不是公式形式，而是判定对象：它检查 agent 最终是否把世界改到目标状态，而不要求工具序列逐步复刻教师轨迹。因此它允许多条合法解，也能给"完成了一部分子目标"的轨迹部分分。代价是 reward 仍在轨迹结束后才汇总；清单拆细能缓解稀疏，却不能自动定位是哪一个早期操作导致后续失败。

> 💡 **批注——generator 的实质**：ScenGenerator 拆 checklist 和编译 check function 的过程，本质是 **用外部 LLM（论文用 GPT-4.1）通过两轮提示词来造 reward 函数**，不是训练了一个专用生成模型。第一轮把任务描述拆成 "Has ..." 格式的独立检查项（每条对应一个布尔可判条件）；第二轮为每条检查项生成 `def check_func(final_state):` 的 Python 函数，再经 AST 语法检查和环境试跑过滤掉不合格的。所以"自动生成"就是**外部 LLM + 提示词工程 + 编译验证过滤**——把原来人工手写 reward 的步骤变成了可规模化的 pipeline。
>
> 这也意味着 EnvScaler **没有解决信用分配问题**。它解决的是"怎么让造 reward 函数不再靠人手工写"，产出的信号仍然是 trajectory-level 的粗粒度 reward。信用分配——"轨迹只有一个总分，模型不知道哪一步导致失败"——是 EnvScaler 交付给下游 RL 的固有挑战，不是它能消除的。

### 4. SFT 和 RL：在同一环境里先学会操作，再用状态反馈继续改进

训练时 agent 看不到完整环境状态，只能根据历史 `H_t` 与当前 observation `o_t` 决定下一步工具调用：

$$
a_t=\pi_\theta(H_t,o_t),\qquad o_{t+1},S_{t+1}=E(a_t,S_t).
$$

论文提供两种 interaction。Non-Conversation 一开始就给全任务信息，重点测工具执行；Conversation 则把信息放在模拟用户对话里，agent 必须追问后才能行动。SFT 先让学生模型模仿教师在环境中的成功轨迹，解决“完全不会操作时直接 RL 很难探索”的冷启动问题；随后 RL 在 Non-Conversation 环境中用上述状态 reward 和 Reinforce++ 继续更新 policy。

因此，SFT 和 RL 不是两套分离的数据配方，而是消费同一批合成环境的两个阶段：前者学会基本操作与交互格式，后者根据状态反馈优化策略。direct-RL 的结果也表明，环境不是万能药：较小模型更容易受稀疏或噪声 reward 影响，SFT 初始化仍然对获得可探索的 policy 很重要。

说白了，EnvScaler 的价值就是**方便做大规模 RL**——大量不同环境的奖励函数设置自动化了，同时也缓解了稀疏奖励问题。

## 证据、定位与边界

### 主实验：它首先证明“合成环境能拿来训练”

论文生成 191 个环境和约 7K 个场景，其中 140 个环境用于 SFT、其余 51 个用于 RL；SFT 使用约 9K 条教师轨迹。评测覆盖 BFCL-v3 Multi-Turn、Tau-Bench 和 ACEBench-Agent，模型为 Qwen3-1.7B、4B、8B。结果的主结论很直接：SFT 已在三种模型上稳定提升，平均增益为 BFCL-MT `+8.67`、Tau-Bench `+4.29`、ACEBench-Agent `+11.57`；叠加 RL 后，多数结果继续提高。

以 Qwen3-8B 为例，分数从 `28.88 / 38.19 / 60.00` 分别提升到 `41.88 / 44.81 / 72.50`。这证明 EnvScaler 生成的环境不是只能在内部自测的玩具数据：它至少能提供一部分可迁移的多轮、多工具训练信号。但主表证明的是“该训练 pipeline 有效”，不是已经证明“环境合成机制本身在所有真实环境中都有效”。

### 三个补充实验：收益从哪里来，又没有来自哪里

- **相似度分析**：训练环境与 BFCL-MT 的主题/工具描述更相似或更不相似，结果只有小幅差别，且两者都优于 base。这支持收益并非只是记住了与测试集相近的 API 文案；更合理的解释是模型学到了一部分可迁移的交互模式。但相似度只按文本 embedding 度量，不能完全排除结构层面的训练-测试重合。
- **环境规模分析**：将 SFT 环境数从少量逐步增加到 141 个时，BFCL-MT 和 ACEBench-Agent 分数持续上升，尤其从 0 到 20 个环境的增益最大。这是“环境供给规模有价值”的直接证据，不过曲线并未说明规模会无限线性增益。
- **交互模式分析**：只训练 Non-Conversation 更擅长任务信息完整的 Base/Long-Context；只训练 Conversation 更有利于缺参数时主动追问；两者混合的总体分数最高。它说明训练内容的多样性不只在领域数量，也在 agent 获得信息的方式。

反证也很重要：Tau-Bench 的提升较弱，Qwen3-1.7B 在最难的 Airline 设置上甚至下降；direct RL 时，小模型也明显不如大模型稳定。这说明更多 sandbox 不能替代复杂规则推理、足够的 base ability 或良好的探索初始化。

### 方法定位：它造的是训练世界，不是新优化器

EnvScaler 可归为 `programmatic environment synthesis + state-based verifier + agent post-training`。它不同于让 LLM 在每一步直接扮演环境：这里的状态和转移由 Python 程序执行，LLM 只在离线阶段合成环境和场景；也不同于从既有工具集、轨迹或 API 文档重建单个环境：它从通用任务资源中发现主题，自动生成环境，并用 dual-agent loop 做质量筛选。

最关键的设计不是“自动写出了多少工具”，而是环境与 reward 共用同一套状态模型。任务检查函数读取的是环境真实维护的 `S_final`，因此 reward 判断的是任务有没有完成，而不是调用格式是否像参考答案。Reinforce++、group normalization 和 KL 等训练细节只是在消费这套环境；它们不是论文的主创新。

### 代价与前提：为什么它还不是现实环境的替代品

| 前提或代价 | 对结论的影响 |
|:--|:--|
| 环境规则、状态和任务由 LLM 合成 | dual-agent 测试只能检查有限调用下的自洽，不能保证与真实业务规则一致。 |
| reward 来自终局状态检查 | 支持多条合法路径，但长轨迹中仍有信用分配困难。 |
| 只覆盖领域化、文本型 stateful environment | 不能直接代表 live web、延迟、网络波动、异步副作用和多模态工具。 |
| 合成需要外部模型与质量筛选 | GPT-4.1/mini 组合下每个环境约 `\$1.02`、每个场景约 `\$0.06`；100 轮评估是主要 token 成本。 |

因此，EnvScaler 最适合被理解为“为 agent 造可训、可判、可扩的受控世界”。它显著降低了环境供给的工程门槛，但没有消除 synthetic-to-real gap。

## 研究卡片

- **问题**：现有 agent 训练缺少大规模、可执行、可自动判定的工具环境；真实环境难用，LLM 模拟不稳，手写 sandbox 又无法扩展。
- **做法**：`SkelBuilder` 从任务反推并生成状态化 Python 环境，再用双 agent 筛掉不自洽的环境；`ScenGenerator` 为每个环境生成初始状态、任务和终局 state check functions；SFT 用成功轨迹冷启动，RL 用最终状态 reward 继续优化。
- **证据**：191 个环境、约 7K 场景在三项外部 benchmark 上带来稳定 SFT 增益；环境数增加时性能持续上升；SFT 与 RL 结合对 8B 模型进一步有效。
- **前提**：环境内部规则足够接近目标任务，终局检查函数确实覆盖任务目标，模型也有足够基础能力在环境中探索。
- **风险**：环境可能只是“内部自洽”而不真实；终局 reward 的信用分配有限；对 live web、异步系统和多模态工具的结论仍为空白。
- **教训**：agent post-training 的优化对象不只有 policy 和 trajectory；环境、状态转移与 verifier 的供给能力本身，往往决定 RL 能不能学到可靠行为。
- **遗留**：如何用真实交互数据校准合成环境的业务规则？如何把过程验证、实际工具成本和异步状态纳入 verifier，缩小 synthetic-to-real gap？
