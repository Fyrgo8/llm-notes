# LLM Notes

> 山中无历日，寒尽不知年

大模型学习笔记仓库，涵盖八股面试、Agent 工程、RL 训练、算法方向选择等主题。

## 目录

### 面试核心高频问题

从原始八股 10 章中提取的高频考点精华，按面试实际考察频率分为 P0/P1/P2 三层，叙事驱动、公式保留、量化数据支撑结论。

| 优先级 | 文件 | 主题 |
|--------|------|------|
| P0 | [Transformer 与 Attention](面试核心高频问题/01-Transformer与Attention.md) | Attention 机制、MHA/MQA/GQA、Flash Attention、KV Cache、Q/K/V 必要性、训练推理差异、Normalization 对比 |
| P0 | [位置编码与长度外推](面试核心高频问题/02-位置编码与长度外推.md) | RoPE 推导、ALiBi、NTK/YaRN 外推 |
| P0 | [强化学习与对齐](面试核心高频问题/03-强化学习与对齐.md) | RLHF 三阶段、PPO、DPO、GRPO、Reward Hacking |
| P1 | [参数高效微调](面试核心高频问题/04-参数高效微调.md) | LoRA/QLoRA/AdaLoRA、PEFT 体系全景、SFT 工程实践、灾难性遗忘 |
| P1 | [分布式训练](面试核心高频问题/05-分布式训练.md) | ZeRO、张量并行、流水线并行、3D 并行 |
| P1 | [推理优化](面试核心高频问题/06-推理优化.md) | Prefill/Decode、PagedAttention/vLLM、投机解码、量化 |
| P2 | [模型架构与设计选择](面试核心高频问题/07-模型架构与设计选择.md) | Decoder-only 原因、Causal vs Prefix LM、Scaling Law、涌现、激活函数、LLaMA 设计、MoE、长上下文 |
| P2 | [Agent 与 RAG](面试核心高频问题/08-Agent与RAG.md) | Agent 架构、Function Calling、RAG 流程与 RAG vs 微调、幻觉分类与缓解、CoT 深入分析 |

### 大模型算法八股

系统性整理的大模型面试知识体系，覆盖从基础到应用的完整链路（原始详细版）：

- [01.大语言模型基础](大模型算法八股/01.大语言模型基础/) — 语言模型、分词、词向量、深度学习基础
- [02.大语言模型架构](大模型算法八股/02.大语言模型架构/) — Transformer、注意力机制、BERT、LLaMA、MoE
- [03.训练数据集](大模型算法八股/03.训练数据集/) — 数据格式、模型参数
- [04.分布式训练](大模型算法八股/04.分布式训练/) — 数据并行、流水线并行、张量并行、DeepSpeed
- [05.有监督微调](大模型算法八股/05.有监督微调/) — LoRA、Adapter、Prompt Tuning、微调实战
- [06.推理](大模型算法八股/06.推理/) — vLLM、TensorRT-LLM、推理优化、量化
- [07.强化学习](大模型算法八股/07.强化学习/) — 策略梯度、PPO、RLHF、DPO
- [08.检索增强RAG](大模型算法八股/08.检索增强rag/) — RAG 技术、Agent
- [09.大语言模型评估](大模型算法八股/09.大语言模型评估/) — 评测体系、幻觉问题
- [10.大语言模型应用](大模型算法八股/10.大语言模型应用/) — 思维链、LangChain

### 论文翻译

- [Agent Harness 工程：综述（中文全文翻译）](agent-harness-engineering-survey-cn.md) — 原文 *"Agent Harness Engineering: A Survey"*（2026, 71页），提出 ETCLOVG 七层分类法，系统梳理 Agent 执行基础设施工程。

### 学习笔记

- [MiniResearcher：低资源多轮搜索 Agent 的 RL 训练全链路](MiniResearcher_学习笔记.md) — 单卡 A100-80GB + Qwen2.5-3B + LoRA，复现并改进 DeepResearcher 训练范式。
- [MiniMind 笔记](MiniMind_笔记.md)

### 大模型算法方向选择

- [AI Infra 2026 深度版](大模型算法方向选择/AI_Infra_2026深度版.md)
- [AI 可入方向](大模型算法方向选择/AI可入方向.md)
- [大模型应用算法落地方法论](大模型算法方向选择/大模型应用算法落地方法论.md)
