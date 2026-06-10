# 面试核心高频问题

> 目标岗位：大模型算法实习（后训练 / RL / Agent 方向）
> 整理依据：从原始八股 10 章中提取高频考点，按面试实际考察频率分层
> 风格说明：叙事驱动、段落优先、公式保留、量化数据支撑结论

### 文档索引

| 优先级 | 文件 | 主题 |
|--------|------|------|
| P0 | 01-Transformer与Attention.md | Attention 机制、MHA/MQA/GQA、Flash Attention、KV Cache、Q/K/V 必要性、训练推理差异、Normalization 对比 |
| P0 | 02-位置编码与长度外推.md | RoPE 推导、ALiBi、NTK/YaRN 外推 |
| P0 | 03-强化学习与对齐.md | RLHF 三阶段、PPO、DPO、GRPO、Reward Hacking |
| P1 | 04-参数高效微调.md | LoRA/QLoRA/AdaLoRA、PEFT 体系全景、SFT 工程实践、灾难性遗忘 |
| P1 | 05-分布式训练.md | ZeRO、张量并行、流水线并行、3D 并行 |
| P1 | 06-推理优化.md | Prefill/Decode、PagedAttention/vLLM、投机解码、量化 |
| P2 | 07-模型架构与设计选择.md | Decoder-only 原因、Causal vs Prefix LM、Scaling Law、涌现、激活函数、LLaMA 设计、MoE、长上下文 |
| P2 | 08-Agent与RAG.md | Agent 架构、Function Calling、RAG 流程与 RAG vs 微调、幻觉分类与缓解、CoT 深入分析 |

### 使用建议

P0 三篇是任何大模型岗位的基础考点，目标是"能脱口而出公式 + 能讲清设计动机 + 能串联到自己项目"。P1 三篇结合项目经验（verl 框架、QLoRA 训练、2×A100 显存优化）可以形成有深度的回答。P2 两篇了解核心设计理由即可，面试中大概率作为追问出现。

复习路径建议：先通读 P0 建立全局认知，然后用"为什么"串联知识点（为什么除 √dk → 为什么用 RoPE → 为什么 GQA → 为什么 Flash Attention），最后用自己项目中的具体数字（采样组大小 G=8、秩 r=64、KV heads=2）来锚定记忆。
