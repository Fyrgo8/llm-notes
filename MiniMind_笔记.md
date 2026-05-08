# MiniMind 全链路：训练 → 推理

64M 参数 | 纯 PyTorch | 结构对齐 Qwen3 | 单卡 3090 可跑完

---

## 模型结构（model/model_minimind.py）

```
token IDs → embed_tokens → 8×Block → RMSNorm → lm_head（与 embed 共享权重）→ 词表概率
```

每层 Block：`RMSNorm → GQA Attention → +残差 → RMSNorm → SwiGLU/MoE → +残差`（Pre-Norm）

| 组件 | 做了什么 | 代码要点 |
|------|---------|---------|
| RMSNorm | 只算均方根归一化 `x/sqrt(mean(x²)+eps)*weight`，比 LayerNorm 省掉减均值步骤，速度更快效果相当 | 先转 FP32 保数值稳定，算完转回 FP16 省显存；weight 是逐维度可学习缩放 |
| RoPE | 在 Attention 内部对 Q/K 按维度分组旋转，每组频率不同——高频管近距离、低频管远距离，位置差越大点积自然衰减 | `rope_theta` 越大低频转越慢、能编码越远位置（主流 1e5~1e6）；YaRN 推理时只缩放低频实现长度外推，高频不动 |
| GQA | Q 8头 KV 4头，每 2 个 Q 共享 1 组 KV。注意力头学到的模式高度冗余，KV 减半几乎不掉效果 | `repeat_kv` 把 4→8 对齐；省的是推理时 KV Cache 显存（生产环境最大瓶颈） |
| SwiGLU | 门控前馈：`SiLU(gate_proj(x)) × up_proj(x)` 做信息筛选，`down_proj` 降回 hidden_size。比传统 ReLU FFN 多 50% 参数但效果明显更好 | 乘法门控（FFN 内部决定让什么通过） ≠ 残差加法保底（层间防梯度消失），功能完全不同 |
| MoE | 把 FFN 换成 N 个并行专家，路由网络（Linear+Softmax）给每个 token 选 top-k 个专家加权求和 | 辅助损失防专家坍塌（所有 token 只走一个专家）；topk 不可导所以用 softmax scores 传梯度 |
| generate | 自回归逐 token 生成，推理时只传最后一个 token + KV Cache 不重算历史，复杂度从 O(n²) 降到 O(n) | 采样链：温度缩放 → 重复惩罚 → top-k（只留前 k 个）→ top-p（累积概率截断）→ 多项式采样 |

---

## 训练链路

### 预训练（train_pretrain.py）

让模型学语言。纯文本 next-token prediction，全 token 参与 loss。

| 工程技巧 | 做了什么 |
|---------|---------|
| AMP 混合精度 | 前向反向用 FP16 省一半显存提速 1.5x，权重更新保持 FP32 避免精度损失 |
| 梯度累积 | N 步内每步算 loss 反向传播（梯度自动累加到 .grad），第 N 步才 optimizer.step()。数学上等效 N 倍 batch_size，拿时间换显存 |
| 梯度裁剪 | `clip_grad_norm_` 把梯度范数限制在阈值内，防止单步更新过大导致 loss 飞掉 |
| 余弦 LR | 从 warmup 峰值到接近 0 的平滑余弦衰减，比 step decay 更稳定，末期能精细调整参数 |
| 断点续训 | 先写临时文件再 rename（原子操作），中途断电不损坏 checkpoint；恢复时加载 epoch/step/optimizer 状态 |
| DDP | 多卡数据并行，每卡各算一份梯度再 all-reduce 取平均，等效增大 batch |
| torch.compile | 把动态 Python 循环编译成静态计算图，减少解释器开销，提速 10~30% |

### SFT（train_full_sft.py）

让模型学对话格式。数据是 `conversations` JSONL。

核心设计：labels 用 -100 mask 掉 user/system 部分，只对 assistant 回复算 loss——模型不学"背诵用户说了什么"，只学"如何回答"。`pre_processing_chat` 20% 概率随机插不同风格 system prompt，训练时的随机性 = 推理时的鲁棒性。

### LoRA（model_lora.py + train_lora.py）

冻结基座，旁路插低秩矩阵：`y = Wx + BAx`。1024×1024 矩阵（100 万参数）用 rank=8 只需训 A[8×1024]+B[1024×8]=1.6 万参数（1.6%）。

流程：加载基座 → 注入 LoRA 层 → `requires_grad=False` 冻结基座（autograd 不算梯度、optimizer 不更新）→ 只训 LoRA 参数 → 保存 LoRA 权重（几 MB）或合并回基座。

### 蒸馏（train_distillation.py）

学生用 KL 散度拟合教师 soft label 分布。暗知识 = 教师输出分布的整体形状——"猫"和"虎"都不是正确答案但教师给"猫"概率远高于"虎"，这个比例关系就是暗知识。比只学 hard label（one-hot）信息量大得多。

### DPO（train_dpo.py）

离线偏好对齐。数据：同一 prompt 的 chosen/rejected 对。

Loss = `-log σ(β × (策略偏好差 - 参考偏好差))`。直觉：让策略相对于参考模型更偏好 chosen。β 控制偏离惩罚力度。不需要 Reward 模型、不需要在线生成，准备好数据对就能跑，稳定省资源。

### GRPO（train_grpo.py）

在线 RL，用组内排名代替 Critic。同一 prompt 生成 G 条回复，奖励做 Z-score 归一化得优势值。去掉了 Critic 整个模块。

| 概念 | 含义 |
|------|------|
| ratio | `exp(new_logp - old_logp)` = 新策略概率 / 旧策略概率，衡量策略变了多少 |
| 裁剪 | 限制 ratio 到 [1-ε, 1+ε]，是限制单步更新幅度防策略突变（trust region），不是"不更新" |
| old_logps | 生成时记录的每个 token 对数概率快照，训练时拿新策略重新算 new_logps 做对比 |
| KL 惩罚 | 防 reward hacking（模型找到奖励模型 bug 刷高分但回答变差），不是防固化 |
| 奖励模型 | Transformer 主体 + reward_head（线性层→标量），用偏好对数据训练。实际打分 = 规则奖励 + 模型评分 |

### PPO（train_ppo.py）

4 模型在线 RL：Actor（策略，更新）+ Critic（价值网络，更新）+ Ref（参考，冻结）+ Reward（打分，冻结）。用 GAE 做优势估计，支持多轮更新 + early stop KL。工程复杂度高（4 模型同步、Critic 不稳定、超参敏感），正在被 GRPO/DPO 替代。

### Agent RL（train_agent.py）

在 GRPO 基础上加多轮 Rollout：生成 → 解析工具调用（`ゅ`...`゜` 标记间的 JSON）→ 执行工具 → 结果注入上下文 → 继续生成，最多 3 轮。`max_turns` 是 Rollout 终止条件，不是 loss 后的约束。

奖励分两路：无工具调用走长度/格式/RM/重复惩罚；有工具调用走标签正确性 + 工具名对齐 + **GT 验证**（最终答案是否含正确答案，核心信号）+ 未完成扣分。clip [-3,3]。

### Rollout 引擎（rollout_engine.py）

| 引擎 | 实现 | 场景 |
|------|------|------|
| Torch | 原生 model.generate()，简单直接 | 调试、小规模训练 |
| SGLang | HTTP API 调远程高性能推理服务 | 大批量训练 |

统一返回 RolloutResult（output_ids / completion_ids / per_token_logps / completions / prompt_lens / completion_mask）。"可插拔"= 推理后端可换，不是能拼任意模型。

---

## 推理链路

### 模型转换（convert_model.py）

| 方向 | 用途 |
|------|------|
| .pth → Qwen3 格式 | 接入 vLLM/ollama/llama.cpp（推荐，生态最好） |
| .pth → MiniMind 格式 | 保留原生结构，AutoModel.from_pretrained 加载 |
| Transformers → .pth | 反向回训练格式 |
| 合并 LoRA | LoRA 权重加回基座导出完整模型，部署时不用额外加载 adapter |

### API 部署（serve_openai_api.py）

FastAPI 实现 `/v1/chat/completions`（兼容 OpenAI 协议）。支持流式/非流式、`reasoning_content`（思考链）、`tool_calls`（工具调用）、`open_thinking`（自适应是否思考）。FP16 + KV Cache + 流式生成。可直接接 FastGPT / Open-WebUI。

---

## 全链路流程图

```
BPE 词表 → 预训练(学语言) → SFT(学对话) →┬─ LoRA / 蒸馏(适配/压缩)
                                           ├─ DPO(离线对齐)
                                           ├─ GRPO / PPO(在线对齐)
                                           └─ Agent RL(工具调用)
                                                    ↓
                                           格式转换 → API 部署
```

各对齐/适配阶段互不依赖，都从 SFT 后出发，实践中选 1~2 种组合。

---

## 核心认知

| 认知 | 展开 |
|------|------|
| 数据 > 算法 | 结构和算法是天花板，数据质量决定能触到几成。所有阶段都适用 |
| 训练 vs 推理 | 训练有 labels 反向传播更新参数；推理无 labels 自回归生成，KV Cache 避免每步重算历史 |
| 对齐的必要性 | 预训练+SFT 会说话但可能有害/啰嗦/幻觉，对齐让输出符合人类偏好（有用、无害、诚实） |
| 残差连接 | F(x)+x 提供梯度直通道绕过连乘衰减；每层只学修正量，最坏情况等于浅层网络不会更差 |
| 工业趋势 | PPO→GRPO/DPO 轻量化；MHA→GQA/MLA 省 KV Cache；Dense→MoE 少激活参数达到大模型效果 |
