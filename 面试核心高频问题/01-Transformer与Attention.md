# Transformer 与 Attention

> 优先级：P0（必背）
> 覆盖范围：Scaled Dot-Product Attention、Multi-Head Attention、MQA/GQA、Flash Attention、KV Cache
> 关联项目：MiniResearcher（Qwen2.5-3B，GQA 推理加速）

### 1. 为什么 Attention 要除以 √dk

设 Q、K 中每个元素独立同分布、均值为 0 方差为 1。点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 是 $d_k$ 个独立随机变量的求和，由方差线性叠加可知其方差为 $d_k$。当 $d_k=128$ 时，点积值的标准差约为 11.3，大致落在 $[-3\sqrt{d_k}, 3\sqrt{d_k}]$ 即 $[-34, 34]$ 的范围内。

softmax 在输入绝对值较大时进入饱和区——输出趋向 one-hot，梯度趋近于 0。具体来说，$y = \text{softmax}(x)$ 的 Jacobian 为 $\text{diag}(y) - yy^T$，当 $y$ 接近 one-hot 时，该矩阵的特征值接近 0，反向传播的梯度信号消失。

除以 $\sqrt{d_k}$ 的效果是将点积方差归一化到 1，对应值域约在 $[-3, 3]$，此时 softmax 处于梯度敏感区，训练稳定收敛。T5 采用了另一种等价做法：不除 $\sqrt{d_k}$，而是将 Q/K 投影层的初始化方差额外除以 $d_k$，同样使点积初始方差为 1。

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 2. Self-Attention 的计算复杂度

对于序列长度 $n$、隐层维度 $d$ 的输入，$QK^T$ 矩阵乘法的时间复杂度为 $O(n^2 d)$，attention 矩阵本身占用 $O(n^2)$ 空间。这意味着序列长度翻倍，计算量翻四倍——这是长上下文模型的核心瓶颈，也是 Flash Attention、稀疏 Attention 等工作的出发点。

### 3. Multi-Head Attention 的设计动机

将 Q/K/V 投影到 $h$ 个不同子空间分别做 attention，再拼接投影回原维度。每个 head 的维度为 $d_k = d_{model}/h$，因此总计算量与单头 attention 相当，但模型获得了同时关注不同位置、不同语义模式的能力。可以类比 CNN 中多通道的概念——不同 head 学到的 attention pattern 确实差异显著（有的关注语法，有的关注语义相似性，有的关注位置邻近）。

降维还有一个工程上的好处：每个 head 的 $d_k$ 较小，attention 矩阵的数值尺度更可控。

### 4. MQA 与 GQA

Multi-Query Attention（MQA）将所有 head 共享同一组 K/V，只有 Q 保留多头。这将 KV Cache 减少到原来的 $1/h$，推理时显存压力大幅下降。代价是质量有损——共享 KV 相当于压缩了模型的表示容量。

Grouped-Query Attention（GQA）是折中方案：将 $h$ 个 Q head 分成 $g$ 组，每组共享一组 KV。当 $g=1$ 退化为 MQA，$g=h$ 退化为 MHA。LLaMA-2 70B 和 Qwen2.5 系列均采用 GQA（例如 Qwen2.5-3B 使用 2 个 KV head），在推理速度和生成质量之间取得了良好平衡。

核心动机始终是：自回归推理时 KV Cache 是显存瓶颈，减少 KV head 数直接减少缓存量和内存带宽占用。

### 5. Flash Attention

标准 attention 实现需要将 $n \times n$ 的 attention 矩阵从 SRAM 写入 HBM 再读回，这一过程受 IO 瓶颈限制——A100 的 HBM 带宽约 2TB/s，而 SRAM 带宽约 19TB/s，差距约 10 倍。

Flash Attention 通过三个关键设计解决这一问题：

**分块计算（Tiling）**：将 Q/K/V 按 block 加载到 SRAM，在片上完成 attention 计算，避免将完整 attention 矩阵写入 HBM。

**Online Softmax**：传统 softmax 需要看完整行才能计算归一化常数。Flash Attention 维护 running max 和 running sum，逐块更新，数学上精确等价于标准 softmax（非近似）。

**反向传播重计算**：前向不保存 attention 矩阵，反向时从 Q/K/V 和输出重新计算梯度所需的中间值。用计算换显存。

最终效果：内存从 $O(n^2)$ 降到 $O(n)$，wallclock 加速 2-4 倍，且输出与标准 attention 在数值上完全一致。Flash Attention 2 进一步优化了 warp 间的工作分配，减少了非矩阵乘法操作的比例。

### 6. KV Cache 机制

自回归解码时，第 $t$ 步生成的 token 只需要与前 $t-1$ 个 token 做 attention。由于前面 token 的 K/V 在每一步都不变，将其缓存下来可以避免 $O(t)$ 的重复计算。

以 Qwen2.5-7B（32 层，8 KV heads，head_dim=128，fp16）为例，序列长度 2048 时单条请求的 KV Cache 为 $2 \times 32 \times 8 \times 128 \times 2048 \times 2 \approx 256\text{MB}$。并发 16 条请求就需要 4GB 纯 KV Cache 开销。这也是 GQA 和量化 KV Cache（int8/int4）的直接驱动力。

### 7. Transformer 中的残差与归一化

**残差连接**为深层网络提供了梯度"高速公路"：即使某一层的变换退化，梯度仍可通过恒等映射直达浅层。对于 100+ 层的 LLM 而言，这是训练收敛的必要条件。

**LayerNorm** 稳定每层输入的分布。现代 LLM 普遍采用 Pre-Norm 结构（先 Norm 后 Attention/FFN），相比 Post-Norm 训练更稳定，不容易梯度爆炸。LLaMA/Qwen 系列进一步用 RMSNorm 替代 LayerNorm——省去减均值步骤，只除均方根值，计算更快，效果相当：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}} \cdot \gamma
$$

### 8. Cross-Attention 与 Self-Attention

Self-Attention 的 Q/K/V 来自同一序列，建模序列内部各位置的依赖关系。Cross-Attention 的 Q 来自 Decoder 当前表示，K/V 来自 Encoder 输出，实现跨序列的信息对齐——典型应用是机器翻译中将源语言信息注入目标语言解码。

在 Decoder-only 架构（GPT/LLaMA/Qwen）中不存在 Cross-Attention，所有交互通过因果 Self-Attention 在统一序列内完成。

### 9. Q/K/V 三组投影的必要性

一个常见面试追问是"为什么不能用同一组向量既做 query 又做 key"。本质原因是 attention 的"匹配"和"被读取"是两种不同的语义功能。Q 代表"我在找什么"，K 代表"我能被谁找到"，V 代表"找到我后应该提取什么信息"——三者分离允许模型在每个子空间上独立学到最适合的映射。

如果令 $Q = K$（即去掉 K 的独立投影），attention 矩阵变为 $XX^T$ 形式（对称），每个位置与自身的匹配分数最高，输出退化为近似恒等映射，丧失了建模非对称关系的能力。实验验证：shared Q/K 模型在机器翻译上 BLEU 下降约 2 个点。V 的独立性则保证了"如何匹配"和"匹配后传递什么"的解耦——在某些 attention head 中，高 attention 权重位置传递的是该位置的语法角色信息而非语义内容，这正是 V 独立投影的价值。

### 10. Transformer 训练与推理的输入差异

训练阶段，Decoder 的输入是完整目标序列的右移版本（teacher forcing）：将 ground truth 的 $[t_0, t_1, ..., t_{n-1}]$ 整体输入，通过因果 mask 让每个位置只看到左边的 token，一次前向即可计算所有位置的 loss。这意味着训练是高度并行的——序列中 $n$ 个位置的预测在一次 forward 中同步完成。

推理阶段则是纯自回归：每步只输入当前 token（或利用 KV Cache 将全部历史压缩到缓存中），前向计算得到下一个 token 的概率分布，采样后追加到序列，循环直到生成结束符。这导致推理无法在序列维度并行化——是"Prefill 快、Decode 慢"的根本原因。

还有一个细节区分：推理首步（Prefill）处理整个 prompt，计算量与训练类似（全部 token 并行过 attention），之后每步（Decode）只处理一个 token。这就是 vLLM 等推理框架区分 prefill batch 和 decode batch 的原因——二者的计算特征截然不同（前者 compute-bound，后者 memory-bound）。

### 11. Normalization 对比

Normalization 的核心目标是稳定各层输入的数值分布，防止训练过程中内部协变量偏移。不同方案在"归一化维度"和"计算复杂度"上做取舍：

**BatchNorm** 沿 batch 维度统计均值和方差，适合 CV 中的固定尺寸输入。但在 NLP 中，batch 内序列长度不一致且 batch 通常较小，BN 的统计量噪声大，效果不稳定。推理时依赖全局 running statistics 也引入了训练-推理不一致的问题。

**LayerNorm** 沿特征维度（隐层 $d$）对每个 token 独立归一化，不依赖 batch 维度统计，天然兼容变长序列。公式为 $\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$，其中 $\mu, \sigma$ 是当前 token 的 $d$ 维向量的均值和标准差。这是 GPT-2/3 和 BERT 使用的方案。

**RMSNorm** 省去了减均值步骤，直接用均方根值归一化：$\text{RMSNorm}(x) = x / \text{RMS}(x) \cdot \gamma$。Zhang & Sennrich (2019) 的实验表明"去中心化"对 Transformer 训练并不关键，省掉它反而减少了约 7-10% 的归一化计算时间。LLaMA、Qwen、Gemma 全系列均采用 RMSNorm。

**pRMSNorm** 是进一步的简化——对隐层的前 $p\%$ 维度计算 RMS 值，再用该值归一化全部维度。Apple 的研究表明取 $p=25\%$ 时效果接近完整 RMSNorm，但推理速度进一步提升。适合部署到边缘设备。

选型总结：对 LLM 训练，Pre-RMSNorm 是当前标准配置，兼顾了稳定性和计算效率。
