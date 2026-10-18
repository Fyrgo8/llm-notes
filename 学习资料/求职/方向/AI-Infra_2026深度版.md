# AI Infra 推荐方向（2026 · 深度版）

AI Infra 是与算法赛道、应用开发赛道并列的第三条核心路径。如果算法岗解决的是"模型能做什么"，应用岗解决的是"怎么变成产品"，那 Infra 岗解决的是"怎么让训练跑得起来、推理扛得住"。2026 年的行业现实是：推理侧基础设施的投入正在超越训练侧，专用推理芯片市场预计突破百亿美元，AI Infra 工程师的供需比达到 1:5 以上。大厂 JD 通常标注为"AI 系统工程师""GPU 集群工程师""推理优化工程师"或"ML Platform Engineer"。

## 前置：基础能力要求

以下能力是所有 AI Infra 子方向的公共门槛，不单独设岗但面试必考。

### 编程与基础框架

**Python / PyTorch Internal**：不是"会用 PyTorch 训模型"——Infra 岗要求理解 PyTorch 的内部机制：Autograd 引擎的反向传播调度、Dispatcher 的算子分发路径、张量的内存生命周期（何时分配、何时释放、引用计数 vs 垃圾回收）、CUDA Stream 的异步执行模型、Eager Mode vs Compile Mode（torch.compile / Inductor）的编译执行路径、以及计算图的构建与优化。2026 年的加分项：理解 PyTorch 2.x 的 TorchDynamo + AOTAutograd + Inductor 编译栈的完整链路，能定位编译失败（graph break）的原因。

**C++ / CUDA Kernel 开发**：算子实现（从数学公式到 CUDA kernel 的完整映射）、全局内存 vs 共享内存 vs 寄存器的访问层次、Warp 级并行与线程划分策略、Bank Conflict 的检测与消除、Kernel Launch 的固定开销与 batching 策略、以及 Occupancy 的计算与调优。2026 年的重要补充：Triton 语言正在成为中高层 kernel 开发的主流选择——不要求手写所有 CUDA kernel，但要理解 Triton 到 PTX 的编译过程和性能模型。

**Transformer / Attention 算法**：MHA / GQA / MQA 的计算量和显存占用差异、Softmax 的数值稳定性（log-sum-exp trick）及其对混合精度训练的影响、RoPE / ALiBi / xPos / YaRN 等位置编码的实现细节和长上下文外推能力比较。2026 年新增：Linear Attention / Mamba / RWKV 等非 Attention 架构的推理特性——虽然 Transformer 仍是主流，但 Infra 工程师需要理解这些替代架构的计算模式差异，因为它们的并行策略和缓存策略完全不同。

### 算法题考察重点

AI Infra 的算法题不是纯 LeetCode——面试官会结合系统场景出题：

- **经典数据结构**：链表 / 树遍历 / LRU / LFU——但通常会追问"如果要支持并发访问呢？"，引出并发数据结构的设计
- **并行与并发**：线程 / 进程 / 锁 / 无锁队列 / CAS / Memory Order——这是 Infra 岗的高频考点，面试中常要求手写一个生产者-消费者模型或并发哈希表
- **LLM 特有算法**：Beam Search / Top-K Sampling / Nucleus Sampling 的实现——要求理解数值精度和性能的 trade-off，比如"Top-K Sampling 的 O(n) 实现 vs O(n log k) 实现在 GPU 上的选择"
- **内存管理**：手动内存池的设计、Slab Allocator、Buddy System——和 KV Cache 管理直接相关
- **调度问题**：多 GPU 任务调度、Pipeline 气泡最小化——本质上是图调度问题

## 方向 1：分布式训练系统（Training Infra）

2026 年 HC 最稳定的 Infra 方向。虽然行业的注意力正在向推理侧转移，但万亿参数级模型的训练需求不减反增——DeepSeek V3 的训练据传使用了上万张 GPU，训练系统的复杂度已经到了"不专门做 Infra 就跑不起来"的程度。

**做什么**：设计和实现支撑大规模模型训练的分布式系统——从并行策略选择到通信优化，从显存管理到训练容错，确保几千甚至上万张 GPU 的训练集群能稳定、高效地运行。

### 细分任务

#### 分布式并行策略设计与实现

这是训练 Infra 的核心课题。数据并行（DDP、ZeRO-1/2/3、FSDP）是基础，但 2026 年的大模型训练早已不是单一并行策略能搞定的——需要 3D 甚至 4D 混合并行（DP + TP + PP + SP）的精细编排。具体包括：

- **张量并行（TP）**：Megatron-style 的列/行切分、All-Reduce 通信模式。2026 年的前沿是 Context Parallel（CP）——把序列维度也并行化，配合 Ring Attention 实现超长序列训练，和传统 SP（Sequence Parallel）的区别在于 CP 是跨节点的
- **流水线并行（PP）**：1F1B / Interleaved Schedule / Zero-Bubble Pipeline 的调度策略。气泡率（bubble rate）是关键指标——从 30%+ 优化到 5% 以下的工程细节，包括 micro-batch 数量选择、chunk 划分、以及异步 pipeline 的实现
- **MoE 专项并行**：Expert Parallel（EP）、Expert-Tensor Parallel（ETP）、Expert-Data Parallel（EDP）的选择和组合。MoE 的 All-to-All 通信是训练瓶颈——2026 年的优化手段包括：通信-计算 overlap、动态路由的负载均衡（auxiliary loss 的调参经验）、以及 Top-K 路由 vs Token Choice 路由的性能差异。DeepSeek-V3 的 MoE 并行策略论文是必读材料
- **自动并行**：Alpa-style 的自动并行策略搜索、以及 Megatron-Core 的半自动配置。真实场景中完全自动化还不成熟，但"给定硬件拓扑和模型结构，快速确定最优并行配置"的能力是面试中的高分项

#### 训练数值与显存优化

- **混合精度训练**：AMP / BF16 / FP8 的使用场景和数值稳定性。2026 年 FP8 训练正在从实验走向生产——NVIDIA Hopper/Blackwell 架构原生支持 FP8，但 FP8 的量化范围（E4M3 vs E5M2）选择、loss scaling 策略、以及不同层的精度分配（Attention 用 BF16、FFN 用 FP8 的混合策略）仍需大量工程调优
- **显存优化全家桶**：Gradient Checkpointing（选择性 recompute vs 全量 recompute 的 trade-off）、Activation Offload（GPU ↔ CPU / NVMe 的异步传输）、Parameter Offload（ZeRO-Infinity）。2026 年的新课题：Selective Recomputation——基于 activation 大小和 recompute 成本的自动决策，而不是简单地"每隔 N 层做一次 checkpoint"
- **FlashAttention 及其演进**：FlashAttention-2 / FlashAttention-3 的实现原理（tiling + online softmax + 显存 IO 优化）。2026 年的扩展包括：FlashDecoding（推理时的 KV cache 友好变体）、以及面向 GQA/MQA 的特化实现
- **长序列训练优化**：Ring Attention（跨设备的序列并行注意力计算）、Sequence Parallel（SP，张量并行内的序列维度切分）、LASP（Long-context Attention via Sequence Parallelism）。当训练序列长度达到 128K-1M 时，Attention 的显存和计算成本呈平方增长，这些技术是唯一的可行方案

#### 优化器

- Adam / AdamW 仍是默认选择，但 2026 年有两个值得关注的演进：Muon / MuonClip（基于矩阵秩的二阶优化器，训练效率更高但实现更复杂，QK-Clip 机制用于稳定 Attention 层的训练）和 SOAP（Shampoo 的高效变体，在某些场景下收敛速度优于 Adam）
- 工程层面更重要的是：优化器状态的显存管理——Adam 的 m 和 v 占总显存的 2/3，ZeRO 对优化器状态的分片策略直接决定了能训多大的模型；以及学习率调度的工程实现——Warmup + Cosine Decay 的参数选择、multi-stage 训练中的 LR 重置策略

#### 训练容错与弹性

- 2026 年大规模训练的最大工程挑战不是并行策略，而是故障恢复。上万张 GPU 的集群中，单卡故障率使得平均每几小时就会有一次训练中断。快速故障检测（NaN/Inf 检测、NCCL timeout 检测、心跳监控）、高效 Checkpoint（异步写入、增量存储、分布式 checkpoint）、以及弹性训练（节点退出/加入后自动重新分片、不中断训练地替换故障节点）是训练 Infra 团队最核心的日常工作
- 在线 Checkpoint：2026 年的前沿是 DCP（Distributed Checkpoint Protocol）——支持在不同并行配置之间 reshape checkpoint，使得训练中途可以动态调整并行度

**需求画像**：深度的分布式系统和 CUDA 编程能力，理解 Megatron / DeepSpeed / FSDP 的内部实现（不是调 API），有大规模训练的实战经验（至少几百张卡）。面试核心考察：给你一个具体的模型和硬件配置（比如"70B 模型、1024 张 H100、序列长度 32K"），你怎么选并行策略？显存怎么算？通信瓶颈在哪？气泡率怎么优化？——能当场算出来并给出合理方案的人，面试基本就过了。

**竞争激烈度**：★★★☆☆——需求大但候选人池子稳定，有 Megatron/DeepSpeed 实战经验的人不多，但也不极度稀缺。传统分布式系统（如 Spark、Flink）背景转型的人需要补上 CUDA 和并行策略的知识。

## 方向 2：推理系统与 Serving（Inference / Serving）

2026 年增长最快的 Infra 方向——行业正在从"训练为王"向"推理为王"转变。原因很直接：模型训练一次，但推理跑无数次；推理的成本和延迟直接影响产品体验和商业可行性。

**做什么**：设计和实现高性能、低延迟、高吞吐的 LLM 推理服务系统——从单卡推理优化到集群级 serving 架构，确保模型能以可接受的成本和延迟为海量用户提供服务。

### 细分任务

#### 解码与缓存系统

- **KV Cache 管理**：LLM 推理中 KV Cache 的显存占用通常超过模型参数本身。PagedAttention（vLLM 的核心创新）将 KV Cache 按 page 管理，消除内存碎片。2026 年的前沿是 Radix Tree Cache（基于前缀树的 KV Cache 复用——相同 system prompt 的多个请求共享 KV Cache）和 Disaggregated KV Cache（KV Cache 和计算分离，存储在独立的内存池中，支持跨请求、甚至跨实例的 Cache 共享）
- **Continuous Batching**：不同于传统的 static batching（等一批请求凑齐再处理），Continuous Batching 在每个 decode step 动态插入/移除请求。2026 年的优化方向包括：Chunked Prefill（将长 prompt 的 prefill 拆成多个 chunk 和 decode 请求交错执行，避免长 prefill 阻塞短请求）和 Priority-based Scheduling（根据请求的 SLA 和等待时间做优先级调度）
- **Speculative Decoding（投机解码）**：用小模型（draft model）快速生成多个候选 token，大模型一次性验证。2026 年的演进包括：EAGLE 系列（EAGLE-1/2/3，用额外的轻量头做投机，不需要独立的 draft model）、Medusa（多头并行投机）、以及 Self-Speculative Decoding（大模型的浅层作为 draft、深层作为验证，无需额外模型）。工程难点在于：acceptance rate 和 draft overhead 的平衡、以及和 KV Cache 管理的耦合

#### 量化与部署优化

- **权重量化**：从 INT8 → FP8 → INT4 → NVFP4 的量化全链路。2026 年的工业标准是：权重 INT4/NVFP4（AWQ / GPTQ / SmoothQuant）+ KV Cache FP8 + Activation FP16。混合精度量化（不同层用不同精度）正在成为主流
- **量化感知训练 vs 训后量化**：QAT 精度更高但成本大，PTQ 快速但精度损失不可控。2026 年的中间路线是 Calibration-based PTQ + 少量微调——先用校准数据确定量化参数，再用少量训练数据微调恢复精度
- **KV Cache 量化**：单独量化 KV Cache（通常用 FP8 或 INT8）可以在几乎不影响输出质量的情况下将 KV Cache 显存减半。这在长上下文场景中尤其重要

#### Serving 引擎与 Runtime

- **vLLM**：2026 年 LLM serving 的事实标准。核心特性：PagedAttention、Continuous Batching、Tensor Parallel serving、Speculative Decoding 集成、LoRA 热加载。适合通用的高吞吐 serving 场景
- **SGLang**：vLLM 的竞争者，核心差异在于 RadixAttention（前缀树级别的 KV Cache 复用）和 Structured Generation（结构化输出的约束解码加速）。在多轮对话和 Agent 场景中表现更优
- **TensorRT-LLM**：NVIDIA 官方的高性能推理引擎，深度绑定 NVIDIA 硬件栈（CUDA / TensorRT / FP8/INT4 kernel）。性能天花板最高但灵活性较低，适合部署稳定的生产模型。2026 年新增 TensorRT Edge-LLM 分支，面向嵌入式/车载场景
- **Triton Inference Server**：模型无关的通用推理服务框架——动态批处理、多模型并发、模型版本管理、A/B 测试路由。通常和 TensorRT-LLM / vLLM 配合使用，作为上层的服务编排层
- **Disaggregated Serving 架构**：2026 年的前沿趋势——将 Prefill 和 Decode 分离到不同的 GPU 集群上（Prefill 是计算密集型，Decode 是访存密集型，硬件需求不同）。Prefill 集群用高算力 GPU（H100/B200），Decode 集群用高带宽 GPU 或定制推理芯片，通过高速网络传递 KV Cache。这从根本上改变了推理集群的设计逻辑

#### 高性能推理 Kernel

这是推理 Infra 中最"硬核"的子方向。核心能力是：理解 GPU 的计算模型（Warp、SM、Tensor Core、内存层次），针对 LLM 推理的具体算子（Attention、GEMM、LayerNorm、RMSNorm、SwiGLU）编写高性能 CUDA/Triton kernel。

关键技术点：分块 Tiling 策略（将矩阵运算切成适合 SM 的 tile）、内外层计算拆分（Attention 的 online softmax + 分块累加）、HBM 带宽 vs Tensor Core 计算的平衡（算子是 compute-bound 还是 memory-bound？MFU/MBU 指标怎么算？）、Warp-level Reduction 的实现。

2026 年的重点 kernel 方向：FlashAttention-3（Hopper 架构的 TMA + WGMMA 利用）、Flash-Linear-Attention（针对 Mamba/RWKV 等线性注意力的高效 kernel）、以及 Fused Kernel（将多个小算子融合成一个 kernel 减少 launch 开销和中间结果的显存读写）。

**需求画像**：深度的 CUDA 编程和系统设计能力，理解 LLM 推理的计算特性（prefill 是 compute-bound、decode 是 memory-bound），有推理引擎的源码级理解（至少精读过 vLLM 或 TensorRT-LLM 的核心模块）。面试核心考察：给你一个具体的推理场景（比如"Llama-70B，batch size 32，序列长度 4K，8 卡 H100，要求 TTFT < 200ms、TPS > 30"），你怎么选引擎、怎么配参数、瓶颈在哪、怎么优化？高性能 kernel 岗位还会要求现场手写一个简化版的 Attention kernel 或 GEMM kernel。

**竞争激烈度**：★★★★☆——2026 年增长最快的方向，需求爆发但有推理引擎源码级经验的人极度稀缺。"会用 vLLM 部署模型"的人很多，"能改 vLLM 核心调度器"的人很少——这就是 Infra 岗和使用者的区别。

## 方向 3：GPU 集群与系统底层（System Infrastructure）

这是 AI Infra 中最"接近硬件"的方向——不是优化模型，而是优化支撑模型运行的整个系统栈：网络、存储、调度、监控。

**做什么**：设计和运维支撑 AI 训练/推理的底层基础设施——从 GPU 集群的网络拓扑到存储系统，从资源调度到故障诊断。确保几千张 GPU 的集群能持续、稳定、高效地运行。

### 细分任务

#### 高性能网络与通信

- 大规模分布式训练的通信量巨大——All-Reduce、All-to-All、All-Gather 的数据传输占训练时间的 20%-40%。核心技术栈：RDMA（Remote Direct Memory Access）——绕过 CPU 直接访问远端 GPU 显存；InfiniBand / RoCEv2——高带宽低延迟的网络协议，2026 年主流集群使用 400 Gbps InfiniBand NDR
- **NCCL 调优**：NVIDIA 的集合通信库是分布式训练的核心——Ring / Tree / Recursive Halving-Doubling 的拓扑选择、通信与计算的 overlap（pipeline 策略）、NCCL 环境变量的调参。2026 年的前沿是 MSCCL（Microsoft Synthesized Collective Communication Library）——基于网络拓扑自动生成最优通信算法
- **MoE All-to-All 通信优化**：MoE 模型的 Expert Parallel 引入了 All-to-All 通信——每个 GPU 上的 token 要根据路由结果发送到持有对应 Expert 的 GPU。这是非规则通信模式，优化难度远大于 All-Reduce。2026 年的手段包括：通信量压缩（量化梯度/激活）、动态路由感知的通信调度、以及计算-通信 overlap
- **多级网络拓扑优化**：大集群通常有 NVLink（卡间）→ InfiniBand（机间）→ 以太网（机柜间）的层次结构，不同层级的带宽差异可达 10-100 倍。并行策略的映射需要感知这个拓扑——TP 放在 NVLink 内、PP 放在同机柜的 InfiniBand 上、DP 放在跨机柜的网络上

#### GPU 集群调度

- **资源调度器**：当几百个训练/推理任务共享一个 GPU 集群时，怎么分配资源？需要考虑：网络拓扑亲和性（同一个训练任务的 GPU 应该在网络距离最近的位置）、NUMA 亲和性（GPU 和 CPU/内存的局部性）、存储亲和性（数据在哪个存储节点上）。2026 年的主流方案：Kubernetes + GPU Operator + 自研调度器插件
- **多租户资源隔离**：Docker + K8s + GPU 虚拟化（NVIDIA MIG / MPS / Time-Slicing）。难点在于：GPU 的共享和隔离远比 CPU 复杂——MIG 提供了硬件级隔离但粒度粗、MPS 允许多进程共享 GPU 但没有严格隔离、Time-Slicing 是时分复用但增加切换开销
- **队列调度策略**：优先级调度、抢占机制（高优先级任务可以抢占低优先级任务的 GPU）、公平调度（保证每个团队的 GPU 配额）、以及弹性调度（根据集群负载动态调整任务的 GPU 数量）
- **SLA 保障与故障诊断**：大规模 GPU 集群的故障率很高——GPU ECC 错误、NVLink 链路降级、InfiniBand 丢包、存储节点宕机。自动化故障检测（Xid 错误监控、NCCL 健康检查、训练 loss 异常检测）→ 自动隔离故障节点 → 自动重调度任务，这套流程的建设是系统 Infra 团队最核心的日常工作

#### 存储与 Checkpoint 系统

- **Checkpoint 存储**：大模型的 checkpoint 动辄几百 GB 到 TB 级别。传统的同步 checkpoint（暂停训练 → 写入存储 → 恢复训练）会造成严重的训练中断。2026 年的标准做法：异步 Checkpoint（在后台异步写入，不阻塞训练）、增量 Checkpoint（只保存变化的参数，减少写入量）、分布式 Checkpoint（每个 rank 并行写入各自的分片，配合 DCP 协议支持不同并行配置的 checkpoint 兼容）
- **高性能存储系统**：训练数据和 checkpoint 的 IO 是常见瓶颈。对象存储（S3/COS/OSS）适合冷数据、并行文件系统（Lustre/GPFS/BeeGFS）适合热数据、NVMe-oF（NVMe over Fabrics）用于高性能缓存层。2026 年的趋势是分层存储——根据数据的访问频率自动在不同存储介质间迁移
- **数据 Pipeline**：训练数据的预处理和加载不能成为瓶颈——预取、异步加载、数据格式优化（WebDataset / Mosaic StreamingDataset）、以及断点恢复策略（确保数据 shuffle 的可重现性）

#### 多模态与 Agent Infra（2026 年新增重点）

- 多模态模型的推理 Infra 和纯文本完全不同——图片/视频的预处理（resize、编码）是额外的计算开销；不同模态的 token 数量差异巨大（一张图可能等于几百个文本 token）；视觉编码器和语言模型可能运行在不同的 GPU 上需要跨设备传输
- **Agent Runtime**：Agent 应用的请求模式和传统的单轮问答完全不同——一个 Agent 任务可能涉及多轮 LLM 调用、工具调用（延迟不可预测）、以及长时间的状态维护。Serving 系统需要支持：长连接管理、中间状态持久化、超时与重试机制、以及高并发下的资源隔离
- **Agentic 推理优化**：Agent 场景下，同一个请求的多轮 LLM 调用之间有大量的 prefix 重叠（system prompt + 对话历史），Prefix Caching 的收益巨大。2026 年的前沿是 Prompt Cache Pool——一个集群级别的共享 prefix cache，不同请求甚至不同用户的公共前缀都可以复用

**需求画像**：深度的 Linux 系统和网络知识 + 分布式系统经验 + GPU 硬件理解。这个方向不要求你会训模型或写 CUDA kernel，但要求你理解 GPU 的硬件架构（SM、内存层次、NVLink 拓扑）和网络协议（InfiniBand、RDMA）。面试核心考察：给你一个集群故障场景（比如"训练任务跑到第 3 小时突然 NCCL timeout，日志显示某个 rank 通信超时"），你怎么排查？怎么从 Xid 错误码、网络拓扑、GPU 状态中定位问题？

**竞争激烈度**：★★★☆☆——需求稳定且在增长。传统云计算/DevOps/SRE 背景的人转型做 GPU 集群运维有天然优势，但需要补上 GPU 硬件和分布式训练的知识。数据中心网络工程师如果懂 RDMA 和 InfiniBand，在这个方向上几乎无竞争。

## 方向 4：系统性能分析与优化（Performance Engineering）

这是一个贯穿训练和推理的横向方向——专注于"为什么慢""瓶颈在哪""怎么优化"。不是独立的产品线，但几乎所有 Infra 团队都需要这个能力。

**做什么**：使用专业的 profiling 工具链对 AI 系统的训练/推理性能进行全栈分析——从 GPU kernel 级别的执行效率到系统级别的通信开销，定位瓶颈并驱动优化。

### 细分任务

#### Profiling 工具栈

- **Nsight Systems**：系统层面的全局视图——CPU 和 GPU 的时间线、CUDA Stream 的并发执行、线程切换、NCCL 通信的时间分布、以及 NVTX（NVIDIA Tools Extension）标记的自定义区间。是性能分析的第一站——先用 Nsight Systems 看全局，找到"时间花在哪了"
- **Nsight Compute**：算子层面的精细分析——针对单个 CUDA kernel 的执行效率分析，包括：Occupancy（SM 利用率）、Memory Throughput（显存带宽利用率）、Compute Throughput（Tensor Core 利用率）、Warp Stall 原因分析。是优化具体 kernel 的核心工具
- **PyTorch Profiler**：框架层面的分析——算子级别的时间分布、CPU-GPU 的数据传输、内存分配模式。和 TensorBoard 集成可以可视化训练过程的性能瓶颈
- **2026 年的新工具**：NVIDIA DCGM（Data Center GPU Manager）——集群级别的 GPU 健康监控和性能指标采集；Holistic Trace Analysis（HTA）——Meta 开源的分布式训练性能分析工具，能自动识别通信瓶颈和负载不均衡；以及各大厂自研的 GPU Profiling 平台（字节的 Neutrino、阿里的 PAI Profiler 等）

#### 性能分析方法论

- **从系统到 Kernel 的分析流程**：Nsight Systems（全局视图）→ 识别热点算子 → Nsight Compute（算子分析）→ 确定是 compute-bound 还是 memory-bound → 针对性优化（如果是 memory-bound 则优化访存模式、如果是 compute-bound 则提高 Tensor Core 利用率）
- **分布式训练的性能分析**：除了单卡性能，还需要分析：通信开销占比、计算-通信 overlap 的有效性、Pipeline 气泡率、负载不均衡（不同 rank 的计算时间差异）。工具包括：NCCL 的 NCCL_DEBUG=INFO 日志分析、以及 HTA 的自动化瓶颈检测
- **Roofline Model**：理论性能上界分析——根据硬件的计算能力（TFLOPS）和内存带宽（TB/s），计算每个算子的理论最优性能，和实际性能对比找差距。这是性能优化工程师的基本功
- **MFU（Model FLOPS Utilization）指标**：训练的整体效率指标——实际达到的 FLOPS 占硬件峰值 FLOPS 的比例。2026 年大厂的公开数据显示：大规模训练的 MFU 通常在 40%-55%，头部团队能做到 55%-65%。从 50% 优化到 60% 的那 10 个百分点，往往就是性能工程师的核心产出

#### 端到端性能优化

- 性能分析不只是"找到瓶颈"，还要"解决瓶颈"。典型的优化路径包括：算子融合（减少 kernel launch 开销和中间结果的显存读写）、通信隐藏（计算和通信的 pipeline overlap）、数据加载优化（预取、异步 IO）、内存管理优化（减少碎片、复用 buffer）
- **编译优化**：torch.compile / Inductor 的编译模式可以自动做算子融合和内存优化，但编译成功率和加速比受模型结构影响很大。Performance Engineer 需要理解编译器的工作原理，定位 graph break 的原因，必要时手动干预编译策略

**需求画像**：深度的 GPU 硬件架构理解 + CUDA profiling 经验 + 性能优化方法论。这个方向的面试通常是 case study 形式：给你一个 Nsight Systems 的 profiling 结果截图，让你分析瓶颈在哪、怎么优化。或者更直接地：给你一段 CUDA kernel 代码，让你分析它的 occupancy 为什么低、怎么改。

**竞争激烈度**：★★☆☆☆——方向极度稀缺但候选人更稀缺。能熟练使用 Nsight 全家桶并基于 profiling 结果给出优化方案的人，在整个行业内都是顶级稀缺人才。传统的高性能计算（HPC）背景转型做这个方向非常对口。

## 方向 5：AI 编译器与算子优化（AI Compiler & Operator Optimization）

2026 年独立性越来越强的方向——从"Performance Engineering 的子集"变成了独立的工程方向。原因是：随着模型架构的多样化（Transformer / Mamba / 混合架构）和硬件的碎片化（NVIDIA / AMD / 国产 GPU / ASIC），"针对特定硬件自动生成高性能算子"成为了刚需。

**做什么**：开发和优化 AI 编译器工具链——从高层计算图到底层硬件指令的完整编译流程，以及针对 LLM 核心算子的手动 / 自动优化。

### 细分任务

#### AI 编译器全栈

- **Graph-Level 优化**：计算图的算子融合、常量折叠、死代码消除、Layout 变换（NCHW vs NHWC）。TorchDynamo + Inductor 是 PyTorch 生态的标准方案；XLA 是 JAX/TensorFlow 的方案；TVM/Apache TVM 是跨框架方案
- **Operator-Level 优化**：针对具体算子（GEMM、Conv、Attention）的代码生成和调优。Triton 语言正在成为这个层面的主流工具——它提供了比 CUDA 更高层的抽象，同时保留了对 tile size、memory layout、pipeline 策略的控制
- **Hardware-Specific Backend**：将中间表示编译到具体硬件指令。NVIDIA 有 PTX/SASS，AMD 有 ROCm/HIP，国产 GPU（华为昇腾、寒武纪）有各自的指令集。2026 年的挑战是：怎么用一套编译器栈适配多种硬件——这是 MLIR（Multi-Level Intermediate Representation）试图解决的问题

#### LLM 专项算子优化

- **Attention Kernel**：这是 LLM 最核心的算子。FlashAttention 系列已经是标杆，但针对不同架构（GQA / MQA / Sliding Window Attention / Sparse Attention）仍需定制化实现
- **GEMM 优化**：矩阵乘法是 Transformer 的主要计算开销。CUTLASS / cuBLAS 提供了高度优化的 GEMM kernel，但针对特定形状（LLM 中常见的"瘦长"矩阵）和量化格式（INT4 × FP16 的混合精度 GEMM）仍有优化空间
- **Fused Kernel**：将 RMSNorm + Residual + SwiGLU 等连续的小算子融合成一个 kernel，减少 kernel launch 开销和中间结果的显存读写。这在 Decode 阶段（memory-bound）尤其重要
- **通信 Kernel**：All-Reduce / All-to-All 的定制实现——利用 NVLink / NVSwitch 的拓扑做环/树优化、将通信和计算 overlap

#### 跨硬件适配

2026 年的一个重要趋势是国产 GPU 适配——华为昇腾（Ascend）、寒武纪（Cambricon）、海光（Hygon）等国产 AI 芯片的生态正在建设中。将 NVIDIA CUDA 上训练好的模型适配到这些硬件上运行，需要：算子层面的重新实现、编译器层面的 backend 开发、以及性能调优。这是一个需求巨大但候选人极少的细分方向。

**需求画像**：编译器原理 + GPU 架构 + 高性能计算的交叉背景。面试通常考察：给你一个 GEMM 的规模和硬件参数，你怎么选 tile size？怎么确定是 compute-bound 还是 memory-bound？怎么设计数据 prefetch 策略？编译器岗位还会考察 IR 设计和优化 pass 的编写。

**竞争激烈度**：★★☆☆☆——极度稀缺。会写编译器又懂 GPU 架构的人全行业可能不超过几千人。对口的背景是：编译器方向的 PhD、高性能计算实验室出身、或者芯片公司的 compiler 团队。

## 校招方向建议

优先**方向 2（推理系统）**。理由：2026 年推理侧的投入超过训练侧，HC 增长最快，而且推理系统的技术栈（vLLM / TensorRT-LLM / 量化 / Serving 架构）相对自成体系，入门路径清晰。校招生如果有 CUDA 编程和系统设计的基础，进入推理系统方向后可以快速产出——比如负责一个量化方案的实现和评估、或者负责 Speculative Decoding 的工程集成。这个方向的能力也有很强的可迁移性——推理优化经验可以平移到端侧部署（算法赛道的方向 C）。

然后是**方向 1（训练系统）**。训练 Infra 的门槛更高（需要理解多种并行策略和大规模集群管理），但技术成长空间也更大。如果你的课题涉及分布式系统、并行计算、或者 GPU 编程，训练方向是最对口的选择。进去之后能接触到的集群规模和技术挑战是其他方向比不了的。

偏硬核系统背景的同学，可以看**方向 3（GPU 集群与系统底层）**——如果你有 Linux 内核、网络协议栈、高性能计算的背景，这个方向的竞争非常小。或者**方向 5（AI 编译器）**——如果你做过编译器、LLVM、或者芯片设计相关的课题，这个方向的稀缺度最高。

**方向 4（性能分析）** 通常不单独设校招岗位，而是融入方向 1-3 的日常工作中。但如果你在面试中能展示出对 GPU profiling 工具的熟练使用和性能分析方法论的理解，对任何 Infra 方向都是巨大的加分。

## 社招方向建议

首选**方向 2（推理系统）**——供不应求、薪资溢价最高。2026 年几乎所有大厂都在扩建推理团队，有 vLLM / TensorRT-LLM 源码级经验的人可以轻松拿到好 offer。特别是：如果你在上一家做过 Serving 引擎的核心开发（调度器、KV Cache 管理、量化集成），这种经验直接平移。

**方向 3（系统底层）** 对传统背景的社招特别友好：

- 云计算 / DevOps / SRE 背景 → GPU 集群调度和运维
- 网络工程师（尤其是懂 RDMA / InfiniBand 的）→ 高性能通信
- 存储工程师 → Checkpoint 和数据 Pipeline

这些转型路径都很自然，只需要补上 GPU 和分布式训练的基础知识。

**方向 5（AI 编译器）** 对芯片公司和编译器团队的社招候选人是最佳选择——你的编译器和硬件经验在 AI Infra 的语境下价值翻倍。特别是有国产 GPU 适配经验的人，在 2026 年的国产化大背景下非常抢手。

**方向 1（训练系统）** 适合有大规模分布式系统经验的社招——如果你在上一家做过 Spark/Flink/Ray 的底层开发，转型做训练 Infra 的并行策略和通信优化非常对口。

一个实际建议：Infra 岗的面试比算法岗更重视"能不能干活"——面试官不会问你太多论文原理，而是直接给场景让你分析、设计、甚至手写代码。准备面试时，与其刷八股不如精读一遍 vLLM / Megatron-LM / DeepSpeed 的源码核心模块，在面试中能说出"这个功能在源码的 XXX 模块实现、核心逻辑是 YYY"远比背诵概念有用。

## 面试趋势与准备策略

**信号 1：系统设计题的权重大幅上升。**

2026 年 Infra 岗的面试中，系统设计占比已经超过 40%。典型题目："设计一个支持 1024 卡训练的分布式训练框架""设计一个高并发 LLM Serving 系统""设计一个集群级别的 KV Cache 共享池"。准备策略：对每个方向的核心系统（训练 → Megatron、推理 → vLLM、集群 → K8s + GPU Operator）做一次完整的架构分析，画出组件图、数据流图、和关键接口。

**信号 2：现场编程题偏向 CUDA / 系统编程。**

不是 LeetCode 风格的算法题——而是："手写一个简化版的 PagedAttention""实现一个 Ring-AllReduce""写一个 KV Cache 的 LRU 淘汰策略"。准备策略：确保你能手写基本的 CUDA kernel（至少 vector add、matrix transpose、reduction），理解 GPU 编程的核心概念（grid/block/thread、shared memory、__syncthreads）。

**信号 3：Profiling 分析成为面试标配环节。**

越来越多的面试会给你一段 Nsight 的 profiling 输出，让你分析瓶颈。准备策略：自己动手跑一遍 Nsight Systems 和 Nsight Compute，分析一个真实的 LLM 训练/推理过程，熟悉输出的每个指标含义。

**信号 4：源码理解是区分度最高的考察点。**

面试官问"vLLM 的 Continuous Batching 是怎么实现的"，能回答"核心逻辑在 scheduler.py 的 _schedule 方法中，每个 step 先处理 waiting 队列的 prefill 请求、再处理 running 队列的 decode 请求、通过 block_manager 分配物理 block"的人，和只能回答"就是动态插入请求"的人，面试评分差距巨大。准备策略：选一个你最熟悉的推理 / 训练框架，精读它的核心模块源码（不需要全部读完，但至少要读懂核心的调度 / 通信 / 内存管理模块），面试时能说出具体的代码结构和实现细节。
