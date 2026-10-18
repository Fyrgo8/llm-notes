# 推理 / 训练 / RL Infra 学习资料

## 总入口

- [zhaochenyang20/Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)：首推，中文资料很多。覆盖 SGLang、verl、slime、FSDP、NCCL、权重同步、训练/推理不一致、RL rollout，已经接近一套 RL Infra 教材。
- [gpu-mode/lectures](https://github.com/gpu-mode/lectures)：GPU/CUDA/AI Infra 基础课。重点看 1/4/8/12/14/17/22/35/39：profiling、内存架构、FlashAttention、Triton、NCCL、vLLM speculative decoding、SGLang、TorchTitan。
- [romitjain/awesome-llm-systems](https://github.com/romitjain/awesome-llm-systems)：论文、博客和系统资料索引，覆盖推理算术、KV Cache、continuous batching、分布式训练、kernel。
- [Stanford CS336 Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems)：动手实现 Transformer 优化和分布式训练，适合补齐系统基础。

## 推理 Infra

### 推荐顺序

1. [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm)：课程化最好，使用 Apple MLX，从 Attention、RoPE 开始，逐步实现 KV Cache、continuous batching、chunked prefill、paged attention。没有 NVIDIA GPU也能学。
2. [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)：约 1200 行 Python，实现调度、KV Cache、prefix caching、tensor parallel、CUDA Graph。适合第一次读推理引擎源码。
3. [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)：约 5000 行，包含 Radix Cache、chunked prefill、overlap scheduling、TP 和 FlashInfer，是从教学实现进入生产框架的最佳桥梁。
4. [sgl-project/sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials)：SGLang 官方 slides 和讲解。
5. [vllm-project/vllm](https://github.com/vllm-project/vllm) / [sgl-project/sglang](https://github.com/sgl-project/sglang)：生产框架，重点读 scheduler、KV cache manager、model runner、worker、distributed executor，不建议从入口文件顺序通读。
6. [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) / [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)：进一步学习 CUDA/CUTLASS kernel、量化、speculative decoding、跨卡推理。

### 学习重点

- 理解请求从进入服务到生成结束的完整生命周期，以及 prefill 和 decode 的计算特征。
- 掌握 KV Cache、PagedAttention、prefix cache、continuous batching 和 chunked prefill。
- 理解 scheduler 如何在吞吐、TTFT、ITL/TPOT、公平性和显存利用率之间取舍。
- 掌握量化、FlashAttention、CUDA Graph、speculative decoding 等核心优化。
- 理解 TP、PP、EP、PD 分离和多节点 serving 的通信与部署方式。
- 能通过 benchmark 和 profiler 定位 CPU 调度、GPU kernel、显存及通信瓶颈。

## 训练 Infra

### 推荐顺序

1. [huggingface/picotron](https://github.com/huggingface/picotron)：首推教学仓库。实现 DP、TP、PP、CP 四维并行，核心文件基本都在 300 行以内，并有配套视频和从零实现仓库。
2. [huggingface/nanotron](https://github.com/huggingface/nanotron)：适合系统学习 3D parallelism、1F1B、ZeRO-1、分布式 checkpoint；配套 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 很强。
3. [pytorch/torchtitan](https://github.com/pytorch/torchtitan)：PyTorch 原生训练栈，代码相对干净，覆盖 FSDP2、TP、PP、CP、activation checkpoint、异步 checkpoint、Float8、profiling。
4. [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)：大规模训练事实标准之一，适合研究 TP/PP/EP、MoE、并行拓扑和 pipeline schedule，但不适合作为第一份源码。
5. [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)：重点理解 ZeRO-1/2/3、offload、通信与显存权衡。
6. [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)：训练 Infra 必学工具，用来测 AllReduce、AllGather、ReduceScatter 带宽以及定位拓扑问题。

## RL Infra / LLM 后训练

1. [Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero)：最小化复现 R1-Zero，适合理解 `prompt → rollout → reward → advantage → policy update`。
2. [huggingface/trl](https://github.com/huggingface/trl)：快速跑通 SFT、DPO、GRPO；API 友好，但隐藏了较多调度和分布式细节。
3. [verl-project/verl](https://github.com/verl-project/verl)：RL Infra 主线首选。HybridFlow 将训练、rollout、reference、reward 数据流显式化，支持 FSDP/Megatron 和 vLLM/SGLang。重点读 PPO architecture、Ray trainer、worker placement、weight update。
4. [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)：Ray + vLLM + DeepSpeed 架构，适合研究 actor/reward/reference/critic 的放置、colocation、异步 rollout 和资源调度。
5. [THUDM/slime](https://github.com/THUDM/slime)：更明确的 Megatron + SGLang 路线，训练、rollout、Data Buffer、权重同步的数据流清晰，适合大规模 RL 和 agentic rollout。
6. [areal-project/AReaL](https://github.com/areal-project/AReaL)：重点是 fully asynchronous RL、off-policy 程度控制和长尾 rollout。
7. [RLinf/RLinf](https://github.com/RLinf/RLinf)：偏具身智能、VLA 和 Agentic AI；如果目标只是语言模型 RL，不建议最先读。
8. [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)：适合 NVIDIA/Megatron 技术栈和多节点生产训练。

