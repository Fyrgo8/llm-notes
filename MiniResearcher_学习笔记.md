# MiniResearcher：低资源多轮搜索 Agent 的 RL 训练全链路

> 时间跨度：2025-05-23 ~ 2025-05-30
> 硬件：1×A100-80GB（AutoDL & BlockElite，从 2×A800-40GB 迁移而来）
> 模型：Qwen2.5-3B-Instruct + LoRA (rank=64)
> 框架：verl 0.2.0.dev + Ray + vLLM + FSDP
> 原始论文：DeepResearcher (GAIR, arXiv:2504.03160)

---

## 一、项目定位与核心命题

DeepResearcher 原论文用 Qwen2.5-7B 全参 + GRPO + 真实 Google 搜索，在端到端 RL 中训练出了一个会主动搜索、交叉验证、自我反思的深度研究 Agent。结果很亮——比 prompt engineering 基线高 28.9 分，比 RAG-RL 高 7.2 分。但这套方案对硬件和搜索环境的依赖极重：7B 全参训练需要大量 A100-80G，真实搜索 API 在国内受限且成本高。

MiniResearcher 的命题是：能不能在单卡 A100-80G 的预算约束下，用 3B 模型 + LoRA 复现并改进这套训练范式？这不只是"缩小版复现"——资源约束倒逼了一系列工程和算法层面的创新：更精巧的优势估计器选择、理论保证的过程奖励注入、防止小模型策略坍缩的课程学习设计、以及一套诊断框架来区分"模型真学会了"和"模型在骗分"。

整个实验体系围绕一个核心问题的五个子问题展开：

```
"如何用 RL 训练一个多轮搜索 Agent"

  ┌─ 算法层：用什么优势估计器？                     → Exp-01
  ├─ 冷启动层：模型不愿意搜索怎么办？               → Exp-02
  ├─ 奖励层：搜索中间过程无信号怎么办？             → Exp-03
  ├─ 效率层：显存不够怎么办？                       → Exp-04
  └─ 诊断层：怎么知道模型是真学会了？               → Exp-05
```

每个子问题的结论 feed 到后续实验：Exp-01 确定了 advantage clipping 策略供 Exp-03b 使用，Exp-02 确定了 entropy/curriculum 超参融入最终方案，Exp-03a 证明了问题在环境不在算法，Exp-05 的监控帮助 Exp-03b 精确定位了 score 不涨的真正原因。

---

## 二、训练循环：一步到底发生了什么

在进入具体实验之前，先建立对整个训练过程的直觉。多轮 Agent RL 和传统 RLHF 的核心差异在于"轨迹内有环境交互"——模型不是生成一次就结束，而是在一个带搜索引擎的环境中执行多步动作。这导致了三个新问题：信用分配更难、mode collapse 更容易、以及需要额外的行为诊断手段。传统 RLHF 只做一次生成就结束，不存在这些问题。

整个训练是一个标准的 on-policy 循环，但每一步都比普通 RLHF 复杂得多：

**第一步：多轮 Rollout。** 从一个 batch 的问题出发，每个问题采样 N 次（GRPO 的 group_size）。对每个问题，vLLM 驱动一个多轮生成循环——模型看到问题后决定是搜索还是直接回答。如果选择搜索，query 被发送给外部 Handler 进程（查离线缓存或实时 API），搜索结果追加到对话历史后模型再次生成。这个循环最多 max_turns 轮，由 curriculum scheduler 动态控制。

**第二步：计算 Log Probability。** Actor 和 Ref 模型对完整轨迹做 teacher-forcing forward pass。vLLM 生成时的 log_prob 在多轮拼接后对不上，必须重新算一遍干净的。两者的差就是逐 token 的 KL divergence 估计。

**第三步：计算 Reward。** 比较最终 answer 与 GT 的 token-level F1，得到 [0, 1] 的终局奖励。如果开启 PBRS，还会在每轮搜索结束位置注入势函数差分的中间奖励。额外信号包括 early-stop penalty、KL penalty、entropy bonus。

**第四步：估计 Advantage。** 对同一问题的 N 条轨迹做 group 内归一化——GRPO 的做法是 (r_i - mean) / std。advantage clipping 在 [-5, 5] 截断防止异常样本。

**第五步：策略更新。** 用 PPO clipped surrogate objective 更新 LoRA 的 A/B 矩阵。Agent 场景通常只做 1 个 epoch，因为轨迹太长、on-policy 偏移更敏感。

---

## 三、Exp-01：优势估计器的选择不是理论问题，是工程问题

### 3.1 问题的来源

GRPO 对 group 内 N 个采样的 reward 做 (r_i - mean) / std 归一化计算 advantage。多轮 Agent 场景下存在一个严重的退化模式：当 group 内所有轨迹的 reward 几乎相同时（高频出现——比如所有轨迹都格式错误得 -1，或者所有轨迹都只搜索了一轮就放弃），std → 0 导致 advantage 爆炸或归零。全员 -1 时 GRPO 归一化后 advantage 全为 0，策略梯度为零，模型完全学不动。

这个问题在单轮 QA 任务中不突出（reward 天然有区分度），但在多轮 Agent 场景中是系统性的——搜索任务的成功依赖多步正确执行，早期大量轨迹落入相同的失败模式。

### 3.2 Dr.GRPO 的设计与它的致命缺陷

Dr.GRPO（Drop-mean GRPO）的修改极为简洁：只除 std，不减 mean。数学上 advantage_i = r_i / std(r)。理论上当全员 reward 为 -1 时 advantage 统一为负值，梯度推动模型远离当前策略；当全员为 0.8 时 advantage 统一为正，强化当前策略。保留了绝对 reward 信号的方向性。

但实验暴露了一个论文里没讨论过的致命问题：当 reward 恒为负时（Agent 训练早期常态），Dr.GRPO 的 advantage 永远为负，模型只有"远离当前策略"的信号没有"靠近好策略"的信号。GRPO 的归一化反而保证了 group 内总有一个最好的样本得到正 advantage，梯度方向更明确。

### 3.3 实验数据

在 A100-80GB 单卡上，关掉搜索环境（do_search=false），让三种估计器各跑 300 步：

| 指标 | GRPO | DrGRPO | RLOO |
|------|------|--------|------|
| pg_loss 范围 | 0.14 ~ 0.52 | 2.2 ~ 14.7 | 1.3 ~ 43.4 |
| grad_norm 范围 | 1.68 ~ 4.19 | 3.5 ~ 40.5 | 45.3 ~ 226.0 |
| advantage scale | ~0.25 | ~4 | ~16 |
| 梯度爆炸频次 | 0/300 | 偶发 | 299/300 |
| reward 改善 | +11.5 | +0.6 | +12.6 |

三种估计器的 advantage scale 差 1~2 个数量级——这是一个方法论层面的发现：用同一学习率直接对比本质上不公平。RLOO 虽然最终 reward 改善与 GRPO 接近（+12.6 vs +11.5），但 grad_norm 几乎每步超过 20，完全靠 gradient clipping 兜底。

### 3.4 结论与后续影响

在 reward 全负的 DeepResearcher Agent 场景下，GRPO 是最稳定的选择。DrGRPO 需要配合 advantage clipping [-5, 5] 才能发挥作用——这个经验值直接被 Exp-03b 采纳。

---

## 四、Exp-02：教会 3B 模型主动搜索的三管齐下

### 4.1 局部最优陷阱

3B 模型在 RL 训练初期表现出明确的 mode collapse：倾向于第 1 轮就输出 `<answer>` 标签终止搜索。原因是格式正确的终止行为本身就能获得基础分（不搜索但格式正确比搜索但格式错误得分高），模型找到了一条"捷径"——不搜索就不会犯搜索错误，稳定拿格式分。这在 outcome reward 下是合理的局部最优，但显然不是全局最优。

### 4.2 三招联用的因果链

**entropy_coeff = 0.01**（vs baseline 的 0.001）：在策略梯度 loss 中加入 entropy bonus，直接鼓励输出分布的多样性。如果太低，模型快速锁死到单一行为模式。注意 entropy bonus 和 KL 惩罚的功能正交——KL 约束"离 Ref 多远"，entropy 约束"当前分布有多集中"。一个策略可以远离 Ref 但高度集中（KL 大但 entropy 低）。

**curriculum（max_turns 递增）**：阶段一只允许搜 1 次，阶段二放到 3 次，阶段三放到 6 次。核心逻辑是降低探索难度——先在压缩的动作空间里学会基本能力，再逐步放开。如果一开始就允许 6 轮搜索，动作空间太大（每轮都要决定搜什么或者停），3B 模型的探索效率极低。

**early_stop_penalty = -0.5**：未满最低搜索轮次就终止的轨迹额外扣 0.5 分。这个惩罚不依赖搜索结果质量（确定性信号），纯粹阻止"不搜索拿格式分"的捷径策略。-0.5 的选择经过消融：-0.1 太小（格式分 0.1-0.2 扣完仍有正收益），-1.0 太大（强迫模型凑轮次，出现无意义重复搜索"what is X" → "X definition" → "X meaning"）。

三者联用的因果链：entropy 保证探索空间不坍缩 → curriculum 降低正确搜索的学习难度 → early_stop_penalty 堵死不搜索的退路 → 模型被"逼"着尝试搜索 → 偶尔搜到有用信息获得正 reward → 强化搜索行为。

### 4.3 效果

8 个 training step 内，search_depth 从 0（完全不搜索）稳步上升至 2.9（平均每轮主动搜索近 3 次），grad_norm 从 10.2 收敛到 2.1。Baseline 在相同步数内 depth 始终为 0，grad_norm 爆炸到 47。搜索行为的涌现是非线性跳变的——不是渐进提升，而是几步内从完全不搜索突然跳到主动多轮搜索。

| 维度 | Baseline | Ours |
|------|----------|------|
| 学会调用工具 | 否 | 是 |
| search_depth 终态 | ≈0 | 2.9 |
| grad_norm 趋势 | 爆炸 (47) | 收敛 (2.1) |

Step 9 因多轮搜索拼接后序列达到 8410 tokens 超过限制而崩溃——模型学会多轮搜索后 response 自然变长，这本身也是"成功"的证据。后续实验将 token 限制提升至 12288~16384。

---

## 五、Exp-03：PBRS 过程奖励——理论正确的信用分配

### 5.1 稀疏 reward 的信用分配困境

原始框架仅在最终 `<answer>` 标签处计算 F1 作为 outcome reward。一条 5-10 轮搜索的轨迹，前面所有搜索行为完全没有梯度信号。如果第 3 轮搜了一个极好的 query 收集到关键信息，但第 7 轮搜了个无关 query 稀释了上下文导致最终 F1 下降，稀疏 reward 只能给出"整条轨迹不好"的信号，无法区分第 3 轮和第 7 轮的贡献差异。

### 5.2 为什么是 PBRS 而不是别的

注入中间奖励有三种常见方案：

PRM（Process Reward Model）需要标注数据训练一个独立打分模型，web search Agent 场景中没有现成的中间步骤标注数据，从头标注成本极高。Rule-based shaping（比如"每搜索一轮 +0.1"）简单但不保证 policy invariance——模型会学会为了过程奖励而凑搜索轮次。PBRS 利用 GT answer 自动计算覆盖度作为势函数，无需额外模型或标注，且 Ng 1999 证明了势函数差分形式 F = γΦ(s') - Φ(s) 不改变最优策略。

势函数定义：Φ(s_t) = token_overlap(accumulated_info_t, GT_answer) / |GT_tokens|。语义清晰——好的搜索覆盖新信息得正奖励，无用搜索得零奖励，重复搜索势差为零不奖不罚。token overlap 而非 embedding similarity 的原因是可加性和幂等性：同一个 token 多轮命中不重复计算，而 embedding 相似度不满足这个性质。

### 5.3 Baseline（3a）：证明纯 outcome reward 不 work

73 步训练后 reward 毫无改善，search_depth 恒定在 1.0-1.6。模型虽然在搜索，但搜到的内容与答案无关（info_gain ≈ 0），所以从 reward 的角度看"搜与不搜没区别"。根本原因是搜索缓存质量极差：模型产生 2296 个不同 query，精确命中率 0%，fuzzy 匹配仅 28% 且大部分返回无关内容。

这个 baseline 的核心价值：证明了"纯 outcome reward + 噪声搜索环境 = 模型无法学习有效搜索策略"，也揭示了搜索环境质量是 Agent RL 训练的硬前提。

### 5.4 PBRS 实验（3b）：数值炼狱与最终稳定

PBRS 实验经历了一场数值稳定性的炼狱。v1 版搜索行为正常（depth 1.1→2.0），但因磁盘满崩溃。v2 续训后 grad_norm/kl 全 NaN——forward 阶段 logits 中每步稳定出现 ~100 个 NaN（vocab_size=151936 的张量中占比 0.000006%），但经 backward 传播后扩散到全部 2900 万梯度元素。

最终 v6 稳定版的修复链展示了 LoRA + RL 场景下数值稳定性的工程复杂度：Adam eps 从 1e-8 提升到 1e-4（默认值在 fp16 下除法溢出）、fp32 upcast + clamp(-1e4, 1e4) + nan_to_num 应用于 logits、autocast(enabled=False) 在 entropy/log_prob 计算时禁用混合精度、entropy_from_logits 中 torch.where(isfinite) 处理 -inf、advantage clipping [-5, 5]（Exp-01 的产出）。

v6 版本的效果：

| 维度 | 3a Baseline | 3b PBRS (v6) |
|------|-------------|-------------|
| pg_loss | 正常但无改善 | 0.4 ~ 2.2 |
| grad_norm | 2608 ~ 23048 | **0.014 ~ 0.057** |
| search_depth | 1.0-1.6 | **1.6-2.5** |
| NaN | 无 | 无（v6 修复后） |

grad_norm 从万级降到百分之一级——下降 5 个数量级。本质原因是 PBRS 把原来只挂在最后一个 token 的信号分散到了中间位置，等价于降低了 credit assignment 的方差。Score 未明显改善的根因是搜索环境质量而非算法失效：info_gain ≈ 0 说明即使模型学会了搜索行为，搜索返回的内容和答案无关。PBRS 是 reward 的"放大器"而非"创造者"——它依赖搜索环境能返回有效信息。

---

## 六、Exp-04：LoRA + Frozen-Ref 的显存工程

### 6.1 为什么不是可选优化，而是工程必需

全参 GRPO 训练需要同时维护 Actor 和 Ref 两份完整模型。最初在 2×A800-40GB 上反复尝试了一整天：FSDP NO_SHARD 模式下，单卡显存是硬瓶颈，激活值不跨卡共享。2×40GB 不等于 80GB——多卡并行只能分摊参数和优化器状态，不能分摊单条序列的激活值。在长序列（4096 tokens）backward 下轻松超过 40GB。截断 response 到 1024 tokens 勉强能跑但严重损害多轮训练效果。最终正确方案是换 A100-80GB 单卡，一步到位。

LoRA 的引入带来一个独特优势：Actor = base + LoRA(ΔW)，Ref = base（共享底座）。KL(π_actor || π_ref) 自然度量的就是 LoRA adapter 引入的策略偏移——与全参微调中单独加载一份完整 Ref 模型在数学上等价，但省掉了 Ref 的 6GB 显存和 12GB optimizer states。训练开始时 LoRA output 为零（B=0），Actor 等于 Ref，KL=0，起点一致。

### 6.2 FSDP 的精细控制

FSDP 通过 lambda wrap policy 确保只对 LoRA 的 A/B 矩阵做 all-gather/reduce-scatter，冻结的 base 参数通过 CPUOffload 卸载到 CPU。通信量从全参的 ~6GB 降至 LoRA 的 ~50MB，降低两个数量级。CPUOffload 的延迟开销约 15%（63s vs 55s per step），但 FSDP 逐 layer 执行时可 prefetch 下一 layer，流水线并行隐藏了大部分 overhead。

### 6.3 LoRA 超参选择

rank=64, alpha=16，effective scaling = alpha/rank = 0.25。rank=16 表达力不足（无法改变 query generation 的分布），rank=128 显存紧张且 FSDP 通信开销回升。alpha < rank 是有意的保守选择：Agent RL 的 grad_norm 在 6000-12000 之间波动，如果 alpha=64（缩放 1.0），梯度波动会直接传导到模型输出导致策略突变。

实测 peak 显存约 50GB（A100-80GB 上运行），KL 从 0.001 缓慢上升到 0.02——说明策略确实在合理范围内偏离 base model。

---

## 七、Exp-05：三维行为监控——单看 reward 曲线远远不够

### 7.1 Reward Hacking 的真实威胁

一个典型案例：模型发现"搜索原问题本身"就能从搜索结果中获得部分 token overlap 分数，于是每轮都搜同一个 query。reward 上升但搜索能力为零。如果只看 reward 曲线，这个 hacking 行为与真正的能力提升看起来完全相同。

### 7.2 三个指标的正交设计

三个维度的选择覆盖了行为评估的三个正交层面：

search depth（行为量）：单条轨迹中 tool_call 的轮次均值。区分"太保守"和"太激进"。query diversity（行为质）：相邻两轮 query 的 BLEU-4，低于 0.3 视为有效策略切换。depth 高但 diversity 低 = 每轮搜同样的东西（hacking）。information gain（行为效）：第 t 轮搜索返回内容与 GT 的新增 token overlap。diversity 高但 info_gain 低 = 搜的虽然不一样但都是垃圾。

三者联合的诊断判据：

| 模式 | reward | depth | diversity | info_gain | 诊断 |
|------|--------|-------|-----------|-----------|------|
| 正常学习 | ↑ | ↑ | 稳定/↑ | ↑ | 健康 |
| Query 重复 hacking | ↑ | 稳定 | ↓↓ | ↓ | 锁定搜索模板 |
| 早停 hacking | ↑ | ↓↓ | N/A | N/A | 靠格式分骗分 |
| 盲搜 | 稳定 | ↑↑ | ↑ | ↓/0 | 搜了但无效 |
| 真正卡住 | 稳定 | 稳定 | 稳定 | 0 | 环境问题 |

### 7.3 实际应用

在 Exp-03b 训练中，监控体系识别出最后一种 pattern——"真正卡住"：reward 无改善、depth 稳定在 1.0-1.6、diversity 维持高位（0.96-1.0）、info_gain ≈ 0。诊断结论：模型策略没有退化（diversity 高说明没有 collapse），问题出在搜索环境质量上。这正是监控体系的核心价值——区分"行为正确但环境无效"与"行为本身有问题"。

阈值不是拍脑袋定的：在未训练模型上跑 200 条 rollout 作为 baseline 分布，取 diversity 的 P25 作为警戒线。这样阈值随任务和模型自动适配。

---

## 八、工程炼狱：让系统跑起来比设计算法更难

### 8.1 显存是物理定律，不接受 workaround

第六章的详细分析已经说明了结论：2×A800-40GB 不等于 80GB，FSDP 分摊的是参数和优化器状态而非单条序列的激活值。在这个问题上浪费了整整一天，各种截断、offload 方案都只是治标。最终迁移到 A100-80GB 单卡之后，所有后续实验（Exp-01 到 Exp-05）一步到位地跑通了，峰值显存约 70GB，余量 10GB 完全够用。这里想强调的教训是：当物理约束卡脖子时，换硬件永远比 workaround 更高效——一天的 debug 时间换来的经验是"应该一开始就换卡"。

### 8.2 搜索缓存：从本地代理到 query 驱动的缓存构建

DeepResearcher 原论文用的是真实 Google 搜索——但在国内 GPU 服务器上根本没有这个条件。整个搜索后端需要自己造。

实际方案是写一个本地搜索代理服务 `/tmp/search_proxy_server.py`，跑在端口 8890 上，完全模拟 SearXNG 的接口规范（POST `/search`，返回 `{"results": [{url, title, content}]}`）。搜索引擎的选择也经过权衡：百度作为主引擎——从国内 IP 访问稳定、中英文 query 都能处理、不需要翻墙；Bing 作为补充引擎——英文 query 的结果质量更高。训练框架那边只认 SearXNG 接口格式，所以代理层做的就是把请求翻译成百度/Bing 的搜索请求，再把结果包装回标准格式。

但真正的坑在缓存构建策略上。初始方案是把各种 query 预搜索一遍存成离线缓存——服务器上堆了 12636 条（49MB），看着很充足。实际一跑就暴露了致命问题：模型产生 2296 个不同 query，精确命中率 0%。原因是这批缓存大多是之前 DuckDuckGo 混合抓取的残留，75% 是英文 query 配中文无关结果的垃圾数据。

解决方案翻转了缓存构建的因果方向：不是"人预设 query → 预搜索 → 希望模型碰上"，而是"先跑一轮 rollout → 从训练日志提取模型实际产生的 query → 用本地代理对这些 query 做真实搜索 → 入库"。第一批高质量缓存 540 条就是这样来的，后续补充到 2231 条后命中率显著提升。训练中还支持缓存热更新——另一个会话实时补充缓存到 13589 条后直接重启 proxy，无需中断训练。

这个经验揭示了一个更深的洞察：DeepResearcher 原论文用真实 Google 搜索之所以 work，"搜索质量高"是一个关键但未被显式讨论的隐含前提。离线训练要复现这个效果，缓存的构建必须是 model-query-driven 的——缓存的 key space 要跟着模型的行为分布走，而不是跟着人的先验走。

### 8.3 序列长度的动态不可预测性

multi-turn agent 的 response 长度不是静态的——取决于搜索结果长度。Step 73 时某样本多轮拼接达 9753 token，超过 ppo_max_token_len_per_gpu=8192，训练直接崩溃。一个异常长的搜索结果就能让整个 batch 挂掉。需要动态截断或 dynamic batching。

### 8.4 NaN 溢出的排查链

Exp-03b 的 NaN 问题是典型的"小概率事件在大规模计算中变成必然"：vocab_size=151936 的 logits 张量中只有 ~100 个 NaN（占比 0.000006%），但 backward 后扩散到全部 2900 万梯度元素。排查方向从 optimizer 到 loss function 到 forward pass 逐层回溯，最终发现是 fp16 精度下特定计算路径的数值不稳定。修复需要 7 处独立的防御措施联合作用。

### 8.5 框架兼容性的时间税

verl 0.2.0.dev + PyTorch 2.6 + Ray + vLLM 的组合有大量未文档化的 breaking change。PyTorch 2.6 将 torch.load 的 weights_only 默认值改为 True，直接导致所有包含 numpy 数组的 checkpoint 加载失败。Ray worker 在 ray stop 后仍占 14GB VRAM 不释放。Handler 进程和 Trainer 进程的启动顺序依赖关系没有文档化（do_search=true 时 Trainer 会无限等待 Handler 的文件信号）。实际开发中 30% 以上时间花在 debug 框架问题上。

---

## 九、未完成项与诚实的边界

score 未明显改善——所有实验中模型的 F1 reward 都在 -0.5 ~ -0.97 之间波动，没有出现持续上升的学习曲线。但这不是算法失败的证据，而是搜索环境质量不足的直接后果。通过三维监控体系的联合诊断，可以明确判断：模型策略正常（diversity 高、depth 在增长），问题在于搜索返回的内容和答案无关（info_gain ≈ 0）。

Exp-06/07 的端到端集成实验因 GPU 时间限制未执行。Curriculum 与 PBRS 的联合使用（理论上两者互补——前者解决冷启动，后者解决中期信用分配）也留待后续验证。v6 稳定版运行到 step 128 时各指标健康但 score 未收敛，需要 500+ steps 观察最终行为。

这些未完成项恰恰构成了项目叙事的诚实结尾：五个方向的子系统各自验证可用，瓶颈被精确定位到搜索环境质量这个单一前置条件上。后续只需补充高质量搜索后端（或覆盖率 90%+ 的离线缓存），即可在已验证的训练管线上完成端到端实验。

---

## 十、关键数字速查

| 结论 | 数据支撑 |
|------|----------|
| 三种策略 8 步内教会工具调用 | search_depth: 0 → 2.9 |
| DrGRPO 稳定性不如 GRPO（reward 全负场景） | grad_norm spike: 40.5 vs GRPO 的 4.2 |
| PBRS + 数值修复后训练完全稳定 | grad_norm: 0.014-0.057，88 步零 NaN |
| LoRA rank=64 在 Agent RL 中可行 | 峰值显存 ~50GB，KL 合理偏移 0.001→0.02 |
| 纯 outcome reward 无法驱动多轮搜索学习 | 73 步 reward 无改善，depth 恒定 1.0-1.6 |
| 搜索缓存质量是实验成功的前置条件 | 精确命中率 0%，info_gain ≈ 0 |
| FSDP 通信量降低两个数量级 | 全参 ~6GB → LoRA ~50MB |
| CPUOffload 速度代价可控 | step time +15%（63s vs 55s） |
