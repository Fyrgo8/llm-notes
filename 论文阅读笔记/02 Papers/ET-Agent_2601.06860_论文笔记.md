# ET-Agent（arXiv:2601.06860）论文笔记

论文：**ET-Agent: Incentivizing Effective Tool-Integrated Reasoning Agent via Behavior Calibration**  
作者：Yifei Chen, Guanting Dong, Zhicheng Dou  
链接：[https://arxiv.org/pdf/2601.06860](https://arxiv.org/pdf/2601.06860)

## 1. 这篇论文到底想解决什么问题

### 一句话问题
这篇论文要解决的是：**TIR（Tool-Integrated Reasoning） agent 不只是会不会做对题，而是会不会以“有效、紧凑、少犯工具调用错误”的方式做对题。**

### 为什么这个问题重要
现在很多 LLM agent 已经会调用搜索、代码解释器之类的工具了，但“会调用工具”不等于“调用得好”。旧工作常常只盯着最终答案是否正确，于是会出现一种很常见的现象：

- 结果可能对了，但中间搜了很多次没用的信息
- 结果可能错了，不是因为模型完全不会，而是因为少调了一次关键工具
- 有时工具调用格式错、代码错、空 query，直接把执行打断
- 有时 reasoning 很长，但真正有信息增益的步骤很少

如果只看 final accuracy，这些问题会被掩盖。可一旦你真的把 agent 放到真实系统里，成本、延迟、稳定性都会被这些行为模式拖垮。  
所以这篇论文的核心视角不是“让模型更会答题”，而是“让模型形成更好的工具使用行为习惯”。

### 旧方法具体卡在哪里
作者点了两类旧方法的不足。

第一类是只做结果优化的 TIR / RL 方法。  
它们会让模型为了正确率疯狂调用工具，因为 reward 里几乎只看结果对不对。于是模型学到的策略可能是“多搜总比少搜安全”，这会造成 redundant tool calls。

第二类是像 DPO 这类 binary preference 方法。  
作者认为这类方法虽然能做行为偏好对齐，但容易把输出压到很窄的动作空间里。TIR 任务的 action space 非常大，因为：

- 同一个问题可以搜不同 query
- 可以先搜再写，或先想再搜
- 搜索次数、顺序、代码调用时机都不同

如果你只做“二选一偏好学习”，模型会更像是在模仿一条好轨迹，而不是学会在广泛动作空间里探索并逐渐校正行为。  
所以作者认为旧方法卡在两个地方：

- **探索不够**：没真正展开 tool-use action space
- **校正不够细**：没有同时处理冗余调用、调用不足、执行错误、推理冗长等问题

---

## 2. 这篇论文最核心的创新是什么

### 作者真正提出了什么
作者提出的是一个完整框架 **ET-Agent**，不是单一 loss trick。它由两部分组成：

- **Self-Evolving Data Flywheel**
- **Behavior Calibration Training**

前者负责把动作空间“撑开”，后者负责把撑开的探索“收回来”，逐步收敛到更优的行为模式。

### 它具体改了什么
它同时改了四层东西：

- **数据构造**：不是只用原始轨迹，而是让模型对对/错轨迹做自我修正、删冗余、补工具、重走后续路径，得到增强数据
- **训练流程**：先 RFT 做 action-space exploration，再做 RL 校正
- **sampling**：不是对所有 query 一视同仁，而是用 Pareto sampling 优先挑“有训练价值”的 query
- **reward 设计**：不是只奖励正确，而是把 correctness、format、tool efficiency、reasoning length 合到一起

### 对照你的分类，它主要动了哪些部位

- `优化目标`：动了。RL 目标不再只是结果正确，而是“正确 × 高效 × 合规格式”
- `reward 设计`：动了，而且这是核心之一
- `advantage / baseline`：有，用 group-normalized reward 做 advantage
- `policy update`：用的是 ARPO，不是作者新发明，但他们把它放进了自己的 curriculum RL 流程
- `sampling`：明显动了，提出了 group-wise Pareto sampling
- `训练工程`：也动了，整套训练是两阶段 + 多轮 curriculum

### 与已有方法最本质的差别
最本质的差别不是“loss 多了一项”，而是：

**作者把“行为校正”当成一个先扩探索、再压收敛的过程。**

旧方法常见两种极端：

- 只做 imitation / DPO，太快收缩，探索不够
- 只做结果导向 RL，探索很多，但行为模式不受控

ET-Agent 试图站在中间：

1. 先用 flywheel + RFT 扩大可见动作空间  
2. 再用 reward + Pareto + curriculum RL 把探索导向“又对又省”的轨迹

这就是它真正想表达的方法论。

---

## 3. 关键公式不要直接略过

这篇论文真正重要的公式不多，核心有三组。

### 3.1 RFT 目标

论文在 Action Space Exploration Fine-tuning 阶段写的是标准监督式目标，本质就是最大化增强轨迹的 token 级对数似然：

\[
\max_\theta \mathbb{E}_{(x,y)}\left[\sum_{t=1}^{|y|}\log P_\theta(y_t|x,y_{<t})\right]
\]

这里：

- `x` 是输入问题
- `y` 是 flywheel 生成并经过质量筛选后的整条轨迹
- `θ` 是当前 policy 参数

这个公式在做什么？

- 它不是在直接优化 reward
- 它是在让模型先学会“这些经过修正/增强的行为轨迹长什么样”

梯度信号从哪里来？

- 来自 teacher forcing 下每个 token 的交叉熵梯度

这一阶段鼓励什么？

- 鼓励模型复现更丰富、更合理的工具使用路径

如果没有这一阶段会怎样？

- 后面的 group RL 很容易遇到组内轨迹太像的问题
- 一旦组内 reward 差异小，advantage 接近 0，RL 信号会弱

所以这一步不是为了最终最优，而是为了给后续 RL 创造“可学的分布差异”。

### 3.2 行为 reward：tool 和 length 的 logistic 分数

论文给了两个效率项：

\[
f_{tool} = \frac{1}{1 + e^{\sigma_{tool}(v_{i,tool}-\bar v_{tool})}}
\]

\[
f_{len} = \frac{1}{1 + e^{\sigma_{len}(v_{i,len}-\bar v_{len})}}
\]

先拆变量：

- `v_{i,tool}`：第 `i` 条轨迹的工具调用次数
- `\bar v_{tool}`：同组轨迹的平均工具调用次数
- `σ_tool`：控制惩罚斜率的超参数
- `v_{i,len}`：第 `i` 条轨迹的 reasoning 长度
- `\bar v_{len}`：同组平均 reasoning 长度
- `σ_len`：长度惩罚斜率

这两个式子直觉上是什么意思？

- 如果一条轨迹的工具调用数 **低于组均值**，那么 `v_i - mean < 0`，指数项更小，`f_tool` 更大
- 如果一条轨迹的 reasoning 长度 **短于组均值**，`f_len` 更大

也就是说，它不是拿绝对阈值惩罚，而是做 **group-relative efficiency reward**。  
这点很重要，因为不同题目的最优工具调用次数本来就不一样。  
作者不想说“所有题都必须少搜”，而是说“在同一题的一组候选里，谁更省、谁更紧凑，就给谁更高分”。

为什么用 logistic，而不是线性扣分？

- logistic 是平滑的
- 不会因为多一次调用就产生特别剧烈的梯度
- 还能通过 `σ` 控制惩罚强度

如果去掉这两项会怎样？

- reward 就会过度偏向 correctness
- 模型会更容易学成“只要能做对，多搜一点也无所谓”

### 3.3 总 reward

论文把 reward 聚合为：

\[
R_i = R_i^{corr}\cdot f_i^{tool}\cdot f_i^{len} + R_i^{format}
\]

这里每项含义是：

- `R_corr`：结果正确性 reward
  - 知识题用 F1
  - 数学题用二值 reward
- `f_tool`：工具调用效率项
- `f_len`：推理长度效率项
- `R_format`：格式惩罚项，格式不合法时给 `-1`

这个 reward 在鼓励什么？

- 先要求答对
- 在答对的前提下，更奖励“工具更省、推理更短”的轨迹
- 同时保证输出格式别坏

为什么是乘法，不是加法？

因为作者想让“效率优化”依附在“正确性”上。  
如果一条轨迹本来就答错了，那么你再短、再省工具，也不应该拿高 reward。  
乘法结构就自然表达了这个逻辑：**先对，再谈省。**

如果只做加法会怎样？

- 可能出现“错得很省”也能得不错分数
- 模型会学到投机行为：少想、少搜、赶紧输出

### 3.4 ARPO 目标和 advantage

论文的 RL 目标是 ARPO 风格的 clipped objective，本质上还是 PPO 家族：

\[
J_{ARPO}(\theta)=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\left(r_t(\theta)A_i,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_i\right)
\right]
\]

其中：

- `G`：一个 group 里的采样轨迹数
- `o_i`：第 `i` 条输出轨迹
- `|o_i|`：轨迹 token 长度
- `r_t(θ)`：新旧 policy 在 token `t` 上的概率比
- `ε`：PPO clipping 超参数
- `A_i`：第 `i` 条轨迹的 advantage

它优化什么？

- 当一条轨迹 reward 高于组内平均，就提高生成这条轨迹 token 的概率
- 当 reward 低于组内平均，就降低概率

论文里的 advantage 近似是组归一化后的 reward：

\[
A_i = \frac{R_i - \bar R}{\mathrm{std}(R)}
\]

这里：

- `\bar R` 是组内平均 reward
- `std(R)` 是组内 reward 标准差

这意味着它没有单独训练一个 value model，而是用 **group-relative baseline** 代替。

这一项在鼓励什么？

- 高于组均值的轨迹被强化
- 低于组均值的轨迹被压制

这一项在约束什么？

- `clip(...)` 限制 policy update 不要一步跳太大
- `std(R)` 归一化防止 reward scale 波动太大

如果去掉 group baseline 会怎样？

- 梯度方差更大
- 同题不同轨迹之间的相对好坏利用不起来

如果组内轨迹太同质会怎样？

- `std(R)` 很小，`A_i` 区分度弱
- RL 学不到东西

这也是他们前面为什么要先做 flywheel + exploration RFT，再做 Pareto sampling。

---

## 4. 训练流程按步骤讲

这篇论文的训练流程很适合按算法管线来理解。

### 1. 输入数据从哪里来

输入起点是原始问题集合 `D_source`。  
每个问题可能来自数学推理任务，也可能来自知识密集任务。

### 2. 如何采样 rollout / response

先让基础模型 `M` 对每个问题生成多条轨迹，形成初始 `D_q^rollout`。  
这些轨迹按答案正确与否分成：

- `Correct Set`
- `Incorrect Set`

### 3. reward / preference 怎么得到

先别急着进 RL。作者先做一个 **Self-Evolving Data Flywheel**：

对正确轨迹：

- `Redundant Modification`：定位第一处冗余工具调用/冗余思考，然后从那里重写后续
- `Global Refinement`：整体压缩推理、保留逻辑一致性

对错误轨迹：

- `Self-Correction`：定位第一处错误推理步骤，修正它，再继续生成后续
- `Hint Injection`：在错误点后面或轨迹末尾插入 hint，鼓励继续搜、继续调工具，修复“工具调用不足”

增强后的轨迹重新并回池子，循环 `R` 轮，得到 `D_aug`。

这里本质上不是 preference pair，而是 **自增强轨迹池**。

### 4. advantage / baseline 怎么算

在 RL 阶段，对某个 query 采样一组轨迹。  
先根据 correctness、tool count、reasoning length、format 算每条轨迹的总 reward `R_i`。  
再按组内均值和标准差归一化，得到：

\[
A_i = \frac{R_i - \bar R}{std(R)}
\]

所以 baseline 是 **group mean**，不是单独 value network。

### 5. loss 怎么组成

整体分两阶段：

第一阶段：RFT loss  
- 对 `D_aug` 里通过质量筛选的轨迹做监督 fine-tuning

第二阶段：ARPO RL loss  
- 对组内采样轨迹做 clipped policy optimization
- reward 用前面那套多目标机制

### 6. 参数怎么更新

第一阶段：

- 用增强后高质量轨迹做 standard LM fine-tuning

第二阶段：

- 用 ARPO 根据组内相对 reward 更新 policy
- 在多个 round 里交替执行：
  - `Group-wise Pareto Sampling`
  - `Curriculum RL Training`

### 7. 最容易不稳定的点在哪

我认为有四个。

- **组内轨迹太像**：reward 差异小，advantage 快没了
- **reward hacking**：模型学会“少搜少想”来刷效率项，但正确性掉了
- **format / tool execution failure**：工具调用本身不合法，reward 结构会被污染
- **早期 exploration 不足**：如果 flywheel 和 RFT 没把动作空间撑开，后面的 RL 只是对狭窄动作空间做局部修补

作者对应的解法是：

- 用 flywheel + RFT 扩探索
- 用 Pareto sampling 选“有分歧、有训练价值”的 query
- 用 correctness × efficiency 的乘法 reward 避免“错得很省”
- 用逐轮减小 `σ_tool` 和 `σ_len` 防 reward hacking

---

## 5. 如果这是 RL / OPD 论文，请额外回答

### reward / preference 信号从哪里来

这篇不是传统 OPD 里“人工偏好对”那种信号来源。  
它的信号主要来自两部分：

- **显式任务结果**
  - 数学题是否做对
  - 知识题 F1 分数
- **行为质量指标**
  - 工具调用次数
  - reasoning 长度
  - 格式是否合法

所以这是一个 **rule-based / programmatic reward** 框架，不是人类 preference 标注驱动的。

### 是 on-policy 还是 off-policy

核心 RL 阶段是 **on-policy**。  
论文自己也明确强调，他们想解决的是在 TIR 大动作空间里做 on-policy behavior calibration。

前面的 flywheel 数据增强 + RFT 更像 offline / supervised warmup，  
但真正校行为的主力阶段是 on-policy ARPO。

### reference policy / KL 的作用是什么

这篇文中没有像 PPO-RLHF 或 DPO 那样显式强调一个 reference policy 的 KL 项。  
它主要继承的是 ARPO / PPO-style clipped update，而不是“显式 KL 正则到 reference model”这条线。

所以你可以把它理解成：

- **稳定性主要靠 clipping**
- **不是靠显式 KL 绑住 reference**

这点也意味着它更依赖 reward 设计和 sampling 质量。

### baseline / group normalization / leave-one-out 的作用是什么

它这里的 baseline 是 **group mean reward**。作用是：

- 降方差
- 把同一题下的多条轨迹做相对比较
- 不需要额外的 value model

group normalization 进一步除以 `std(R)`，作用是：

- 让不同 batch / 不同题目的 reward scale 更可比
- 避免某组 reward 方差特别大，把梯度主导掉

它不是 RLOO 那种 leave-one-out baseline。  
所以这篇更接近 **GRPO / group-relative PPO** 路线，而不是 RLOO 路线。

### 优化粒度是 token、step、trajectory 还是 group

可以分三层看：

- **reward 赋值粒度**：trajectory 级
- **baseline 比较粒度**：group 级
- **policy update 粒度**：token 级（通过 trajectory advantage 回传到每个 token）

所以最准确的说法是：

**trajectory-level reward, group-relative normalization, token-level policy gradient update。**

### 它和 PPO / DPO / GRPO / RLOO / REINFORCE++ 的关系是什么

- 和 `PPO`：很近。ARPO 目标本质是 PPO-family 的 clipped ratio objective
- 和 `DPO`：形成对照。作者明确认为 DPO 在 TIR 场景里容易收缩动作空间，探索不足
- 和 `GRPO`：最接近。因为它也是 group-wise relative optimization，不用 value model，advantage 来自组内相对 reward
- 和 `RLOO`：不像。它没有 leave-one-out baseline 的那个结构
- 和 `REINFORCE++`：比纯 REINFORCE 稳，因为有 clipping 和 group normalization

如果硬要归类，这篇论文的方法学血统更接近：

**“exploration-enhanced, group-relative PPO/GRPO-style agent RL framework”**

而不是纯 OPD。

---

## 6. 实验部分不要只报结果

### 最关键的实验是哪一个

我认为最关键的实验不是主表，而是下面这三个合起来看：

- **主结果表 Table 1**
- **action space distribution 分析 Figure 7**
- **ablation Table 2，尤其是 `w/o Pareto`、`w/o Reward`、`w/o σDecrease`**

### 它到底证明了什么

#### 1. Table 1 证明了什么
Table 1 证明的是：

**ET-Agent 不只是更省工具，它在正确率和效率两个维度上同时更强。**

这点很关键。  
因为如果只是效率更高，可能意味着它保守了；  
如果只是正确率更高，可能意味着它更激进地多搜了。  
而 ET-Agent 试图证明自己不是在 correctness 和 efficiency 之间做单边牺牲，而是在同时改善两者。

#### 2. Figure 7 证明了什么
Figure 7 更像机制证据。  
它表明：

- 从 base instruct 到 RFT，输出分布更分散  
  说明 flywheel + RFT 确实把动作空间探索撑开了
- 从 RFT 到 RL，分布又变紧  
  说明 RL 阶段不是继续发散，而是在已有广泛探索上收敛到更优轨迹

这张图实际上是在给整篇方法的核心叙事背书：  
**先扩探索，再压收敛。**

#### 3. Table 2 证明了什么
ablation 是最有价值的部分，因为它在检验“每个部件是不是都真有用”。

- `w/o Flywheel` 变差：说明探索预热确实重要
- `w/o Pareto` 变差：说明不是所有 query 都一样有训练价值，采样策略有贡献
- `w/o Reward` 变差：说明行为 reward 不是摆设
- `w/o σDecrease` 正确率掉得很厉害：说明 reward hacking 真存在，而 curriculum 调 sigma 确实在防它

### 是证明“方法有效”，还是只证明“某个实现细节有效”

整体上，它证明的是“**这套框架整体有效**”，不只是某个小 trick。  
但要更严格地说：

- `主表 + 多指标分析`：证明整体 pipeline 有效
- `Figure 7`：更偏机制验证
- `σDecrease` ablation：更偏实现细节验证

所以不是所有实验都在证明同一件事。  
你读的时候要分清：

- 哪些实验在证明“框架思想成立”
- 哪些实验在证明“这个工程细节确实有帮助”

### baseline 是否公平

有一部分是相对公平的，也有一部分需要保留意见。

相对公平的地方：

- 同时对比 direct reasoning、single-TIR、multi-TIR
- 使用统一工具配置
- 还专门把 WebSailor 用 Wikipedia 数据重训，尽量对齐环境

需要保留意见的地方：

- 不同 baseline 原生设计目标并不一样，有些本来就不是“效率优先”
- 他们把效率定义成 `correctness / tool_calls`，这是合理但偏作者立场的指标
- 知识题训练和测试都 heavily 依赖 Wikipedia / local retrieval，泛化到真实 web 搜索场景还没完全说明

### 有没有可能存在别的解释

有，至少有三个。

- 性能提升部分可能来自 **数据增强本身**，不全是 RL 行为校正
- Pareto sampling 可能本质上是在做“更聪明的数据重加权”，而不是一定只有作者解释的“防组内梯度退化”
- efficiency 提升可能部分来自“输出更短了”，而不是更深层的行为理解提升

不过作者用 Figure 7 和 ablation 至少说明：  
这不太像单个因素就能解释完的结果，更像是多组件协同。

---

## 7. 这篇论文的前提、代价和风险

### 这套方法成立依赖什么前提

它至少依赖四个前提。

- **工具环境可执行、可观测**：你得能判断工具调用是否成功、调用了几次、调用后有没有结果
- **行为质量能被简单指标近似**：比如“调用次数更少”通常更高效，“reasoning 更短”通常更好
- **同题多轨迹之间存在可比较性**：否则 group-relative reward 没意义
- **模型有足够能力做自修正**：否则 flywheel 生成出来的增强轨迹质量有限

### 它的代价是什么

代价主要有三类。

- **采样代价高**：每题要采多轨迹，还要做多轮飞轮增强
- **训练流程复杂**：不是一次 SFT 或一次 RL 就完事
- **reward 工程成本高**：要设计 correctness、format、tool count、reasoning length 等一整套指标

从论文实现细节看，它还用了：

- 10.5k RFT 数据
- RL 做 3 轮 curriculum
- 每题采样 16 条轨迹估计 dispersion
- 4 张 A800

这说明它不是一个很轻的 recipe。

### 在什么场景下可能失效

我认为以下场景会更危险。

- **高噪声 reward 场景**：如果 correctness evaluator 不稳，group reward 直接失真
- **长轨迹、多分支任务**：tool count 少不一定更好，过度惩罚可能压掉必要探索
- **工具成本不均匀**：一次 web search 和一次 code execution 的代价可能完全不同，但它现在基本按“次数”处理
- **真实开放网络环境**：local Wikipedia 和 live web 的噪声结构差很多

### 如果模型更小、reward 更 noisy、轨迹更长，会怎样

#### 模型更小
- flywheel 的自修正质量会下降
- 错轨迹修正可能修不回来，甚至越修越乱
- RL 阶段更容易塌到短而错的策略

#### reward 更 noisy
- group-relative ranking 会不稳定
- Pareto dispersion 的“高价值 query”筛选会受污染
- curriculum 可能把错误方向越推越强

#### 轨迹更长
- reasoning length 惩罚会更敏感
- 很可能压制真正需要长链思考的问题
- 组内方差增大，优化更难稳

所以这套方法并不是普适地“少搜少想就是好”，它隐含假设是：
**在当前任务分布里，很多无效行为确实表现为冗余搜和冗长想。**

---

## 8. 用更直白的话重新解释一遍

把这篇论文当成一个工程问题，其实很直白。

### 旧方法哪里不好

旧 agent 像一个刚学会用搜索和代码工具的新人：

- 遇到不会的题会乱搜
- 有时明明该再查一步，却过早下结论
- 有时代码写坏了，工具直接报错
- 有时为了保险，一件事反复搜很多次

你如果只拿“最后答对没答对”来训他，他很可能学到的是：

- 宁可多搜，也别漏搜
- 宁可多写点思维过程，也别冒险省略

于是系统越来越重，越来越慢。

### 新方法怎么补

ET-Agent 的补法分两步。

第一步，**先让他多见一些“更好的做法”**。  
把原来对的轨迹删冗余、把错的轨迹补工具、把有问题的步骤修一修，形成一个增强后的示范池，然后先做一轮 RFT。

第二步，**再在在线采样里教他学会选更好的那条路**。  
同一题采很多条解法，比较谁更对、谁更省、谁更短、谁格式更稳，然后把高 reward 那些方向逐渐放大。

### 为什么补得上

因为它不是一上来就要求 RL 自己从很差的分布中学会全部东西。  
它先用 flywheel 把“可探索空间”展开，再用 group RL 把空间里的好行为收敛出来。

一句更口语的话：

**先让模型学会“原来这题还能这么搜、这么调工具、这么改错”，再让它学会“这些解法里哪种最值”。**

### 代价是什么

代价就是：

- 训练更复杂
- 采样更贵
- reward 更依赖人工设计
- 很容易过度奖励“短”和“省”，必须靠 curriculum 慢慢收

---

## 9. 帮我提炼“这篇论文最值得学的精华”

### 1. 最值得学的机制

**“先扩探索，再做相对校正”** 这个训练结构最值得学。  
很多 agent RL 工作一上来就做 reward optimization，但如果动作空间没撑开，后面优化只是局部修补。  
这篇论文把 `data flywheel -> RFT -> group RL` 串成一个闭环，这个思路很有价值。

### 2. 最值得学的实验设计

**Figure 7 的分布迁移分析** 很值得学。  
它不是只报主指标，而是试图证明：  
RFT 阶段让分布变散，RL 阶段让分布变精。  
这种“机制图 + 主结果 + ablation”三件套，比单纯涨点数更有说服力。

### 3. 最值得学的研究思路

**把行为模式当成优化对象，而不是只把最终正确率当目标。**  
这其实是很强的研究视角转移。  
对 agent 来说，tool efficiency、execution reliability、reasoning conciseness 都可能是和 correctness 同等重要的目标。

---

## 10. 我的论文笔记

- 一句话问题：  
现有 TIR agent 往往只为最终正确率优化，导致出现冗余工具调用、工具调用不足、执行错误和冗长推理；论文要解决的是如何在保持正确率的同时，把这些行为模式校正得更有效。

- 核心创新：  
提出 ET-Agent 框架，把行为校正拆成两步：先用 Self-Evolving Data Flywheel + RFT 扩大工具使用动作空间，再用 Pareto sampling + 多目标 reward + ARPO 做 group-wise curriculum RL，把探索逐渐收敛到更优轨迹。

- 关键机制：  
`R_corr × f_tool × f_len + R_format` 这套 reward 让效率优化依附于正确性；组内归一化 advantage 让同题多轨迹做相对比较；Pareto sampling 只挑在 correctness dispersion 和 tool-count dispersion 上“有学习价值”的 query，减少组内同质化导致的梯度衰减。

- 实验真正证明了什么：  
主表证明 ET-Agent 在正确率和效率上能同时提升；Figure 7 证明方法不是单纯靠“更短输出”刷分，而是先扩探索后压收敛；ablation 证明 flywheel、Pareto sampling、reward 设计和 sigma curriculum 都不是可有可无的工程附件。

- 风险 / 前提：  
方法依赖可执行的工具环境、较可靠的 correctness reward、以及“更少工具调用/更短推理通常更优”的任务前提；在小模型、高噪声 reward、超长轨迹或真实开放 web 环境里，可能出现 reward hacking、必要探索被压掉、泛化不足等问题。

- 我最该学的点：  
最该学的是它的研究方法论：先证明问题确实存在，再把方法拆成“扩探索”和“校行为”两段，最后用分布分析和针对性 ablation 去证明每一段都在起作用，而不是只交一张主结果表。
