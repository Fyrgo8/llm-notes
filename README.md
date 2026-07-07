# LLM Notes

这不是一个“把 Markdown 堆在一起”的仓库，而是一套逐步成形的 LLM 学习与论文阅读资产。

仓库当前分成两层。根目录放的是少量但相对成篇、可以直接阅读的主题笔记，适合快速理解某个问题域的主线；`论文阅读笔记/` 则是一套独立的 Obsidian vault，更像一个长期维护的论文阅读系统，强调索引、模板、摘录、主题沉淀和持续复用，而不只是单篇摘要。

## 仓库里有什么

根目录目前有四份核心内容，分别承担不同角色。

- [MiniMind_笔记.md](./MiniMind_%E7%AC%94%E8%AE%B0.md)
  围绕 MiniMind 的训练到推理全链路学习笔记，重点在模型结构、训练流程、实现路径和工程理解，适合用来建立“小模型从零到可跑”的整体心智模型。
- [MiniResearcher_学习笔记.md](./MiniResearcher_%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
  聚焦低资源多轮搜索 Agent 的 RL 训练实践，内容更偏实验与系统实现，适合已经熟悉基础概念、想进一步看真实训练栈和问题拆解的人。
- [agent-harness-engineering-survey-cn.md](./agent-harness-engineering-survey-cn.md)
  《Agent Harness Engineering: A Survey》中文全文翻译，偏基础设施与工程体系视角，适合作为理解 Agent execution harness 的系统入口。
- `论文阅读笔记/`
  一套独立的论文阅读库，不和根目录几篇长笔记混写，而是按论文管理工作流来组织。

如果只想快速读成篇内容，从根目录这三篇开始就够了。如果你想把论文阅读变成一套可累积、可导航、可复用的系统，再进入 `论文阅读笔记/`。

## 论文阅读笔记 Vault

`论文阅读笔记/` 的定位不是“论文摘要仓库”，而是一个面向长期积累的阅读工作台。它把阅读过程拆成入口、索引、单篇笔记、跨论文主题、模板、附件和摘录几个层次，这样后续新增内容时不容易重新退化成“文件越来越多、但越来越难找”的状态。

核心目录如下。

- `00 Inbox`
  暂存未整理的论文、想法、待处理材料。
- `01 MOC`
  整个 vault 的导航层，包含论文总览、论文看板、最近论文、研究主题、作者索引、阅读状态看板等入口。
- `02 Papers`
  单篇论文笔记与精读结果，当前已经有 `EnvScaler`、`ET-Agent`、`Tool-Star` 等条目。
- `03 Topics`
  按主题聚合的跨论文笔记，例如 `Agentic RL`、`Tool Use`、`GRPO`、`RLHF`。
- `04 Authors`
  作者维度归档。
- `05 Venues`
  会议 / 期刊维度归档。
- `06 Daily Notes`
  阅读过程中的日常记录。
- `07 Templates`
  论文笔记、主题笔记、深读笔记等模板。
- `08 Attachments`
  附件与辅助材料。
- `09 Excerpts`
  摘录、intake 和记要。
- `copilot`
  配套的 prompt 资产。
- `.obsidian`
  Vault 配置与插件配置。

## 推荐阅读路径

第一次进入这个仓库，最顺的路径通常不是从目录树最深处开始翻，而是先确定你要解决的问题是哪一类。

1. 如果你想先看成篇、相对完整的技术笔记：
   从 [MiniMind_笔记.md](./MiniMind_%E7%AC%94%E8%AE%B0.md) 或 [MiniResearcher_学习笔记.md](./MiniResearcher_%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md) 开始。
2. 如果你想看 Agent 基础设施的系统综述：
   直接读 [agent-harness-engineering-survey-cn.md](./agent-harness-engineering-survey-cn.md)。
3. 如果你想进入论文阅读工作流本身：
   先看 `论文阅读笔记/01 MOC/论文总览.md`，再看 `论文阅读笔记/01 MOC/论文看板.md`，最后进入 `论文阅读笔记/02 Papers/` 的单篇笔记。

这个顺序背后的考虑是：先拿到导航，再决定深入单篇，通常比直接扎进某个文件更省时间。

## 使用方式

- 只想读内容：直接在 GitHub 浏览 Markdown。
- 想完整使用论文库：用 Obsidian 打开 `论文阅读笔记/`。
- 想把这套结构迁移到自己的阅读系统：优先参考 `01 MOC/` 的入口组织方式和 `07 Templates/` 的模板设计，而不是只复制单篇笔记正文。
