Super-FunSearch: 具备元认知与跨域知识检索的进化型 Agent 系统
第一部分：产品需求文档 (PRD) 摘要
1.1 产品愿景与定位
传统的代码生成 Agent（包括原版 FunSearch）在解决复杂算法问题时，高度依赖大模型的“随机变异（Random Mutation）”和盲目抽卡。本系统旨在将其升级为**“定向进化系统”**。通过引入多级反思（Reflection）、四层动态知识库（RAG）和跨学科启发的元思想（Meta-Thoughts），让 Agent 能够像人类算法大师一样：诊断失败根本原因 -> 检索跨界解决模式 -> 融合生成高分代码 -> 提炼成功经验并自我进化。

1.2 核心模块与功能描述
执行与评估沙盒 (Evaluator/Sandbox): * 隔离执行 LLM 生成的代码，防止系统崩溃。

返回执行跑分 (Score) 和具体的错误反馈 (Error Trace / Bad Cases)。

反思与诊断分诊台 (Reflector & Router):

接收沙盒反馈，进行错误分级（语法报错、逻辑缺陷跑分低、打破纪录 SOTA）。

将具体的失败测试用例，抽象总结为“核心症状 (Symptoms)”。

四层动态图谱知识库 (4-Tier RAG Knowledge Base):

解耦存储哲学元思想 (L1)、跨领域模式 (L2)、具体问题域 (L3) 和针对具体问题的高分战术 (L4)。

支持基于“症状”的语义相似度检索，以及基于“问题域”的硬过滤。

思维链代码生成器 (Thought-Guided Generator):

融合“诊断症状”和“检索到的跨界知识”，强制 LLM 先输出融合思路，再输出 Python 代码。

知识提炼器 (Knowledge Extractor) [自我进化引擎]:

当跑出历史最高分时，自动分析代码，提炼并总结具体的 L4 战术，写入知识库，实现系统的越用越聪明。

第二部分：四层动态知识库数据结构 (JSON Schema V3.0)
请将此结构作为数据库初始化的种子文件（seed_knowledge.json）。系统在运行 RAG 时，主要对 applicable_symptoms 进行向量化 (Embedding)。

JSON
{
  "L1_Meta_Thoughts": [
    {
      "meta_id": "META_001",
      "name": "历史与动量 (Memory & Momentum)",
      "core_philosophy": "打破马尔可夫性（只看当前状态）。引入对历史状态的记忆，用过去的经验（惩罚或奖励）来修正当前的决策权重。",
      "applicable_symptoms": [
        "当前贪心策略导致后期无路可走",
        "算法在某个局部反复震荡",
        "每次都做出相同的错误选择，缺乏自适应性"
      ]
    },
    {
      "meta_id": "META_002",
      "name": "懒惰与延迟决策 (Lazy Evaluation & Look-ahead)",
      "core_philosophy": "不到万不得已不计算全部结果，或者不立刻做出不可逆的决定。先模拟、评估，再落子。",
      "applicable_symptoms": [
        "过于贪婪地填满当前空间，导致后续缺乏连续大空间",
        "每次计算全部状态导致运行严重超时"
      ]
    }
  ],

  "L2_Cross_Domain_Patterns": [
    {
      "pattern_id": "PATT_001",
      "linked_meta_id": "META_001",
      "domain": "Deep Learning",
      "name": "动量优化器 (Momentum Optimizer)",
      "mechanism": "记录上一次梯度的方向，如果在同一个方向持续下降，则加速；如果方向改变，则减速。",
      "generator_prompt_hint": "在启发式评分公式中，尝试加入一个历史记录数组。比如：priority = greedy_score - penalty_history[item_type]。"
    },
    {
      "pattern_id": "PATT_002",
      "linked_meta_id": "META_002",
      "domain": "Operating System",
      "name": "写时复制 / 按需计算 (Copy-on-Write / Lazy Tag)",
      "mechanism": "在树结构或大数组中，先用一个 Tag 标记这片区域需要修改，直到真正有请求访问这里时，才执行计算。",
      "generator_prompt_hint": "不要用双层 for 循环每次更新所有箱子的剩余空间。尝试引入 Lazy 机制，只在当前箱子塞不下时，再去更新/查找下一个可用箱子。"
    }
  ],

  "L3_Problem_Domains": [
    {
      "domain_id": "PROB_BIN_PACKING_1D",
      "name": "一维装箱问题 (1D Bin Packing)",
      "description": "将一组一维物品放入容量固定的最少数量的箱子中。",
      "base_constraints": ["物品大小不能超过箱子容量", "物品不可分割"],
      "evaluation_metric": "使用的箱子总数最少，或平均空间利用率最高"
    }
  ],

  "L4_Specific_Tactics": [
    {
      "tactic_id": "TAC_BIN_001",
      "linked_domain_id": "PROB_BIN_PACKING_1D",
      "linked_pattern_ids": ["PATT_002"],
      "name": "延迟大件匹配 (Deferred Large Item Assignment)",
      "applicable_symptoms": [
        "大件物品最后无法放入，小件物品把箱子分割得很碎"
      ],
      "tactic_description": "遇到接近箱子容量一半的中大件物品时，延迟决策。先尝试将它们两两配对，或者与特小件物品打包成一个虚拟的 'Super Item'，然后再寻找箱子放入。",
      "provenance": {
        "score_improvement": "Seed",
        "author": "Human_Expert"
      }
    }
  ]
}
第三部分：系统工作流与数据流 (The Flywheel)
本章节定义了系统的主干循环控制流。在代码实现中，这部分逻辑应集成在主调度器（Main Loop）中。

阶段 1：任务初始化 (Initialization)
系统加载 L3_Problem_Domains 获取当前任务目标（例如：PROB_BIN_PACKING_1D）。

Generator 根据 L3 的描述，生成初代基础算法代码（通常是基础贪心算法）。

阶段 2：执行与评估 (Execution & Evaluation)
Sandbox 接收生成的 Python 代码并运行测试用例。

产出执行结果：Score (跑分) 和 Logs (运行日志、Bad Cases、Error Trace)。

阶段 3：分诊与反思 (Triage & Reflection)
Reflector 接收阶段 2 的产出，并根据预设阈值/状态进行分发：

分支 A (Local Bug - 语法错误): 直接将 Error Trace 打包，发送给 Generator 触发**“局部修复 Prompt”**。不查知识库。

分支 B (Global Drawback - 分数低下/遇到瓶颈): 将 Bad Cases 发送给 LLM 进行**“专科诊断”**，强制输出 1-2 句话的抽象 Symptom (症状)。进入【阶段 4】。

分支 C (New SOTA - 创造新高分): 进入【阶段 6】知识提炼。

阶段 4：降维知识检索 (RAG Retrieval)
携带分支 B 提取出的 Symptom 以及当前的 domain_id (如 PROB_BIN_PACKING_1D) 查询知识库：

优先战术匹配 (L4 检索): 硬过滤 linked_domain_id 相符的 L4 节点，对它们的 applicable_symptoms 进行向量相似度计算，提取 Top-1 匹配的战术。

跨界灵感启发 (L2/L1 检索): 如果 L4 没有匹配项（或相似度低于阈值，说明当前问题域的战术已耗尽），则在全局的 L2 和 L1 节点中计算症状相似度，提取跨界模式（如物理退火、OS 缓存）作为灵感。

阶段 5：定向变异生成 (Thought-Guided Generation)
将以下三要素拼接组合为 Global Prompt 喂给大模型：

Original Code (失败的原代码)

Diagnosed Symptom (反思出的核心盲点)

Retrieved Knowledge (阶段 4 检索到的 L4 战术或 L2 跨界模式及代码提示)

动作： 强制大模型先输出 <thinking> 思考如何将检索到的知识应用到当前代码中，然后输出全新的 <python> 代码。随后返回【阶段 2】形成闭环。

阶段 6：知识提炼与进化 (Knowledge Distillation - 触发分支 C)
唤醒 Knowledge Extractor Agent。

输入打破纪录的 SOTA 代码。

Agent 分析代码，自动生成符合 L4 规范的 JSON 节点（提取名称、适用症状、战术描述）。

自动分配新的 tactic_id，关联当前的 domain_id，持久化写入本地 JSON/向量库。系统完成一次自我进化。