# Super-FunSearch PRD v2：基于方案 B 的定向进化 Agent 系统

## 第一部分：架构总览

### 1.1 设计理念

在原版 FunSearch 的 **Island 进化算法**基础上，增加 **Reflector（反思分诊）+ RAG（知识检索）+ Thought-Guided Generation（思想引导生成）** 作为加速通道。两套机制互补共存：

- **Island 进化**负责「广度探索」— 维持种群多样性、淘汰弱 island、迁移强 founder
- **Reflector + RAG**负责「深度引导」— 诊断失败原因、检索跨界知识、定向突变

所有生成的程序（无论来自哪条路径）最终都注册进 `ProgramsDatabase` 的 island/cluster 系统，平等参与进化竞争。

### 1.2 核心改进点（相对原版 FunSearch）

| 原版 FunSearch | Super-FunSearch v2 |
|---|---|
| LLM 盲目续写代码 | 两步调用：先生成策略思想 (thought)，再根据思想生成代码 |
| 生成失败 → 静默丢弃 | 生成失败 → Reflector 分诊 → 局部修复 / 全局定向突变 |
| 无记忆，每次从零探索 | 四层动态知识库，检索历史经验和跨界灵感 |
| 无自我进化 | 打破 SOTA 时自动提炼成功经验，写入知识库 |

---

## 第二部分：系统模块设计

### 2.1 模块总览

```
┌─────────────────────────────────────────────────────────────┐
│                     funsearch.main()                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Programs     │  │   Sampler    │  │    Evaluator      │ │
│  │ Database     │◄─┤  (主循环)    │──►│  (沙盒执行+评分)  │ │
│  │ (island进化) │  │              │  │                   │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘ │
│         │                 │                    │            │
│         │          ┌──────▼───────┐     ┌──────▼──────────┐ │
│         │          │   LLMAPI     │     │   Reflector     │ │
│         │          │ (两步调用)   │     │   (分诊+诊断)   │ │
│         │          │ ①生成thought │     │                 │ │
│         │          │ ②生成code   │     └────────┬────────┘ │
│         │          └──────────────┘              │          │
│         │                              ┌────────▼────────┐ │
│         │                              │  KnowledgeBase  │ │
│         │                              │  (四层RAG检索)  │ │
│         │                              └────────┬────────┘ │
│         │                              ┌────────▼────────┐ │
│         │                              │ Knowledge       │ │
│         │                              │ Extractor       │ │
│         │                              │ (SOTA知识提炼)  │ │
│         │                              └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 各模块职责

#### 模块 1：ProgramsDatabase（保留原版，小改）

- **职责**：维护 island/cluster 进化种群
- **改动**：`Function` 数据结构增加 `thought` 字段，使得每个程序都附带产生它的策略描述。`Island._generate_prompt()` 在拼接 prompt 时，在每个版本函数前附上其 thought 作为注释，让 LLM 能看到策略演进脉络

#### 模块 2：Sampler + LLMAPI（核心改造）

- **职责**：主循环调度，两步 LLM 调用
- **两步调用设计**：
  - **第一步 `generate_thought()`**：输入上下文（父代 thought、诊断症状、RAG 知识等），输出纯自然语言的策略描述
  - **第二步 `generate_code_from_thought()`**：输入 thought + 函数签名，输出纯 Python 代码
- **为什么两步而非一步**：
  - thought 需要被独立存储、独立进化、独立检索
  - 思路好但代码烂时，只需重试第二步，不用重新想策略
  - 第二步输出纯代码，可直接进入 `_trim_function_body()` 的 AST 解析，无需额外文本解析
  - 每步职责单一，输出格式天然清晰，不依赖 LLM 遵守 XML 标签格式

#### 模块 3：Evaluator + Sandbox（改造返回值）

- **职责**：沙盒执行代码，返回评分和反馈
- **改动**：`analyse()` 从 void 方法改为返回 `EvalResult` 结构体，包含分数、错误信息、执行状态等完整反馈。`Sandbox.run()` 执行异常时需要把 error message 传回来，而不是只返回 `(None, False)`

#### 模块 4：Reflector（新建）

- **职责**：接收评估反馈，分诊并提取症状
- **两级处理**：
  - Level 1 `triage()`：纯规则判断，不调 LLM，速度快
  - Level 2 `diagnose_symptom()`：调 LLM 做抽象症状提取，仅分支 B 触发

#### 模块 5：KnowledgeBase（新建）

- **职责**：四层知识存储与语义检索
- **检索策略**：先精准匹配 L4 → 匹配不到升维到 L2/L1

#### 模块 6：KnowledgeExtractor（新建）

- **职责**：分析 SOTA 代码，提炼 L4 战术写入知识库
- **触发条件**：仅当出现新的历史最高分时

---

## 第三部分：主循环工作流（飞轮）

### 3.1 四条路径

主循环中每个 sample 经过评估后，进入 Reflector 分诊，分流到四条路径之一：

```
                        ┌──────────────┐
                        │ Database     │
                        │ get_prompt() │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │ 路径0：正常进化      │
                    │ ① generate_thought  │
                    │ ② generate_code     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Evaluator.analyse() │
                    │ → EvalResult        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Reflector.triage()  │
                    └──┬─────┬────────┬───┘
                       │     │        │
          ┌────────────┘     │        └───────────┐
          ▼                  ▼                    ▼
    ┌───────────┐   ┌──────────────┐    ┌──────────────┐
    │ 分支A     │   │ 分支B        │    │ 分支C        │
    │ 局部修复  │   │ 全局诊断+RAG │    │ SOTA知识提炼 │
    └─────┬─────┘   └──────┬───────┘    └──────┬───────┘
          │                │                    │
          ▼                ▼                    ▼
    只重试第二步     两步定向生成         Extractor提炼
    (保留thought,   (新thought+新code)   → 写入L4知识库
     修复code)
          │                │
          ▼                ▼
    Evaluator再评估  Evaluator再评估
          │                │
          └───────┬────────┘
                  ▼
    Database.register_program()
    （所有产出都进入 island/cluster 竞争）
```

### 3.2 路径 0：正常进化（占大多数迭代）

这是原版 FunSearch 逻辑的增强版，在每次 LLM 调用时加入 thought 层：

1. `Database.get_prompt()` → 取出父代代码和父代 thought
2. **LLM 第一步** `generate_thought()`：输入父代 thought 列表，要求改进/融合，输出新 thought
3. **LLM 第二步** `generate_code_from_thought()`：输入新 thought + 函数签名，输出纯 Python 代码
4. `Evaluator.analyse(code)` → 得到 `EvalResult`
5. 将 `(thought, code, score)` 注册进 Database
6. 进入 `Reflector.triage()` 判断是否触发分支 A/B/C

### 3.3 分支 A：局部修复（thought 不变，只修代码）

**触发条件**：`EvalResult.error_trace` 非空（语法错误、运行时崩溃、类型错误等）

**行为**：
1. 保留原来的 thought 不变（策略没问题，只是代码实现有 bug）
2. **仅重新调用 LLM 第二步**：将 thought + 原代码 + error_trace 一起给 LLM，要求修复
3. 最多重试 `max_fix_attempts` 次（默认 2 次）
4. 修复成功 → 连同原 thought 一起注册进 Database
5. 修复失败 → 放弃，主循环继续下一轮

**Prompt 模板**：
```
你的策略思路是：{thought}
你上一次生成的代码报错了：
{error_trace}
请根据策略思路重新实现，修复上述错误。只输出 Python 代码。
```

### 3.4 分支 B：全局诊断 + RAG 定向生成（换一个 thought）

**触发条件**：代码能跑通但分数低（低于父代或低于 baseline），或连续多轮分数未提升（停滞）

**行为**：
1. `Reflector.diagnose_symptom(code, score, bad_cases)` → 调 LLM 提取 1-2 句抽象症状
2. `KnowledgeBase.search(symptom, domain_id)` → 分级检索：
   - 优先匹配 L4（具体问题战术），硬过滤 domain_id
   - L4 无匹配或相似度低于阈值 → 扩展到 L2（跨域模式）和 L1（元思想）
3. **LLM 第一步**（定向思想生成）：将旧 thought + 症状 + RAG 知识一起给 LLM，要求提出全新策略
4. **LLM 第二步**：根据新 thought 生成代码
5. 评估 → 注册进 Database

**诊断 Prompt 模板**：
```
以下代码在测试中表现不佳，得分为 {score}：
{code}
不要修改代码。请分析当前算法在策略层面的根本盲点是什么？
用 1-2 句话抽象总结出核心症状。
```

**定向思想生成 Prompt 模板**：
```
当前策略：{old_thought}
诊断出的核心症状：{symptom}
参考以下跨界知识：
  - 名称：{knowledge.name}
  - 机制：{knowledge.mechanism}
  - 应用提示：{knowledge.prompt_hint}
请提出一个全新的改进策略（2-3句话描述），融合上述跨界知识来解决当前症状。
```

### 3.5 分支 C：SOTA 知识提炼（自我进化引擎）

**触发条件**：`EvalResult.reduced_score > 历史最高分`

**行为**：
1. 正常注册进 Database（这步不变，不阻塞主循环）
2. 调用 `KnowledgeExtractor.extract_tactic(code, thought, score, domain_id)`
3. Extractor 通过 LLM 分析 SOTA 代码，自动生成一条符合 L4 schema 的战术 JSON
4. 写入 `KnowledgeBase`，持久化到本地文件
5. 后续迭代的分支 B 检索时，可以命中这条新战术

**提炼 Prompt 模板**：
```
以下代码在 {domain} 问题上取得了历史最高分 {score}：
{code}
该代码的策略思路是：{thought}
请分析这段代码为什么能取得高分，并按以下 JSON 格式提炼一条可复用的战术：
{
  "name": "战术名称",
  "applicable_symptoms": ["该战术能解决的问题症状1", "症状2"],
  "tactic_description": "2-3句话的战术描述"
}
```

---

## 第四部分：四层动态知识库 Schema（v3.0）

此结构作为种子文件 `seed_knowledge.json` 初始化知识库。系统运行时主要对 `applicable_symptoms` 字段做向量化 (Embedding) 用于语义检索。

```json
{
  "L1_Meta_Thoughts": [
    {
      "meta_id": "META_001",
      "name": "历史与动量 (Memory & Momentum)",
      "core_philosophy": "打破马尔可夫性。引入对历史状态的记忆，用过去的经验来修正当前的决策权重。",
      "applicable_symptoms": [
        "当前贪心策略导致后期无路可走",
        "算法在某个局部反复震荡",
        "每次都做出相同的错误选择，缺乏自适应性"
      ]
    },
    {
      "meta_id": "META_002",
      "name": "懒惰与延迟决策 (Lazy Evaluation & Look-ahead)",
      "core_philosophy": "不到万不得已不做不可逆的决定。先模拟、评估，再落子。",
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
      "mechanism": "记录上一次梯度的方向，同方向持续下降则加速，方向改变则减速。",
      "applicable_symptoms": [
        "算法在某个局部反复震荡",
        "缺乏对历史决策的记忆"
      ],
      "generator_prompt_hint": "在启发式评分公式中加入历史记录数组。例如：priority = greedy_score - penalty_history[item_type]。"
    },
    {
      "pattern_id": "PATT_002",
      "linked_meta_id": "META_002",
      "domain": "Operating System",
      "name": "写时复制 / 按需计算 (Copy-on-Write / Lazy Tag)",
      "mechanism": "先用 Tag 标记需要修改的区域，直到真正有请求访问时才执行计算。",
      "applicable_symptoms": [
        "过于贪婪地填满当前空间，导致后续缺乏连续大空间",
        "双层循环每次更新所有状态导致超时"
      ],
      "generator_prompt_hint": "不要每次更新所有箱子的剩余空间。引入 Lazy 机制，只在当前箱子塞不下时再查找下一个。"
    }
  ],

  "L3_Problem_Domains": [
    {
      "domain_id": "PROB_BIN_PACKING_1D",
      "name": "一维装箱问题 (1D Bin Packing)",
      "description": "将一组一维物品放入容量固定的最少数量的箱子中。",
      "base_constraints": ["物品大小不能超过箱子容量", "物品不可分割"],
      "evaluation_metric": "使用的箱子总数最少（分数为负的平均箱子数）"
    }
  ],

  "L4_Specific_Tactics": [
    {
      "tactic_id": "TAC_BIN_001",
      "linked_domain_id": "PROB_BIN_PACKING_1D",
      "linked_pattern_ids": ["PATT_002"],
      "name": "延迟大件匹配 (Deferred Large Item Assignment)",
      "applicable_symptoms": [
        "大件物品最后无法放入",
        "小件物品把箱子分割得很碎"
      ],
      "tactic_description": "遇到接近箱子容量一半的中大件物品时，延迟决策。先尝试将它们两两配对，或者与特小件物品打包成虚拟 Super Item，然后再寻找箱子放入。",
      "provenance": {
        "score_improvement": "Seed",
        "author": "Human_Expert"
      }
    }
  ]
}
```

### 检索策略

1. **L4 精准匹配**：硬过滤 `linked_domain_id` 相符的节点 → 对 `applicable_symptoms` 做向量相似度 → 取 Top-1
2. **L2/L1 扩展检索**：当 L4 无匹配（或相似度 < 阈值）时，在全局 L2 和 L1 节点中检索，引入跨界灵感
3. **Embedding 方案**：轻量级使用 `sentence-transformers`（如 `all-MiniLM-L6-v2`）本地模型，或调用 API 的 Embedding 接口。知识库规模小（几十到几百条），直接用 numpy 余弦相似度即可，不需要向量数据库

---

## 第五部分：数据结构设计

### 5.1 Function（扩展 `code_manipulation.py`）

```python
@dataclasses.dataclass
class Function:
    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    thought: str | None = None           # 新增：产生此代码的策略描述
    score: int | None = None
    global_sample_nums: int | None = None
    sample_time: float | None = None
    evaluate_time: float | None = None
```

### 5.2 EvalResult（新增，`evaluator.py`）

```python
@dataclasses.dataclass
class EvalResult:
    function: code_manipulation.Function | None  # 解析后的函数对象
    program: str                                  # 完整可执行程序
    scores_per_test: dict                         # 各测试用例的分数
    reduced_score: float | None                   # 汇总分数
    error_trace: str | None                       # 执行异常时的错误信息
    is_valid: bool                                # 是否成功执行并得到有效分数
    registered: bool                              # 是否已注册进 database
```

### 5.3 TriageResult（新增，`reflector.py`）

```python
@dataclasses.dataclass
class TriageResult:
    branch: str              # "local_bug" | "global_drawback" | "new_sota" | "normal"
    error_trace: str | None  # 分支A时有值
    symptom: str | None      # 分支B时，由 diagnose_symptom() 填充
```

---

## 第六部分：代码结构与文件清单

### 6.1 目录结构

```
implementation/
├── funsearch.py              [大改] 主入口，初始化新组件并传给 Sampler
├── sampler.py                [大改] LLM 基类增加两步调用接口；Sampler 主循环加入分诊分支
├── evaluator.py              [改]   analyse() 返回 EvalResult；Sandbox 异常时传回 error message
├── programs_database.py      [小改] _generate_prompt() 附带 thought 信息
├── code_manipulation.py      [小改] Function 增加 thought 字段
├── config.py                 [改]   新增 ReflectorConfig、KnowledgeBaseConfig
├── profile.py                [小改] 记录 thought、诊断类型等
├── evaluator_accelerate.py   [不动]
│
├── reflector.py              [新建] Reflector 类：triage() + diagnose_symptom()
├── knowledge_base.py         [新建] KnowledgeBase 类：四层知识存储与语义检索
├── knowledge_extractor.py    [新建] KnowledgeExtractor 类：SOTA 代码 → L4 战术提炼
└── seed_knowledge.json       [新建] 知识库种子数据
```

### 6.2 各模块接口定义

#### `reflector.py`

```python
class Reflector:
    def __init__(self, llm_class, config: ReflectorConfig):
        """复用同一个 LLM 接口做诊断"""

    def triage(self, eval_result: EvalResult, best_score: float) -> TriageResult:
        """
        分诊台（纯规则判断，不调 LLM）：
        - error_trace 非空 → local_bug
        - reduced_score > best_score → new_sota
        - 其余 → global_drawback
        """

    def diagnose_symptom(self, code: str, score: float, thought: str) -> str:
        """
        专科诊断（调 LLM，仅分支B触发）：
        输入失败代码和分数，输出 1-2 句抽象症状描述
        """
```

#### `knowledge_base.py`

```python
class KnowledgeBase:
    def __init__(self, config: KnowledgeBaseConfig):
        """加载 seed JSON，初始化 embedding 模型，构建向量索引"""

    def search(self, symptom: str, domain_id: str) -> dict | None:
        """
        分级检索：
        1. L4 硬过滤 domain_id → 向量匹配 applicable_symptoms → Top-1
        2. L4 无匹配 → L2/L1 全局搜索
        返回匹配到的知识节点 dict，或 None
        """

    def add_tactic(self, tactic: dict) -> None:
        """分支C触发时，写入新的 L4 战术并持久化"""
```

#### `knowledge_extractor.py`

```python
class KnowledgeExtractor:
    def __init__(self, llm_class):
        """复用 LLM 接口"""

    def extract_tactic(self, code: str, thought: str, score: float, domain_id: str) -> dict:
        """
        分析 SOTA 代码，通过 LLM 生成符合 L4 schema 的战术 JSON
        """
```

#### `sampler.py`（LLM 基类扩展）

```python
class LLM(ABC):
    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """原有方法，保留用于兼容"""

    def generate_thought(self, context: str) -> str:
        """第一步：生成/进化策略思想（纯自然语言）"""

    def generate_code_from_thought(self, thought: str, function_header: str) -> str:
        """第二步：根据思想生成纯 Python 代码"""

    def _call_api(self, prompt: str) -> str:
        """底层 LLM API 调用（从 _draw_sample 中提取复用）"""
```

#### `sampler.py`（Sampler 主循环改造）

```python
class Sampler:
    def __init__(self, database, evaluators, samples_per_prompt,
                 max_sample_nums, llm_class,
                 reflector, knowledge_base, extractor):  # 新增三个组件
        ...

    def sample(self, **kwargs):
        while not self._should_stop():
            prompt = self._database.get_prompt()

            for _ in range(self._samples_per_prompt):
                # 路径0：两步生成
                thought = self._llm.generate_thought(思想进化上下文)
                code = self._llm.generate_code_from_thought(thought, 函数签名)
                eval_result = evaluator.analyse(code, ...)

                # 分诊
                triage = self._reflector.triage(eval_result, best_score)

                if triage.branch == "local_bug":
                    self._handle_local_fix(thought, eval_result, ...)
                elif triage.branch == "global_drawback":
                    self._handle_global_diagnosis(eval_result, ...)
                elif triage.branch == "new_sota":
                    self._handle_sota_extraction(eval_result, ...)

    def _handle_local_fix(self, thought, eval_result, ...):
        """分支A：保留 thought，重试代码生成（最多 max_fix_attempts 次）"""

    def _handle_global_diagnosis(self, eval_result, ...):
        """分支B：诊断症状 → RAG 检索 → 定向生成新 thought + code"""

    def _handle_sota_extraction(self, eval_result, ...):
        """分支C：提炼知识写入 L4"""
```

#### `config.py`（新增配置）

```python
@dataclasses.dataclass(frozen=True)
class ReflectorConfig:
    enable_reflection: bool = True
    max_fix_attempts: int = 2          # 分支A局部修复最多重试次数
    stagnation_rounds: int = 5         # 连续多少轮不提升视为停滞（可选，辅助分支B判断）

@dataclasses.dataclass(frozen=True)
class KnowledgeBaseConfig:
    seed_path: str = "seed_knowledge.json"
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.6  # L4 匹配阈值，低于此值升级到 L2/L1 检索
    persist_path: str = "knowledge.json"  # 运行时知识库持久化路径
```

---

## 第七部分：开发计划

### 优先级排序

| 优先级 | 任务 | 改动文件 | 难度 | 说明 |
|---|---|---|---|---|
| **P0** | LLMAPI 重构：提取 `_call_api()`，实现 `generate_thought()` + `generate_code_from_thought()` | notebook / 入口文件 | 低 | 后续所有模块都依赖两步调用能跑通 |
| **P0** | 改造 `evaluator.py`：`analyse()` 返回 `EvalResult`，Sandbox 异常时带回 error message | evaluator.py | 低 | Reflector 的输入来源 |
| **P1** | 新建 `reflector.py` | reflector.py | 中 | triage 是纯规则；diagnose_symptom 是一次 LLM 调用 |
| **P1** | 新建 `knowledge_base.py` + `seed_knowledge.json` | knowledge_base.py | 中 | JSON 读写 + embedding + 余弦相似度 |
| **P2** | 改造 `sampler.py` 主循环：接入 Reflector/KB/Extractor，实现三条分支 | sampler.py | 高 | 最复杂的整合工作 |
| **P2** | 新建 `knowledge_extractor.py` | knowledge_extractor.py | 低 | 本质是一次结构化 LLM 调用 |
| **P2** | `code_manipulation.py` 加 thought 字段 | code_manipulation.py | 低 | 加一个 dataclass 字段 |
| **P2** | `programs_database.py` prompt 生成附带 thought | programs_database.py | 中 | 改 `_generate_prompt()` |
| **P3** | `config.py` 增加新配置项 | config.py | 低 | 加两个 dataclass |
| **P3** | `funsearch.py` 初始化新组件并传给 Sampler | funsearch.py | 低 | 接线工作 |
| **P3** | `profile.py` 记录 thought、诊断类型 | profile.py | 低 | 扩展日志字段 |

### 建议实施顺序

```
第一阶段（基础设施）:  P0 任务
  → 验证两步 LLM 调用能正常工作
  → 验证 EvalResult 能正确返回错误信息

第二阶段（核心能力）:  P1 任务
  → Reflector 能正确分诊
  → KnowledgeBase 能正确检索

第三阶段（整合联调）:  P2 任务
  → 主循环跑通完整飞轮
  → 端到端测试

第四阶段（完善）:      P3 任务
  → 配置、日志、初始化等收尾
```

---

## 第八部分：与原版 FunSearch 的兼容性

### 设计原则

1. **所有新增功能可通过配置开关关闭**：设置 `enable_reflection=False` 时，系统退化为原版 FunSearch + thought 两步调用
2. **Database 接口不变**：`register_program()` 的调用方式保持一致，新增的 thought 字段是 optional 的
3. **原版 `draw_samples()` 保留**：`LLM` 基类的原方法保留用于兼容，新增方法是扩展而非替换
4. **Evaluator 向后兼容**：`EvalResult` 是新增的返回包装，如果不需要，`analyse()` 内部仍然可以执行原有的注册逻辑
