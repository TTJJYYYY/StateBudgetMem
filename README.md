# StateBudgetMem

**资源受限端侧个人智能体的时态一致性长期记忆管理**  
<<<<<<< HEAD
*Temporally Consistent Long-Term Memory for Resource-Constrained On-Device Agents*
=======
*Temporally Consistent Long-Term Memory for Resource-Constrained On-Device Personal Agents*

## 项目简介

长期运行的个人智能体会持续积累用户偏好、健康状态、居住地点和任务目标等信息。当这些信息发生变化时，普通语义检索可能同时取回当前状态和已经过期的历史记忆，从而导致错误回答。

本项目研究：

- 如何表示用户状态的更新、替代和临时失效；
- 如何区分当前有效状态与历史版本；
- 如何根据问题类型选择正确的记忆；
- 如何在有限存储、token 和检索延迟下完成记忆管理。

项目当前处于**研究工程与基础基线阶段**。现已完成确定性的离线 TF-IDF 检索基线和统一方法接口 v0.1；MemoryBank、版本管理、双视图和查询路由仍在后续开发中。

---

## 当前已完成

- `MemoryRecord`、`QueryRecord`、`Scenario` 公共数据结构；
- 受控状态变化数据；
- TF-IDF + Cosine Similarity Top-K 检索；
- Recall@K、Valid Recall@K、Stale Retrieval Rate；
- 检索 token 成本与检索延迟统计；
- CLI 实验入口与 JSON/CSV 结果保存；
- 统一方法接口：`reset()`、`ingest()`、`retrieve()`；
- 统一输出结构：`RetrievedMemory`、`MethodResult`；
- TF-IDF Adapter 与自动化测试。

当前实现不代表完整的 StateBudgetMem 方法，也不包含正式的 LLM 回答生成、MemoryBank 接入、版本推断、双视图路由或预算优化。

---

## 四条研究路线

项目第一阶段按照四个模块并行开发。四个文件夹对应四条研究路线，但最终会组合成同一个系统。

### `baselines/`：基础记忆系统与对照基线

负责：

- 维护现有 TF-IDF 基线；
- 整理并接入 MemoryBank；
- 实现语义相关性、时间衰减、重要性和记忆强化等基础机制；
- 在相同数据和指标下比较 TF-IDF 与 MemoryBank。

目的：建立可靠对照组，观察普通语义检索和传统长期记忆方法是否会取回过期记忆。

主要参考：MemoryBank、Generative Agents、Mem0。

### `versioning/`：状态版本管理

负责：

- 判断新旧记忆之间的关系；
- 实现 `ADD`、`MERGE`、`SUPERSEDE`、`TEMP_INVALIDATE`、`DELETE`、`NOOP`；
- 维护有效时间、当前状态和历史版本；
- 构建用户状态变化链。

目的：让系统知道哪条记忆当前有效，哪条已经被替代、失效或暂时覆盖。

主要参考：A-MEM、Mem0、STALE、MemConflict。

### `views/`：当前状态与历史版本双视图

负责：

- 构建 `Current View`，保存当前有效状态；
- 构建 `History View`，保存完整历史版本；
- 维护和检索两个视图；
- 比较统一记忆库、仅当前状态和双视图三种方案。

目的：减少旧记忆对当前问题的干扰，同时保留历史状态与变化过程的查询能力。

主要参考：LongMemEval、MemConflict、A-MEM。

### `routing/`：查询分类与记忆路由

负责：

- 分类 `CURRENT`、`HISTORICAL`、`CHANGE`、`GENERAL`；
- 根据查询类型选择当前视图、历史视图、两个视图或不检索个人记忆；
- 判断记忆对当前问题是否适用；
- 评测查询分类准确率和不必要检索率。

目的：把版本管理和双视图串成完整的时态一致性检索流程。

主要参考：MemConflict、LongMemEval、STALE、Memora。

四条路线的组合关系为：

```text
基础记忆与基线
        ↓
状态版本管理
        ↓
当前 / 历史双视图
        ↓
查询分类与路由
        ↓
完整 StateBudgetMem
```

---

## 统一开发接口

四个模块可以采用不同算法，但必须共用统一的数据、调用方式和结果格式。

### 公共输入

- `MemoryRecord`：一条记忆；
- `QueryRecord`：一个问题；
- `Scenario`：一组记忆及其对应问题。

禁止在各模块中重新定义另一套 Memory、Query 或 Scenario。

### 公共调用方式

所有可比较方法最终需要支持：

```python
method.reset()
method.ingest(memories)
result = method.retrieve(
    query,
    top_k=3,
    token_budget=None,
    mutate=False,
)
```

### 公共输出

所有方法返回 `MethodResult`，其中包含：

- 检索到的记忆 ID；
- 分数和排名；
- 记忆来源视图；
- 可选的查询类型预测；
- 总 token 成本；
- 检索延迟。

完整规范见 [`docs/UNIFIED_SPEC.md`](docs/UNIFIED_SPEC.md)。
>>>>>>> ba900d42c9450c7df9e9737f2bedadadbdce7427

StateBudgetMem studies how a long-running personal agent can keep the current
user state correct, preserve historical versions, avoid stale-memory misuse,
and retrieve useful memories under storage and context-token budgets.

This repository is the cleaned, structured union of the former `main`,
`routing`, and `feature/tfidf-baseline-framework` branches. The original demo,
MemoryBank baseline, controlled datasets, previous results, routing code, and
versioning implementation are preserved; duplicate v1/v2 files and local IDE
artifacts are not.

<<<<<<< HEAD
## Current status

Implemented:

- deterministic TF-IDF controlled baseline and 44 controlled scenarios;
- MemoryBank/FAISS baseline, lightweight agents, Memora adapters, answer
  evaluation, stale-memory analysis, and full Gradio comparison demo;
- structured memory preprocessing;
- state-versioning engine and tests;
- rule-based and LLM query routing;
- shared schemas, interfaces, metrics, CLI, and preserved experiment outputs.

Next major stage:

- Current View and History View;
- a unified end-to-end pipeline;
- budget-aware selection and final StateBudgetMem visualization.

## Collaboration-oriented structure

```text
StateBudgetMem/
├── configs/                         # reproducible experiment configuration
├── data/
│   ├── controlled/                  # 12 baseline + 32 temporal scenarios
│   └── external/memora/             # optional external dataset instructions
├── docs/
│   ├── ARCHITECTURE.md              # module boundaries and data flow
│   ├── TEAM_WORKFLOW.md             # four-person collaboration rules
│   ├── MIGRATION_FROM_THREE_BRANCHES.md
│   └── baselines/MEMORYBANK_BASELINE.md
├── examples/                        # minimal public-API examples
├── tools/
│   ├── memorybank/                  # baseline-specific analysis entry points
│   └── routing/                     # prompt and real-API debugging tools
├── results/                         # preserved and newly generated outputs
├── tests/                           # mirrors the source-module structure
│   ├── baselines/memorybank/
│   ├── baselines/tfidf/
│   ├── evaluation/
│   ├── integration/
│   ├── routing/
│   ├── schemas/
│   └── versioning/
└── src/statebudgetmem/
    ├── interfaces.py                # single public contract facade
    ├── core/                        # shared online/experiment protocols
    ├── schemas/                     # MemoryRecord, QueryRecord, MethodResult
    ├── data/                        # controlled-data loading
    ├── preprocessing/               # dialogue → structured memory
    ├── baselines/
    │   ├── memorybank/              # system, agents, data, eval, staleness, demo
    │   └── tfidf/                   # retriever, adapter, controlled runner
    ├── versioning/                  # matching, operations, graph, resolver
    ├── views/                       # Current/History views — next stage
    ├── routing/                     # rule and LLM routers
    ├── retrieval/                   # shared Retriever/Embedder protocols
    ├── evaluation/                  # method-independent retrieval metrics
    ├── apps/                        # reserved for the final system demo
    └── cli.py
```

The key rule is: **method-specific code stays together; shared contracts and
metrics stay method-independent**. Tests mirror the source path, so a module and
its tests are easy to locate.
=======
```text
StateBudgetMem/
├── configs/                         # 实验配置
├── data/controlled/                 # 自建受控数据
├── docs/                            # 研究路线、任务文档和统一规范
├── results/                         # 原始结果与汇总结果
├── src/statebudgetmem/
│   ├── baselines/                   # 路线一：基础基线与 MemoryBank
│   ├── versioning/                  # 路线二：状态版本管理
│   ├── views/                       # 路线三：当前 / 历史双视图
│   ├── routing/                     # 路线四：查询分类与路由
│   ├── core/                        # 公共方法接口
│   ├── schemas/                     # 公共数据与结果结构
│   ├── retrieval/                   # 当前 TF-IDF 检索实现
│   ├── evaluation/                  # 公共评测指标
│   ├── experiments/                 # 实验流程
│   └── cli.py                       # 命令行入口
├── tests/                           # 自动化测试
├── AGENTS.md                        # Codex 开发约束
├── pyproject.toml
└── README.md
```
>>>>>>> ba900d42c9450c7df9e9737f2bedadadbdce7427

## Shared interfaces

<<<<<<< HEAD
All modules use one public import path:

```python
from statebudgetmem.interfaces import (
    # Online memory-system layer
    MemoryPiece,
    MemorySystem,
    MemoryType,
    MemoryStatus,
    UpdateOperation,
    VersionManager,
    ViewManager,
    QueryRouter,
    ViewType,

    # Controlled-experiment layer
    MemoryRecord,
    QueryRecord,
    MemoryMethod,
    MethodResult,
    QueryType,
    RetrievedMemory,
    Scenario,
)
```

`MemoryPiece` / `MemorySystem` describe a live memory backend such as
MemoryBank. `MemoryRecord` / `QueryRecord` / `MemoryMethod` / `MethodResult`
describe reproducible controlled experiments. These layers are related but not
duplicates. Do not create private copies of these types inside a feature module.

## Install and verify

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[test]"
pytest -q
```

Optional components:

```bash
python -m pip install -e ".[memorybank]"  # FAISS + embeddings
python -m pip install -e ".[llm]"         # OpenAI-compatible APIs
python -m pip install -e ".[demo]"        # MemoryBank + Gradio
```

## Main commands
=======
## 数据集

当前仓库包含两套受控数据：

```text
data/controlled/baseline_scenarios.jsonl
data/controlled/temporal_challenge_v1.jsonl
```

- `baseline_scenarios.jsonl`：用于快速验证数据加载、检索、指标和结果输出；
- `temporal_challenge_v1.jsonl`：用于测试状态更新、历史版本、过期记忆和时态问题。

公开数据集 Memora 暂时作为本地外部数据保存，待四个模块初步合并后再统一适配和评测。

---

## 安装

需要 Python 3.11 或更高版本。

```bash
python -m pip install -e ".[test]"
```

当前基础运行依赖为 `pydantic`。TF-IDF 检索器使用 Python 标准库实现，可完全离线运行。

---

## 运行基础实验

```bash
python -m statebudgetmem.cli run --config configs/baseline.yaml
```

实验结果保存至：

```text
results/raw/<run_id>.jsonl
results/summaries/<run_id>.json
results/summaries/<run_id>.csv
```

---

## 运行测试

```bash
pytest -q
```

---

## 第一阶段对比目标

后续将逐步比较：

```text
TF-IDF
→ MemoryBank
→ Version-Aware
→ Version-Aware + Dual View
→ Version-Aware + Dual View + Query Routing
```

共同评价指标包括：

- Recall@K；
- Valid Recall@K；
- Stale Retrieval Rate；
- 总检索 token 成本；
- 检索延迟。

各模块还可增加自己的专项指标，但不得改变公共指标定义。
>>>>>>> ba900d42c9450c7df9e9737f2bedadadbdce7427

Controlled TF-IDF baseline:

<<<<<<< HEAD
```bash
statebudgetmem run --config configs/baseline.yaml
```

Query routing:

```bash
statebudgetmem route "我现在还喜欢吃辣吗？"
python tools/routing/debug_routing.py --dry-run --query "我的饮食习惯怎么变化的？"
```

MemoryBank evaluation and stale-memory analysis:

```bash
statebudgetmem evaluate-memorybank --output results/memorybank/evaluation.json
statebudgetmem analyze-staleness --backend tfidf

# Full original-style utilities:
python tools/memorybank/run_evaluation.py --output results/memorybank/evaluation.json
python tools/memorybank/analyze_staleness.py --mode demo
```

MemoryBank visual comparison:

```bash
statebudgetmem-demo
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) before changing shared
interfaces, and use [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md) for team
coordination.
=======
## 协作规则

建议四个方向分别使用功能分支：

```text
feature/memorybank-baseline
feature/state-versioning
feature/dual-view
feature/query-routing
```

公共区域包括：

```text
src/statebudgetmem/schemas/
src/statebudgetmem/core/
src/statebudgetmem/evaluation/
src/statebudgetmem/experiments/
src/statebudgetmem/cli.py
pyproject.toml
```

修改公共区域前需要先进行组内沟通。各成员应优先在自己负责的模块中开发，并通过 Pull Request 合并。

---

## 后续计划

1. 接入 MemoryBank 基线；
2. 实现状态版本更新操作；
3. 构建当前状态与历史版本双视图；
4. 实现查询分类与视图路由；
5. 统一接入 Memora 等公开数据集；
6. 加入 token / 存储预算检索；
7. 完成消融、压力测试、端侧性能测试和研究展示。
>>>>>>> ba900d42c9450c7df9e9737f2bedadadbdce7427
