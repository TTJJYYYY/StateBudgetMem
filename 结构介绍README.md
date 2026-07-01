# StateBudgetMem

面向资源受限端侧智能体的长期记忆管理研究项目。

本仓库整合了 MemoryBank 基线、TF-IDF 受控基线、状态版本管理、查询路由、统一评测接口与可视化组件。下面只说明项目文件结构及各模块职责。

## 目录结构

```text
StateBudgetMem/
├── configs/                 # 实验与模块配置
├── data/                    # 受控数据和外部数据集
├── docs/                    # 研究设计、接口和迁移说明
├── examples/                # 最小可运行示例
├── results/                 # 实验输出与历史结果
├── tools/                   # 调试、评测和批量运行工具
├── tests/                   # 自动测试，结构与 src 对应
├── src/
│   └── statebudgetmem/      # 正式 Python 包
├── README.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## 正式代码

```text
src/statebudgetmem/
├── interfaces.py
├── cli.py
├── core/
├── schemas/
├── data/
├── preprocessing/
├── baselines/
│   ├── memorybank/
│   └── tfidf/
├── versioning/
├── routing/
├── views/
├── retrieval/
├── evaluation/
└── apps/
```

### `interfaces.py`

公共接口统一入口。

其他模块应优先从这里导入公共类型：

```python
from statebudgetmem.interfaces import (
    MemoryPiece,
    MemorySystem,
    MemoryRecord,
    QueryRecord,
    MemoryMethod,
    MethodResult,
    QueryType,
    ViewType,
    UpdateOperation,
)
```

不要在各模块中重复定义这些公共类型。

### `core/`

保存项目的基础协议。

```text
core/
├── online.py       # 在线记忆系统接口
└── method.py       # 受控实验方法接口
```

其中：

- `MemoryPiece`、`MemorySystem` 面向实际运行的记忆系统；
- `MemoryRecord`、`QueryRecord`、`MemoryMethod`、`MethodResult` 面向统一实验与评测。

### `schemas/`

保存受控数据和实验结果的数据结构，包括：

- 记忆记录；
- 查询记录；
- 场景定义；
- 检索结果；
- 查询类型；
- 状态标注。

### `preprocessing/`

负责把原始文本转换为结构化记忆。

```text
preprocessing/
├── api_parser.py
├── rule_parser.py
├── normalizer.py
├── models.py
└── pipeline.py
```

### `baselines/memorybank/`

MemoryBank 基线的完整实现。

```text
memorybank/
├── system.py       # MemoryBank 和 TF-IDF MemoryBank 后端
├── agents.py       # 无记忆和带记忆 Agent
├── datasets.py     # 内置历史、探针问题和 Memora 数据读取
├── evaluator.py    # 回答对比、批量评测和结果导出
├── staleness.py    # 过期记忆分析
├── demo.py         # Gradio 对比展示
└── README.md
```

MemoryBank 的系统、评测、分析和 Demo 都集中在该目录中。

### `baselines/tfidf/`

确定性 TF-IDF 受控基线。

```text
tfidf/
├── retriever.py    # TF-IDF + Cosine Top-K 检索
├── adapter.py      # 接入统一实验接口
├── runner.py       # 运行受控实验
└── README.md
```

### `versioning/`

状态版本管理模块。

```text
versioning/
├── engine.py
├── graph.py
├── matcher.py
├── classifier.py
├── resolver.py
├── updater.py
├── validator.py
├── adapters.py
├── models.py
├── operations.py
└── exceptions.py
```

该模块负责状态匹配、更新关系判断、版本图维护和一致性校验。

支持的主要操作包括：

```text
ADD
MERGE
SUPERSEDE
TEMP_INVALIDATE
RESTORE
DELETE
NOOP
```

### `routing/`

查询分类与视图路由模块。

```text
routing/
├── router.py
├── models.py
├── prompts.py
├── config.yaml
└── README.md
```

主要查询类型为：

```text
CURRENT
HISTORICAL
CHANGE
GENERAL
```

对应视图关系为：

```text
CURRENT     → 当前状态视图
HISTORICAL  → 历史版本视图
CHANGE      → 当前与历史视图
GENERAL     → 不读取个人记忆
```

### `views/`

当前状态视图与历史版本视图模块。

该目录用于保存：

- Current View；
- History View；
- 版本更新后的视图同步；
- 视图查询接口。

### `retrieval/`

公共检索与排序模块。

该目录用于保存：

- 视图内检索；
- 时间过滤；
- 过期风险排序；
- Token 预算选择；
- 可复用检索协议。

TF-IDF 专属实现仍保留在 `baselines/tfidf/` 中。

### `evaluation/`

全项目共享的评价指标与实验工具。

主要用于计算：

- Recall@K；
- Valid Recall@K；
- Stale Retrieval Rate；
- Token 成本；
- 检索延迟。

MemoryBank 专属评测保留在 `baselines/memorybank/` 中。

### `apps/`

完整 StateBudgetMem 可视化应用入口。

MemoryBank 基线 Demo 位于：

```text
baselines/memorybank/demo.py
```

最终系统 Demo 放在 `apps/` 中，用于展示版本链、双视图、路由结果、检索候选和资源消耗。

### `cli.py`

项目命令行入口，负责统一调用实验、路由、评测和分析功能。

## 数据目录

```text
data/
├── controlled/
│   ├── baseline_scenarios.jsonl
│   └── temporal_challenge_v1.jsonl
└── external/
    └── memora/
```

- `controlled/`：项目自建受控数据，供所有方法共同使用；
- `external/memora/`：外部 Memora 数据集放置位置。

## 测试目录

```text
tests/
├── baselines/
│   ├── memorybank/
│   └── tfidf/
├── routing/
├── versioning/
├── schemas/
├── evaluation/
└── integration/
```

测试目录与正式代码目录对应。

例如：

```text
src/statebudgetmem/routing/router.py
tests/routing/test_router.py
```

跨模块测试统一放在：

```text
tests/integration/
```

## 工具目录

```text
tools/
├── memorybank/
│   ├── run_evaluation.py
│   └── analyze_staleness.py
└── routing/
    ├── debug_routing.py
    └── run_real_routing.py
```

`tools/` 只保存运行入口和调试脚本，可复用逻辑仍放在 `src/statebudgetmem/` 中。

## 示例目录

```text
examples/
├── memorybank_quickstart.py
└── tfidf_controlled_baseline.py
```

用于快速查看各基线的最小调用方式。

## 结果目录

```text
results/
├── raw/
├── summaries/
└── memorybank/
```

- `raw/`：逐查询原始结果；
- `summaries/`：汇总指标；
- `memorybank/`：MemoryBank 相关历史分析结果。

## 文档目录

```text
docs/
```

用于保存研究方案、架构说明、接口设计、实验计划和历史迁移记录。根目录 README 只负责说明仓库结构。
