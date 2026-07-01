# AGENTS.md

## 1. 项目名称

**StateBudgetMem：资源受限端侧个人智能体的时态一致性长期记忆管理**

本项目是一个研究型项目，目标不是开发通用聊天应用，而是研究：

- 动态用户状态下的长期记忆表示；
- 新旧状态的更新、替代、临时失效与恢复；
- 当前状态与历史版本的分离管理；
- 查询相关的记忆有效性判断；
- Token、存储、延迟和内存受限条件下的记忆检索；
- 过期记忆误用的评测与降低。

---

## 2. 开始工作前必须阅读

执行任何任务前，先阅读：

1. `docs/StateBudgetMem_research_plan.md`
<<<<<<< HEAD
2. `docs/ARCHITECTURE.md` 与 `docs/TEAM_WORKFLOW.md`
3. 当前任务说明文件，例如 `docs/CODEX_TASK_XXX.md`
4. 相关模块的现有代码、测试和配置
5. `README.md` 中的运行方式与目录约定
=======
2. 当前任务说明文件，例如 `docs/CODEX_TASK_XXX.md`
3. 相关模块的现有代码、测试和配置
4. `README.md` 中的运行方式与目录约定
>>>>>>> ba900d42c9450c7df9e9737f2bedadadbdce7427

其中：

- `docs/StateBudgetMem_research_plan.md` 是研究目标和研究路线的最高依据；
- 当前任务说明文件是本次开发范围和验收标准的最高依据；
- 不得自行修改研究问题、实验指标或对照方法。

---

## 3. 研究优先级

所有设计和实现按以下优先级排序：

1. 当前有效状态判断正确；
2. 历史版本能够完整保留和查询；
3. 记忆有效性与当前查询相关；
4. 减少过期记忆的检索和使用；
5. 实验可复现；
6. Token、存储、延迟和内存开销可测量；
7. 系统结构清晰、模块可替换；
8. 展示界面和工程美化。

如发生冲突，优先保证研究正确性和实验可复现性，不优先追求界面、功能数量或代码复杂度。

---

## 4. 明确不做什么

除非任务文件明确要求，否则不要实现：

- 通用聊天机器人；
- 复杂前端；
- 用户登录、账号体系和云端后台；
- 与研究无关的推荐、搜索或社交功能；
- 大规模模型训练；
- 自行修改论文研究主线；
- 未经验证的复杂知识图谱；
- 必须依赖付费 API 才能运行的基础流程；
- 无法复现的“演示型结果”。

---

## 5. 技术与开发约定

### 5.1 基础环境

- Python 版本：3.11 或更高；
- 测试框架：`pytest`；
- 配置文件：优先使用 YAML；
- 数据格式：优先使用 JSON、JSONL、CSV；
- 数据模型：优先使用 `pydantic`；
- 基础检索：可使用 `scikit-learn`；
- 数据处理：可使用 `pandas` 和 `numpy`；
- 绘图：优先使用 `matplotlib`；
- 命令行：使用 `argparse`、`typer` 或等价轻量方案。

### 5.2 依赖原则

- 第一版必须可以离线运行；
- 单元测试不得依赖外部 API；
- LLM、Embedding 和向量数据库必须通过可替换接口接入；
- 不要在基础版本中引入不必要的重型框架；
- 新增依赖前，先说明其必要性；
- 依赖版本必须写入 `pyproject.toml` 或锁定文件。

---

## 6. 推荐目录结构

```text
StateBudgetMem/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
├── data/
│   ├── controlled/
<<<<<<< HEAD
│   └── external/
├── docs/
├── examples/
├── tools/
│   ├── memorybank/
│   └── routing/
├── src/statebudgetmem/
│   ├── interfaces.py
│   ├── core/
│   ├── schemas/
│   ├── data/
│   ├── preprocessing/
│   ├── baselines/
│   │   ├── memorybank/
│   │   └── tfidf/
│   ├── versioning/
│   ├── views/
│   ├── routing/
│   ├── retrieval/
│   ├── evaluation/
│   ├── apps/
│   └── cli.py
├── tests/                 # mirror src module ownership
└── results/
```

按方法聚合专属代码：MemoryBank 的系统、Agent、数据适配、评测、过期分析和 Demo 都放在 `baselines/memorybank/`；TF-IDF 的检索、适配器和实验运行器都放在 `baselines/tfidf/`。只把真正跨方法复用的协议和指标放进 `core/`、`retrieval/`、`evaluation/`。

=======
│   ├── public/
│   └── processed/
├── docs/
├── src/
│   └── statebudgetmem/
│       ├── schemas/
│       ├── data/
│       ├── extraction/
│       ├── memory_store/
│       ├── versioning/
│       ├── views/
│       ├── routing/
│       ├── retrieval/
│       ├── generation/
│       ├── evaluation/
│       ├── experiments/
│       └── cli.py
├── tests/
├── results/
│   ├── raw/
│   ├── summaries/
│   └── figures/
└── scripts/
```

>>>>>>> ba900d42c9450c7df9e9737f2bedadadbdce7427
不要随意创建重复目录。新增目录或模块前，先检查现有结构。

---

## 7. 核心接口

模块之间必须通过小型、清晰、可替换的接口连接。

建议至少保留以下接口：

```python
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class MemoryStore(Protocol):
    def add(self, memory: "MemoryRecord") -> None:
        ...

    def list(self, subject: str | None = None) -> list["MemoryRecord"]:
        ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: "QueryRecord",
        memories: list["MemoryRecord"],
        top_k: int,
    ) -> list["RetrievedMemory"]:
        ...


class VersionManager(Protocol):
    def apply(
        self,
        new_memory: "MemoryRecord",
        existing_memories: list["MemoryRecord"],
    ) -> "VersionUpdateResult":
        ...


class QueryRouter(Protocol):
    def classify(self, query: "QueryRecord") -> "QueryType":
        ...


class Evaluator(Protocol):
    def evaluate(self, prediction, reference) -> dict[str, float]:
        ...
```

不要让某个模块直接依赖具体模型、数据库或外部 API。

---

## 8. 核心数据模型

### 8.1 MemoryRecord

至少包含：

- `memory_id`
- `subject`
- `attribute`
- `value`
- `text`
- `event_time`
- `valid_from`
- `valid_to`
- `status`
- `memory_type`
- `importance`
- `confidence`
- `token_cost`
- `supersedes`
- `temporarily_invalidates`
- `metadata`

### 8.2 QueryRecord

至少包含：

- `query_id`
- `text`
- `query_type`
- `reference_time`
- `gold_relevant_memory_ids`
- `gold_valid_memory_ids`
- `gold_stale_memory_ids`

### 8.3 查询类型

使用显式枚举：

```text
CURRENT
HISTORICAL
CHANGE
GENERAL
```

### 8.4 更新操作

使用显式枚举：

```text
ADD
MERGE
SUPERSEDE
TEMP_INVALIDATE
RESTORE
DELETE
NOOP
```

不要用模糊字符串代替固定枚举。

---

## 9. 实验要求

### 9.1 公平比较

所有对照方法必须尽量使用：

- 相同数据；
- 相同生成模型；
- 相同嵌入模型；
- 相同 Token 预算；
- 相同 Top-K；
- 相同随机种子；
- 相同生成参数；
- 相同硬件或清晰记录硬件差异。

### 9.2 每次实验必须记录

- 方法名称；
- 配置文件；
- 数据集名称和版本；
- 随机种子；
- Top-K；
- Token 预算；
- 存储预算；
- 模型名称；
- 运行时间；
- 输出文件路径；
- Git 提交版本（如可获得）；
- 硬件信息（端侧实验必须记录）。

### 9.3 结果目录

```text
results/
├── raw/          # 原始逐样本结果
├── summaries/    # 汇总指标
└── figures/      # 自动生成的图表
```

不得只把结果打印在终端中。

### 9.4 不得伪造结果

- 不得编造实验数字；
- 不得手工修改结果文件以迎合预期；
- 不得只报告最优运行而隐藏失败运行；
- 未实际执行命令时，不得声称测试或实验已经通过；
- 如果实验失败，必须保存错误日志并说明原因。

---

## 10. 核心评价指标

优先实现和保存以下指标：

### 检索指标

- Recall@K
- Valid Recall@K
- Stale Retrieval Rate
- 平均检索 Token 成本
- 检索延迟

### 回答指标

后续任务中可加入：

- Overall Answer Accuracy
- Current-State Accuracy
- Historical Accuracy
- Change-Reasoning Accuracy
- Stale Usage Rate
- Abstention Accuracy
- FAMA

### 资源指标

- 本地数据库大小；
- 向量索引大小；
- 写入延迟；
- 更新延迟；
- 检索延迟；
- 峰值内存；
- 平均 Token 使用量。

---

## 11. 开发流程

执行每个任务时按以下顺序进行：

1. 阅读研究计划和当前任务文件；
2. 检查现有仓库；
3. 写一份简短实施计划；
4. 只实现当前任务范围；
5. 编写或更新测试；
6. 运行测试；
7. 运行任务规定的复现命令；
8. 保存机器可读结果；
9. 更新 README 或任务文档；
10. 汇报修改内容、命令、结果和限制。

如果当前任务依赖尚未完成的模块，不要自行扩大范围。先实现最小接口或停止并报告依赖问题。

---

## 12. 测试要求

每个功能至少覆盖：

- 正常输入；
- 边界情况；
- 非法输入；
- 空数据；
- 确定性；
- 已知失败模式。

涉及时间和版本关系时，必须测试：

- 永久替代；
- 临时失效；
- 状态恢复；
- 历史查询；
- 当前查询；
- 同一属性多次更新；
- 无法确定当前状态；
- 时间相同或缺失时的处理。

完成任务前至少运行：

```bash
pytest -q
python -m statebudgetmem.cli --help
```

如果任务文件定义了额外命令，也必须执行。

---

## 13. 代码质量要求

- 使用类型注解；
- 函数和类保持单一职责；
- 研究公式和指标必须写清楚定义；
- 复杂逻辑必须有注释；
- 避免重复代码；
- 避免全局可变状态；
- 随机过程必须可设置种子；
- 文件路径不得硬编码为个人电脑路径；
- 错误信息必须明确；
- 不要吞掉异常；
- 不要为了“代码更高级”而过度设计。

---

## 14. Git 与任务边界

- 每次任务只做一个明确目标；
- 不要把无关重构混入研究功能；
- 不要随意重命名研究术语；
- 不要同时改动数据 Schema、检索接口和实验格式，除非任务明确要求；
- 修改公共接口时，必须同步更新测试和文档；
- 不要删除原始实验结果；
- 不要覆盖别人未提交的工作。

建议提交信息格式：

```text
feat: add version-aware memory updates
test: add temporal transition cases
docs: document baseline experiment
fix: correct stale retrieval metric
```

---

## 15. 完成标准

一个任务只有同时满足以下条件才算完成：

```text
任务说明
+ 实现代码
+ 自动测试
+ 可复现命令
+ 机器可读结果
+ 文档更新
+ 限制说明
```

仅有代码、仅有截图、仅有终端输出或仅口头声称“完成”，都不算完成。

---

## 16. 每次任务结束时的汇报格式

请按以下格式汇报：

```text
## 已完成

- 完成了什么
- 满足了哪些验收条件

## 修改文件

- path/to/file_1
- path/to/file_2

## 执行命令

- pytest -q
- python -m statebudgetmem.cli ...

## 测试与运行结果

- 通过多少项测试
- 生成了哪些结果文件
- 是否出现警告或失败

## 假设与限制

- 当前实现依赖哪些假设
- 哪些能力尚未实现
- 下一步建议是什么
```

不得只回复“已完成”或“代码已经准备好”。

---

## 17. 当前项目推进顺序

除非任务文件另有说明，项目按以下顺序推进：

1. 确定性离线基线；
2. 状态版本管理；
3. 当前状态与历史版本双视图；
4. 查询类型路由；
5. 过期风险感知检索；
6. Token 和存储预算；
7. 主实验和消融实验；
8. LLM 与 Embedding 适配；
9. 公开数据集；
10. 端侧资源测试；
11. 展示系统。

在核心实验完成前，不要优先开发展示界面。
