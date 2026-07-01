# CODEX_TASK_001：第一阶段——确定性离线基线与实验骨架

## 1. 阶段目标

建立 StateBudgetMem 项目的最小可运行实验系统。

本阶段只解决三个问题：

1. 项目代码结构能否稳定运行；
2. 能否加载受控记忆数据并执行基础 Top-K 检索；
3. 能否自动计算“相关记忆、有效记忆和过期记忆”的检索指标。

本阶段的意义是建立后续版本管理、双视图、查询路由和预算检索共同依赖的基础接口与实验框架。

> 本阶段不追求方法创新，不生成最终论文结论。

---

## 2. 开始前必须阅读

Codex 开始工作前必须阅读：

1. `AGENTS.md`
2. `docs/StateBudgetMem_research_plan.md`
3. 本任务文件
4. 当前仓库中的 `README.md`、配置文件和已有测试

不得自行修改研究目标、核心术语和实验主线。

---

## 3. 本阶段必须完成的内容

### 3.1 创建项目骨架

建立以下目录结构：

```text
StateBudgetMem/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   └── baseline.yaml
├── data/
│   └── controlled/
│       └── baseline_scenarios.jsonl
├── docs/
├── src/
│   └── statebudgetmem/
│       ├── __init__.py
│       ├── schemas/
│       ├── data/
<<<<<<< HEAD
│       ├── baselines/
│       │   └── tfidf/
│       ├── evaluation/
=======
│       ├── retrieval/
│       ├── evaluation/
│       ├── experiments/
>>>>>>> ba900d42c9450c7df9e9737f2bedadadbdce7427
│       └── cli.py
├── tests/
└── results/
    ├── raw/
    └── summaries/
```

要求：

- 使用 Python 3.11 或更高版本；
- 使用 `pyproject.toml` 管理依赖；
- 不依赖任何外部 API；
- 不要求 API Key；
- 所有路径必须使用相对路径或配置文件；
- 不得写死个人电脑路径。

---

### 3.2 定义核心数据模型

使用 `pydantic` 或等价方案定义以下模型。

#### MemoryRecord

至少包含：

```text
memory_id
subject
attribute
value
text
event_time
valid_from
valid_to
status
memory_type
importance
confidence
token_cost
supersedes
temporarily_invalidates
metadata
```

建议枚举：

```text
status:
CURRENT
HISTORICAL
INVALIDATED
UNKNOWN
```

#### QueryRecord

至少包含：

```text
query_id
text
query_type
reference_time
gold_relevant_memory_ids
gold_valid_memory_ids
gold_stale_memory_ids
```

查询类型必须使用显式枚举：

```text
CURRENT
HISTORICAL
CHANGE
GENERAL
```

#### Scenario

至少包含：

```text
scenario_id
description
memories
queries
```

要求：

- 时间字段使用明确格式；
- 空字段必须允许显式为空；
- 非法枚举、重复 ID 和错误时间格式应抛出清晰错误。

---

### 3.3 构建受控测试数据

创建：

```text
data/controlled/baseline_scenarios.jsonl
```

至少包含 12 个场景，覆盖：

1. 饮食偏好变化；
2. 临时健康限制；
3. 居住地点变更；
4. 临时出差；
5. 学习或工作状态变化；
6. 日程安排变化。

每个场景至少包含：

- 2—4 条记忆；
- 1 个当前状态问题；
- 1 个历史状态问题；
- 可选的状态变化问题；
- 明确标注相关记忆、有效记忆和过期记忆。

必须至少设计一个典型失败案例：

```text
旧记忆：用户可以接受微辣
新记忆：用户因胃部不适暂时不能吃辣
查询：今天适合吃什么？
```

普通 TF-IDF Top-K 检索应有可能把语义相关但已经过期的“可以接受微辣”检索出来，用于证明基础检索存在过期记忆误用风险。

注意：

- 这批数据仅用于验证实验流程；
- 不得把这批小规模数据的结果写成最终论文结论；
- 数据必须人工可检查。

---

### 3.4 定义检索接口

定义可替换接口：

```python
from typing import Protocol

class Retriever(Protocol):
    def retrieve(
        self,
        query: QueryRecord,
        memories: list[MemoryRecord],
        top_k: int,
    ) -> list[RetrievedMemory]:
        ...
```

同时定义：

```python
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...
```

本阶段只实现：

```text
TF-IDF + Cosine Similarity
```

要求：

- 使用离线实现；
- 检索顺序必须可复现；
- 相同分数时使用稳定排序；
- 输出每条记忆的相似度分数；
- 不实现版本过滤；
- 不实现查询路由；
- 不实现预算优化。

本阶段故意保留普通检索的缺陷，作为后续方法的基线。

---

### 3.5 实现评价指标

至少实现以下指标。

#### Recall@K

```text
检索结果中命中的相关记忆数量
÷
标准相关记忆数量
```

#### Valid Recall@K

```text
检索结果中命中的当前有效记忆数量
÷
标准有效记忆数量
```

#### Stale Retrieval Rate

```text
检索结果中过期记忆数量
÷
检索记忆总数
```

#### Average Retrieved Token Cost

```text
检索结果中所有记忆 token_cost 的平均值
```

#### Retrieval Latency

记录单次检索耗时，单位至少支持毫秒。

要求：

- 指标函数必须独立于具体 Retriever；
- 分母为零时必须有明确处理；
- 每个指标必须有单元测试；
- 本阶段不计算最终回答准确率；
- 本阶段不调用大模型生成回答。

---

### 3.6 实现命令行入口

提供可运行命令：

```bash
python -m statebudgetmem.cli run   --config configs/baseline.yaml
```

配置文件至少包含：

```yaml
method: tfidf_topk
dataset_path: data/controlled/baseline_scenarios.jsonl
top_k: 3
random_seed: 42
results_dir: results
```

命令执行后必须生成：

```text
results/raw/<run_id>.jsonl
results/summaries/<run_id>.json
results/summaries/<run_id>.csv
```

---

### 3.7 原始结果格式

每一条查询的原始结果至少包含：

```text
run_id
scenario_id
query_id
query_text
query_type
retrieved_memory_ids
retrieved_scores
retrieved_valid_flags
retrieved_stale_flags
recall_at_k
valid_recall_at_k
stale_retrieval_rate
average_token_cost
retrieval_latency_ms
method
top_k
random_seed
```

不得只把结果打印在终端。

---

### 3.8 汇总结果格式

汇总文件至少包含：

```text
query_count
mean_recall_at_k
mean_valid_recall_at_k
mean_stale_retrieval_rate
mean_token_cost
mean_retrieval_latency_ms
method
top_k
dataset_path
random_seed
```

CSV 与 JSON 中的核心数值必须一致。

---

### 3.9 编写测试

至少包含：

#### 数据模型测试

- 正常数据可以解析；
- 非法时间格式报错；
- 非法查询类型报错；
- 缺失必要字段报错；
- 重复 ID 能被发现或明确处理。

#### 检索测试

- 相同输入得到相同排序；
- Top-K 数量正确；
- 相同分数时排序稳定；
- 空记忆列表有明确结果；
- `top_k` 大于记忆数量时不崩溃。

#### 指标测试

- Recall@K 计算正确；
- Valid Recall@K 计算正确；
- Stale Retrieval Rate 计算正确；
- 空集合处理正确；
- Token 成本计算正确。

#### CLI 测试

- 命令能够成功运行；
- 结果目录自动创建；
- JSONL、JSON、CSV 文件成功生成；
- 输出内容能够重新读取。

---

## 4. 本阶段明确不做

本阶段禁止实现：

- LLM 回答生成；
- OpenAI 或其他外部 API；
- Dense Embedding；
- 向量数据库；
- 版本更新操作；
- `ADD / SUPERSEDE / TEMP_INVALIDATE / RESTORE`；
- 当前状态视图；
- 历史版本视图；
- 查询路由；
- Token 预算优化；
- 存储淘汰；
- Streamlit、网页或聊天界面；
- LongMemEval、Memora、STALE、MemConflict 数据集接入；
- 最终论文图表；
- 最终研究结论。

发现这些需求时，应停止扩展并保持当前任务边界。

---

## 5. 验收标准

本阶段只有同时满足以下条件才算完成。

### 5.1 测试通过

```bash
pytest -q
```

必须全部通过。

### 5.2 CLI 可运行

```bash
python -m statebudgetmem.cli run   --config configs/baseline.yaml
```

必须正常退出。

### 5.3 结果文件真实存在

必须生成：

```text
results/raw/*.jsonl
results/summaries/*.json
results/summaries/*.csv
```

### 5.4 基线问题被实际观察到

至少一个样例中：

- TF-IDF 检索出语义相关但已经过期的记忆；
- 对应的 `Stale Retrieval Rate` 大于 0；
- 原始结果文件中可以定位该案例。

### 5.5 文档完整

`README.md` 至少说明：

- 环境安装；
- 数据格式；
- 基线运行命令；
- 测试命令；
- 结果文件位置；
- 当前阶段的限制。

### 5.6 无外部依赖

- 不需要 API Key；
- 不需要联网；
- 单元测试可以完全离线运行。

---

## 6. Codex 执行步骤

Codex 应按以下顺序工作：

1. 阅读 `AGENTS.md`、研究计划和本任务文件；
2. 检查仓库现状；
3. 创建 `docs/implementation_plan_task001.md`；
4. 写出简洁实施计划；
5. 创建项目骨架；
6. 实现数据模型；
7. 创建受控测试数据；
8. 实现 TF-IDF 检索；
9. 实现指标；
10. 实现 CLI 与结果输出；
11. 编写测试；
12. 运行测试；
13. 运行基线实验；
14. 检查结果文件；
15. 汇报完成情况和限制。

---

## 7. Codex 开始任务时使用的提示词

```text
请阅读：

1. AGENTS.md
2. docs/StateBudgetMem_research_plan.md
3. docs/CODEX_TASK_001_BASELINE.md

只执行 Task 001，不要尝试完成整个研究项目。

首先检查当前仓库，并将简洁的实施计划写入：

docs/implementation_plan_task001.md

然后按照任务文档完成：

- Python 项目骨架；
- MemoryRecord、QueryRecord 和 Scenario 数据模型；
- 小规模受控测试数据；
- 离线 TF-IDF Top-K 检索；
- Recall@K、Valid Recall@K、Stale Retrieval Rate、平均 Token 成本和检索延迟；
- 命令行实验入口；
- JSONL、JSON 和 CSV 结果输出；
- pytest 测试；
- README 使用说明。

要求：

- 不依赖任何 API Key；
- 不调用外部模型；
- 不开发聊天界面；
- 不实现版本管理、双视图、查询路由和预算检索；
- 不修改研究目标；
- 不编造实验结果；
- 所有结果必须由实际命令生成；
- 至少包含一个语义相似但已过期的记忆被普通检索取出的案例。

完成后必须实际执行：

pytest -q

以及：

python -m statebudgetmem.cli run --config configs/baseline.yaml

最后按 AGENTS.md 中规定的格式汇报：

1. 已完成内容；
2. 修改文件；
3. 执行命令；
4. 测试与运行结果；
5. 生成的结果文件；
6. 假设与限制。
```

---

## 8. 阶段完成后的下一步

本阶段验收通过后，才进入：

```text
Task 002：状态版本管理
```

下一阶段将实现：

```text
ADD
MERGE
SUPERSEDE
TEMP_INVALIDATE
RESTORE
NOOP
```

但不得提前在本阶段实现。
