# Unified Development Interface v0.2 — FROZEN

**Owner:** 穆泓烨

**Frozen on:** 2026-07-13

**Scope:** 第一轮公平检索实验与 MemoryBank/StateBudgetMem Adapter 并行开发

除资源精细测量和持久化协议外，本规范中的接口与语义正式冻结。兼容的
adapter-local metadata 可以增加；字段删除、重命名或语义改变必须由接口负责人批准。

## 1. 公共记录与方法接口

所有方法复用 `statebudgetmem.interfaces` 导出的 `MemoryRecord`、`QueryRecord`、
`Scenario`、`MethodResult` 和 `RetrievedMemory`，不得在功能目录复制同类 schema。
统一 runner 在调用方法前清空 query 的三个 gold ID 列表。

```python
class MemoryMethod(Protocol):
    @property
    def name(self) -> str: ...
    def reset(self) -> None: ...
    def ingest(self, memories: list[MemoryRecord]) -> None: ...
    def retrieve(
        self,
        query: QueryRecord,
        *,
        top_k: int,
        token_budget: int | None = None,
        mutate: bool = False,
    ) -> MethodResult: ...
```

`MethodResult` 字段不变。`latency_ms` 定义为从进入 adapter 的 `retrieve()` 到完整
`MethodResult` 返回的端到端耗时，必须覆盖候选检索、过滤、排序、预算裁剪和允许的
状态更新；不得只记录向量搜索耗时。

## 2. 方法构造上下文

```python
class MethodBuildContext(BaseModel):
    experiment: ExperimentConfig
    work_dir: Path

MethodFactory = Callable[[MethodBuildContext], MemoryMethod]
```

`MethodRegistry.create(name, context)` 必须把同一个上下文传给 factory。Adapter 从
`context.experiment` 取得 embedding、retrieval、forgetting、reinforcement 和 query
生命周期配置，从 `work_dir` 取得该方法本次运行的私有输出目录。配置不得只抄入结果
metadata 而不影响构造或运行。

embedding dimension 由构造后的 embedder/index 读取，不是必填实验字段。

## 3. 冻结的最小实验配置

第一轮共同配置包含：

- `embedding_backend`、`embedding_model`；
- `top_k`、`candidate_k`，且 `candidate_k >= top_k`；
- `token_budget`、`token_counter_name`；
- `forgetting_enabled`、`forgetting_threshold`（0 到 1）；
- `exclude_forgotten`、`reinforcement_enabled`；
- `query_state_policy`：只能为 `independent` 或 `sequential`；
- dataset、methods、seed、repeat 和结果路径。

`top_k` 是最终返回上限，`candidate_k` 是统一候选检索深度。第一轮默认关闭强化，避免
查询间状态污染。

## 4. Scenario 与 query 生命周期

- `reset()` 清除当前 scenario 的记忆、FAISS/index 内容、ID 映射、版本/视图/路由的
  可变状态、访问计数和强化/遗忘日志；不得残留上一个 scenario 的记忆状态。
- 已加载的只读 embedding model 可以在 scenario 间复用，以避免重复加载；它的参数、
  缓存语义和配置不得随 scenario 改变。scenario 数据及向量索引不可复用。
- 每次检索的逻辑时间必须来自 `QueryRecord.reference_time`，不得用墙钟时间替代。
- `independent`：每条 query 前执行 `reset()` 并重新 ingest 同一 scenario，任何 query
  都不继承其他 query 的 strength、last_accessed、access_count 或强化日志。
- `sequential`：每个 scenario 只 reset/ingest 一次，按数据集中固定的 query 顺序运行，
  后续 query 共享此前合法产生的状态。

MemoryBank Core 的 `retrieve_with_metadata(..., reinforce=True)` 保持旧行为；设置
`reinforce=False` 时不得改变 memory 或 bank 的 strength、last_accessed、access_count，
也不得产生强化 after-state/log 记录。

## 5. 公平候选检索边界

MemoryBank Core 和 StateBudgetMem 必须使用同一 embedding 实例配置、FAISS 候选检索、
相似度定义、`candidate_k` 和 MemoryBank 候选评分方式。StateBudgetMem 的
Versioning/Views/Routing 可以改变哪些 memory ID 有资格进入候选集合，但进入集合的记忆
不得换用不同 embedding、检索器、相似度或评分公式。最终共同固定数据、query time、
top-k、token budget、seed 和资源测量口径。

## 6. MemoryBank 当前真实评分语义

以 `baselines/memorybank/system.py` 的 `_retrieval_candidate_row()` 为准：

```text
age_hours       = max(0, (query_time - memory.timestamp) / 3600)
time_decay      = exp(-age_hours / 168)
strength_factor = 1 + 0.3 * strength
ranking_score   = semantic_score * strength_factor * time_decay

elapsed_units   = max(0, query_time - last_accessed) / decay_interval_sec
retention       = exp(-elapsed_units / strength)  # strength <= 0 时为 0
is_forgotten    = retention < forgetting_threshold
```

最终按 `ranking_score` 降序排列。`strength` 通过 `strength_factor` 参与排序；
`retention` 不参与排序。forgotten flag 只用于 metadata/log，以及
`exclude_forgotten=True` 时从 ranking pool 过滤。因此在其他状态完全相同时，
`forgetting_enabled=True` 且 `exclude_forgotten=False` 不改变检索结果或顺序。

Adapter 中的 `forgetting_enabled=False` 应跳过遗忘过滤/遗忘状态推进；若开启，则使用上述
threshold 和同一 query time。不得把 retention 偷偷乘入 ranking score。

## 7. 输出、注册与烟测

`total_token_cost` 等于返回项 `token_cost` 之和，并不得超过 token budget。方法输出不得
包含 gold 有效性或 stale 标签。Runner 通过 `MethodRegistry` 注册 Adapter，并在唯一运行
目录生成 `raw.jsonl`、`summary.json`、`summary.csv`、`environment.json`。

`data/controlled/interface_smoke_v1.jsonl` 只验证集成，不作为正式实验结果。现有
`tfidf_topk` 必须持续通过该 smoke。

## 8. 本轮未冻结（P1）

- save/load 的 `PersistableMethod` 协议、checkpoint 格式和恢复一致性；
- 精细 resource profiler、冷启动/模型加载计时、峰值内存和索引磁盘归因；
- 不同硬件间的资源结果归一化。

这些 P1 项不得阻塞 Adapter 的 reset/ingest/retrieve 开发，也不得通过私有实现反向改变
本规范已经冻结的生命周期和延迟语义。
