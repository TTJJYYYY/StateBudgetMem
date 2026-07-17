# Dual Views 等价性分析

## 1. 分析目标

本分析用于解释当前正式 fair comparison 中
`memorybank_dual_views` 与 `memorybank_core`
结果完全一致的原因，并判断该现象属于实现设计、数据问题
还是实验配置问题。

## 2. 正式结果来源

- 原始逐查询结果：
  `results/fair_comparison/per_query_results.jsonl`
- 等价性明细：
  `results/fair_comparison_by_type/dual_views_equivalence.csv`
- 数据集：
  `data/controlled/temporal_challenge_v1.jsonl`
- Query 数：96
- 配置：
  - `top_k = 3`
  - `candidate_k = 20`
  - `token_budget = 256`
  - `random_seed = 42`
  - `repeat = 1`
  - `embedding_backend = sentence_transformer`
  - `embedding_model = all-MiniLM-L6-v2`

冻结数据集中包含：

| Query Type | 数量 |
|---|---:|
| CURRENT | 32 |
| HISTORICAL | 32 |
| CHANGE | 32 |
| GENERAL | 0 |

## 3. 逐查询等价性结果

`memorybank_core` 与 `memorybank_dual_views`
按照 `query_id + repeat_index` 逐条配对比较。

| 比较项 | 等价率 |
|---|---:|
| Retrieved Memory IDs | 100% |
| Retrieved Scores | 100% |
| Recall@K | 100% |
| Valid Recall@K | 100% |
| Stale Retrieval Rate | 100% |
| 所有比较项同时一致 | 100% |

按 Query Type 统计：

| Query Type | 查询数 | 完全等价率 |
|---|---:|---:|
| CURRENT | 32 | 100% |
| HISTORICAL | 32 | 100% |
| CHANGE | 32 | 100% |
| GENERAL | 0 | 无样本 |

因此，这不是总体平均值恰好相同，而是 96 条查询的检索结果逐条相同。

## 4. 按 Query Type 的指标证据

### CURRENT

| 方法 | Recall@K | Valid Recall@K | Stale Rate |
|---|---:|---:|---:|
| memorybank_core | 0.5243 | 0.8385 | 0.1667 |
| memorybank_dual_views | 0.5243 | 0.8385 | 0.1667 |

### HISTORICAL

| 方法 | Recall@K | Valid Recall@K | Stale Rate |
|---|---:|---:|---:|
| memorybank_core | 0.5594 | 0.2344 | 0.3438 |
| memorybank_dual_views | 0.5594 | 0.2344 | 0.3438 |

### CHANGE

| 方法 | Recall@K | Valid Recall@K | Stale Rate |
|---|---:|---:|---:|
| memorybank_core | 0.5443 | 0.5443 | 0.0000 |
| memorybank_dual_views | 0.5443 | 0.5443 | 0.0000 |

## 5. 原因分析

当前 Dual Views adapter 采用的是
`current_and_history_no_router` 策略。

它并未根据真实 Query Type 执行以下选择：

- CURRENT → Current View
- HISTORICAL → History View
- CHANGE → Current View + History View

而是对全部查询统一合并 Current View 与 History View。

当两个视图去重后的并集覆盖完整记忆集合时，
Dual Views 与 MemoryBank Core 获得相同的 eligible memory 集合。
之后二者继续复用相同的：

- MiniLM embedding；
- FAISS 检索；
- MemoryBank 候选评分；
- forgetting 设置；
- `candidate_k`；
- `top_k`；
- token budget。

因此，二者最终产生完全相同的检索结果。

## 6. 排除其他原因

### 6.1 不是随机误差

96 条查询的 Retrieved IDs 和指标均逐条相同，
因此不能解释为总体平均值的偶然重合。

### 6.2 不是预算配置差异

两种方法使用完全一致的：

- `top_k = 3`
- `candidate_k = 20`
- `token_budget = 256`
- `seed = 42`

### 6.3 不是 embedding 差异

两种方法均使用：

`all-MiniLM-L6-v2 + FAISS`

### 6.4 不主要是数据缺少状态变化

数据集中包含 CURRENT、HISTORICAL 和 CHANGE 各 32 条，
且存在 supersedes 和 temporarily invalidates 关系。
因此不能简单归因于数据集中不存在状态变化。

## 7. 结论

当前 `memorybank_dual_views == memorybank_core`
主要是 adapter 设计导致的退化行为。

该结果不能说明“双视图思想无效”，只能说明当前实现尚未进行
query-aware view selection。Dual Views 在当前配置下实际上等价于
对完整记忆集合执行 MemoryBank Core 检索。

## 8. 后续建议

在组长确认允许修改正式逻辑后，可考虑：

- CURRENT 只检索 Current View；
- HISTORICAL 检索 History View；
- CHANGE 联合检索 Current View 和 History View；
- GENERAL 不检索个人记忆；
- 修改后作为补充实验保存，不能覆盖当前正式结果。

## 9. 限制

冻结数据集没有 GENERAL 查询，因此无法判断 Dual Views 在真正
GENERAL 查询中能否正确避免个人记忆检索。
