# Rule Routing 误差分析

## 1. 分析目标

本文分析 `statebudgetmem_rule` 相比 `statebudgetmem_oracle` 在 Recall@K 和 Valid Recall@K 上明显偏低的原因，并区分以下问题：

1. Query Type 判断错误；
2. 错误回退到 GENERAL；
3. 候选记忆集合过度收缩；
4. Dense Retrieval 排序失败。

本分析只解释当前正式实验结果，不修改 Router、Adapter 或其他正式实验逻辑。

---

## 2. 正式结果来源

本分析使用以下正式结果：

- 逐查询结果：

  `results/fair_comparison/per_query_results.jsonl`

- 按 Query Type 分组结果：

  `results/fair_comparison_by_type/by_query_type.csv`

- 路由混淆矩阵：

  `results/fair_comparison_by_type/routing_confusion.csv`

- Rule 错误案例：

  `results/fair_comparison_by_type/rule_error_cases.jsonl`

正式实验配置：

```yaml
dataset_path: data/controlled/temporal_challenge_v1.jsonl
top_k: 3
candidate_k: 20
token_budget: 256
random_seed: 42
repeat: 1
embedding_backend: sentence_transformer
embedding_model: all-MiniLM-L6-v2
query_state_policy: independent
reinforcement_enabled: false
```

数据集共包含：

| Query Type | 数量 |
|---|---:|
| CURRENT | 32 |
| HISTORICAL | 32 |
| CHANGE | 32 |
| GENERAL | 0 |
| 总计 | 96 |

冻结数据集不包含 GENERAL 查询，因此本文只能分析其他类型被错误预测为 GENERAL 的情况，不能评价 Router 在真实 GENERAL 查询上的准确率。

---

## 3. Rule 与 Oracle 整体结果

| 方法 | Recall@K | Valid Recall@K | Stale Retrieval Rate | Mean Token Proxy |
|---|---:|---:|---:|---:|
| `statebudgetmem_rule` | 0.2506 | 0.3455 | 0.0035 | 14.8438 |
| `statebudgetmem_oracle` | 0.4758 | 0.7335 | 0.0035 | 29.2917 |

Rule 相比 Oracle：

- Recall@K 低约 `0.2252`；
- Valid Recall@K 低约 `0.3880`；
- Stale Retrieval Rate 基本相同；
- 平均 Token Proxy 只有 Oracle 的约一半。

Rule 与 Oracle 使用相同的：

- MiniLM embedding；
- FAISS；
- MemoryBank Dense Retriever；
- `candidate_k = 20`；
- `top_k = 3`；
- `token_budget = 256`；
- forgetting 配置；
- scoped retrieval 接口。

两者的主要差别是 Query Type 的来源：

- Rule 使用规则路由；
- Oracle 直接读取数据集中的真实 `query_type`。

因此，Rule 与 Oracle 的明显性能差距主要来自路由和候选集合构造，而不是 Dense Retrieval 后端不同。

---

## 4. 路由混淆矩阵

正式结果中的路由混淆矩阵如下：

| Gold Query Type | Pred CURRENT | Pred HISTORICAL | Pred CHANGE | Pred GENERAL | Pred NONE | Total |
|---|---:|---:|---:|---:|---:|---:|
| CURRENT | 25 | 0 | 0 | 7 | 0 | 32 |
| HISTORICAL | 0 | 1 | 0 | 31 | 0 | 32 |
| CHANGE | 1 | 0 | 19 | 12 | 0 | 32 |
| GENERAL | 0 | 0 | 0 | 0 | 0 | 0 |
| TOTAL | 26 | 1 | 19 | 50 | 0 | 96 |

总体预测正确数为：

```text
25 + 1 + 19 = 45
```

总体路由准确率为：

```text
45 / 96 = 46.875%
```

按 Query Type 统计：

| Query Type | 正确数 | 总数 | Routing Accuracy |
|---|---:|---:|---:|
| CURRENT | 25 | 32 | 78.125% |
| HISTORICAL | 1 | 32 | 3.125% |
| CHANGE | 19 | 32 | 59.375% |
| GENERAL | 0 | 0 | 无样本 |
| 总体 | 45 | 96 | 46.875% |

其中 HISTORICAL 的识别率最低，32 条中只有 1 条被正确识别。

---

## 5. 按 Query Type 的检索表现

## 5.1 CURRENT

| 方法 | Recall@K | Valid Recall@K | Stale Rate | Empty Retrieval Rate | Mean Token Proxy |
|---|---:|---:|---:|---:|---:|
| `statebudgetmem_rule` | 0.3717 | 0.6563 | 0.0104 | 0.2188 | 23.7500 |
| `statebudgetmem_oracle` | 0.4920 | 0.8750 | 0.0104 | 0.0000 | 30.5625 |

CURRENT 中有 7 条查询被误判为 GENERAL：

```text
7 / 32 = 21.875%
```

这与 Rule 在 CURRENT 上的空检索率 `0.21875` 完全一致。

说明这些查询在错误回退为 GENERAL 后没有进入个人记忆检索流程，从而导致 Recall@K 和 Valid Recall@K 明显低于 Oracle。

---

## 5.2 HISTORICAL

| 方法 | Recall@K | Valid Recall@K | Stale Rate | Empty Retrieval Rate | Mean Token Proxy |
|---|---:|---:|---:|---:|---:|
| `statebudgetmem_rule` | 0.0000 | 0.0000 | 0.0000 | 0.9688 | 0.7813 |
| `statebudgetmem_oracle` | 0.3911 | 0.7813 | 0.0000 | 0.0000 | 25.4688 |

HISTORICAL 中有 31 条查询被预测为 GENERAL：

```text
31 / 32 = 96.875%
```

Rule 的 HISTORICAL 空检索率同样为：

```text
0.96875 = 96.875%
```

这说明 HISTORICAL 几乎完全没有进入正确的历史记忆候选集合。

HISTORICAL 是当前 Rule Router 表现偏低的最主要来源。

---

## 5.3 CHANGE

| 方法 | Recall@K | Valid Recall@K | Stale Rate | Empty Retrieval Rate | Mean Token Proxy |
|---|---:|---:|---:|---:|---:|
| `statebudgetmem_rule` | 0.3802 | 0.3802 | 0.0000 | 0.3750 | 20.0000 |
| `statebudgetmem_oracle` | 0.5443 | 0.5443 | 0.0000 | 0.0000 | 31.8438 |

CHANGE 中有 12 条查询被预测为 GENERAL：

```text
12 / 32 = 37.5%
```

Rule 在 CHANGE 上的空检索率也是：

```text
0.375 = 37.5%
```

另外有 1 条 CHANGE 查询被预测为 CURRENT，属于错误时间视图选择。

---

## 6. Rule 错误案例统计

Rule 比 Oracle 表现差的查询共有：

```text
51
```

错误模式统计如下：

| Error Pattern | 次数 |
|---|---:|
| `fallback_to_general` | 50 |
| `eligibility_filter_overpruning` | 45 |
| `wrong_temporal_view` | 1 |

这里的 51 是 Rule 与 Oracle 存在路由错误或检索质量差距的案例数；
50 是其中被 Rule Router 预测为 GENERAL 的案例数。两者是不同口径。

这些错误案例中：

- Mean Recall Gap：`0.4239`
- Mean Valid Recall Gap：`0.7304`

需要注意，`eligibility_filter_overpruning` 在多数案例中并不是与路由问题无关的第二个独立根因，而是错误回退到 GENERAL 后的直接结果。

当个人状态查询被错误预测为 GENERAL 时，通常会同时发生：

1. `predicted_query_type = GENERAL`；
2. 个人记忆候选集合为空；
3. Retrieved Memory IDs 为空；
4. Token Proxy 接近 0；
5. Recall@K 为 0；
6. Valid Recall@K 为 0。

因此，当前最主要的根因是 Router 的 GENERAL fallback，而候选池过度收缩是该错误路由带来的后果。

---

## 7. 典型错误案例

## 7.1 CHANGE 查询被错误预测为 GENERAL

查询：

```text
我的暑假旅行目的地是怎么改的？
```

结果：

| 字段 | 值 |
|---|---|
| Gold Query Type | CHANGE |
| Predicted Query Type | GENERAL |
| Rule Recall@K | 0.0 |
| Oracle Recall@K | 1.0 |
| Rule Retrieved IDs | 空 |
| Error Patterns | `fallback_to_general`、`eligibility_filter_overpruning` |

该问题包含“怎么改的”，语义上明显询问状态变化，但当前 CHANGE 规则没有覆盖这一自然语言表达。

---

## 7.2 月份表达未被识别为 HISTORICAL

查询：

```text
三月份我更偏好哪种书籍形式？
```

结果：

| 字段 | 值 |
|---|---|
| Gold Query Type | HISTORICAL |
| Predicted Query Type | GENERAL |
| Rule Recall@K | 0.0 |
| Oracle Recall@K | 0.75 |
| Rule Retrieved IDs | 空 |

其他类似失败案例包括：

```text
二月份我对辣味是什么态度？
二月份我工作日通常怎么去公司？
三月份我的主力手机是什么系统？
五月份我通常每天喝多少咖啡？
六月二十日我的长期饮食限制有哪些？
```

这些问题都包含明确的月份或具体日期，但 Router 没有将这类时间表达识别为 HISTORICAL 信号。

---

## 7.3 当前时间表达覆盖不足

查询：

```text
我今晚应该服用多少毫克？
```

结果：

| 字段 | 值 |
|---|---|
| Gold Query Type | CURRENT |
| Predicted Query Type | GENERAL |
| Rule Recall@K | 0.0 |
| Oracle Recall@K | 0.75 |

另一个案例：

```text
这周我还能按原计划跑五公里吗？
```

结果：

| 字段 | 值 |
|---|---|
| Gold Query Type | CURRENT |
| Predicted Query Type | GENERAL |
| Rule Recall@K | 0.0 |
| Oracle Recall@K | 0.60 |

说明“今晚”“这周”等当前时间表达没有被完整覆盖。

---

## 7.4 CHANGE 自然语言表达覆盖不足

查询：

```text
我的暑假旅行目的地是怎么改的？
```

当前规则未识别“怎么改的”为 CHANGE。

类似的自然语言变化表达还可能包括：

```text
改成了什么
后来变成了什么
原来和现在有什么区别
还保持以前的状态吗
从原来的状态变成什么了
```

如果 Router 只依赖少量固定关键词或字面模板，就容易将这些问题回退为 GENERAL。

---

## 8. GENERAL fallback 是主要问题

共有 50 条错误案例出现：

```text
fallback_to_general
```

混淆矩阵中也显示：

```text
CURRENT → GENERAL：7
HISTORICAL → GENERAL：31
CHANGE → GENERAL：12
总计：50
```

即 96 条查询中超过一半被预测为 GENERAL：

```text
50 / 96 = 52.083%
```

但冻结数据集本身没有任何真实 GENERAL 查询。

这说明当前 Router 的默认 fallback 策略过于激进。

未命中关键词并不代表问题属于 GENERAL，尤其是包含“我”“我的”等明显个人状态表达的问题。

---

## 9. Dense Retrieval 不是当前主要瓶颈

Rule 和 Oracle 使用完全相同的 Dense Retrieval 后端。

如果主要瓶颈来自 MiniLM 或 FAISS 排序，那么 Rule 与 Oracle 的差距不应如此明显，因为两者在获得相同候选集合时会执行相同的检索过程。

当前结果显示：

- Rule Recall@K：0.2506
- Oracle Recall@K：0.4758
- Rule Valid Recall@K：0.3455
- Oracle Valid Recall@K：0.7335

Oracle 只替换了 Query Type 来源，就获得了明显提升。

因此可以判断，当前主要瓶颈为：

1. Query Type 判断错误；
2. 错误路由导致候选池为空或过窄；
3. 正确记忆在 Dense Retrieval 前已被排除。

不应优先通过提高 `candidate_k` 或更换 embedding 模型解决该问题。

---

## 10. 初步改进方向

根据当前错误案例，Router 应优先补充以下规则。

### 10.1 CURRENT 时间表达

包括但不限于：

```text
今天
今晚
这周
本周
现在
目前
当前
最近
这两天
```

### 10.2 HISTORICAL 时间表达

包括但不限于：

```text
一月份至十二月份
具体月份
具体日期
去年
上个月
当时
那时候
原来
最开始
之前那段时间
过去一段时间
```

### 10.3 CHANGE 表达

包括但不限于：

```text
怎么改的
改成了什么
后来变成
从……到……
从……变成……
原来和现在
是否还
不再
先……后来……
```

### 10.4 默认 fallback

对于包含明显个人状态信号的问题，不应直接回退 GENERAL。

个人状态信号可以包括：

```text
我
我的
喜欢
习惯
工作
饮食
住址
通勤
药物
计划
设备
宠物
```

具体 fallback 策略需要由组长确认后再修改。

---

## 11. 结论

`statebudgetmem_rule` Recall 偏低的主要原因是 Rule Router 覆盖不足，而不是 Dense Retrieval 能力不足。

最主要的问题包括：

1. HISTORICAL 的月份、日期和历史时间段表达覆盖严重不足；
2. CHANGE 的“怎么改的”等自然语言表达没有被识别；
3. CURRENT 的“今晚”“这周”等时间表达覆盖不足；
4. 未命中规则后大量错误回退到 GENERAL；
5. GENERAL fallback 导致个人记忆候选集合为空；
6. 候选集合为空进一步导致 Recall@K 和 Valid Recall@K 为 0。

当前结果支持优先改进 Router 规则、fallback 策略和 Router 测试，而不是优先更换 embedding 或增加候选数量。

---

## 12. 限制

本分析存在以下限制：

1. 当前冻结数据集只有 96 条查询；
2. GENERAL 类型没有真实样本；
3. 无法评价 Router 对真实 GENERAL 查询的准确率；
4. 不应针对当前 96 条查询逐句硬编码；
5. 新规则必须加入同义改写和负例测试；
6. 修改 Router 后应生成新的补充实验结果；
7. 修复后结果不能覆盖当前正式结果。
