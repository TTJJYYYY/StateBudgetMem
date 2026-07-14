# M1 数据与公平实验协议

## 1. 文档目的

本文档规定 StateBudgetMem 第一轮统一实验（M1）的数据版本、方法矩阵、公共参数、运行流程、结果管理和公平性约束。

M1 的目标是让所有方法：

* 使用同一份固定数据；
* 遵守同一统一 Runner 接口；
* 使用相同的检索数量和 token 预算；
* 使用相同的随机种子与查询状态策略；
* 对 MemoryBank 系方法复用相同的 dense retrieval 后端；
* 生成可追踪、可校验和可复现的结构化结果。

本文档只定义实验协议，不修改 MemoryBank、Versioning、Views、Routing 或统一 Runner 的内部算法。

---

## 2. 固定数据集

### 2.1 数据文件

M1 使用以下数据文件：

```text
data/controlled/temporal_challenge_v1.jsonl
```

对应 manifest：

```text
data/controlled/manifests/temporal_challenge_v1.manifest.json
```

### 2.2 数据版本

| 字段          | 值                                                                  |
| ----------- | ------------------------------------------------------------------ |
| 数据集名称       | `temporal_challenge`                                               |
| 数据集版本       | `v1`                                                               |
| Schema      | `statebudgetmem.schemas.Scenario`                                  |
| 冻结状态        | `true`                                                             |
| Scenario 数量 | 32                                                                 |
| Memory 数量   | 193                                                                |
| Query 数量    | 96                                                                 |
| SHA-256     | `f93331a2d93588fa8931efb4484fce577f5a5c9c4e679c51d4bb0192af6c8dd9` |

### 2.3 Query 类型覆盖

| Query 类型     | 数量 |
| ------------ | -: |
| `CURRENT`    | 32 |
| `HISTORICAL` | 32 |
| `CHANGE`     | 32 |
| `GENERAL`    |  0 |

当前版本针对时态一致性问题，均衡覆盖 `CURRENT`、`HISTORICAL` 和 `CHANGE`。

`GENERAL` 在 v1 中未覆盖，manifest 中明确记录为 0。不得为了满足数量要求而临时向冻结数据中插入未经设计和校验的查询。

如果后续需要加入 `GENERAL`，必须创建新的版本化数据文件，例如：

```text
temporal_challenge_v2.jsonl
```

不得直接覆盖或修改已经冻结的 v1 文件。

### 2.4 Memory 状态覆盖

| Memory 状态     |  数量 |
| ------------- | --: |
| `CURRENT`     | 147 |
| `HISTORICAL`  |  38 |
| `INVALIDATED` |   4 |
| `UNKNOWN`     |   4 |

### 2.5 版本关系覆盖

| 关系                        | 边数量 |
| ------------------------- | --: |
| `supersedes`              |  25 |
| `temporarily_invalidates` |  11 |

---

## 3. 数据校验

数据校验入口：

```text
src/statebudgetmem/data/validation.py
```

命令：

```powershell
python -m statebudgetmem.data.validation `
  --dataset data/controlled/temporal_challenge_v1.jsonl `
  --manifest data/controlled/manifests/temporal_challenge_v1.manifest.json
```

校验内容包括：

1. JSONL 能否通过公共 `Scenario` schema 解析；
2. 数据集是否为空；
3. `scenario_id` 是否全局唯一；
4. `memory_id` 是否全局唯一；
5. `query_id` 是否全局唯一；
6. `supersedes` 引用是否有效；
7. `temporarily_invalidates` 引用是否有效；
8. 版本关系是否存在自引用；
9. Query 类型覆盖统计；
10. Memory 状态覆盖统计；
11. 数据文件 SHA-256；
12. manifest 中的路径是否为仓库相对路径；
13. manifest 中的数量、覆盖信息和校验值是否与实际数据一致。

对应测试：

```text
tests/integration/test_controlled_dataset_validation.py
```

---

## 4. M1 方法矩阵

M1 最低方法矩阵如下：

| 序号 | 方法                            | 方法注册名             | 当前状态          |
| -: | ----------------------------- | ----------------- | ------------- |
|  1 | TF-IDF Top-K                  | `tfidf_topk`      | 已接入并验证        |
|  2 | MemoryBank Core               | `memorybank_core` | 已接入并验证        |
|  3 | MemoryBank + Versioning       | 待成员 B 确认          | 等待 Adapter 合并 |
|  4 | MemoryBank + Dual Views       | 待成员 B 确认          | 等待 Adapter 合并 |
|  5 | StateBudgetMem Rule Routing   | 待成员 B 确认          | 等待 Adapter 合并 |
|  6 | StateBudgetMem Oracle Routing | 待成员 B 确认          | 等待 Adapter 合并 |

不得在方法尚未注册时自行虚构注册名或伪造运行结果。

成员 B 的方法合并后，需要：

1. 确认 Registry 中的真实方法名；
2. 为每种方法增加独立配置；
3. 将新配置加入公平参数测试；
4. 使用同一个正式数据集运行；
5. 检查所有方法的公共参数完全一致。

---

## 5. 公平实验公共参数

M1 主实验固定参数：

| 参数                      | 固定值                                           |
| ----------------------- | --------------------------------------------- |
| `dataset_path`          | `data/controlled/temporal_challenge_v1.jsonl` |
| `top_k`                 | 3                                             |
| `candidate_k`           | 20                                            |
| `token_budget`          | 32                                            |
| `random_seed`           | 42                                            |
| `repeat`                | 1                                             |
| `forgetting_enabled`    | `true`                                        |
| `forgetting_threshold`  | 0.3                                           |
| `exclude_forgotten`     | `false`                                       |
| `reinforcement_enabled` | `false`                                       |
| `query_state_policy`    | `independent`                                 |
| `token_counter_name`    | `memory_record_token_cost`                    |

所有正式方法配置必须保持以上字段一致。

不同方法允许变化的字段仅包括：

```text
methods
results_dir
embedding_backend
embedding_model
```

其中 embedding 字段只能因为方法类型不同而变化。

---

## 6. Dense retrieval 公平性要求

所有 MemoryBank 系方法必须共享：

```yaml
embedding_backend: hash
embedding_model: deterministic_hash_embedding
```

并复用统一的：

* embedding 模型；
* FAISS 索引；
* 相似度计算；
* MemoryBank 候选评分；
* 遗忘逻辑；
* token budget 截断逻辑。

StateBudgetMem 方法只能通过以下模块改变候选记忆 ID 集合：

* Versioning；
* Current View；
* History View；
* Rule Routing；
* Oracle Routing。

不得为某个 StateBudgetMem 方法单独实现新的 embedding、FAISS、相似度或 MemoryBank 排名公式。

TF-IDF 不使用 dense embedding，因此配置为：

```yaml
embedding_backend: method_default
embedding_model: method_default
```

TF-IDF 与 dense 方法不要求使用相同的检索算法，但必须使用相同的：

* 数据集；
* `top_k`；
* `candidate_k`；
* `token_budget`；
* 随机种子；
* repeat；
* query-state 策略；
* token 统计口径。

---

## 7. 查询状态隔离

M1 主实验统一使用：

```yaml
reinforcement_enabled: false
query_state_policy: independent
```

原因是正式公平比较需要保证：

* 每个 Query 的结果不受前一个 Query 影响；
* 调整 Query 顺序不会改变某个 Query 的输出；
* MemoryBank 强化状态不会跨 Query 传播；
* 不同方法在相同、独立的查询条件下进行比较。

`sequential` 模式可作为后续状态强化实验，不属于 M1 主实验默认配置。

---

## 8. Oracle Routing 约束

Oracle Routing 的 Oracle 只表示“已知 Query 类型”，不表示“已知正确答案”。

Oracle Routing 可以读取：

```text
query_type
```

合法值包括：

```text
CURRENT
HISTORICAL
CHANGE
GENERAL
```

Oracle Routing 禁止读取：

```text
gold_memory_ids
gold_answer
expected_memory_ids
target_memory_ids
任何直接指向正确记忆或正确答案的字段
```

Oracle 的作用只能是选择：

* Current View；
* History View；
* 当前和历史联合视图；
* General 查询的合法默认候选视图。

Oracle 不得利用 gold memory ID 直接构造候选集合，否则属于 gold leakage，结果不能进入正式比较。

---

## 9. 当前配置文件

当前已经完成：

```text
configs/fair_experiments/m1_tfidf_topk.yaml
configs/fair_experiments/m1_memorybank_core.yaml
```

对应配置测试：

```text
tests/integration/test_fair_experiment_configs.py
```

该测试负责检查：

* 正式数据与 manifest 一致；
* 配置中的方法名正确；
* 数据集路径一致；
* 公平参数完全一致；
* `candidate_k >= top_k`；
* token budget 为正数；
* 随机种子固定；
* reinforcement 已关闭；
* query-state 策略为 `independent`；
* MemoryBank 使用统一 dense 后端；
* TF-IDF 不声明使用 dense 后端；
* 每种方法使用独立结果目录；
* 结果路径为仓库相对路径。

---

## 10. 当前可复现命令

### 10.1 数据校验

```powershell
python -m statebudgetmem.data.validation `
  --dataset data/controlled/temporal_challenge_v1.jsonl `
  --manifest data/controlled/manifests/temporal_challenge_v1.manifest.json
```

### 10.2 定向测试

```powershell
python -m pytest tests/integration/test_controlled_dataset_validation.py -q
python -m pytest tests/integration/test_fair_experiment_configs.py -q
```

### 10.3 完整回归

```powershell
python -m pytest -q
```

当前已验证：

```text
321 passed
```

### 10.4 TF-IDF

```powershell
python -m statebudgetmem.unified_runner `
  --config configs/fair_experiments/m1_tfidf_topk.yaml
```

### 10.5 MemoryBank Core

```powershell
python -m statebudgetmem.unified_runner `
  --config configs/fair_experiments/m1_memorybank_core.yaml
```

---

## 11. 已完成的链路验证结果

以下结果仅用于验证正式数据、配置和统一 Runner 可以完整运行，不作为最终论文结论。

| 方法              | Query 数 | Recall@K | Valid Recall@K | Stale Retrieval Rate | Token Cost |
| --------------- | ------: | -------: | -------------: | -------------------: | ---------: |
| TF-IDF Top-K    |      96 |   0.6043 |         0.7240 |               0.2483 |      27.51 |
| MemoryBank Core |      96 |   0.2015 |         0.1424 |               0.0069 |      26.15 |

初步现象：

* TF-IDF 的 recall 和 valid recall 较高；
* TF-IDF 更容易同时检索到已经过期的历史状态；
* MemoryBank Core 的 stale retrieval rate 较低；
* 当前确定性 hash embedding 下，MemoryBank Core 的 recall 较低。

这些结果不能直接证明某种方法整体更好，因为当前 MemoryBank 使用的是用于离线复现的确定性 hash embedding，而不是高质量语义 embedding。

正式结论需要等待：

1. StateBudgetMem 四种方法全部接入；
2. 六种方法使用同一配置完成运行；
3. 结果绑定同一数据校验值和代码 commit；
4. 必要消融实验完成；
5. 结果不经过手工修改。

---

## 12. 结果管理规范

正式实验结果必须记录：

* Git commit；
* 数据集路径；
* 数据集 SHA-256；
* 完整配置；
* 方法注册名；
* 随机种子；
* repeat；
* 环境信息；
* 原始逐 Query 结果；
* 汇总 JSON；
* 汇总 CSV。

禁止：

* 手工修改结果文件；
* 把不同 commit 的结果混在一次比较中；
* 使用修改后的数据但继续标记为 v1；
* 使用不同 token budget 比较方法；
* 使用不同 candidate-k 比较方法；
* 提交模型权重；
* 提交临时缓存或索引；
* 提交本机绝对路径；
* 提交重复 timestamp 的调试结果。

默认情况下，开发运行产生的：

```text
results/fair_experiments/
```

不提交仓库。

只有经过团队确认的正式结果，才能按照项目统一规范绑定 commit、配置、数据校验值和 seed 后归档。

---

## 13. 已知问题

### 13.1 `GENERAL` 尚未覆盖

`temporal_challenge_v1` 当前没有 `GENERAL` Query。

处理方式：

* M1 时态实验明确报告 `GENERAL=0`；
* 不直接修改冻结 v1；
* 如需 General 对照，创建新数据版本或单独数据集。

### 13.2 StateBudgetMem 方法尚未全部注册

当前只有：

```text
tfidf_topk
memorybank_core
```

其余方法配置等待成员 B 合并后补充。

### 13.3 Run ID 使用 smoke 前缀

当前正式配置运行后，统一 Runner 生成的 run ID 仍可能包含：

```text
unified_smoke_seed42
```

这是统一 Runner 的公共命名行为，不影响结果内容，但正式归档前应由统一 Runner 维护者确认是否修改命名。

成员 C 不直接修改公共 Runner。

---

## 14. 后续工作

1. 等待成员 B 合并四种 StateBudgetMem Adapter；
2. 确认真实 Registry 方法名；
3. 增加四个正式 YAML 配置；
4. 扩展公平配置测试；
5. 确保所有 MemoryBank 系方法共享相同 dense 后端；
6. 运行完整六方法矩阵；
7. 增加 `top_k`、`candidate_k` 和 `token_budget` 的单变量消融；
8. 绑定 commit、配置、数据 SHA-256 和随机种子；
9. 形成第一轮正式实验结果。
