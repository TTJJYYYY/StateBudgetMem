# M1 数据与公平实验协议

## 1. 实验目标

M1 用于统一比较以下六种方法：

1. `tfidf_topk`
2. `memorybank_core`
3. `memorybank_versioning`
4. `memorybank_dual_views`
5. `statebudgetmem_rule`
6. `statebudgetmem_oracle`

所有方法必须使用同一数据集、统一 Runner、相同检索预算和随机种子。

---

## 2. 固定数据集

数据文件：

```text
data/controlled/temporal_challenge_v1.jsonl
```

Manifest：

```text
data/controlled/manifests/temporal_challenge_v1.manifest.json
```

数据统计：

| 项目 | 数量 |
|---|---:|
| Scenario | 32 |
| Memory | 193 |
| Query | 96 |
| CURRENT | 32 |
| HISTORICAL | 32 |
| CHANGE | 32 |
| GENERAL | 0 |

SHA-256：

```text
8eb0757dd676105a17ecfcc365ba500a10e952f67b955f7aae5c925c0541dc2d
```

`temporal_challenge_v1.jsonl` 已冻结。后续如需修改数据，必须创建新的版本文件，不能直接覆盖 v1。

数据校验命令：

```powershell
python -m statebudgetmem.data.validation `
  --dataset data/controlled/temporal_challenge_v1.jsonl `
  --manifest data/controlled/manifests/temporal_challenge_v1.manifest.json
```

---

## 3. 方法矩阵

| 方法 | Registry 名称 | 后端 |
|---|---|---|
| TF-IDF Top-K | `tfidf_topk` | Lexical |
| MemoryBank Core | `memorybank_core` | Dense |
| MemoryBank + Versioning | `memorybank_versioning` | Dense |
| MemoryBank + Dual Views | `memorybank_dual_views` | Dense |
| Rule Routing | `statebudgetmem_rule` | Dense |
| Oracle Routing | `statebudgetmem_oracle` | Dense |

机器可读矩阵：

```text
configs/fair_experiments/m1_method_matrix.json
```

---

## 4. 公平实验参数

本 PR 中存在两套可复现实验入口，不能混为同一组结果：

- `configs/fair_comparison/*.yaml` 对应已经生成的正式分析结果
  `results/fair_comparison/per_query_results.jsonl`，使用
  `sentence_transformer + all-MiniLM-L6-v2` 和 `token_budget: 256`。
- `configs/fair_experiments/m1_*.yaml` 是 M1 离线可复现矩阵，使用
  `hash + deterministic_hash_embedding` 和 `token_budget: 32`，用于
  CI/无模型环境下验证公平配置、方法注册和数据冻结校验。

M1 离线矩阵中，六种方法统一使用：

```yaml
dataset_path: data/controlled/temporal_challenge_v1.jsonl
top_k: 3
candidate_k: 20
token_budget: 32
random_seed: 42
repeat: 1

forgetting_enabled: true
forgetting_threshold: 0.3
exclude_forgotten: false

reinforcement_enabled: false
query_state_policy: independent

token_counter_name: memory_record_token_cost
```

允许因方法不同而变化的字段：

```text
methods
results_dir
embedding_backend
embedding_model
```

---

## 5. Dense 后端约束

以下五种方法必须使用相同的 dense 后端：

```text
memorybank_core
memorybank_versioning
memorybank_dual_views
statebudgetmem_rule
statebudgetmem_oracle
```

统一配置：

```yaml
embedding_backend: hash
embedding_model: deterministic_hash_embedding
```

五种方法必须复用相同的：

- embedding；
- MemoryBank；
- FAISS；
- 相似度计算；
- 候选评分；
- 遗忘逻辑；
- token budget 截断；
- scoped retrieval；
- `allowed_memory_ids` 候选限制入口。

Versioning、Views 和 Routing 只能改变候选记忆 ID 集合，不能重新实现 dense retrieval。

TF-IDF 使用：

```yaml
embedding_backend: method_default
embedding_model: method_default
```

---

## 6. Oracle Routing 限制

Oracle Routing 只能读取：

```text
query_type
```

禁止读取：

```text
gold_memory_ids
gold_answer
expected_memory_ids
target_memory_ids
```

Oracle 只表示已知问题类型，不能直接获得正确记忆或正确答案。

---

## 7. 主实验配置

```text
configs/fair_experiments/m1_tfidf_topk.yaml
configs/fair_experiments/m1_memorybank_core.yaml
configs/fair_experiments/m1_memorybank_versioning.yaml
configs/fair_experiments/m1_memorybank_dual_views.yaml
configs/fair_experiments/m1_statebudgetmem_rule.yaml
configs/fair_experiments/m1_statebudgetmem_oracle.yaml
```

六种配置必须通过统一 Runner 解析，并使用互不重复的结果目录。

---

## 8. 单变量消融实验

主实验基线：

```yaml
top_k: 3
candidate_k: 20
token_budget: 32
```

消融范围：

| 参数 | 取值 |
|---|---|
| `top_k` | 1、3、5 |
| `candidate_k` | 5、20、40 |
| `token_budget` | 16、32、64 |

主配置已经代表基线值，因此消融目录只生成非基线配置：

```text
top_k: 1, 5
candidate_k: 5, 40
token_budget: 16, 64
```

配置总数：

```text
6 种方法 × 3 个参数 × 2 个非基线值 = 36
```

生成命令：

```powershell
python scripts/generate_m1_ablation_configs.py
```

检查命令：

```powershell
python scripts/generate_m1_ablation_configs.py --check
```

消融配置目录：

```text
configs/fair_experiments/ablations/
```

---

## 9. 测试命令

成员 C 定向测试：

```powershell
python -m pytest `
  tests/integration/test_controlled_dataset_validation.py `
  tests/integration/test_fair_experiment_configs.py `
  tests/integration/test_m1_ablation_configs.py `
  tests/integration/test_m1_method_matrix.py -q
```

完整回归：

```powershell
python -m pytest -q
```

---

## 10. 六方法运行

```powershell
$configs = @(
    "m1_tfidf_topk.yaml",
    "m1_memorybank_core.yaml",
    "m1_memorybank_versioning.yaml",
    "m1_memorybank_dual_views.yaml",
    "m1_statebudgetmem_rule.yaml",
    "m1_statebudgetmem_oracle.yaml"
)

foreach ($config in $configs) {
    python -m statebudgetmem.unified_runner `
      --config "configs/fair_experiments/$config"
}
```

每种方法应满足：

```text
query_count = 96
top_k = 3
candidate_k = 20
token_budget = 32
random_seed = 42
repeat = 1
query_state_policy = independent
reinforcement_enabled = false
```

---

## 11. 结果管理

每次运行会生成：

```text
raw.jsonl
summary.json
summary.csv
environment.json
```

正式结果必须绑定：

- Git commit；
- 数据集 SHA-256；
- 完整配置；
- 方法 Registry 名称；
- 随机种子；
- 环境信息；
- 原始结果和汇总结果。

禁止：

- 手工修改结果；
- 混用不同 commit 的结果；
- 不同方法使用不同预算；
- 提交模型权重、缓存或临时索引；
- 提交本机绝对路径；
- Oracle 读取 gold 字段。

开发阶段产生的结果默认不提交。

删除调试结果：

```powershell
Remove-Item -Recurse -Force results/fair_experiments/m1
```

---

## 12. 当前交付内容
已完成：

- 固定版本数据集；
- dataset manifest；
- SHA-256 校验；
- 数据校验模块与测试；
- 六方法主实验配置；
- 公平参数一致性测试；
- 六方法机器可读矩阵；
- 36 个单变量消融配置；
- 消融配置生成器与测试；
- Oracle 防 gold leakage 约束；
- M1 实验协议。
