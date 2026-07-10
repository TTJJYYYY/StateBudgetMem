# MemoryBank Phase 1 Reproduction Dataset

本目录保存 StateBudgetMem Phase 1 使用的小型、可离线复现的 MemoryBank-style 数据集。

## 数据规模

- 5 个虚拟用户；
- 每个用户 7 天对话，共 35 个 user-day；
- 140 条 dialog memory；
- 5 条全局事件摘要和 5 条全局用户画像；
- 50 个 probing questions；
- 150 个可寻址 memory source。

问题覆盖：

```text
memory_recall      22
event_summary       7
user_portrait       8
negative_memory     5
temporal_memory     8
```

## 目录

```text
users/*.json              用户对话与人工 fixture
probing_questions.jsonl   问题、参考答案和 gold labels
```

## 构建与校验

在项目根目录运行：

```powershell
python tools\memorybank\build_reproduction_dataset.py
pytest tests\baselines\memorybank\test_datasets.py -q
```

生成结果：

```text
results/memorybank/reproduction_storage/summary.json
```

`summary.json` 是从 `users/*.json` 生成的 fixture 结果，不是独立的数据源。

## 重要说明

- 所有用户均为人工构造的虚拟用户，不包含真实个人数据；
- summary 和 portrait 是 `manual_fixture`，不是 LLM 自动生成结果；
- `gold_memory_ids` 必须指向真实存在且属于同一用户的 memory；
- `negative_memory` 的 `gold_memory_ids` 必须为空；
- 正式多用户评测应隔离每位用户的 MemoryBank，避免跨用户记忆污染。

完整字段定义、校验规则、限制以及与原论文设置的差异见：

```text
docs/memorybank_phase1_dataset.md
```