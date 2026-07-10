# Phase 1: On-Device MemoryBank Baseline — Reproduction Report

> Phase 1 复现负责：唐婕颖（MemoryBank 机制）、黄筱婷（数据集）、**费哲瀚（评测与报告）**

## 1. 复现目标

在已有 MemoryBank prototype 基础上，构建完整的端侧 MemoryBank Baseline：
- 不依赖云端 API
- 三层存储模型（dialog / summary / portrait）
- Ebbinghaus 遗忘曲线
- 端侧资源可量化记录

## 2. 当前复现的 MemoryBank 机制

| 机制 | 状态 | 说明 |
|------|------|------|
| FAISS 检索 | ✅ | embedding + index search |
| Memory Strength | ✅ | recall 后 S+=1 |
| last_accessed 更新 | ✅ | recall 后更新 |
| Ebbinghaus 遗忘 R=exp(-t/S) | ✅ | `t` 默认以一天为单位，可通过 retention time unit 配置 |
| forgetting log | ✅ | 输出遗忘事件 |
| 可选 hard exclusion | ✅ | `--exclude-forgotten` 作为端侧/消融策略 |
| build_augmented_prompt() | ✅ | retrieval context + portrait + summary |
| hash embedding (CI) | ✅ | 确定性本地 embedding |
| sentence-transformer embedding | ✅ | 本地语义 embedding（首次可能需下载或使用本地路径） |

## 3. 使用的数据集

- **正式**：`data/memorybank_reproduction/`（黄筱婷构建）
  - 5 用户，每用户 5-10 天对话
  - 40-60 probing questions
  - gold_memory_ids + expected_keywords
- **Smoke**：`default_paper_storage_spec()`（内置样例，用于 CI）

## 4. 运行命令

```bash
# Smoke (CI)
python tools/memorybank/run_phase1_baseline.py --smoke

# 正式
python tools/memorybank/run_phase1_baseline.py

# 指定参数
python tools/memorybank/run_phase1_baseline.py \
  --embedding-backend hash \
  --exclude-forgotten \
  --top-k 5

# Budget sweep
python tools/memorybank/run_budget_sweep.py --quick
```

## 5. 指标定义

### Proxy Metrics（关键词覆盖度）
- `memory_retrieval_accuracy`：检索结果中包含 relevant_keywords 的比例
- `response_correctness`：回答中包含 expected_keywords 的比例
- `contextual_coherence`：回答与 query + retrieved memories 的重叠度
- `stale_retrieval_rate`：检索结果中包含 stale_keywords 的比例

### Gold-Label Metrics（精确标注）
- `gold_precision`：检索结果中 gold memories 的占比
- `gold_recall`：gold memories 中被检索到的占比
- `gold_f1`：F1 = 2×P×R/(P+R)

> **重要说明**：proxy metrics 是关键词级近似，不等同于 MemoryBank 原论文的人工评测。

## 6. 实验结果

（跑完后填写）

## 7. 端侧资源开销

（跑完后填写）

## 8. 与 MemoryBank 原论文差异

| 维度 | 原论文 | 本复现 |
|------|--------|--------|
| embedding | English: MiniLM; Chinese: Text2vec | sentence-transformer / hash |
| 用户数 | 15 | 5 |
| 天数 | 10 | 5-10 |
| 评测方式 | 人工标注 | gold labels + keyword proxy |
| LLM 调用 | 云端 LLM 生成 summary | fixture（预设） |
| 遗忘策略 | 基于 Ebbinghaus retention 和 recall reinforcement 的简化更新机制 | 默认记录 retention；另提供可选 hard threshold exclusion 作为端侧/消融策略；`update_forgetting_with_log()` 中 forgotten 后 `strength *= 0.5` 是项目遗留简化，不是论文原始规定 |
| SiliconFriend 微调 | 有 | 无 |

`--exclude-forgotten` 不是原论文唯一指定行为。默认模式保持基线兼容：
记录 retention 和 forgotten 标记，但不强制从 retrieval/prompt 中移除。
开启该开关后，低于 threshold 的候选不进入最终 Top-K、不进入 prompt、
也不会因本次 query 被强化。

当前实现中，retention 公式的 `t` 使用
`elapsed_seconds / decay_interval_sec`；默认 `decay_interval_hours=24.0`，
即 24 小时对应 `t=1`。raw retrieval rows 和 forgetting events 会记录
`elapsed_hours`、`elapsed_time_units` 和 `retention_time_unit_hours`，方便确认
当前使用的时间单位。这里仅更新机制说明，不填写尚未产生的正式实验结果。

## 9. 当前限制

- summary/portrait 是 fixture，不是 LLM 实时生成
- proxy metrics 是近似值，不等同于人工评测
- 数据集规模小于原论文
- 没有多用户交叉验证

## 10. 是否完成端侧 MemoryBank Baseline

（跑完实验后填写结论）

## 11. Phase 2 衔接

Phase 1 的 MemoryBank Baseline 将作为 Phase 2 的对照组：
- Versioning 引擎可以替代简单的 forgetting 机制
- Views 系统可以替代单一的 retrieval 路径
- Routing 模块可以替代固定 top-k 检索
- 端侧资源记录框架可直接复用
