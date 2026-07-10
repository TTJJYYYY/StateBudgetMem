# MemoryBank Phase 1 复现数据集说明

## 1. 数据集目标

`data/memorybank_reproduction/` 是 StateBudgetMem Phase 1 使用的、可离线复现的 MemoryBank-style 小型数据集。它的目标不是复刻 MemoryBank 原论文的全部实验数据，而是为当前项目提供一套：

- 能被 `run_phase1_baseline.py` 直接读取的数据；
- 同时包含对话、事件摘要和用户画像的三层记忆输入；
- 具有明确 `gold_memory_ids`、参考答案和关键词标签的 probing questions；
- 不依赖云端 API、在不同机器上可重复构建和校验的数据；
- 能覆盖事实回忆、事件总结、用户画像、负例和跨时间变化等长期记忆能力的数据。

该数据集由虚拟用户和人工编写的 fixture 构成，不包含真实用户隐私数据。

---

## 2. 目录结构

```text
data/
├── controlled/
├── external/
└── memorybank_reproduction/
    ├── users/
    │   ├── user_001.json
    │   ├── user_002.json
    │   ├── user_003.json
    │   ├── user_004.json
    │   └── user_005.json
    ├── probing_questions.jsonl
    └── README.md
```

相关代码和生成结果：

```text
src/statebudgetmem/baselines/memorybank/datasets.py
tools/memorybank/build_reproduction_dataset.py
tests/baselines/memorybank/test_datasets.py
results/memorybank/reproduction_storage/summary.json
```

其中：

- `users/*.json` 是原始对话、每日 fixture、全局摘要和全局画像的事实来源；
- `probing_questions.jsonl` 保存评测问题和 gold labels；
- `datasets.py` 负责加载、建立 memory catalog 和严格校验；
- `build_reproduction_dataset.py` 校验数据并生成可复现的 summary/portrait fixture 文件；
- `summary.json` 是生成结果，不是独立的数据源。

---

## 3. 当前数据规模

当前版本的数据统计如下：

| 项目 | 数量 |
|---|---:|
| 虚拟用户 | 5 |
| 每位用户对话天数 | 7 天 |
| user-day 总数 | 35 |
| 每天对话轮次 | 4 条消息，即 2 组 user-assistant 交互 |
| dialog memory | 140 |
| 全局事件摘要 memory | 5 |
| 全局用户画像 memory | 5 |
| 可寻址 memory source 总数 | 150 |
| probing questions | 50 |
| 每位用户问题数 | 10 |

所有用户的对话日期范围均为：

```text
2026-06-20 至 2026-06-26
```

### 3.1 问题类型分布

| `question_type` | 数量 | 主要评测内容 |
|---|---:|---|
| `memory_recall` | 22 | 回忆明确出现过的事实、事件、偏好或计划 |
| `event_summary` | 7 | 根据一天或多天经历总结主要事件 |
| `user_portrait` | 8 | 根据长期行为和偏好回答用户特征 |
| `negative_memory` | 5 | 判断某件未发生的事情是否存在于记忆中 |
| `temporal_memory` | 8 | 回答先后状态、变化过程或跨日信息 |
| **合计** | **50** |  |

每位用户至少包含：

- 1 个 `negative_memory`；
- 1 个 `user_portrait`；
- 1 个 `temporal_memory`。

---

## 4. 用户数据字段定义

每个 `users/user_XXX.json` 表示一位虚拟用户。

### 4.1 顶层字段

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `user_id` | string | 是 | 用户唯一标识，例如 `user_001` |
| `profile` | object | 是 | 用户的基础设定，用于编写一致的对话场景 |
| `days` | array | 是 | 按日期组织的对话与每日 fixture |
| `global_event_summary` | string | 是 | 对全部日期事件的人工全局总结 |
| `global_user_portrait` | string | 是 | 根据全部对话形成的人工长期画像 |
| `global_memory_ids` | object | 建议 | 显式记录全局摘要和画像使用的 memory ID |

`profile` 当前通常包含：

```json
{
  "name": "Lin",
  "age": 21,
  "occupation": "college student",
  "personality": "curious, self-motivated, and willing to learn",
  "interests": ["programming", "data analysis", "technology"]
}
```

`profile` 主要用于保持数据编写的一致性。评测问题应优先引用对话、摘要或画像中可以追溯的内容，而不是只引用未被 MemoryBank 写入的 profile 字段。

### 4.2 `days` 字段

每个 day object 的结构如下：

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `date` | string | 是 | 日期，格式为 `YYYY-MM-DD` |
| `dialogues` | array | 是 | 当日对话消息 |
| `daily_event_summary` | string | 是 | 当日事件摘要 fixture |
| `daily_personality` | string | 是 | 根据当日表现归纳的人格或状态 fixture |

### 4.3 `dialogues` 字段

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `memory_id` | string | 是 | 对话 memory 的唯一标识 |
| `role` | string | 是 | `user` 或 `assistant` |
| `content` | string | 是 | 对话正文 |
| `timestamp` | string | 是 | ISO 风格时间戳 |

对话 memory ID 使用以下形式：

```text
<user_id>_day<两位日期序号>_dialog<两位消息序号>
```

示例：

```text
user_001_day02_dialog03
```

所有对话 ID 必须：

- 在整个数据集中唯一；
- 以对应的 `user_id` 开头；
- 与 probing question 中的 `gold_memory_ids` 完全一致。

### 4.4 全局 memory ID

全局事件摘要和用户画像采用确定性 ID：

```text
<user_id>_global_event_summary
<user_id>_global_user_portrait
```

例如：

```text
user_001_global_event_summary
user_001_global_user_portrait
```

当前 B3 memory catalog 可寻址的内容包括：

- 140 条 dialog memory；
- 5 条 global event summary；
- 5 条 global user portrait。

每日的 `daily_event_summary` 和 `daily_personality` 会进入 fixture 构建结果，但当前没有独立、稳定的 gold memory ID，因此 probing questions 不应直接把每日 fixture 当作唯一 gold ID。

---

## 5. Probing question 字段定义

`probing_questions.jsonl` 每行是一个独立 JSON object，必须包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `query_id` | string | 问题唯一标识，例如 `q001` |
| `user_id` | string | 问题所属用户 |
| `question` | string | 输入给记忆系统的问题 |
| `question_type` | string | 五种受支持的问题类型之一 |
| `reference_answer` | string | 确定性参考答案 |
| `gold_memory_ids` | array[string] | 回答该问题所需的真实 memory ID |
| `expected_keywords` | array[string] | 用于 deterministic proxy evaluation 的关键词 |

示例：

```json
{
  "query_id": "q006",
  "user_id": "user_001",
  "question": "How did my career plan change over time?",
  "question_type": "temporal_memory",
  "reference_answer": "I changed from preparing for graduate school to pursuing software and AI development.",
  "gold_memory_ids": [
    "user_001_day01_dialog03",
    "user_001_day05_dialog01",
    "user_001_day07_dialog03",
    "user_001_global_event_summary"
  ],
  "expected_keywords": ["graduate school", "software", "AI"]
}
```

### 5.1 五种问题类型

#### `memory_recall`

针对明确出现过的单一事实或事件，例如学习内容、饮食、活动、职业目标。

#### `event_summary`

要求归纳某段时间中的主要经历。可以同时引用原始对话和 `global_event_summary`。

#### `user_portrait`

要求描述用户的长期倾向、性格或行为特征。通常引用：

```text
<user_id>_global_user_portrait
```

#### `negative_memory`

询问一件数据中没有发生的事情。其约束是：

- `gold_memory_ids` 必须为空数组；
- `reference_answer` 必须明确否定；
- `expected_keywords` 必须同时包含否定词和具体对象；
- 具体对象不能真实出现在该用户的 memory 文本中。

示例：

```json
{
  "query_id": "q009",
  "user_id": "user_001",
  "question": "Did I ever mention learning Java as my main goal?",
  "question_type": "negative_memory",
  "reference_answer": "No, I did not mention learning Java as my main goal.",
  "gold_memory_ids": [],
  "expected_keywords": ["No", "Java"]
}
```

#### `temporal_memory`

要求理解至少两个时间点之间的先后关系、状态变化或发展过程。此类问题通常需要多个 `gold_memory_ids`。

---

## 6. Fixture summary / portrait 的来源

MemoryBank 原始方案中的 summary 和 portrait 通常由模型生成。为了保证 Phase 1 在无网络、无云端 API 的环境下可重复运行，本数据集采用 **fixture mode**。

### 6.1 Fixture 的生成原则

当前 fixture 由数据编写者人工整理，而不是运行时调用 LLM 生成：

- `daily_event_summary`：从同一天的 `dialogues` 中概括事件；
- `daily_personality`：根据同一天用户表现做简短归纳；
- `global_event_summary`：综合 7 天事件，保留主要发展过程；
- `global_user_portrait`：综合长期兴趣、行为和目标形成画像。

每个 fixture 都应满足：

1. 能从原始对话追溯；
2. 不增加对话中完全没有依据的新事实；
3. 不把 profile 中的设定直接伪装成对话事实；
4. 明确标记为 fixture，不声称是模型自动生成结果。

### 6.2 构建结果中的来源标记

运行构建脚本后，`summary.json` 会记录：

```json
{
  "summary_mode": "fixture",
  "portrait_mode": "fixture",
  "source": "manual_fixture"
}
```

因此，结果文件能够明确区分人工 fixture 和未来可能接入的 local summarizer 输出。

---

## 7. 构建与校验方法

### 7.1 环境准备

项目要求 Python 3.11 或更高版本。推荐在独立 Conda 环境中安装：

```powershell
conda create -n statebudgetmem python=3.11 -y
conda activate statebudgetmem
cd D:\memorybank\StateBudgetMem
python -m pip install -e ".[memorybank,test]"
```

### 7.2 生成 fixture summary

在项目根目录运行：

```powershell
python tools\memorybank\build_reproduction_dataset.py
```

当前版本的预期输出为：

```text
Validated reproduction dataset: 5 users, 35 user-days, 50 probes, 150 addressable memory sources.
Saved fixture summary to results\memorybank\reproduction_storage\summary.json
```

生成文件：

```text
results/memorybank/reproduction_storage/summary.json
```

脚本不会盲目覆盖结果。它会先调用 `load_reproduction_dataset()` 完成严格校验；数据不合法时会直接报错并停止。

### 7.3 运行数据集测试

```powershell
pytest tests\baselines\memorybank\test_datasets.py -q
```

当前版本预期：

```text
11 passed
```

运行全部 MemoryBank 测试：

```powershell
pytest tests\baselines\memorybank -q
```

当前已验证结果：

```text
45 passed
```

### 7.4 运行正式 Phase 1 baseline

```powershell
python tools\memorybank\run_phase1_baseline.py
```

快速只运行第一位用户：

```powershell
python tools\memorybank\run_phase1_baseline.py --quick
```

使用确定性的本地 hash embedding：

```powershell
python tools\memorybank\run_phase1_baseline.py --embedding-backend hash
```

---

## 8. 自动校验规则

`datasets.py` 会执行以下检查：

### 8.1 用户和 memory 检查

- `user_id` 不为空且不重复；
- 用户文件必须包含合法的 `profile` 和 `days`；
- 每条 dialog 必须有非空 `memory_id`；
- memory ID 必须以所属 `user_id` 开头；
- memory ID 在整个数据集中不得重复。

### 8.2 Probe 字段检查

- 每行必须是合法 JSON object；
- 七个必需字段必须全部存在；
- `query_id` 必须唯一；
- `user_id` 必须对应真实用户；
- `question_type` 必须属于五种受支持类型；
- `question` 和 `reference_answer` 不得为空。

### 8.3 Gold label 检查

- 非负例至少有一个 `gold_memory_id`；
- 负例的 `gold_memory_ids` 必须为空；
- 所有 gold ID 必须真实存在；
- gold memory 必须属于问题对应的用户；
- 同一问题不能重复写入相同 gold ID。

### 8.4 Keyword 检查

- `expected_keywords` 不得为空；
- 单个关键词不能过短或过于宽泛；
- 关键词不得重复；
- 每个关键词必须能在 `reference_answer` 中找到；
- 负例必须包含明确否定词和问题特定对象。

### 8.5 覆盖度检查

- 数据集整体必须覆盖五种问题类型；
- 每个用户至少覆盖 `negative_memory`、`user_portrait` 和 `temporal_memory`。

---

## 9. 与 MemoryBank 原论文数据设置的差异

本数据集是课程项目中的小规模、可控 reproduction dataset，不应表述为 MemoryBank 原论文完整数据集的复刻。

| 维度 | MemoryBank 原论文实验设置 | StateBudgetMem Phase 1 |
|---|---:|---:|
| 用户数量 | 15 | 5 |
| 每位用户对话天数 | 10 天 | 7 天 |
| probing questions | 194 | 50 |
| 用户来源 | 原论文实验参与者/设定 | 人工构造的虚拟用户 |
| summary / portrait | 模型生成流程 | 人工 fixture |
| 运行环境 | 包含论文中的模型流程 | 可完全离线运行 |
| 评测标签 | 原论文评测方案 | `gold_memory_ids` + reference answer + keyword proxy |
| 数据目标 | 验证论文完整系统 | 验证端侧 baseline 的数据、检索和评测闭环 |

因此，当前结果适合用于：

- 检查 MemoryBank baseline 能否完整跑通；
- 比较不同 embedding、top-k、forgetting 和预算配置；
- 为 Phase 2 的 Versioning、Views 和 Routing 提供统一对照数据。

当前结果不适合直接用于：

- 复述原论文中的最终性能结论；
- 宣称达到与原论文相同的数据规模或人工评测质量；
- 用 50 个问题推断真实用户群体中的通用表现。

---

## 10. 当前限制

1. **数据规模较小**：只有 5 位虚拟用户和 50 个问题，主要用于课程项目中的受控复现。
2. **对话为人工构造**：语言和主题分布比真实长期对话更规则，难以覆盖噪声、隐含表达和矛盾信息。
3. **Fixture 不是模型输出**：summary 和 portrait 不能衡量自动 Memory Writing 的生成质量，只能隔离并验证后续存储与检索流程。
4. **中英文混合**：对话主要为中文，问题和参考答案主要为英文；这有利于测试跨语言 embedding，但 hash embedding 下语义检索能力有限。
5. **关键词指标是 proxy**：`expected_keywords` 只能近似判断回答正确性，不能替代人工或强模型评测。
6. **每日 fixture 尚无稳定 gold ID**：当前可寻址 gold memory 只包括 dialog、全局事件摘要和全局画像。
7. **多用户运行需要保持隔离**：正式评测时应为每位用户建立独立 MemoryBank，或在检索中强制按 `user_id` 过滤，避免不同虚拟用户的记忆互相干扰；全局 summary/portrait 也应按用户分别维护。
8. **时间跨度固定**：所有用户均为连续 7 天，尚未覆盖数周或数月的长时间遗忘与复习场景。

---

## 11. 后续扩展方向

后续可以在不破坏现有数据接口的前提下扩展：

- 将用户数扩展到 10 至 15；
- 将每位用户时间跨度扩展到 10 天以上；
- 增加状态冲突、过期偏好、临时失效和恢复等问题；
- 为每日 summary / personality 分配稳定 memory ID；
- 增加中文 probing questions，并分别报告单语和跨语言结果；
- 接入本地 summarizer，和人工 fixture 做对照；
- 增加 paraphrase probes，降低关键词重合带来的评测偏差；
- 为 Versioning、Current View 和 History View 增加专门的 temporal gold labels。

---

## 12. B4 验收清单

- [x] 说明数据集目标；
- [x] 说明用户数量和对话天数；
- [x] 说明 probing questions 数量和类型分布；
- [x] 定义用户、对话、fixture 和 probe 字段；
- [x] 说明 fixture summary / portrait 的来源；
- [x] 给出数据构建与测试命令；
- [x] 说明自动校验规则；
- [x] 列出当前限制；
- [x] 对比 MemoryBank 原论文的 10 天、15 用户和 194 questions 设置。