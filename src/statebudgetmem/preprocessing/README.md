# preprocessing

`preprocessing` 模块负责将原始自然语言记忆转换为项目统一使用的 `MemoryPiece`。

它是整个 StateBudgetMem 的数据入口，不负责最终的版本更新。它只负责把自然语言整理成结构化记忆，供后续 `baselines`、`versioning`、`views` 等模块使用。

## 1. 功能概述

整体流程：

```text
RawMemoryInput
      ↓
preprocessing
      ↓
StructuredMemory
      ↓
MemoryPiece
      ↓
baselines / versioning / views
```

当前支持三种解析模式：

```text
rule    规则解析，离线可运行，适合作为 baseline
api     外部 API 解析，适合处理复杂自然语言
hybrid  优先使用 API，失败后回退到规则解析
```

推荐默认使用 `hybrid`，测试时使用 `rule`。

## 2. 文件说明

```text
preprocessing/
├── README.md
├── __init__.py
├── models.py
├── normalizer.py
├── rule_parser.py
├── api_parser.py
└── pipeline.py
```

### `models.py`

定义预处理模块内部使用的数据结构：

- `RawMemoryInput`：原始自然语言输入；
- `StructuredMemory`：预处理后的结构化草稿；
- `PreprocessConfig`：预处理配置；
- `OperationHint`：给 `versioning` 的更新提示。

其中 `StructuredMemory` 最终会转换为项目已有的 `MemoryRecord`。

### `normalizer.py`

负责文本清洗和字段标准化，包括：

- 去除简单口语填充词；
- 切分长文本；
- 清洗抽取出的 `value`；
- 统一 `attribute` 名称；
- 推断 `memory_type`；
- 估算 `token_cost`。

示例：

```text
住处 / 住址 / 城市 / 居住地 -> home_location
早餐 / 早饭 -> breakfast
过敏 -> allergy
```

### `rule_parser.py`

规则版解析器，不依赖外部 API。

主要用于：

- 离线可复现实验；
- 没有 API key 时保证项目可运行；
- `hybrid` 模式下作为 fallback。

当前支持常见表达：

```text
我对花生过敏 -> allergy=花生
我现在住在北京 -> home_location=北京
我喜欢喝茶 -> preference=like:喝茶
我不喝咖啡 -> preference=avoid:咖啡
早餐通常吃燕麦 -> breakfast=燕麦
我以前住上海，现在搬到北京 -> previous_value=上海, value=北京, operation_hint=SUPERSEDE
```

### `api_parser.py`

外部 API 版解析器，用于处理复杂自然语言。

特点：

- 使用 API 进行语义解析；
- 使用 JSON Schema 约束输出格式；
- 不在代码中写死 API key；
- 从环境变量 `OPENAI_API_KEY` 读取 key；
- 输出统一转换为 `StructuredMemory`。

没有 API key 时，`api` 模式会报错；`hybrid` 模式会自动回退到规则解析。

### `pipeline.py`

预处理主流程入口。

核心类：

```python
MemoryPreprocessor
```

主要功能：

- 根据配置选择 `rule`、`api` 或 `hybrid`；
- 调用对应 parser；
- 按置信度过滤结果；
- 生成 `memory_id`；
- 将 `StructuredMemory` 转换为 `MemoryRecord`；
- 生成 `Scenario`。

### `__init__.py`

统一导出常用类，方便其他模块导入：

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, RawMemoryInput, PreprocessConfig
```

## 3. 输入接口

预处理输入为 `RawMemoryInput`。

最小示例：

```json
{
  "raw_id": "r1",
  "text": "我以前住上海，现在搬到北京了。",
  "observed_at": "2026-06-29"
}
```

## 4. 输出接口

预处理最终输出项目已有的 `MemoryRecord`。

额外预处理信息存入：

```python
MemoryPiece 的 tags / source / confidence / query_types 等字段
```

当前约定字段：

| 字段 | 说明 |
|---|---|
| `source_raw_id` | 原始输入 ID |
| `previous_value` | 旧值 |
| `operation_hint` | 更新操作提示 |
| `evidence_span` | 原文证据片段 |
| `needs_review` | 是否建议人工检查 |
| `parser` | 使用的解析器：`rule` 或 `api` |
| `source_type` | 原始输入来源 |
| `speaker` | 说话人 |

示例：

```json
{
  "source_raw_id": "r1",
  "previous_value": "上海",
  "operation_hint": "SUPERSEDE",
  "evidence_span": "我以前住上海，现在搬到北京了",
  "needs_review": false,
  "parser": "rule"
}
```

## 5. 使用方式

准备原始数据：

```json
{"raw_id":"r1","text":"我以前住上海，现在搬到北京了。","observed_at":"2026-06-29"}
{"raw_id":"r2","text":"我最近不喝咖啡了，改喝茶。早餐通常吃燕麦。","observed_at":"2026-06-29"}
{"raw_id":"r3","text":"我对花生过敏。现在住在上海。","observed_at":"2026-06-29"}
```

规则模式：

```bash
python -m statebudgetmem.cli preprocess \
  --input data/raw/user_notes.jsonl \
  --output data/processed/user_scenarios.jsonl \
  --scenario-id S_USER_RAW \
  --parser rule
```

API 模式：

```bash
python -m statebudgetmem.cli preprocess \
  --input data/raw/user_notes.jsonl \
  --output data/processed/user_scenarios.jsonl \
  --scenario-id S_USER_RAW \
  --parser api
```

Hybrid 模式：

```bash
python -m statebudgetmem.cli preprocess \
  --input data/raw/user_notes.jsonl \
  --output data/processed/user_scenarios.jsonl \
  --scenario-id S_USER_RAW \
  --parser hybrid
```

## 6. 与其他模块的关系

### 与 `schemas/`

`schemas/` 定义全项目正式数据结构，例如 `MemoryRecord`、`Scenario`。

`preprocessing/models.py` 只定义预处理阶段使用的临时结构，例如 `RawMemoryInput`、`StructuredMemory`。

最终输出仍然以 `MemoryRecord` 为准。

### 与 `versioning/`

`preprocessing` 不直接做版本更新，只提供：

```text
previous_value
operation_hint
evidence_span
```

最终如何更新旧记忆，由 `versioning` 模块决定。

### 与 `views/`

`views` 不需要重新解析自然语言，只使用处理后的 `MemoryRecord` 构建 Current View 和 History View。

### 与 `baselines/`

`baselines` 可以直接使用 preprocessing 输出的 `Scenario` 文件进行实验。

## 7. 后续 TODO

- 确认 `operation_hint` 枚举是否与 `versioning` 完全一致；
- 确认 `previous_value` 是否足够表达旧状态；
- 确认 `metadata` 中的字段是否需要上移到公共 schema；
- 确认 API 模型和 key 管理方式；
- 补充更多规则和测试样例。

## 8. 注意事项

- 不要把 API key 写进代码或提交到 Git；
- 测试不要强依赖外部 API，建议使用 `rule` 模式；
- `operation_hint` 只是提示，不是最终更新结果；
- 不要在 preprocessing 中重新定义 `MemoryRecord`；
- 规则版是 baseline 和 fallback，不要求覆盖所有复杂自然语言。
