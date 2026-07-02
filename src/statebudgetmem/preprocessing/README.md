# preprocessing

`preprocessing` 模块负责将原始自然语言消息解析成结构化记忆草稿，并提供到实验层统一格式的转换入口。

它不负责最终的版本更新，也不维护 `Current View / History View`。它只负责把自然语言整理成结构化字段，供后续 `versioning`、`views`、`baselines`、`evaluation` 使用。

---

## 1. 接口分层

本项目目前有两层记忆接口：

```text
在线层接口
└── MemoryPiece
    └── 适合在线记忆系统、demo、MemorySystem 风格接口

实验层接口
└── MemoryRecord
    └── 适合 controlled dataset、versioning、views、evaluation
```

因此 preprocessing 的推荐流程是：

```text
RawMessage / message tuples
        ↓
MemoryPreprocessor
        ↓
ParsedMemory
        ↓
record_adapter.py
        ↓
MemoryRecord
        ↓
versioning / views / baselines / evaluation
```

模块仍然保留旧流程：

```text
ParsedMemory
        ↓
MemoryPiece
```

但如果要接 `versioning/views`，不建议优先走 `MemoryPiece`，因为 `MemoryPiece` 没有一等字段 `attribute / value / previous_value / operation`，会丢掉部分结构化信息。

---

## 2. 文件说明

```text
preprocessing/
├── README.md
├── __init__.py
├── models.py
├── normalizer.py
├── rule_parser.py
├── api_parser.py
├── pipeline.py
└── record_adapter.py
```

### `models.py`

定义 preprocessing 内部数据结构：

```text
RawMessage
ParsedMemory
PreprocessConfig
```

其中：

* `RawMessage`：原始消息；
* `ParsedMemory`：结构化记忆草稿；
* `PreprocessConfig`：预处理配置；
* `messages_to_raw_messages()`：把简单消息元组转换为 `RawMessage`。

`ParsedMemory` 中保留了对后续版本管理重要的字段：

```text
attribute
value
previous_value
operation
evidence_span
confidence
tags
```

这些字段会通过 `record_adapter.py` 转入 `MemoryRecord`。

---

### `normalizer.py`

负责文本清洗和字段标准化，例如：

```text
住处 / 住址 / 城市 / 居住地 -> home_location
早餐 / 早饭 -> breakfast
过敏 -> allergy
```

这样可以减少不同表达方式对后续版本管理的影响。

---

### `rule_parser.py`

规则解析器，不依赖外部 API。

主要用于：

* 离线可复现实验；
* 没有 API key 时保证项目可运行；
* `hybrid` 模式下作为 fallback。

当前支持部分常见表达：

```text
我对花生过敏 -> allergy=花生
我现在住在北京 -> home_location=北京
我喜欢喝茶 -> preference=like:茶
我不喝咖啡 -> preference=avoid:咖啡
早餐通常吃燕麦 -> breakfast=燕麦
我以前住上海，现在搬到北京 -> previous_value=上海, value=北京, operation=SUPERSEDE
```

`rule_parser.py` 不需要覆盖所有复杂自然语言场景，它的定位是 baseline 和 fallback。

---

### `api_parser.py`

外部 API 解析器，用于复杂自然语言场景。

可用环境变量：

```text
SBM_API_KEY
DEEPSEEK_API_KEY
OPENAI_API_KEY
SBM_API_BASE_URL
SBM_API_MODEL
```

注意：

```text
不要把 API key 写入代码；
不要把 API key 提交到 Git；
自动测试不要强依赖外部 API。
```

---

### `pipeline.py`

预处理主入口：

```python
MemoryPreprocessor
```

常用方法：

```python
parse_messages()              # message tuples -> ParsedMemory
preprocess_messages()         # message tuples -> MemoryPiece
parse_messages_to_records()   # message tuples -> MemoryRecord，推荐接 views/versioning
```

推荐用法：

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig

preprocessor = MemoryPreprocessor(
    PreprocessConfig(parser_type="rule")
)

records = preprocessor.parse_messages_to_records([
    ("user", "我以前住上海，现在搬到北京了。", "2026-06-29")
])
```

---

### `record_adapter.py`

统一适配器，用于把 preprocessing 输出接入实验链路。

主要函数：

```python
parsed_memory_to_record()
parsed_memories_to_records()
memory_piece_to_record()
memory_pieces_to_records()
preprocess_messages_to_records()
```

推荐优先使用：

```text
ParsedMemory -> MemoryRecord
```

不推荐优先使用：

```text
ParsedMemory -> MemoryPiece -> MemoryRecord
```

因为后者是有损转换。

---

## 3. 使用方式

### 3.1 输出 ParsedMemory

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig

messages = [
    ("user", "我以前住上海，现在搬到北京了。", "2026-06-29")
]

preprocessor = MemoryPreprocessor(
    PreprocessConfig(parser_type="rule")
)

parsed = preprocessor.parse_messages(messages)

for item in parsed:
    print(item.attribute, item.value, item.previous_value, item.operation)
```

---

### 3.2 推荐：输出 MemoryRecord，接 versioning/views

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig
from statebudgetmem.views import DualViewMemoryMethod

messages = [
    ("user", "我以前住上海，现在搬到北京了。", "2026-06-29")
]

preprocessor = MemoryPreprocessor(
    PreprocessConfig(parser_type="rule")
)

records = preprocessor.parse_messages_to_records(messages)

method = DualViewMemoryMethod()
method.ingest(records)
```

也可以使用函数式入口：

```python
from statebudgetmem.preprocessing import preprocess_messages_to_records

records = preprocess_messages_to_records([
    ("user", "我以前住上海，现在搬到北京了。", "2026-06-29")
])
```

---

### 3.3 保留：输出 MemoryPiece

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig

preprocessor = MemoryPreprocessor(
    PreprocessConfig(parser_type="rule")
)

memory_pieces = preprocessor.preprocess_messages([
    ("user", "我对花生过敏。", "2026-06-29")
])
```

`MemoryPiece` 更适合在线接口或 demo；实验比较建议使用 `MemoryRecord`。

---

## 4. 与其他模块的关系

### 与 `versioning`

`preprocessing` 提供结构化提示：

```text
operation = SUPERSEDE
previous_value = 上海
value = 北京
```

`versioning` 根据 `MemoryRecord.metadata["versioning_intent"]` 等字段判断是否替代旧状态、合并、删除或临时失效。

---

### 与 `views`

`views` 不重新解析自然语言，只使用 `MemoryRecord` 构建：

```text
Current View
History View
Dual View
```

因此接入方式是：

```text
ParsedMemory -> MemoryRecord -> views
```

---

### 与 `baselines/evaluation`

`baselines/tfidf` 和 `evaluation` 的受控实验接口同样使用：

```text
MemoryRecord / QueryRecord / MethodResult
```

因此 preprocessing 接入实验链路时，也应优先输出 `MemoryRecord`。

---

## 5. 是否需要重写 preprocessing

不需要整体重写。

当前 `rule_parser.py`、`api_parser.py`、`pipeline.py` 的职责划分是合理的：

```text
rule_parser.py   负责离线规则抽取
api_parser.py    负责复杂语义抽取
pipeline.py      负责统一调度
models.py        负责内部结构
```

本次只需要补充 `record_adapter.py`，并在 `pipeline.py` 增加：

```python
parse_raw_messages_to_records()
parse_messages_to_records()
```

就能把 preprocessing 和 `views/versioning` 串起来。

---

## 6. 注意事项

* 自动测试不要强依赖外部 API；
* `rule_parser.py` 是 baseline 和 fallback，不要求覆盖所有复杂表达；
* `api_parser.py` 负责处理复杂自然语言；
* `operation` 是预处理阶段的可观察提示，不是最终版本更新结果；
* 最终版本关系由 `versioning` 决定；
* 实验链路优先使用 `MemoryRecord`；
* 在线链路可以继续使用 `MemoryPiece`。

---

## 7. 测试方式

测试 preprocessing：

```bash
PYTHONPATH=src pytest tests/preprocessing -q
```

如果要同时测试和 views 的衔接：

```bash
PYTHONPATH=src pytest tests/preprocessing tests/views -q
```

全量测试：

```bash
PYTHONPATH=src pytest -q
```
