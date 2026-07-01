# preprocessing

`preprocessing` 模块负责将原始自然语言记忆转换为项目统一使用的 `MemoryPiece`。

它是 StateBudgetMem 的数据入口之一，不负责最终的版本更新，也不维护 Current View / History View。它只负责把自然语言整理成结构化记忆，供后续 `baselines`、`versioning`、`views` 等模块使用。

## 1. 功能概述

整体流程：

```text
RawMessage
    ↓
preprocessing
    ↓
ParsedMemory
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

推荐：

- 测试和离线实验使用 `rule`；
- 演示或复杂自然语言处理使用 `hybrid`；
- 只想测试外部 API 效果时使用 `api`。

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

定义 preprocessing 内部使用的数据结构。

主要包括：

- `RawMessage`：原始自然语言输入；
- `ParsedMemory`：预处理后的结构化草稿；
- `PreprocessConfig`：预处理配置；
- `messages_to_raw_messages()`：将简单消息列表转换为 `RawMessage`。

`ParsedMemory` 最终会转换成项目统一接口中的 `MemoryPiece`。

### `normalizer.py`

负责文本清洗和字段标准化。

主要功能：

- 清洗原始文本；
- 切分长文本；
- 清洗抽取出的值；
- 统一属性名称；
- 估算简单 token cost。

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

当前支持部分常见表达：

```text
我对花生过敏 -> allergy=花生
我现在住在北京 -> home_location=北京
我喜欢喝茶 -> preference=like:茶
我不喝咖啡 -> preference=avoid:咖啡
早餐通常吃燕麦 -> breakfast=燕麦
我以前住上海，现在搬到北京 -> previous_value=上海, value=北京, operation=SUPERSEDE
```

### `api_parser.py`

外部 API 版解析器，用于处理复杂自然语言。

特点：

- 使用外部 API 进行语义解析；
- 使用结构化输出约束返回格式；
- 不在代码中写死 API key；
- 从环境变量读取 API key；
- 输出统一转换为 `ParsedMemory`。

可用的 API key 环境变量：

```text
SBM_API_KEY
DEEPSEEK_API_KEY
OPENAI_API_KEY
```

如果没有 API key，`api` 模式会报错；`hybrid` 模式会自动回退到规则解析器。

### `pipeline.py`

预处理主流程入口。

核心类：

```python
MemoryPreprocessor
```

主要功能：

- 根据配置选择 `rule`、`api` 或 `hybrid`；
- 调用对应 parser；
- 按置信度过滤解析结果；
- 将 `RawMessage` 转换为 `ParsedMemory`；
- 将 `ParsedMemory` 转换为 `MemoryPiece`。

### `__init__.py`

统一导出常用类，方便其他模块导入。

示例：

```python
from statebudgetmem.preprocessing import (
    MemoryPreprocessor,
    PreprocessConfig,
    RawMessage,
)
```

## 3. 输入接口

preprocessing 的输入是 `RawMessage`。

最小示例：

```python
RawMessage(
    role="user",
    content="我以前住上海，现在搬到北京了。",
    timestamp="2026-06-29",
)
```

也可以使用简单消息元组：

```python
messages = [
    ("user", "我以前住上海，现在搬到北京了。", "2026-06-29")
]
```

## 4. 输出接口

preprocessing 的输出是项目统一使用的 `MemoryPiece`。

`MemoryPiece` 来自：

```python
from statebudgetmem.interfaces import MemoryPiece
```

preprocessing 不重新定义全项目的记忆格式，而是将内部的 `ParsedMemory` 转换为统一的 `MemoryPiece`，供后续模块使用。

## 5. 使用方式

### 规则模式：输入 `RawMessage`

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig, RawMessage

messages = [
    RawMessage(
        role="user",
        content="我以前住上海，现在搬到北京了。",
        timestamp="2026-06-29",
    )
]

preprocessor = MemoryPreprocessor(
    PreprocessConfig(parser_type="rule")
)

memory_pieces = preprocessor.preprocess_raw_messages(messages)
```

### 规则模式：输入简单元组

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig

messages = [
    ("user", "我以前住上海，现在搬到北京了。", "2026-06-29")
]

preprocessor = MemoryPreprocessor(
    PreprocessConfig(parser_type="rule")
)

memory_pieces = preprocessor.preprocess_messages(messages)
```

### Hybrid 模式

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig

messages = [
    ("user", "我最近不喝咖啡了，改喝茶。早餐通常吃燕麦。", "2026-06-29")
]

preprocessor = MemoryPreprocessor(
    PreprocessConfig(parser_type="hybrid")
)

memory_pieces = preprocessor.preprocess_messages(messages)
```

### API 模式

使用 API 模式前，需要安装可选依赖：

```bash
pip install -e ".[llm]"
```

然后设置环境变量，例如：

```bash
export OPENAI_API_KEY="your_api_key"
```

或：

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

然后使用：

```python
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig

preprocessor = MemoryPreprocessor(
    PreprocessConfig(parser_type="api")
)
```

注意：不要把 API key 写入代码，也不要提交到 Git。

## 6. 与其他模块的关系

### 与 `interfaces.py`

`interfaces.py` 定义全项目统一接口，例如：

- `MemoryPiece`
- `MemoryType`
- `MemoryStatus`
- `UpdateOperation`
- `QueryType`

preprocessing 最终输出 `MemoryPiece`，从而与 baseline 和后续模块保持一致。

### 与 `baselines`

`baselines` 可以直接接收 preprocessing 输出的 `MemoryPiece`，用于检索实验和基线比较。

### 与 `versioning`

preprocessing 不直接做版本更新，只提供初步解析结果和更新提示。

例如：

```text
我以前住上海，现在搬到北京了。
```

preprocessing 可以解析出：

```text
attribute = home_location
previous_value = 上海
value = 北京
operation = SUPERSEDE
```

最终是否让旧记忆失效、是否建立版本关系，由 `versioning` 模块决定。

### 与 `views`

`views` 不需要重新解析自然语言，只使用处理后的 `MemoryPiece` 构建 Current View 和 History View。

## 7. 后续 TODO

- 确认 `operation` 是否完全满足 `versioning` 的需要；
- 确认 `previous_value` 是否足够表达旧状态；
- 确认解析出的结构化字段是否需要进一步统一命名；
- 补充更多规则解析样例；
- 补充 preprocessing 的单元测试；
- 视情况增加命令行入口或数据文件读写工具。

## 8. 注意事项

- 不要提交 API key；
- 自动测试不要强依赖外部 API；
- `rule_parser.py` 是 baseline 和 fallback，不要求覆盖所有复杂表达；
- `api_parser.py` 负责处理更复杂的自然语言；
- `operation` 只是预处理阶段的提示，不是最终版本更新结果；
- preprocessing 不应该重新定义全项目的最终记忆格式，应使用 `statebudgetmem.interfaces.MemoryPiece`。
