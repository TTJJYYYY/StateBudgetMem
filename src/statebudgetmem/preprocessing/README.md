# preprocessing

本目录负责把非结构化自然语言转换成项目统一使用的 `MemoryRecord`。

## 目标

它是整个 StateBudgetMem 项目的数据入口：

```text
raw text / chat / note
        ↓
preprocessing
        ↓
Scenario / MemoryRecord
        ↓
baselines / versioning / views
