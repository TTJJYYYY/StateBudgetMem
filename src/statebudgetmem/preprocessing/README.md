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

本模块支持三种解析模式：

rule：只用规则解析，离线可跑，适合作为 baseline；
api：只用外部 API，适合处理复杂自然语言；
hybrid：优先 API，失败后回退规则版，适合演示和项目集成。
输出格式

预处理额外信息放在 MemoryRecord.metadata：

{
  "previous_value": "上海",
  "operation_hint": "SUPERSEDE",
  "evidence_span": "我以前住上海，现在搬到北京了",
  "needs_review": false,
  "parser": "api"
}
TODO

需要和 versioning/、views/ 小组统一确认结构化字段协议。

