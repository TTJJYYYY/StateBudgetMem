# StateBudgetMem / Routing 模块

> 时态一致性记忆系统的 **查询分类与记忆路由** 子模块

## 概述

本模块负责把用户的自然语言查询分类为 4 种时态类型, 从而决定从哪个记忆视图检索:

```
用户查询 ──► LLMQueryRouter ──► QueryType ──► ViewType ──► 记忆视图
                                  │
                                  ├─ CURRENT    → 只检索当前有效记忆
                                  ├─ HISTORICAL → 只检索历史快照
                                  ├─ CHANGE     → 同时检索当前+历史 (用于对比)
                                  └─ GENERAL    → 不检索个人记忆 (走通用知识)
```

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `QueryRecord` | `models.py` | 查询数据模型 (Pydantic v2), 含 `text` / `reference_time` |
| `QueryType` | `schemas/records.py`（统一复用） | 枚举: CURRENT / HISTORICAL / CHANGE / GENERAL |
| `QueryRouter` | `router.py` | 路由协议 (Protocol), 定义 `classify` 方法 |
| `LLMQueryRouter` | `router.py` | 基于大模型的实现 (OpenAI 兼容 API) |
| `RuleBasedRouter` | `router.py` | 纯规则兜底实现 (离线可用) |
| `SYSTEM_PROMPT` | `prompts.py` | 系统提示词 (针对 Memora 长周期场景调优) |
| `FEW_SHOT_EXAMPLES` | `prompts.py` | Few-shot 示例 (8 条, 覆盖 4 种类型) |
| `config.yaml` | `config.yaml` | YAML 配置 (模型/超参/降级策略) |

## 快速开始

### 离线模式 (无需 API Key)

```python
from statebudgetmem.routing import RuleBasedRouter, QueryRecord, QueryType

router = RuleBasedRouter()
qtype = router.classify(QueryRecord(text="我现在还喜欢吃辣吗?"))
print(qtype)  # → QueryType.CHANGE
```

### 在线模式 (需 API Key)

```python
from statebudgetmem.routing import LLMQueryRouter, QueryRecord

router = LLMQueryRouter(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
qtype = router.classify(QueryRecord(text="我现在还喜欢吃辣吗?"))
print(qtype)  # → QueryType.CHANGE
```

## Memora 场景调优

本模块的 prompt 针对 [Memora](https://github.com/) 数据集特征调优:

- **长周期状态演进**: 用户偏好/习惯/状态在数周到数月间持续变化
- **时态线索敏感**: 对"以前/现在/最近/还…吗/改了/换成"等时间信号高度敏感
- **状态切换检测**: "我现在还 X 吗?" → CHANGE (隐含"之前 X, 现在是否仍 X"对比)
- **通用知识排除**: 天气/数学/常识 → GENERAL (不检索个人记忆)

Few-shot 示例覆盖:
1. CURRENT: "我现在适合吃什么?" / "我目前的工作是什么?"
2. HISTORICAL: "我去年这个时候在做什么?" / "我大学时候喜欢吃什么?"
3. CHANGE: "我的饮食习惯是怎么变化的?" / "我现在还喜欢吃辣吗?"
4. GENERAL: "今天北京天气怎么样?" / "帮我算一下 123 乘以 456"

## 鲁棒性设计

1. **JSON 解析**: 容忍 Markdown 代码块 / 多余文本 / 不同 key 名 (`query_type` / `type` / `category`)
2. **降级策略**: 解析失败 / 超时 / 异常 → 返回 `fallback_type` (默认 GENERAL)
3. **空查询处理**: 空字符串直接返回 fallback, 跳过 LLM 调用
4. **日志记录**: 所有异常经 `logging` 记录后再降级, 不吞掉关键错误
5. **统计追踪**: `get_stats()` 返回调用次数 / 成功数 / 降级数

## 测试

```bash
# 离线单元测试 (116 个, 无需网络)
pytest tests/test_routing.py -v

# 全部测试
pytest -q
```

测试覆盖:
- JSON 解析鲁棒性 (15+ 用例)
- QueryRecord 数据模型校验
- RuleBasedRouter 规则分类
- LLMQueryRouter 正常路径 (Mock OpenAI client)
- LLMQueryRouter 边界与异常 (空串/非标准JSON/超时/异常)
- 离线测试隔离验证
- interfaces.QueryRouter ABC 兼容性
- Memora 长周期场景模拟
