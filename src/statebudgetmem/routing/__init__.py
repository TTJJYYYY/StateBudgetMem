"""
statebudgetmem.routing — 查询分类与记忆路由

本子包实现时态一致性记忆系统的"查询分类与记忆路由"功能:

    用户查询 → [LLMQueryRouter] → QueryType → ViewType → 记忆视图

核心组件:
    - QueryRecord      : 查询数据模型 (Pydantic, 含 text / reference_time)
    - QueryRouter      : 路由协议 (Protocol, 定义 classify 方法)
    - LLMQueryRouter   : 基于大模型的实现 (OpenAI 兼容 API)
    - RuleBasedRouter  : 纯规则兜底实现 (离线可用)
    - QueryType        : 查询类型枚举 (复用自 interfaces)
    - ViewType         : 记忆视图枚举 (复用自 interfaces)

快速开始::

    from statebudgetmem.routing import LLMQueryRouter, QueryRecord, QueryType

    router = LLMQueryRouter(api_key="sk-xxx", base_url="...", model="deepseek-chat")
    qtype = router.classify(QueryRecord(text="我现在还喜欢吃辣吗?"))
    print(qtype)  # → QueryType.CHANGE

离线测试 (无需 API Key)::

    from statebudgetmem.routing import RuleBasedRouter, QueryRecord
    router = RuleBasedRouter()
    qtype = router.classify(QueryRecord(text="我以前喜欢吃什么?"))
    print(qtype)  # → QueryType.HISTORICAL
"""

from __future__ import annotations

from statebudgetmem.core import MemoryPiece, QueryRouter as QueryRouterABC, ViewType
from statebudgetmem.schemas import QueryType
from .models import QueryRecord, RoutingQueryRecord
from .prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_messages
from .router import (
    LLMQueryRouter,
    QueryRouter,
    RuleBasedRouter,
    load_config,
    parse_query_type_from_response,
)

__all__ = [
    # 数据模型
    "RoutingQueryRecord",
    "QueryRecord",  # backward-compatible alias
    # 协议与实现
    "QueryRouter",
    "QueryRouterABC",
    "LLMQueryRouter",
    "RuleBasedRouter",
    # 枚举 (复用自 interfaces)
    "QueryType",
    "ViewType",
    "MemoryPiece",
    # Prompt
    "SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "build_messages",
    # 工具函数
    "parse_query_type_from_response",
    "load_config",
]

