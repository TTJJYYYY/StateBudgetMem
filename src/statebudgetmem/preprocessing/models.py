from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Tuple

from statebudgetmem.interfaces import (
    MemoryPiece,
    MemoryStatus,
    MemoryType,
    QueryType,
    UpdateOperation,
)


@dataclass
class RawMessage:
    """预处理模块接收的原始消息。

    对齐主接口中的 messages: [(role, content, timestamp), ...]。
    """

    role: str
    content: str
    timestamp: str | float | int
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.role, self.content, str(self.timestamp))


@dataclass
class ParsedMemory:
    """预处理后的结构化记忆草稿。

    这不是新的全局接口，只是 preprocessing 内部结果。
    最终通过 to_memory_piece() 转成项目统一的 MemoryPiece。
    """

    content: str
    timestamp: float
    memory_type: MemoryType = MemoryType.FACT
    operation: UpdateOperation = UpdateOperation.ADD

    attribute: Optional[str] = None
    value: Optional[str] = None
    previous_value: Optional[str] = None
    evidence_span: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.8
    source: Optional[str] = None
    needs_review: bool = False

    def to_memory_piece(self) -> MemoryPiece:
        memory_id = _stable_memory_id(
            self.content,
            self.timestamp,
            self.attribute or "",
            self.value or "",
        )

        tags = list(dict.fromkeys(self.tags + _auto_tags(self)))

        return MemoryPiece(
            content=self.content,
            timestamp=self.timestamp,
            memory_type=self.memory_type,
            memory_id=memory_id,
            version=1,
            parent_id=None,
            status=MemoryStatus.ACTIVE,
            validity_period=(self.timestamp, None),
            tags=tags,
            confidence=self.confidence,
            source=self.source,
            query_types=_default_query_types(self),
        )


@dataclass
class PreprocessConfig:
    """预处理配置。

    parser_type:
    - rule: 只用规则解析
    - api: 只用外部 API
    - hybrid: 优先 API，失败后回退规则解析
    """

    parser_type: str = "hybrid"
    api_model: str = "deepseek-chat"
    fallback_to_rule: bool = True
    keep_note_fallback: bool = True
    min_confidence: float = 0.0


def parse_timestamp(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ]:
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue

    try:
        return float(text)
    except ValueError:
        return datetime.now().timestamp()


def messages_to_raw_messages(
    messages: Iterable[Tuple[str, str, str | float | int]],
) -> list[RawMessage]:
    return [
        RawMessage(role=role, content=content, timestamp=timestamp)
        for role, content, timestamp in messages
    ]


def _stable_memory_id(*parts: str | float) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _auto_tags(memory: ParsedMemory) -> list[str]:
    tags: list[str] = ["preprocessed"]

    if memory.attribute:
        tags.append(memory.attribute)

    if memory.operation:
        tags.append(f"op:{memory.operation.value}")

    if memory.previous_value:
        tags.append("has_previous_value")

    if memory.needs_review:
        tags.append("needs_review")

    return tags


def _default_query_types(memory: ParsedMemory) -> list[QueryType]:
    if memory.operation == UpdateOperation.SUPERSEDE:
        return [QueryType.CURRENT, QueryType.CHANGE]

    if memory.previous_value:
        return [QueryType.CURRENT, QueryType.CHANGE]

    return [QueryType.CURRENT]
