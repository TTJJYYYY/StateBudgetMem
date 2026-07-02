from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

from statebudgetmem.interfaces import MemoryPiece
from statebudgetmem.preprocessing.models import ParsedMemory
from statebudgetmem.schemas import MemoryRecord, MemoryStatus as RecordMemoryStatus
from statebudgetmem.versioning.operations import UpdateOperation

_DEFAULT_SUBJECT = "user"
_DEFAULT_IMPORTANCE = 0.7
_ADAPTER_NAME = "preprocessing.record_adapter"


def parsed_memory_to_record(
    memory: ParsedMemory,
    *,
    subject: str = _DEFAULT_SUBJECT,
    memory_id: str | None = None,
    dimensions: Mapping[str, str] | None = None,
    importance: float = _DEFAULT_IMPORTANCE,
    token_cost: int | None = None,
    valid_from: date | str | None = None,
    valid_to: date | str | None = None,
    status: RecordMemoryStatus | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MemoryRecord:
    """Convert a preprocessing ParsedMemory into the canonical MemoryRecord.

    ParsedMemory keeps structured fields such as attribute, value,
    previous_value and operation. This direct conversion is therefore the
    preferred bridge from preprocessing to versioning/views experiments.
    """

    event_time = _date_from_timestamp(memory.timestamp)
    operation = _coerce_operation(memory.operation)
    record_metadata = _metadata_from_parsed(memory, operation)
    record_metadata.update(dict(metadata or {}))

    return MemoryRecord(
        memory_id=memory_id or memory.to_memory_piece().memory_id,
        subject=subject,
        attribute=_non_empty(memory.attribute, fallback="fact"),
        value=_non_empty(memory.value, fallback=memory.content),
        text=_non_empty(memory.content, fallback=_build_text(memory.attribute, memory.value)),
        event_time=event_time,
        valid_from=_coerce_date(valid_from) or event_time,
        valid_to=_coerce_date(valid_to),
        status=_coerce_record_status(status) or _default_status_for_operation(operation),
        memory_type=_enum_value(memory.memory_type),
        importance=_clamp01(importance),
        confidence=_clamp01(memory.confidence),
        token_cost=token_cost if token_cost is not None else estimate_token_cost(memory.content),
        dimensions=dict(dimensions or {}),
        supersedes=[],
        temporarily_invalidates=[],
        metadata=record_metadata,
    )


def parsed_memories_to_records(
    memories: Iterable[ParsedMemory],
    *,
    subject: str = _DEFAULT_SUBJECT,
    dimensions: Mapping[str, str] | None = None,
    importance: float = _DEFAULT_IMPORTANCE,
) -> list[MemoryRecord]:
    """Convert a batch of ParsedMemory objects into MemoryRecord objects."""

    return [
        parsed_memory_to_record(
            memory,
            subject=subject,
            dimensions=dimensions,
            importance=importance,
        )
        for memory in memories
    ]


def memory_piece_to_record(
    memory: MemoryPiece,
    *,
    subject: str = _DEFAULT_SUBJECT,
    attribute: str | None = None,
    value: str | None = None,
    dimensions: Mapping[str, str] | None = None,
    importance: float = _DEFAULT_IMPORTANCE,
    token_cost: int | None = None,
    event_time: date | str | None = None,
    valid_from: date | str | None = None,
    valid_to: date | str | None = None,
    status: RecordMemoryStatus | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MemoryRecord:
    """Best-effort conversion from online MemoryPiece to MemoryRecord.

    MemoryPiece is the online-layer format and does not have first-class
    attribute/value/operation fields. Pass attribute/value explicitly whenever
    possible. For preprocessing outputs, prefer parsed_memory_to_record().
    """

    inferred_attribute = attribute or _infer_attribute_from_tags(memory.tags)
    inferred_value = value or _infer_value_from_content(memory.content)
    start, end = _split_validity_period(memory.validity_period)
    record_event_time = _coerce_date(event_time) or _date_from_timestamp(memory.timestamp)

    record_metadata: dict[str, Any] = {
        "adapter": _ADAPTER_NAME,
        "adapter_source": "MemoryPiece",
        "adapter_note": "lossy conversion; pass attribute/value explicitly when possible",
        "source": memory.source,
        "online_status": _enum_value(memory.status),
        "online_tags": list(memory.tags),
        "online_query_types": [_enum_value(item) for item in memory.query_types],
        "online_strength": memory.strength,
        "online_last_accessed": memory.last_accessed,
        "online_access_count": memory.access_count,
    }
    record_metadata.update(dict(metadata or {}))

    return MemoryRecord(
        memory_id=_non_empty(memory.memory_id, fallback=_stable_piece_id(memory)),
        subject=subject,
        attribute=_non_empty(inferred_attribute, fallback="fact"),
        value=_non_empty(inferred_value, fallback=memory.content),
        text=_non_empty(memory.content, fallback=str(inferred_value or "")),
        event_time=record_event_time,
        valid_from=_coerce_date(valid_from) or start or record_event_time,
        valid_to=_coerce_date(valid_to) or end,
        status=_coerce_record_status(status) or _map_online_status(memory.status),
        memory_type=_enum_value(memory.memory_type),
        importance=_clamp01(importance),
        confidence=_clamp01(memory.confidence),
        token_cost=token_cost if token_cost is not None else estimate_token_cost(memory.content),
        dimensions=dict(dimensions or {}),
        supersedes=[memory.parent_id] if memory.parent_id else [],
        temporarily_invalidates=[],
        metadata=record_metadata,
    )


def memory_pieces_to_records(
    memories: Iterable[MemoryPiece],
    *,
    subject: str = _DEFAULT_SUBJECT,
    dimensions: Mapping[str, str] | None = None,
    importance: float = _DEFAULT_IMPORTANCE,
) -> list[MemoryRecord]:
    """Convert a batch of MemoryPiece objects into MemoryRecord objects."""

    return [
        memory_piece_to_record(
            memory,
            subject=subject,
            dimensions=dimensions,
            importance=importance,
        )
        for memory in memories
    ]


def preprocess_messages_to_records(
    messages: Iterable[tuple[str, str, str | float | int]],
    *,
    subject: str = _DEFAULT_SUBJECT,
    dimensions: Mapping[str, str] | None = None,
    config: Any | None = None,
) -> list[MemoryRecord]:
    """Convenience wrapper: raw message tuples -> ParsedMemory -> MemoryRecord."""

    from statebudgetmem.preprocessing.pipeline import MemoryPreprocessor

    preprocessor = MemoryPreprocessor(config=config)
    parsed = preprocessor.parse_messages(messages)
    return parsed_memories_to_records(parsed, subject=subject, dimensions=dimensions)


def estimate_token_cost(text: str) -> int:
    """Deterministic dependency-free token-cost approximation for experiments."""

    normalized = str(text or "").strip()
    if not normalized:
        return 1

    whitespace_tokens = [part for part in normalized.replace("\n", " ").split(" ") if part]
    if len(whitespace_tokens) >= 3:
        return max(1, int(round(len(whitespace_tokens) * 1.3)))

    return max(1, int(round(len(normalized) / 1.6)))


def _metadata_from_parsed(memory: ParsedMemory, operation: UpdateOperation) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "adapter": _ADAPTER_NAME,
        "adapter_source": "ParsedMemory",
        "versioning_intent": operation.value,
        "operation_hint": operation.value,
        "preprocessing_attribute": memory.attribute,
        "preprocessing_value": memory.value,
        "preprocessing_previous_value": memory.previous_value,
        "evidence_span": memory.evidence_span,
        "source": memory.source,
        "preprocessing_tags": list(memory.tags),
        "needs_review": memory.needs_review,
    }

    if operation is UpdateOperation.DELETE:
        metadata["delete_request"] = True
    elif operation is UpdateOperation.TEMP_INVALIDATE:
        metadata["temporary"] = True
    elif operation is UpdateOperation.MERGE:
        metadata["merge_request"] = True
    elif operation is UpdateOperation.RESTORE:
        metadata["restore_signal"] = True

    return {key: value for key, value in metadata.items() if value is not None}


def _default_status_for_operation(operation: UpdateOperation) -> RecordMemoryStatus:
    if operation is UpdateOperation.NOOP:
        return RecordMemoryStatus.UNKNOWN

    if operation is UpdateOperation.DELETE:
        return RecordMemoryStatus.INVALIDATED

    return RecordMemoryStatus.CURRENT


def _map_online_status(status: Any) -> RecordMemoryStatus:
    raw = _enum_value(status).strip().lower()

    if raw == "active":
        return RecordMemoryStatus.CURRENT

    if raw == "superseded":
        return RecordMemoryStatus.HISTORICAL

    if raw in {"temp_invalid", "deleted"}:
        return RecordMemoryStatus.INVALIDATED

    return RecordMemoryStatus.UNKNOWN


def _coerce_record_status(value: RecordMemoryStatus | str | None) -> RecordMemoryStatus | None:
    if value is None:
        return None

    if isinstance(value, RecordMemoryStatus):
        return value

    return RecordMemoryStatus(str(value).strip().upper())


def _coerce_operation(value: Any) -> UpdateOperation:
    if isinstance(value, UpdateOperation):
        return value

    return UpdateOperation(value)


def _coerce_date(value: date | str | None) -> date | None:
    if value is None:
        return None

    if isinstance(value, date):
        return value

    return date.fromisoformat(str(value))


def _date_from_timestamp(timestamp: float | int) -> date:
    return datetime.fromtimestamp(float(timestamp)).date()


def _split_validity_period(
    validity_period: tuple[float, float | None] | None,
) -> tuple[date | None, date | None]:
    if validity_period is None:
        return None, None

    start, end = validity_period
    start_date = _date_from_timestamp(start)
    end_date = _date_from_timestamp(end) if end is not None else None
    return start_date, end_date


def _enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def _non_empty(value: Any, *, fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def _build_text(attribute: str | None, value: str | None) -> str:
    if attribute and value:
        return f"用户的{attribute}是{value}"

    return str(value or attribute or "memory")


def _infer_attribute_from_tags(tags: Iterable[str]) -> str | None:
    ignored_prefixes = ("op:",)
    ignored = {
        "preprocessed",
        "needs_review",
        "has_previous_value",
        "profile",
        "preference",
        "habit",
        "health",
        "control",
        "delete",
    }

    for tag in tags:
        text = str(tag).strip()
        if not text or text in ignored or text.startswith(ignored_prefixes):
            continue
        return text

    return None


def _infer_value_from_content(content: str) -> str:
    text = str(content or "").strip()

    for marker in ("是", "为", "=", ":", "："):
        if marker in text:
            candidate = text.rsplit(marker, 1)[-1].strip()
            if candidate:
                return candidate

    return text


def _stable_piece_id(memory: MemoryPiece) -> str:
    raw = f"{memory.content}|{memory.timestamp}|{_enum_value(memory.memory_type)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
