from __future__ import annotations

from statebudgetmem.schemas import MemoryRecord, MemoryStatus
from statebudgetmem.versioning import StateKey, UpdateOperation, VersioningEngine


def _record(
    memory_id: str,
    value: str,
    event_time: str,
    *,
    metadata: dict[str, object] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject="user",
        attribute="location",
        value=value,
        text=f"user location is {value}",
        event_time=event_time,
        status=MemoryStatus.CURRENT,
        memory_type="state",
        importance=1.0,
        confidence=1.0,
        token_cost=6,
        metadata=metadata or {},
    )


def test_preprocessing_style_operation_hint_reaches_versioning_engine() -> None:
    engine = VersioningEngine()
    engine.ingest(_record("m1", "shanghai", "2026-01-01"))

    result = engine.ingest(
        _record(
            "m2",
            "beijing",
            "2026-02-01",
            metadata={"operation_hint": "SUPERSEDE"},
        )
    )

    assert result.results[0].decision.operation is UpdateOperation.SUPERSEDE
    assert result.results[0].decision.target_memory_ids == ["m1"]
    current = engine.resolve_current(StateKey(subject="user", attribute="location"))
    assert [item.memory_id for item in current] == ["m2"]


def test_conflicting_upstream_hints_are_conservative_noop() -> None:
    engine = VersioningEngine()
    engine.ingest(_record("m1", "shanghai", "2026-01-01"))

    result = engine.ingest(
        _record(
            "m2",
            "beijing",
            "2026-02-01",
            metadata={
                "versioning_intent": "ADD",
                "operation_hint": "DELETE",
            },
        )
    )

    assert result.results[0].decision.operation is UpdateOperation.NOOP
    current = engine.resolve_current(StateKey(subject="user", attribute="location"))
    assert [item.memory_id for item in current] == ["m1"]
