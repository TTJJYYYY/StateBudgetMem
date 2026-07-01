from __future__ import annotations

from statebudgetmem.schemas import MemoryRecord, MemoryStatus
from statebudgetmem.versioning import (
    MemoryRecordAdapter,
    UpdateOperation,
    normalize_versioning_metadata,
)


def _memory(metadata: dict[str, object]) -> MemoryRecord:
    return MemoryRecord(
        memory_id="m2",
        subject="user",
        attribute="location",
        value="beijing",
        text="user now lives in beijing",
        event_time="2026-02-01",
        status=MemoryStatus.CURRENT,
        memory_type="state",
        importance=1.0,
        confidence=1.0,
        token_cost=6,
        metadata=metadata,
    )


def test_legacy_operation_hint_is_promoted_to_canonical_intent() -> None:
    observation = MemoryRecordAdapter().to_observations(
        _memory({"operation_hint": "supersede"})
    )[0]

    assert observation.metadata["versioning_intent"] == "SUPERSEDE"
    assert "operation_hint" not in observation.metadata


def test_target_ids_and_boolean_hints_are_normalized() -> None:
    metadata = normalize_versioning_metadata(
        {
            "versioning_target_ids": ["m1", "m1", "  m0  "],
            "temporary": "yes",
        }
    )

    assert metadata["versioning_target_ids"] == ["m1", "m0"]
    assert metadata["temporary"] is True


def test_conflicting_intents_fail_closed_without_mutating_input() -> None:
    source = {
        "versioning_intent": "ADD",
        "operation_hint": "DELETE",
        "nested": {"items": [1, 2]},
    }
    metadata = normalize_versioning_metadata(source)

    assert metadata["versioning_intent"] == UpdateOperation.NOOP.value
    assert metadata["needs_review"] is True
    assert "conflicts" in metadata["versioning_contract_error"]
    assert source["nested"] == {"items": [1, 2]}
    assert source["operation_hint"] == "DELETE"
