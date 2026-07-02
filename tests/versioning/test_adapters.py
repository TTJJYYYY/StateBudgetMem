from __future__ import annotations

from statebudgetmem.schemas import MemoryRecord, MemoryStatus
from statebudgetmem.versioning import MemoryRecordAdapter, UpdateOperation


def _base_memory() -> dict[str, object]:
    return {
        "memory_id": "m1",
        "subject": "user",
        "attribute": "location",
        "value": "shanghai",
        "text": "user lives in shanghai",
        "event_time": "2026-01-01",
        "status": MemoryStatus.CURRENT,
        "memory_type": "state",
        "importance": 1.0,
        "confidence": 1.0,
        "token_cost": 5,
        "dimensions": {"city_type": "primary"},
        "metadata": {"nested": {"items": [1, 2]}},
    }


def test_adapter_deep_copies_metadata() -> None:
    memory = MemoryRecord(**_base_memory())
    observation = MemoryRecordAdapter().to_observations(memory)[0]

    observation.metadata["nested"]["items"].append(3)

    assert memory.metadata == {"nested": {"items": [1, 2]}}


def test_adapter_bridges_supersedes_field_into_versioning_contract() -> None:
    memory = MemoryRecord(**_base_memory(), supersedes=["old_location"])
    observation = MemoryRecordAdapter().to_observations(memory)[0]

    assert observation.metadata["versioning_intent"] == UpdateOperation.SUPERSEDE.value
    assert observation.metadata["versioning_target_ids"] == ["old_location"]


def test_adapter_bridges_temporary_invalidation_field_into_contract() -> None:
    memory = MemoryRecord(
        **_base_memory(),
        temporarily_invalidates=["stable_location"],
    )
    observation = MemoryRecordAdapter().to_observations(memory)[0]

    assert (
        observation.metadata["versioning_intent"]
        == UpdateOperation.TEMP_INVALIDATE.value
    )
    assert observation.metadata["versioning_target_ids"] == ["stable_location"]
    assert observation.metadata["temporary"] is True
