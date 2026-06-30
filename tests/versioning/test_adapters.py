from __future__ import annotations

from statebudgetmem.schemas import MemoryRecord, MemoryStatus
from statebudgetmem.versioning import MemoryRecordAdapter


def test_adapter_deep_copies_metadata_and_ignores_gold_compatibility_fields() -> None:
    base = dict(
        memory_id="m1",
        subject="user",
        attribute="location",
        value="shanghai",
        text="user lives in shanghai",
        event_time="2026-01-01",
        status=MemoryStatus.CURRENT,
        memory_type="state",
        importance=1.0,
        confidence=1.0,
        token_cost=5,
        dimensions={"city_type": "primary"},
        metadata={"nested": {"items": [1, 2]}},
    )
    first = MemoryRecord(**base, supersedes=["x"], temporarily_invalidates=[])
    second = MemoryRecord(
        **{**base, "status": MemoryStatus.HISTORICAL},
        supersedes=[],
        temporarily_invalidates=["y"],
    )
    adapter = MemoryRecordAdapter()
    first_observation = adapter.to_observations(first)[0]
    second_observation = adapter.to_observations(second)[0]
    assert first_observation == second_observation
    first_observation.metadata["nested"]["items"].append(3)
    assert first.metadata == {"nested": {"items": [1, 2]}}
