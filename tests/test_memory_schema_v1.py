from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

import statebudgetmem.schemas
import statebudgetmem.schemas.annotations as annotation_module
import statebudgetmem.schemas.records as records_module
from statebudgetmem.schemas import MemoryAnnotation, MemoryRecord


def test_memory_record_loads_without_dimensions() -> None:
    memory = MemoryRecord.model_validate(_memory_payload())

    assert memory.dimensions == {}


def test_memory_record_dimensions_defaults_are_independent() -> None:
    first = MemoryRecord.model_validate(_memory_payload(memory_id="M1"))
    second = MemoryRecord.model_validate(_memory_payload(memory_id="M2"))

    first.dimensions["condition"] = "temporary"

    assert second.dimensions == {}


def test_memory_record_dimensions_store_multiple_conditions() -> None:
    payload = _memory_payload()
    payload["dimensions"] = {
        "city": "Shanghai",
        "day_type": "weekday",
        "time_of_day": "morning",
    }

    memory = MemoryRecord.model_validate(payload)

    assert memory.dimensions == {
        "city": "Shanghai",
        "day_type": "weekday",
        "time_of_day": "morning",
    }


@pytest.mark.parametrize(
    "dimensions",
    [
        {1: "weekday"},
        {"day_type": 1},
    ],
)
def test_memory_record_dimensions_reject_non_string_keys_or_values(
    dimensions: dict[object, object],
) -> None:
    payload = _memory_payload()
    payload["dimensions"] = dimensions

    with pytest.raises(ValidationError, match="dimensions"):
        MemoryRecord.model_validate(payload)


def test_memory_annotation_serializes_and_deserializes() -> None:
    annotation = MemoryAnnotation(
        memory_id="M1",
        gold_status="CURRENT",
        gold_operation="ADD",
        gold_target_memory_ids=["M0"],
        gold_supersedes=["M0"],
        gold_temporarily_invalidates=[],
        metadata={"annotator": "test"},
    )

    assert MemoryAnnotation.model_validate_json(annotation.model_dump_json()) == annotation


def test_memory_annotation_defaults_are_independent() -> None:
    first = MemoryAnnotation(memory_id="M1")
    second = MemoryAnnotation(memory_id="M2")

    first.gold_target_memory_ids.append("M0")
    first.gold_supersedes.append("M0")
    first.gold_temporarily_invalidates.append("M3")
    first.metadata["source"] = "test"

    assert second.gold_target_memory_ids == []
    assert second.gold_supersedes == []
    assert second.gold_temporarily_invalidates == []
    assert second.metadata == {}


def test_schemas_do_not_import_versioning() -> None:
    schema_sources = "\n".join(
        [
            inspect.getsource(statebudgetmem.schemas),
            inspect.getsource(records_module),
            inspect.getsource(annotation_module),
        ]
    )

    assert "statebudgetmem.versioning" not in schema_sources
    assert "versioning.operations" not in schema_sources


def _memory_payload(memory_id: str = "M1") -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "subject": "user",
        "attribute": "diet.preference.spicy",
        "value": "avoid",
        "text": "The user avoids spicy food.",
        "event_time": "2026-06-01",
        "valid_from": "2026-06-01",
        "valid_to": None,
        "status": "CURRENT",
        "memory_type": "preference",
        "importance": 0.8,
        "confidence": 0.9,
        "token_cost": 8,
        "supersedes": [],
        "temporarily_invalidates": [],
        "metadata": {},
    }
