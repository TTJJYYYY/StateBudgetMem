from __future__ import annotations

import pytest
from pydantic import ValidationError

from statebudgetmem.data import load_scenarios
from statebudgetmem.schemas import MemoryRecord, QueryRecord, Scenario


def test_controlled_dataset_loads() -> None:
    scenarios = load_scenarios("data/controlled/baseline_scenarios.jsonl")
    assert len(scenarios) >= 12
    assert all(len(scenario.memories) >= 2 for scenario in scenarios)
    assert all(len(scenario.queries) >= 2 for scenario in scenarios)


def test_memory_rejects_invalid_time() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord.model_validate(
            {
                "memory_id": "M1",
                "subject": "user",
                "attribute": "food",
                "value": "x",
                "text": "bad date",
                "event_time": "2026/01/01",
                "valid_from": None,
                "valid_to": None,
                "status": "CURRENT",
                "memory_type": "preference",
                "importance": 0.5,
                "confidence": 0.5,
                "token_cost": 1,
                "supersedes": [],
                "temporarily_invalidates": [],
                "metadata": {},
            }
        )


def test_query_rejects_invalid_query_type() -> None:
    with pytest.raises(ValidationError):
        QueryRecord.model_validate(
            {
                "query_id": "Q1",
                "text": "question",
                "query_type": "NOWISH",
                "reference_time": "2026-06-29",
                "gold_relevant_memory_ids": [],
                "gold_valid_memory_ids": [],
                "gold_stale_memory_ids": [],
            }
        )


def test_missing_required_field_is_error() -> None:
    with pytest.raises(ValidationError):
        QueryRecord.model_validate(
            {
                "query_id": "Q1",
                "query_type": "CURRENT",
                "reference_time": "2026-06-29",
            }
        )


def test_duplicate_memory_id_is_error() -> None:
    memory = {
        "memory_id": "M1",
        "subject": "user",
        "attribute": "food",
        "value": "x",
        "text": "memory",
        "event_time": "2026-01-01",
        "valid_from": "2026-01-01",
        "valid_to": None,
        "status": "CURRENT",
        "memory_type": "preference",
        "importance": 0.5,
        "confidence": 0.5,
        "token_cost": 1,
        "supersedes": [],
        "temporarily_invalidates": [],
        "metadata": {},
    }
    with pytest.raises(ValidationError, match="duplicate memory_id"):
        Scenario.model_validate(
            {
                "scenario_id": "S",
                "description": "duplicate",
                "memories": [memory, memory],
                "queries": [],
            }
        )
