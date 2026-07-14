from __future__ import annotations

import inspect
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")

from statebudgetmem.baselines.memorybank.adapter import (
    MemoryBankMethod,
    memory_record_to_piece,
)
from statebudgetmem.core.experiment import ExperimentConfig, MethodBuildContext
from statebudgetmem.core.method import MemoryMethod
from statebudgetmem.schemas.records import MemoryRecord, MemoryStatus, QueryRecord, QueryType


def _context(**overrides) -> MethodBuildContext:
    values = {
        "dataset_path": Path("fixture.jsonl"),
        "embedding_backend": "hash",
        "embedding_model": "deterministic_hash_embedding",
        "top_k": 2,
        "candidate_k": 3,
        "reinforcement_enabled": False,
    }
    values.update(overrides)
    return MethodBuildContext(
        experiment=ExperimentConfig(**values), work_dir=Path("results/test-memorybank")
    )


def _memory(memory_id: str, text: str, token_cost: int = 4) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject="user",
        attribute="preference.food",
        value=text,
        text=text,
        event_time=date(2026, 5, 1),
        valid_from=date(2026, 5, 1),
        status=MemoryStatus.CURRENT,
        memory_type="preference",
        importance=0.8,
        confidence=0.9,
        token_cost=token_cost,
    )


def _query(reference_time: date = date(2026, 5, 10)) -> QueryRecord:
    return QueryRecord(
        query_id="q1",
        text="spicy food",
        query_type=QueryType.CURRENT,
        reference_time=reference_time,
    )


def test_memory_record_mapping_preserves_identity_and_temporal_fields() -> None:
    record = _memory("m1", "spicy food")
    piece = memory_record_to_piece(record)

    assert piece.memory_id == "m1"
    assert piece.content == record.text
    assert piece.confidence == record.confidence
    assert piece.validity_period is not None
    assert piece.last_accessed == piece.timestamp


def test_retrieve_scoped_signature_is_keyword_only_and_stable() -> None:
    signature = inspect.signature(MemoryBankMethod.retrieve_scoped)

    assert inspect.signature(MemoryBankMethod.retrieve) == inspect.signature(
        MemoryMethod.retrieve
    )
    assert signature.parameters["allowed_memory_ids"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["top_k"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["token_budget"].default is None
    assert signature.parameters["mutate"].default is False


@pytest.mark.parametrize(
    "allowed_memory_ids",
    [
        {"m2"},
        frozenset({"m2"}),
        ["m2"],
        ("m2",),
    ],
)
def test_retrieve_scoped_accepts_collections_and_only_returns_allowed_ids(
    allowed_memory_ids,
) -> None:
    method = MemoryBankMethod(_context())
    method.ingest(
        [
            _memory("m1", "spicy food"),
            _memory("m2", "running shoes"),
            _memory("m3", "quiet library"),
        ]
    )
    bank = method.bank
    encoder = method.embedding_model
    ordinary = method.retrieve(_query(), top_k=3, mutate=False)

    result = method.retrieve_scoped(
        _query(),
        allowed_memory_ids=allowed_memory_ids,
        top_k=3,
        mutate=False,
    )

    assert method.bank is bank
    assert method.embedding_model is encoder
    assert ordinary.metadata["core_retrieval"]["scoped_retrieval"] is False
    assert [item.memory_id for item in result.retrieved_memories] == ["m2"]
    core = result.metadata["core_retrieval"]
    assert core["scoped_retrieval"] is True
    assert core["allowed_memory_count"] == 1
    assert core["matched_allowed_memory_count"] == 1
    json.dumps(result.model_dump(mode="json"), sort_keys=True)


def test_retrieve_scoped_empty_missing_and_budgeted_results_are_empty() -> None:
    method = MemoryBankMethod(_context(reinforcement_enabled=True))
    method.ingest([_memory("m1", "spicy food", token_cost=5)])
    piece = method.bank.memories_by_id["m1"]
    before = (piece.strength, piece.last_accessed, piece.access_count, method.bank.access_count)

    empty = method.retrieve_scoped(
        _query(), allowed_memory_ids=set(), top_k=1, mutate=True
    )
    missing = method.retrieve_scoped(
        _query(), allowed_memory_ids={"missing"}, top_k=1, mutate=True
    )
    budgeted = method.retrieve_scoped(
        _query(),
        allowed_memory_ids={"m1"},
        top_k=1,
        token_budget=4,
        mutate=True,
    )

    for result in (empty, missing, budgeted):
        assert result.retrieved_memories == []
        assert result.total_token_cost == 0
        assert result.metadata["reinforcement_applied"] is False
    assert (piece.strength, piece.last_accessed, piece.access_count, method.bank.access_count) == before


def test_retrieve_scoped_reinforces_only_final_budget_selection() -> None:
    method = MemoryBankMethod(_context(reinforcement_enabled=True))
    method.ingest(
        [
            _memory("m1", "spicy food", token_cost=5),
            _memory("m2", "spicy restaurant", token_cost=5),
        ]
    )
    before = {
        memory_id: (piece.strength, piece.access_count)
        for memory_id, piece in method.bank.memories_by_id.items()
    }

    result = method.retrieve_scoped(
        _query(),
        allowed_memory_ids={"m1", "m2"},
        top_k=1,
        token_budget=5,
        mutate=True,
    )

    assert len(result.retrieved_memories) == 1
    selected_id = result.retrieved_memories[0].memory_id
    for memory_id, piece in method.bank.memories_by_id.items():
        old_strength, old_access_count = before[memory_id]
        if memory_id == selected_id:
            assert piece.strength == old_strength + 1
            assert piece.access_count == old_access_count + 1
        else:
            assert (piece.strength, piece.access_count) == (
                old_strength,
                old_access_count,
            )


def test_adapter_respects_top_k_token_budget_and_reference_time() -> None:
    method = MemoryBankMethod(_context())
    method.ingest(
        [
            _memory("m1", "spicy food", token_cost=5),
            _memory("m2", "spicy restaurant", token_cost=5),
            _memory("m3", "running", token_cost=5),
        ]
    )

    result = method.retrieve(_query(), top_k=2, token_budget=5, mutate=False)

    assert len(result.retrieved_memories) == 1
    assert result.total_token_cost == 5
    assert result.metadata["query_time"] == "2026-05-10"
    assert result.metadata["core_retrieval"]["candidate_k"] == 3
    assert result.latency_ms >= 0.0


def test_reinforcement_disabled_has_no_state_side_effects() -> None:
    method = MemoryBankMethod(_context(reinforcement_enabled=False))
    method.ingest([_memory("m1", "spicy food")])
    piece = method.bank.memories_by_id["m1"]
    before = (piece.strength, piece.last_accessed, piece.access_count, method.bank.access_count)

    result = method.retrieve(_query(), top_k=1, mutate=True)

    assert (piece.strength, piece.last_accessed, piece.access_count, method.bank.access_count) == before
    assert result.metadata["reinforcement_applied"] is False


def test_sequential_reinforcement_uses_query_reference_time() -> None:
    method = MemoryBankMethod(_context(reinforcement_enabled=True))
    method.ingest([_memory("m1", "spicy food")])

    first = method.retrieve(_query(date(2026, 5, 10)), top_k=1, mutate=True)
    second = method.retrieve(_query(date(2026, 5, 11)), top_k=1, mutate=True)
    piece = method.bank.memories_by_id["m1"]

    assert first.metadata["reinforcement_applied"] is True
    assert second.metadata["reinforcement_applied"] is True
    assert piece.strength == 3.0
    assert piece.access_count == 2
    assert piece.last_accessed == datetime(
        2026, 5, 11, tzinfo=timezone.utc
    ).timestamp()


def test_reset_reuses_encoder_but_clears_scenario_state() -> None:
    method = MemoryBankMethod(_context())
    encoder = method.embedding_model
    method.ingest([_memory("m1", "spicy food")])

    method.reset()

    assert method.embedding_model is encoder
    assert method.bank.memories == []
    assert method.bank.memories_by_id == {}
    assert method.bank.access_count == 0


def test_independent_queries_do_not_depend_on_query_order() -> None:
    method = MemoryBankMethod(_context(reinforcement_enabled=True))
    memories = [_memory("m1", "spicy food"), _memory("m2", "running")]
    early = _query(date(2026, 5, 10))
    late = _query(date(2026, 5, 11))

    def run(order: list[QueryRecord]) -> dict[date, list[str]]:
        outputs = {}
        for query in order:
            method.reset()
            method.ingest(memories)
            result = method.retrieve(query, top_k=2, mutate=True)
            outputs[query.reference_time] = [
                item.memory_id for item in result.retrieved_memories
            ]
        return outputs

    assert run([early, late]) == run([late, early])
