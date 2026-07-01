from __future__ import annotations

import pytest

from statebudgetmem.baselines.tfidf_adapter import TfidfMemoryMethod
from statebudgetmem.data import load_scenarios
from statebudgetmem.schemas.records import QueryRecord
from statebudgetmem.schemas.results import MethodResult


class GoldGuardQueryRecord(QueryRecord):
    def __getattribute__(self, name: str):
        if name in {
            "gold_relevant_memory_ids",
            "gold_valid_memory_ids",
            "gold_stale_memory_ids",
        }:
            raise AssertionError(f"adapter read gold field: {name}")
        return super().__getattribute__(name)


def test_tfidf_adapter_exposes_name_reset_ingest_and_method_result() -> None:
    scenario = load_scenarios("data/controlled/baseline_scenarios.jsonl")[0]
    method = TfidfMemoryMethod()

    assert method.name == "tfidf_topk"
    method.reset()
    method.ingest(scenario.memories)

    result = method.retrieve(scenario.queries[0], top_k=3)

    assert isinstance(result, MethodResult)
    assert result.method_name == method.name
    assert result.query_id == scenario.queries[0].query_id


def test_tfidf_adapter_returns_input_ids_consecutive_ranks_costs_and_flat_view() -> None:
    scenario = load_scenarios("data/controlled/baseline_scenarios.jsonl")[0]
    method = TfidfMemoryMethod()
    method.ingest(scenario.memories)

    result = method.retrieve(scenario.queries[0], top_k=3)
    input_ids = {memory.memory_id for memory in scenario.memories}

    assert {item.memory_id for item in result.retrieved_memories} <= input_ids
    assert [item.rank for item in result.retrieved_memories] == list(
        range(1, len(result.retrieved_memories) + 1)
    )
    assert result.total_token_cost == sum(
        item.token_cost for item in result.retrieved_memories
    )
    assert {item.source_view for item in result.retrieved_memories} == {"flat"}


def test_tfidf_adapter_reset_clears_previous_scenario_memories() -> None:
    scenarios = load_scenarios("data/controlled/baseline_scenarios.jsonl")
    method = TfidfMemoryMethod()
    method.ingest(scenarios[0].memories)
    method.reset()
    method.ingest(scenarios[1].memories)

    result = method.retrieve(scenarios[1].queries[0], top_k=99)
    second_ids = {memory.memory_id for memory in scenarios[1].memories}

    assert result.retrieved_memories
    assert {item.memory_id for item in result.retrieved_memories} <= second_ids


def test_tfidf_adapter_does_not_read_query_gold_fields() -> None:
    scenario = load_scenarios("data/controlled/baseline_scenarios.jsonl")[0]
    guarded_query = GoldGuardQueryRecord.model_validate(scenario.queries[0].model_dump())
    method = TfidfMemoryMethod()
    method.ingest(scenario.memories)

    result = method.retrieve(guarded_query, top_k=3)

    assert isinstance(result, MethodResult)


def test_method_result_validates_rank_and_total_token_contract() -> None:
    scenario = load_scenarios("data/controlled/baseline_scenarios.jsonl")[0]
    method = TfidfMemoryMethod()
    method.ingest(scenario.memories)
    result = method.retrieve(scenario.queries[0], top_k=2)
    payload = result.model_dump()
    payload["total_token_cost"] += 1

    with pytest.raises(ValueError, match="total_token_cost"):
        MethodResult.model_validate(payload)
