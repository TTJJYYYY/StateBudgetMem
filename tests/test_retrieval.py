from __future__ import annotations

from statebudgetmem.data import load_scenarios
from statebudgetmem.retrieval import TfidfCosineRetriever
from statebudgetmem.schemas import MemoryRecord, QueryRecord


def test_retrieval_is_deterministic() -> None:
    scenario = load_scenarios("data/controlled/baseline_scenarios.jsonl")[0]
    retriever = TfidfCosineRetriever()
    first = retriever.retrieve(scenario.queries[0], scenario.memories, top_k=2)
    second = retriever.retrieve(scenario.queries[0], scenario.memories, top_k=2)
    assert [item.memory.memory_id for item in first] == [item.memory.memory_id for item in second]
    assert [item.score for item in first] == [item.score for item in second]


def test_top_k_and_large_top_k() -> None:
    scenario = load_scenarios("data/controlled/baseline_scenarios.jsonl")[0]
    retriever = TfidfCosineRetriever()
    assert len(retriever.retrieve(scenario.queries[0], scenario.memories, top_k=1)) == 1
    assert len(retriever.retrieve(scenario.queries[0], scenario.memories, top_k=99)) == len(
        scenario.memories
    )


def test_empty_memory_list_returns_empty() -> None:
    scenario = load_scenarios("data/controlled/baseline_scenarios.jsonl")[0]
    assert TfidfCosineRetriever().retrieve(scenario.queries[0], [], top_k=3) == []


def test_tie_breaking_keeps_input_order() -> None:
    memories = [
        _memory("M1", "alpha beta"),
        _memory("M2", "alpha beta"),
    ]
    query = QueryRecord(
        query_id="Q",
        text="alpha",
        query_type="CURRENT",
        reference_time="2026-06-29",
        gold_relevant_memory_ids=["M1", "M2"],
        gold_valid_memory_ids=["M1", "M2"],
        gold_stale_memory_ids=[],
    )
    retrieved = TfidfCosineRetriever().retrieve(query, memories, top_k=2)
    assert [item.memory.memory_id for item in retrieved] == ["M1", "M2"]


def _memory(memory_id: str, text: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject="user",
        attribute="test",
        value="value",
        text=text,
        event_time="2026-01-01",
        valid_from="2026-01-01",
        valid_to=None,
        status="CURRENT",
        memory_type="test",
        importance=0.5,
        confidence=0.5,
        token_cost=5,
        supersedes=[],
        temporarily_invalidates=[],
        metadata={},
    )
