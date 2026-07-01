from __future__ import annotations

from statebudgetmem.evaluation import (
    average_retrieved_token_cost,
    recall_at_k,
    stale_retrieval_rate,
    valid_recall_at_k,
)
from statebudgetmem.schemas import MemoryRecord, RetrievedMemory


def test_recall_at_k() -> None:
    assert recall_at_k(["M1", "M3"], ["M1", "M2"]) == 0.5


def test_valid_recall_at_k() -> None:
    assert valid_recall_at_k(["M2"], ["M2", "M4"]) == 0.5


def test_stale_retrieval_rate() -> None:
    assert stale_retrieval_rate(["M1", "M2", "M3"], ["M1", "M3"]) == 2 / 3


def test_empty_denominators_return_zero() -> None:
    assert recall_at_k(["M1"], []) == 0.0
    assert valid_recall_at_k(["M1"], []) == 0.0
    assert stale_retrieval_rate([], ["M1"]) == 0.0
    assert average_retrieved_token_cost([]) == 0.0


def test_average_token_cost() -> None:
    retrieved = [
        RetrievedMemory(memory=_memory("M1", 4), score=1.0, rank=1),
        RetrievedMemory(memory=_memory("M2", 8), score=0.5, rank=2),
    ]
    assert average_retrieved_token_cost(retrieved) == 6.0


def _memory(memory_id: str, token_cost: int) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject="user",
        attribute="test",
        value="value",
        text="text",
        event_time="2026-01-01",
        valid_from="2026-01-01",
        valid_to=None,
        status="CURRENT",
        memory_type="test",
        importance=0.5,
        confidence=0.5,
        token_cost=token_cost,
        supersedes=[],
        temporarily_invalidates=[],
        metadata={},
    )
