from __future__ import annotations

import time
from typing import Any

from statebudgetmem.baselines.tfidf.retriever import TfidfCosineRetriever
from statebudgetmem.schemas.records import MemoryRecord, QueryRecord
from statebudgetmem.schemas.results import MethodResult, RetrievedMemory


class TfidfMemoryMethod:
    """Adapter from the existing flat TF-IDF retriever to the unified method API."""

    name = "tfidf_topk"

    def __init__(self, retriever: TfidfCosineRetriever | None = None) -> None:
        self._retriever = retriever or TfidfCosineRetriever()
        self._memories: list[MemoryRecord] = []

    def reset(self) -> None:
        self._memories = []

    def ingest(self, memories: list[MemoryRecord]) -> None:
        self._memories = list(memories)

    def retrieve(
        self,
        query: QueryRecord,
        *,
        top_k: int,
        token_budget: int | None = None,
        mutate: bool = False,
    ) -> MethodResult:
        started = time.perf_counter()
        retrieved = self._retriever.retrieve(query, self._memories, top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0

        candidates = [
            RetrievedMemory(
                memory_id=item.memory.memory_id,
                score=item.score,
                rank=item.rank,
                token_cost=item.memory.token_cost,
                source_view="flat",
                metadata={},
            )
            for item in retrieved
        ]
        unified_memories = _apply_token_budget(candidates, token_budget)
        for rank, item in enumerate(unified_memories, start=1):
            item.rank = rank
        return MethodResult(
            method_name=self.name,
            query_id=query.query_id,
            retrieved_memories=unified_memories,
            predicted_query_type=None,
            total_token_cost=sum(item.token_cost for item in unified_memories),
            latency_ms=latency_ms,
            metadata=_adapter_metadata(token_budget=token_budget, mutate=mutate),
        )


def _adapter_metadata(*, token_budget: int | None, mutate: bool) -> dict[str, Any]:
    return {
        "token_budget": token_budget,
        "mutate": mutate,
        "source_retriever": "TfidfCosineRetriever",
    }


def _apply_token_budget(
    candidates: list[RetrievedMemory],
    token_budget: int | None,
) -> list[RetrievedMemory]:
    if token_budget is None:
        return candidates
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative or None")
    selected: list[RetrievedMemory] = []
    used = 0
    for candidate in candidates:
        if used + candidate.token_cost > token_budget:
            continue
        selected.append(candidate)
        used += candidate.token_cost
    return selected
