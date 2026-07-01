from __future__ import annotations

import time
from typing import Any

from statebudgetmem.retrieval import TfidfCosineRetriever
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

        unified_memories = [
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
