from __future__ import annotations

from typing import Protocol

from statebudgetmem.schemas import MemoryRecord, QueryRecord, RetrievedMemory


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: QueryRecord,
        memories: list[MemoryRecord],
        top_k: int,
    ) -> list[RetrievedMemory]:
        ...
