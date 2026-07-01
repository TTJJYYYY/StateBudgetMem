from __future__ import annotations

from typing import Protocol

from statebudgetmem.schemas.records import MemoryRecord, QueryRecord
from statebudgetmem.schemas.results import MethodResult


class MemoryMethod(Protocol):
    @property
    def name(self) -> str:
        ...

    def reset(self) -> None:
        ...

    def ingest(self, memories: list[MemoryRecord]) -> None:
        ...

    def retrieve(
        self,
        query: QueryRecord,
        *,
        top_k: int,
        token_budget: int | None = None,
        mutate: bool = False,
    ) -> MethodResult:
        ...
