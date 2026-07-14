"""Unified experiment adapter for the real stateful MemoryBank Core."""

from __future__ import annotations

import time
from collections.abc import Collection
from datetime import date, datetime, time as datetime_time, timezone
from typing import Any, Callable

from statebudgetmem.baselines.memorybank.embeddings import build_embedding_model
from statebudgetmem.baselines.memorybank.system import MemoryBank
from statebudgetmem.core.experiment import MethodBuildContext
from statebudgetmem.core.online import (
    MemoryPiece,
    MemoryStatus as OnlineMemoryStatus,
    MemoryType,
)
from statebudgetmem.schemas.records import (
    MemoryRecord,
    MemoryStatus as RecordMemoryStatus,
    QueryRecord,
)
from statebudgetmem.schemas.results import MethodResult, RetrievedMemory


class MemoryBankMethod:
    """Map controlled experiment records onto the stateful MemoryBank backend."""

    name = "memorybank_core"

    def __init__(
        self,
        context: MethodBuildContext,
        *,
        embedding_model: Any | None = None,
        bank_factory: Callable[..., MemoryBank] = MemoryBank,
    ) -> None:
        self._context = context
        self._config = context.experiment
        self._embedding_model = embedding_model or build_embedding_model(
            self._config.embedding_backend, self._config.embedding_model
        )
        self._bank_factory = bank_factory
        self._records: dict[str, MemoryRecord] = {}
        self._bank = self._new_bank()

    @property
    def bank(self) -> MemoryBank:
        """Expose the shared dense backend for composed StateBudgetMem adapters."""
        return self._bank

    @property
    def embedding_model(self) -> Any:
        """Return the scenario-reused encoder shared by dense method variants."""
        return self._embedding_model

    def _new_bank(self) -> MemoryBank:
        return self._bank_factory(
            forgetting_threshold=self._config.forgetting_threshold,
            embedding_model=self._embedding_model,
        )

    def reset(self) -> None:
        self._records = {}
        self._bank = self._new_bank()

    def ingest(self, memories: list[MemoryRecord]) -> None:
        self._records = {memory.memory_id: memory for memory in memories}
        for memory in memories:
            self._bank.ingest_piece(memory_record_to_piece(memory))

    def retrieve(
        self,
        query: QueryRecord,
        *,
        top_k: int,
        token_budget: int | None = None,
        mutate: bool = False,
    ) -> MethodResult:
        return self._retrieve_impl(
            query,
            allowed_memory_ids=None,
            top_k=top_k,
            token_budget=token_budget,
            mutate=mutate,
        )

    def retrieve_scoped(
        self,
        query: QueryRecord,
        *,
        allowed_memory_ids: Collection[str],
        top_k: int,
        token_budget: int | None = None,
        mutate: bool = False,
    ) -> MethodResult:
        return self._retrieve_impl(
            query,
            allowed_memory_ids=frozenset(allowed_memory_ids),
            top_k=top_k,
            token_budget=token_budget,
            mutate=mutate,
        )

    def _retrieve_impl(
        self,
        query: QueryRecord,
        *,
        allowed_memory_ids: Collection[str] | None,
        top_k: int,
        token_budget: int | None,
        mutate: bool,
    ) -> MethodResult:
        started = time.perf_counter()
        query_time = _date_timestamp(query.reference_time)
        core_result = self._bank.retrieve_with_metadata(
            query=query.text,
            top_k=self._config.candidate_k,
            candidate_k=self._config.candidate_k,
            current_time=query_time,
            exclude_forgotten=(
                self._config.forgetting_enabled and self._config.exclude_forgotten
            ),
            reinforce=False,
            allowed_memory_ids=allowed_memory_ids,
        )
        candidates = [
            _to_retrieved_memory(item, self._records[item["memory_id"]])
            for item in core_result["memories"]
            if item["memory_id"] in self._records
        ]
        selected = _select_with_budget(candidates, top_k, token_budget)
        reinforce = self._config.reinforcement_enabled and mutate
        if reinforce:
            self._bank.reinforce_memory_ids(
                [item.memory_id for item in selected], current_time=query_time
            )
        for rank, item in enumerate(selected, start=1):
            item.rank = rank
        latency_ms = (time.perf_counter() - started) * 1000.0
        return MethodResult(
            method_name=self.name,
            query_id=query.query_id,
            retrieved_memories=selected,
            total_token_cost=sum(item.token_cost for item in selected),
            latency_ms=latency_ms,
            metadata={
                "candidate_k": self._config.candidate_k,
                "token_budget": token_budget,
                "forgetting_enabled": self._config.forgetting_enabled,
                "exclude_forgotten": (
                    self._config.forgetting_enabled
                    and self._config.exclude_forgotten
                ),
                "reinforcement_applied": reinforce,
                "query_time": query.reference_time.isoformat(),
                "source_retriever": "MemoryBank/FAISS IndexFlatIP",
                "core_retrieval": core_result,
            },
        )


def memory_record_to_piece(memory: MemoryRecord) -> MemoryPiece:
    """Deterministically preserve controlled-record identity in MemoryBank Core."""
    try:
        memory_type = MemoryType(memory.memory_type)
    except ValueError:
        memory_type = MemoryType.FACT
    return MemoryPiece(
        content=memory.text,
        timestamp=_date_timestamp(memory.event_time),
        memory_type=memory_type,
        memory_id=memory.memory_id,
        status=_map_status(memory.status),
        validity_period=(
            _date_timestamp(memory.valid_from or memory.event_time),
            _date_timestamp(memory.valid_to) if memory.valid_to else None,
        ),
        tags=[memory.subject, memory.attribute, memory.memory_type],
        confidence=memory.confidence,
        source=str(memory.metadata.get("source", "controlled_experiment")),
        last_accessed=_date_timestamp(memory.event_time),
    )


def _map_status(status: RecordMemoryStatus) -> OnlineMemoryStatus:
    return {
        RecordMemoryStatus.CURRENT: OnlineMemoryStatus.ACTIVE,
        RecordMemoryStatus.HISTORICAL: OnlineMemoryStatus.SUPERSEDED,
        RecordMemoryStatus.INVALIDATED: OnlineMemoryStatus.TEMP_INVALID,
        RecordMemoryStatus.UNKNOWN: OnlineMemoryStatus.CONFLICTING,
    }[status]


def _date_timestamp(value: date) -> float:
    return datetime.combine(value, datetime_time.min, tzinfo=timezone.utc).timestamp()


def _to_retrieved_memory(
    item: dict[str, Any], record: MemoryRecord
) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=record.memory_id,
        score=float(item["composite_score"]),
        rank=int(item["retrieval_rank"]),
        token_cost=record.token_cost,
        source_view="flat",
        metadata={
            key: value
            for key, value in item.items()
            if key not in {"memory_id", "content", "retrieval_rank", "score"}
        },
    )


def _select_with_budget(
    candidates: list[RetrievedMemory], top_k: int, token_budget: int | None
) -> list[RetrievedMemory]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if token_budget is not None and token_budget < 0:
        raise ValueError("token_budget must be non-negative or None")
    selected: list[RetrievedMemory] = []
    used = 0
    for candidate in candidates:
        if token_budget is not None and used + candidate.token_cost > token_budget:
            continue
        selected.append(candidate)
        used += candidate.token_cost
        if len(selected) == top_k:
            break
    return selected
