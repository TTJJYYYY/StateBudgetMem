"""Eligibility-only StateBudgetMem adapter over the shared MemoryBank method.

The current MemoryBankMethod does not yet expose scoped retrieval.  This module
therefore owns only version/view/routing decisions and keeps the unavailable
dense call behind one method.  It must not fall back to retrieve-then-filter.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from statebudgetmem.baselines.memorybank.adapter import MemoryBankMethod
from statebudgetmem.baselines.memorybank.system import MemoryBank
from statebudgetmem.core.experiment import MethodBuildContext
from statebudgetmem.routing.models import RoutingQueryRecord
from statebudgetmem.routing.router import RuleBasedRouter
from statebudgetmem.schemas.records import MemoryRecord, QueryRecord, QueryType
from statebudgetmem.schemas.results import MethodResult
from statebudgetmem.views.manager import RecordViewManager


class StateBudgetMemMode(str, Enum):
    VERSIONING = "versioning"
    DUAL_VIEWS = "dual_views"
    RULE_ROUTING = "rule_routing"
    ORACLE_ROUTING = "oracle_routing"


_METHOD_NAMES: dict[StateBudgetMemMode, str] = {
    StateBudgetMemMode.VERSIONING: "memorybank_versioning",
    StateBudgetMemMode.DUAL_VIEWS: "memorybank_dual_views",
    StateBudgetMemMode.RULE_ROUTING: "statebudgetmem_rule",
    StateBudgetMemMode.ORACLE_ROUTING: "statebudgetmem_oracle",
}


@dataclass(frozen=True)
class EligibilityDecision:
    effective_query_type: QueryType
    predicted_query_type: QueryType | None
    eligible_memory_ids: frozenset[str]
    source_view: str
    router_source: str
    selection_policy: str
    metadata: Mapping[str, Any]


class StateBudgetMemDenseMethod:
    """Compose versioning, views, and routing before shared dense retrieval."""

    def __init__(
        self,
        context: MethodBuildContext,
        *,
        method_name: str,
        mode: StateBudgetMemMode,
        base_method: MemoryBankMethod | None = None,
        view_manager: RecordViewManager | None = None,
        router: RuleBasedRouter | None = None,
    ) -> None:
        expected_name = _METHOD_NAMES.get(mode)
        if expected_name is None or method_name != expected_name:
            raise ValueError(
                f"method_name {method_name!r} does not match mode {mode!r}; "
                f"expected {expected_name!r}"
            )

        self._name = method_name
        self._mode = mode
        self._base_method = base_method or MemoryBankMethod(context)
        self._view_manager = view_manager or RecordViewManager()
        self._router = (
            router or RuleBasedRouter(fallback_type=QueryType.GENERAL)
            if mode is StateBudgetMemMode.RULE_ROUTING
            else None
        )
        self._input_memory_ids: frozenset[str] = frozenset()
        self._last_eligibility_decision: EligibilityDecision | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> StateBudgetMemMode:
        return self._mode

    @property
    def base_method(self) -> MemoryBankMethod:
        return self._base_method

    @property
    def bank(self) -> MemoryBank:
        return self._base_method.bank

    @property
    def embedding_model(self) -> Any:
        return self._base_method.embedding_model

    @property
    def view_manager(self) -> RecordViewManager:
        return self._view_manager

    @property
    def last_eligibility_decision(self) -> EligibilityDecision | None:
        return self._last_eligibility_decision

    def reset(self) -> None:
        self._base_method.reset()
        self._view_manager.reset()
        self._input_memory_ids = frozenset()
        self._last_eligibility_decision = None

    def ingest(self, memories: list[MemoryRecord]) -> None:
        records = list(memories)
        memory_ids = [memory.memory_id for memory in records]
        duplicates = sorted(
            memory_id for memory_id in set(memory_ids) if memory_ids.count(memory_id) > 1
        )
        if duplicates:
            self.reset()
            raise ValueError(f"duplicate memory_id values: {duplicates}")

        self.reset()
        try:
            self._base_method.ingest(records)
            self._view_manager.ingest(records)

            input_ids = frozenset(memory_ids)
            bank_ids = frozenset(self.bank.memories_by_id)
            view_ids = frozenset(self._view_manager.memories_by_id)
            if not (input_ids == bank_ids == view_ids):
                raise ValueError(
                    "memory ID mismatch after ingest: "
                    f"input={sorted(input_ids)}, bank={sorted(bank_ids)}, "
                    f"views={sorted(view_ids)}"
                )
            self._input_memory_ids = input_ids
        except Exception:
            self.reset()
            raise

    def _to_routing_query(self, query: QueryRecord) -> RoutingQueryRecord:
        return RoutingQueryRecord(
            text=query.text,
            reference_time=query.reference_time.isoformat(),
        )

    def _effective_query_type(
        self, query: QueryRecord
    ) -> tuple[QueryType, QueryType | None, str, str]:
        if self._mode is StateBudgetMemMode.VERSIONING:
            return (
                QueryType.CURRENT,
                None,
                "none",
                "current_only_no_router",
            )

        if self._mode is StateBudgetMemMode.DUAL_VIEWS:
            return (
                QueryType.CHANGE,
                None,
                "none",
                "current_and_history_no_router",
            )

        if self._mode is StateBudgetMemMode.RULE_ROUTING:
            if self._router is None:  # defensive invariant
                raise RuntimeError("rule routing mode requires RuleBasedRouter")
            predicted = self._router.classify(self._to_routing_query(query))
            return predicted, predicted, "rule", "rule_routed"

        return (
            query.query_type,
            query.query_type,
            "oracle_query_type",
            "oracle_routed",
        )

    def _current_and_history_records(self, query: QueryRecord) -> list[MemoryRecord]:
        records_by_id = {
            memory.memory_id: memory
            for memory in self._view_manager.current_records(
                reference_time=query.reference_time
            )
        }
        for memory in self._view_manager.history_records():
            records_by_id.setdefault(memory.memory_id, memory)
        return [records_by_id[memory_id] for memory_id in sorted(records_by_id)]

    def _resolve_eligibility(self, query: QueryRecord) -> EligibilityDecision:
        (
            effective_query_type,
            predicted_query_type,
            router_source,
            selection_policy,
        ) = self._effective_query_type(query)

        if effective_query_type is QueryType.GENERAL:
            records: list[MemoryRecord] = []
            source_view = "none"
        elif self._mode is StateBudgetMemMode.DUAL_VIEWS:
            records = self._current_and_history_records(query)
            source_view = "current_and_history"
        elif effective_query_type is QueryType.CURRENT:
            records = self._view_manager.current_records(
                reference_time=query.reference_time
            )
            source_view = "current"
        elif effective_query_type is QueryType.HISTORICAL:
            records = self._view_manager.point_in_time_records(
                reference_time=query.reference_time
            )
            source_view = "history"
        else:
            records = self._current_and_history_records(query)
            source_view = "current_and_history"

        decision = EligibilityDecision(
            effective_query_type=effective_query_type,
            predicted_query_type=predicted_query_type,
            eligible_memory_ids=frozenset(memory.memory_id for memory in records),
            source_view=source_view,
            router_source=router_source,
            selection_policy=selection_policy,
            metadata=MappingProxyType({"mode": self._mode.value}),
        )
        self._last_eligibility_decision = decision
        return decision

    def _retrieve_from_memorybank(
        self,
        query: QueryRecord,
        *,
        allowed_memory_ids: set[str],
        top_k: int,
        token_budget: int | None,
        mutate: bool,
    ) -> MethodResult:
        return self._base_method.retrieve_scoped(
            query,
            allowed_memory_ids=allowed_memory_ids,
            top_k=top_k,
            token_budget=token_budget,
            mutate=mutate,
        )

    def retrieve(
        self,
        query: QueryRecord,
        *,
        top_k: int,
        token_budget: int | None = None,
        mutate: bool = False,
    ) -> MethodResult:
        started = time.perf_counter()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if token_budget is not None and token_budget < 0:
            raise ValueError("token_budget must be non-negative or None")

        decision = self._resolve_eligibility(query)
        if not decision.eligible_memory_ids:
            return self._empty_result(query, decision, started=started)

        base_result = self._retrieve_from_memorybank(
            query,
            allowed_memory_ids=set(decision.eligible_memory_ids),
            top_k=top_k,
            token_budget=token_budget,
            mutate=mutate,
        )
        metadata = dict(base_result.metadata)
        metadata.update(self._decision_metadata(decision))
        return MethodResult(
            method_name=self.name,
            query_id=query.query_id,
            retrieved_memories=base_result.retrieved_memories,
            predicted_query_type=decision.predicted_query_type,
            total_token_cost=base_result.total_token_cost,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )

    def _empty_result(
        self,
        query: QueryRecord,
        decision: EligibilityDecision,
        *,
        started: float,
    ) -> MethodResult:
        metadata = self._decision_metadata(decision)
        metadata["skipped_dense_retrieval"] = True
        return MethodResult(
            method_name=self.name,
            query_id=query.query_id,
            retrieved_memories=[],
            predicted_query_type=decision.predicted_query_type,
            total_token_cost=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )

    def _decision_metadata(self, decision: EligibilityDecision) -> dict[str, Any]:
        return {
            "statebudgetmem_mode": self._mode.value,
            "effective_query_type": decision.effective_query_type.value,
            "router_source": decision.router_source,
            "source_view": decision.source_view,
            "eligible_memory_count": len(decision.eligible_memory_ids),
            "base_method_name": "memorybank_core",
            "selection_policy": decision.selection_policy,
        }
