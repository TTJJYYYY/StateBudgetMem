from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from statebudgetmem.schemas import MemoryRecord, QueryRecord, QueryType
from statebudgetmem.versioning.engine import VersioningEngine
from statebudgetmem.versioning.models import StateKey
from statebudgetmem.views.models import ViewDecision, ViewName, ViewPolicy
from statebudgetmem.views.selectors import (
    all_state_keys,
    current_memory_ids,
    history_memory_ids,
    ordered_records_by_ids,
    point_in_time_memory_ids,
    records_by_id,
)


class RecordViewManager:
    """Maintain query-aware Current and History views.

    The manager delegates all temporal semantics to ``VersioningEngine`` and
    only maps resolved memory IDs back to public ``MemoryRecord`` objects.
    """

    def __init__(
        self,
        *,
        version_manager: VersioningEngine | None = None,
        policy: ViewPolicy | None = None,
    ) -> None:
        self.version_manager = version_manager or VersioningEngine()
        self.policy = policy or ViewPolicy()
        self._memories: list[MemoryRecord] = []
        self._memories_by_id: dict[str, MemoryRecord] = {}

    @property
    def memories(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._memories)

    @property
    def memories_by_id(self) -> dict[str, MemoryRecord]:
        return dict(self._memories_by_id)

    def reset(self) -> None:
        self.version_manager.reset()
        self._memories = []
        self._memories_by_id = {}

    def ingest(self, memories: Iterable[MemoryRecord]) -> None:
        self.reset()
        self._memories = sorted(
            list(memories),
            key=lambda item: (item.event_time, item.memory_id),
        )
        self._memories_by_id = records_by_id(self._memories)
        self.version_manager.ingest_many(self._memories, sort_by_event_time=True)

    def current_records(
        self,
        *,
        reference_time: date | str | None = None,
    ) -> list[MemoryRecord]:
        ids = current_memory_ids(self.version_manager, reference_time=reference_time)
        return ordered_records_by_ids(self._memories_by_id, ids)

    def point_in_time_records(
        self,
        *,
        reference_time: date | str,
    ) -> list[MemoryRecord]:
        ids = point_in_time_memory_ids(
            self.version_manager,
            reference_time=reference_time,
        )
        return ordered_records_by_ids(self._memories_by_id, ids)

    def history_records(
        self,
        *,
        state_keys: Iterable[StateKey] | None = None,
    ) -> list[MemoryRecord]:
        keys = tuple(state_keys) if state_keys is not None else all_state_keys(self._memories)
        ids = history_memory_ids(self.version_manager, state_keys=keys)
        return ordered_records_by_ids(self._memories_by_id, ids)

    def flat_records(self) -> list[MemoryRecord]:
        return list(self._memories)

    def records_for_query(self, query: QueryRecord, *, view: ViewName) -> list[MemoryRecord]:
        if query.query_type is QueryType.GENERAL:
            return []

        if view is ViewName.FLAT:
            return self.flat_records()

        if view is ViewName.CURRENT:
            # A current-state question must be resolved at the query time so
            # temporary states that have expired are not treated as current.
            if (
                query.query_type in {QueryType.CURRENT, QueryType.CHANGE}
                and not self.policy.current_as_of_latest
            ):
                return self.current_records(reference_time=query.reference_time)
            return self.current_records()

        if view is ViewName.HISTORY:
            if query.query_type is QueryType.HISTORICAL:
                return self.point_in_time_records(reference_time=query.reference_time)
            if (
                query.query_type is QueryType.CHANGE
                and self.policy.history_for_change_queries
            ):
                return self.history_records()
            return self.history_records()

        if view is ViewName.DUAL:
            return self.dual_records(query)

        raise ValueError(f"unsupported view: {view}")

    def dual_records(self, query: QueryRecord) -> list[MemoryRecord]:
        decision = self.route(query)
        seen: set[str] = set()
        merged: list[MemoryRecord] = []

        for view in decision.selected_views:
            records = self.records_for_query(query, view=view)
            for memory in records:
                if memory.memory_id in seen:
                    continue
                seen.add(memory.memory_id)
                merged.append(memory)

        merged.sort(key=lambda item: (item.event_time, item.memory_id))
        return merged

    def route(self, query: QueryRecord) -> ViewDecision:
        if query.query_type is QueryType.CURRENT:
            return ViewDecision(
                query_type=query.query_type,
                selected_views=[ViewName.CURRENT],
                reason="current query uses the state snapshot at query.reference_time",
            )

        if query.query_type is QueryType.HISTORICAL:
            return ViewDecision(
                query_type=query.query_type,
                selected_views=[ViewName.HISTORY],
                reason="historical query uses the state snapshot valid at reference_time",
            )

        if query.query_type is QueryType.CHANGE:
            return ViewDecision(
                query_type=query.query_type,
                selected_views=[ViewName.CURRENT, ViewName.HISTORY],
                reason="change query needs both the current snapshot and full version history",
            )

        return ViewDecision(
            query_type=query.query_type,
            selected_views=[],
            reason="general query should not retrieve personal memory",
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "memory_count": len(self._memories),
            "current_memory_ids": sorted(
                memory.memory_id for memory in self.current_records()
            ),
            "history_memory_ids": sorted(
                memory.memory_id for memory in self.history_records()
            ),
            "versioning": self.version_manager.snapshot(),
        }
