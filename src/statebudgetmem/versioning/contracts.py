from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Protocol, runtime_checkable

from statebudgetmem.schemas import MemoryRecord
from statebudgetmem.versioning.models import (
    BatchUpdateResult,
    ResolvedState,
    StateKey,
    ValidationReport,
)
from statebudgetmem.versioning.operations import UpdateOperation


@runtime_checkable
class VersionManager(Protocol):
    """Public contract implemented by :class:`VersioningEngine`.

    This replaces the older online-only ``VersionManager`` abstraction whose
    methods did not match the versioning engine that is actually used by the
    research pipeline.
    """

    def reset(self) -> None: ...

    def ingest(self, memory: MemoryRecord) -> BatchUpdateResult: ...

    def ingest_many(
        self,
        memories: Iterable[MemoryRecord],
        *,
        sort_by_event_time: bool = True,
    ) -> BatchUpdateResult: ...

    def resolve_at(
        self,
        state_key: StateKey,
        reference_time: date | str,
    ) -> tuple[ResolvedState, ...]: ...

    def resolve_current(
        self,
        state_key: StateKey,
        *,
        reference_time: date | str | None = None,
    ) -> tuple[ResolvedState, ...]: ...

    def current_view(
        self,
        *,
        reference_time: date | str | None = None,
    ) -> dict[StateKey, tuple[ResolvedState, ...]]: ...

    def history(self, state_key: StateKey) -> tuple[ResolvedState, ...]: ...

    def lineage(self, memory_id: str) -> tuple[ResolvedState, ...]: ...

    def validate(self) -> ValidationReport: ...

    def snapshot(self) -> dict[str, object]: ...


__all__ = ["UpdateOperation", "VersionManager"]
