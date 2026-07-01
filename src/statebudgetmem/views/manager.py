"""
views.manager — MemoryViewManager implementation

Consumes VersioningEngine to build:
- Current View: active memories resolved at a reference point
- History View: all versions for a given state key

Also implements the ViewManager ABC from core.online for compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from statebudgetmem.core.online import ViewManager, ViewType
from statebudgetmem.versioning.engine import VersioningEngine
from statebudgetmem.versioning.models import (
    ResolvedState,
    StateKey,
    StateObservation,
)
from statebudgetmem.versioning.operations import ComputedStatus


# ---- Public output type ----
class ViewEntry:
    """A single entry in a memory view, wrapping a ResolvedState."""

    __slots__ = (
        "memory_id",
        "state_key",
        "value",
        "text",
        "effective_from",
        "effective_to",
        "status",
        "subject",
        "attribute",
        "dimensions",
    )

    def __init__(self, state: ResolvedState, observation: StateObservation | None = None) -> None:
        self.memory_id = state.memory_id
        self.state_key = str(state.state_key)
        self.value = state.value
        self.text = observation.text if observation else ""
        self.effective_from = state.effective_from
        self.effective_to = state.effective_to
        self.status = _status_label(state.computed_status)
        self.subject = state.state_key.subject
        self.attribute = state.state_key.attribute
        self.dimensions = state.state_key.dimension_map()

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "state_key": self.state_key,
            "value": self.value,
            "text": self.text,
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "status": self.status,
            "subject": self.subject,
            "attribute": self.attribute,
            "dimensions": self.dimensions,
        }

    @property
    def content(self) -> str:
        """Human-readable single-line representation for retrieval."""
        parts = [f"[{self.subject}] {self.attribute}: {self.value}"]
        if self.text and self.text != self.value:
            parts.append(f"({self.text})")
        return " ".join(parts)


def _status_label(computed_status: ComputedStatus) -> str:
    return computed_status.value


# ---- ViewManager implementation ----
class MemoryViewManager(ViewManager):
    """
    Bridge between versioning and retrieval.

    Wraps a ``VersioningEngine`` to produce memory views suitable for
    downstream retrieval and answer generation.
    """

    def __init__(self, engine: VersioningEngine | None = None) -> None:
        self._engine = engine or VersioningEngine()

    @property
    def engine(self) -> VersioningEngine:
        return self._engine

    # ---- ViewManager ABC ----
    def get_current_view(self, **filters: Any) -> list[dict[str, Any]]:
        """Return all currently-active memory entries as dicts."""
        return [entry.to_dict() for entry in self.current_entries()]

    def get_history_view(
        self, memory_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return full history — optionally scoped to a state-key lineage."""
        if memory_id:
            return [entry.to_dict() for entry in self.lineage_entries(memory_id)]
        return [entry.to_dict() for entry in self.all_history_entries()]

    def sync_views(self, operation: Any, memory: Any) -> None:
        """No-op: views are derived on-demand from the versioning engine."""
        return

    # ---- Primary API ----
    def current_entries(
        self, *, reference_time: date | str | None = None
    ) -> list[ViewEntry]:
        """All CURRENT entries resolved at *reference_time*."""
        view = self._engine.current_view(reference_time=reference_time)
        entries: list[ViewEntry] = []
        for _key, states in view.items():
            for state in states:
                obs = self._engine._observations.get(state.memory_id)
                entries.append(ViewEntry(state, obs))
        entries.sort(key=lambda e: (e.subject, e.attribute, e.memory_id))
        return entries

    def history_entries(self, state_key: StateKey) -> list[ViewEntry]:
        """All historical entries (incl. superseded) for *state_key*."""
        history = self._engine.history(state_key)
        entries: list[ViewEntry] = []
        for state in history:
            obs = self._engine._observations.get(state.memory_id)
            entries.append(ViewEntry(state, obs))
        entries.sort(key=lambda e: (e.effective_from, e.memory_id))
        return entries

    def all_history_entries(self) -> list[ViewEntry]:
        """Flattened history across all state keys."""
        entries: list[ViewEntry] = []
        seen: set[str] = set()
        for node in self._engine.graph.nodes:
            if node.memory_id in seen:
                continue
            seen.add(node.memory_id)
            obs = self._engine._observations.get(node.memory_id)
            if obs is None:
                continue
            entries.append(
                ViewEntry(
                    ResolvedState(
                        memory_id=node.memory_id,
                        state_key=node.state_key,
                        value=obs.value,
                        effective_from=obs.effective_from,
                        effective_to=obs.valid_to,
                        computed_status=node.computed_status,
                        reasons=[],
                    ),
                    obs,
                )
            )
        entries.sort(key=lambda e: (e.effective_from, e.memory_id))
        return entries

    def lineage_entries(self, memory_id: str) -> list[ViewEntry]:
        """Entries in the same version lineage as *memory_id*."""
        lineage = self._engine.lineage(memory_id)
        entries: list[ViewEntry] = []
        for state in lineage:
            obs = self._engine._observations.get(state.memory_id)
            entries.append(ViewEntry(state, obs))
        entries.sort(key=lambda e: (e.effective_from, e.memory_id))
        return entries

    def select_view(
        self,
        view_type: ViewType,
        *,
        state_key: StateKey | None = None,
        reference_time: date | str | None = None,
    ) -> list[ViewEntry]:
        """
        Dispatch to the right view based on routing output.

        Returns
        -------
        list[ViewEntry]
            CURRENT → current_entries()
            HISTORY → history_entries(state_key) or all_history_entries()
            BOTH    → current + history (deduplicated)
            NONE    → empty list
        """
        if view_type is ViewType.CURRENT:
            return self.current_entries(reference_time=reference_time)
        if view_type is ViewType.HISTORY:
            if state_key:
                return self.history_entries(state_key)
            return self.all_history_entries()
        if view_type is ViewType.BOTH:
            current = self.current_entries(reference_time=reference_time)
            current_ids = {e.memory_id for e in current}
            if state_key:
                history = [
                    e
                    for e in self.history_entries(state_key)
                    if e.memory_id not in current_ids
                ]
            else:
                history = [
                    e
                    for e in self.all_history_entries()
                    if e.memory_id not in current_ids
                ]
            return current + history
        # ViewType.NONE
        return []

    # ---- Convenience ----
    def to_texts(self, entries: list[ViewEntry]) -> list[str]:
        """Convert entries to retrieval-ready text lines."""
        return [entry.content for entry in entries]
