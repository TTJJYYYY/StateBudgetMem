from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date

from statebudgetmem.versioning.exceptions import MissingObservationError
from statebudgetmem.versioning.graph import VersionGraph
from statebudgetmem.versioning.models import (
    ResolvedState,
    StateKey,
    StateObservation,
    VersionNode,
)
from statebudgetmem.versioning.operations import ComputedStatus, VersionRelation


_REPLACEMENT_RELATIONS = {
    VersionRelation.SUPERSEDES,
    VersionRelation.MERGES_INTO,
    VersionRelation.RESTORES,
    VersionRelation.DELETES,
}


class VersionResolver:
    """Resolve exact state slots at a reference date from graph semantics."""

    def __init__(
        self,
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation],
    ) -> None:
        self.graph = graph
        self.observations_by_id = observations_by_id

    def resolve_at(
        self,
        state_key: StateKey,
        reference_time: date | str,
    ) -> tuple[ResolvedState, ...]:
        reference_time = self._coerce_date(reference_time)
        active: list[tuple[VersionNode, StateObservation, list[str]]] = []
        for node in self.graph.nodes:
            if node.state_key != state_key:
                continue
            observation = self._observation(node.memory_id)
            reasons: list[str] = []
            if node.computed_status is ComputedStatus.DELETED:
                continue
            if not self._interval_active(node, observation, reference_time):
                continue
            if self._replaced_at(node.memory_id, reference_time):
                continue
            invalidators = self._active_invalidators(node.memory_id, reference_time)
            if invalidators:
                continue
            reasons.append("node validity interval contains reference_time")
            reasons.append("no active replacement relation at reference_time")
            reasons.append("no active temporary invalidator at reference_time")
            active.append((node, observation, reasons))

        active.sort(
            key=lambda item: (
                -self._effective_from(item[0], item[1]).toordinal(),
                item[0].memory_id,
            )
        )
        conflict = len(active) > 1
        return tuple(
            ResolvedState(
                memory_id=node.memory_id,
                state_key=node.state_key,
                value=observation.value,
                effective_from=self._effective_from(node, observation),
                effective_to=self._effective_to(node, observation),
                computed_status=node.computed_status,
                is_conflict=conflict,
                reasons=[*reasons, *( ["multiple states are active"] if conflict else [] )],
            )
            for node, observation, reasons in active
        )

    def resolve_current(
        self,
        state_key: StateKey,
        *,
        reference_time: date | str | None = None,
    ) -> tuple[ResolvedState, ...]:
        at = self._coerce_date(reference_time) if reference_time is not None else self.latest_known_time()
        if at is None:
            return ()
        return self.resolve_at(state_key, at)

    def current_view(
        self,
        *,
        reference_time: date | str | None = None,
    ) -> dict[StateKey, tuple[ResolvedState, ...]]:
        at = self._coerce_date(reference_time) if reference_time is not None else self.latest_known_time()
        if at is None:
            return {}
        keys = sorted(
            {node.state_key for node in self.graph.nodes},
            key=str,
        )
        return {
            key: resolved
            for key in keys
            if (resolved := self.resolve_at(key, at))
        }

    def history(self, state_key: StateKey) -> tuple[ResolvedState, ...]:
        history: list[ResolvedState] = []
        for node in self.graph.nodes:
            if node.state_key != state_key:
                continue
            observation = self._observation(node.memory_id)
            history.append(
                ResolvedState(
                    memory_id=node.memory_id,
                    state_key=node.state_key,
                    value=observation.value,
                    effective_from=self._effective_from(node, observation),
                    effective_to=self._effective_to(node, observation),
                    computed_status=node.computed_status,
                    reasons=["historical version node"],
                )
            )
        history.sort(key=lambda item: (item.effective_from, item.memory_id))
        return tuple(history)

    def lineage(self, memory_id: str) -> tuple[ResolvedState, ...]:
        result: list[ResolvedState] = []
        for node in self.graph.lineage(memory_id):
            observation = self._observation(node.memory_id)
            result.append(
                ResolvedState(
                    memory_id=node.memory_id,
                    state_key=node.state_key,
                    value=observation.value,
                    effective_from=self._effective_from(node, observation),
                    effective_to=self._effective_to(node, observation),
                    computed_status=node.computed_status,
                    reasons=["member of the same version lineage"],
                )
            )
        result.sort(key=lambda item: (item.effective_from, item.memory_id))
        return tuple(result)

    def latest_known_time(self) -> date | None:
        if not self.observations_by_id:
            return None
        return max(observation.event_time for observation in self.observations_by_id.values())

    def _active_invalidators(self, memory_id: str, at: date) -> tuple[str, ...]:
        active: list[str] = []
        for edge in self.graph.outgoing_edges(
            memory_id,
            VersionRelation.TEMP_INVALIDATES,
        ):
            if edge.effective_from > at:
                continue
            invalidator = self.graph.get_node(edge.successor_id)
            observation = self._observation(edge.successor_id)
            if invalidator.computed_status is ComputedStatus.DELETED:
                continue
            if not self._interval_active(invalidator, observation, at):
                continue
            if self._replaced_at(invalidator.memory_id, at):
                continue
            active.append(invalidator.memory_id)
        return tuple(active)

    def _replaced_at(self, memory_id: str, at: date) -> bool:
        return any(
            edge.relation in _REPLACEMENT_RELATIONS and edge.effective_from <= at
            for edge in self.graph.outgoing_edges(memory_id)
        )

    @staticmethod
    def _effective_from(node: VersionNode, observation: StateObservation) -> date:
        return node.valid_from or observation.valid_from or observation.event_time

    @staticmethod
    def _effective_to(
        node: VersionNode,
        observation: StateObservation,
    ) -> date | None:
        values = [
            value
            for value in (node.valid_to, observation.valid_to)
            if value is not None
        ]
        return min(values) if values else None

    def _interval_active(
        self,
        node: VersionNode,
        observation: StateObservation,
        at: date,
    ) -> bool:
        effective_from = self._effective_from(node, observation)
        effective_to = self._effective_to(node, observation)
        return at >= effective_from and (effective_to is None or at < effective_to)


    @staticmethod
    def _coerce_date(value: date | str) -> date:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        raise TypeError("reference_time must be a date or ISO date string")

    def _observation(self, memory_id: str) -> StateObservation:
        try:
            return self.observations_by_id[memory_id]
        except KeyError as exc:
            raise MissingObservationError(
                f"missing observation for memory_id: {memory_id}"
            ) from exc
