from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from statebudgetmem.versioning.exceptions import InvalidDecisionError
from statebudgetmem.versioning.graph import VersionGraph
from statebudgetmem.versioning.models import (
    StateObservation,
    UpdateDecision,
    UpdateResult,
    VersionEdge,
    VersionNode,
)
from statebudgetmem.versioning.operations import (
    ComputedStatus,
    UpdateOperation,
    VersionRelation,
)


class VersionUpdater:
    """Apply a validated update decision to a mutable VersionGraph."""

    def apply(
        self,
        graph: VersionGraph,
        new_observation: StateObservation,
        decision: UpdateDecision,
        observations_by_id: Mapping[str, StateObservation],
    ) -> UpdateResult:
        self._validate_decision(graph, new_observation, decision)
        operation = decision.operation
        if operation is UpdateOperation.NOOP:
            return UpdateResult(decision=decision, skipped=True)
        if operation is UpdateOperation.ADD:
            return self._apply_add(graph, new_observation, decision)
        if operation is UpdateOperation.SUPERSEDE:
            return self._apply_replacement(
                graph,
                new_observation,
                decision,
                relation=VersionRelation.SUPERSEDES,
            )
        if operation is UpdateOperation.MERGE:
            return self._apply_replacement(
                graph,
                new_observation,
                decision,
                relation=VersionRelation.MERGES_INTO,
            )
        if operation is UpdateOperation.TEMP_INVALIDATE:
            return self._apply_temp_invalidate(graph, new_observation, decision)
        if operation is UpdateOperation.RESTORE:
            return self._apply_restore(
                graph,
                new_observation,
                decision,
                observations_by_id,
            )
        if operation is UpdateOperation.DELETE:
            return self._apply_delete(graph, new_observation, decision)
        raise InvalidDecisionError(f"unsupported update operation: {operation}")

    def _apply_add(
        self,
        graph: VersionGraph,
        observation: StateObservation,
        decision: UpdateDecision,
    ) -> UpdateResult:
        graph.add_node(self._new_node(observation, ComputedStatus.CURRENT))
        return UpdateResult(
            decision=decision,
            created_node_ids=[observation.memory_id],
        )

    def _apply_replacement(
        self,
        graph: VersionGraph,
        observation: StateObservation,
        decision: UpdateDecision,
        *,
        relation: VersionRelation,
    ) -> UpdateResult:
        self._require_targets(decision)
        updated: list[str] = []
        graph.add_node(self._new_node(observation, ComputedStatus.CURRENT))
        edges: list[VersionEdge] = []
        effective_from = observation.effective_from
        for target_id in decision.target_memory_ids:
            target = graph.get_node(target_id)
            target.computed_status = ComputedStatus.HISTORICAL
            target.valid_to = self._closed_valid_to(target.valid_to, effective_from)
            target.invalidated_by = [
                item for item in target.invalidated_by if item != observation.memory_id
            ]
            graph.update_node(target)
            updated.append(target_id)
            edge = self._edge(
                target_id,
                observation.memory_id,
                relation,
                effective_from,
                decision,
            )
            graph.add_edge(edge)
            edges.append(edge)
        return UpdateResult(
            decision=decision,
            created_node_ids=[observation.memory_id],
            updated_node_ids=updated,
            created_edges=edges,
        )

    def _apply_temp_invalidate(
        self,
        graph: VersionGraph,
        observation: StateObservation,
        decision: UpdateDecision,
    ) -> UpdateResult:
        self._require_targets(decision)
        graph.add_node(self._new_node(observation, ComputedStatus.CURRENT))
        updated: list[str] = []
        edges: list[VersionEdge] = []
        for target_id in decision.target_memory_ids:
            target = graph.get_node(target_id)
            target.computed_status = ComputedStatus.TEMP_INVALIDATED
            target.invalidated_by = list(
                dict.fromkeys([*target.invalidated_by, observation.memory_id])
            )
            graph.update_node(target)
            updated.append(target_id)
            edge = self._edge(
                target_id,
                observation.memory_id,
                VersionRelation.TEMP_INVALIDATES,
                observation.effective_from,
                decision,
            )
            graph.add_edge(edge)
            edges.append(edge)
        return UpdateResult(
            decision=decision,
            created_node_ids=[observation.memory_id],
            updated_node_ids=updated,
            created_edges=edges,
        )

    def _apply_restore(
        self,
        graph: VersionGraph,
        observation: StateObservation,
        decision: UpdateDecision,
        observations_by_id: Mapping[str, StateObservation],
    ) -> UpdateResult:
        del observations_by_id
        self._require_targets(decision)
        restore_source_ids = list(
            dict.fromkeys(decision.metadata.get("restore_source_ids", []))
        )
        temporary_ids = list(
            dict.fromkeys(decision.metadata.get("temporary_target_ids", []))
        )
        if not restore_source_ids:
            restore_source_ids = [
                target_id
                for target_id in decision.target_memory_ids
                if graph.get_node(target_id).computed_status
                in {ComputedStatus.TEMP_INVALIDATED, ComputedStatus.HISTORICAL}
            ]
        if not temporary_ids:
            temporary_ids = [
                target_id
                for target_id in decision.target_memory_ids
                if target_id not in restore_source_ids
            ]
        if not restore_source_ids:
            raise InvalidDecisionError("RESTORE requires at least one restore source")

        graph.add_node(self._new_node(observation, ComputedStatus.CURRENT))
        updated: list[str] = []
        edges: list[VersionEdge] = []
        effective_from = observation.effective_from

        for temporary_id in temporary_ids:
            temporary = graph.get_node(temporary_id)
            temporary.computed_status = ComputedStatus.HISTORICAL
            temporary.valid_to = self._closed_valid_to(
                temporary.valid_to,
                effective_from,
            )
            graph.update_node(temporary)
            updated.append(temporary_id)
            edge = self._edge(
                temporary_id,
                observation.memory_id,
                VersionRelation.SUPERSEDES,
                effective_from,
                decision,
            )
            graph.add_edge(edge)
            edges.append(edge)

        for source_id in restore_source_ids:
            source = graph.get_node(source_id)
            source.computed_status = ComputedStatus.HISTORICAL
            source.invalidated_by = [
                item for item in source.invalidated_by if item not in temporary_ids
            ]
            graph.update_node(source)
            updated.append(source_id)
            edge = self._edge(
                source_id,
                observation.memory_id,
                VersionRelation.RESTORES,
                effective_from,
                decision,
            )
            graph.add_edge(edge)
            edges.append(edge)

        return UpdateResult(
            decision=decision,
            created_node_ids=[observation.memory_id],
            updated_node_ids=list(dict.fromkeys(updated)),
            created_edges=edges,
        )

    def _apply_delete(
        self,
        graph: VersionGraph,
        observation: StateObservation,
        decision: UpdateDecision,
    ) -> UpdateResult:
        self._require_targets(decision)
        graph.add_node(self._new_node(observation, ComputedStatus.DELETED))
        updated: list[str] = []
        edges: list[VersionEdge] = []
        effective_from = observation.effective_from
        for target_id in decision.target_memory_ids:
            target = graph.get_node(target_id)
            target.computed_status = ComputedStatus.DELETED
            target.valid_to = self._closed_valid_to(target.valid_to, effective_from)
            graph.update_node(target)
            updated.append(target_id)
            edge = self._edge(
                target_id,
                observation.memory_id,
                VersionRelation.DELETES,
                effective_from,
                decision,
            )
            graph.add_edge(edge)
            edges.append(edge)
        return UpdateResult(
            decision=decision,
            created_node_ids=[observation.memory_id],
            updated_node_ids=updated,
            created_edges=edges,
        )

    @staticmethod
    def _new_node(
        observation: StateObservation,
        status: ComputedStatus,
    ) -> VersionNode:
        return VersionNode(
            memory_id=observation.memory_id,
            state_key=observation.state_key,
            computed_status=status,
            valid_from=observation.effective_from,
            valid_to=observation.valid_to,
            invalidated_by=[],
            metadata=deepcopy(observation.metadata),
        )

    @staticmethod
    def _closed_valid_to(current: object, effective_from: object):
        if current is None:
            return effective_from
        return min(current, effective_from)

    @staticmethod
    def _edge(
        predecessor_id: str,
        successor_id: str,
        relation: VersionRelation,
        effective_from: object,
        decision: UpdateDecision,
    ) -> VersionEdge:
        return VersionEdge(
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            relation=relation,
            effective_from=effective_from,
            confidence=decision.confidence,
            reason=decision.reason,
            metadata={"operation": decision.operation.value},
        )

    @staticmethod
    def _require_targets(decision: UpdateDecision) -> None:
        if not decision.target_memory_ids:
            raise InvalidDecisionError(
                f"{decision.operation.value} requires at least one target memory"
            )

    @staticmethod
    def _validate_decision(
        graph: VersionGraph,
        observation: StateObservation,
        decision: UpdateDecision,
    ) -> None:
        if decision.new_memory_id != observation.memory_id:
            raise InvalidDecisionError(
                "decision.new_memory_id must equal observation.memory_id"
            )
        if decision.state_key != observation.state_key:
            raise InvalidDecisionError(
                "decision.state_key must equal observation.state_key"
            )
        if observation.memory_id in graph and decision.operation is not UpdateOperation.NOOP:
            raise InvalidDecisionError(
                f"version node already exists: {observation.memory_id}"
            )
        for target_id in decision.target_memory_ids:
            if target_id not in graph:
                raise InvalidDecisionError(f"unknown decision target: {target_id}")
            if target_id == observation.memory_id:
                raise InvalidDecisionError("a decision cannot target the incoming memory itself")
