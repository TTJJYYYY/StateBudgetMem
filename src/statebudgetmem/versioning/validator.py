from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from statebudgetmem.versioning.graph import VersionGraph
from statebudgetmem.versioning.models import (
    StateObservation,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from statebudgetmem.versioning.operations import ComputedStatus, VersionRelation


class VersionGraphValidator:
    """Validate structural and state-transition invariants."""

    def validate(
        self,
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation] | None = None,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        node_ids = {node.memory_id for node in graph.nodes}

        for edge in graph.edges:
            if edge.predecessor_id not in node_ids or edge.successor_id not in node_ids:
                issues.append(
                    self._error(
                        "MISSING_EDGE_ENDPOINT",
                        f"edge endpoint is missing: {edge.predecessor_id}->{edge.successor_id}",
                        [edge.predecessor_id, edge.successor_id],
                    )
                )
            if edge.predecessor_id == edge.successor_id:
                issues.append(
                    self._error(
                        "SELF_EDGE",
                        f"self edge is not allowed: {edge.predecessor_id}",
                        [edge.predecessor_id],
                    )
                )

        if self._has_cycle(graph):
            issues.append(
                self._error(
                    "GRAPH_CYCLE",
                    "version graph contains a directed cycle",
                    [],
                )
            )

        signatures: set[tuple[str, str, VersionRelation]] = set()
        for edge in graph.edges:
            signature = (edge.predecessor_id, edge.successor_id, edge.relation)
            if signature in signatures:
                issues.append(
                    self._error(
                        "DUPLICATE_EDGE",
                        f"duplicate relation edge: {signature}",
                        [edge.predecessor_id, edge.successor_id],
                    )
                )
            signatures.add(signature)

        for node in graph.nodes:
            for invalidator_id in node.invalidated_by:
                if invalidator_id not in node_ids:
                    issues.append(
                        self._error(
                            "UNKNOWN_INVALIDATOR",
                            f"node {node.memory_id} references unknown invalidator {invalidator_id}",
                            [node.memory_id, invalidator_id],
                        )
                    )
                    continue
                matching_edge = any(
                    edge.predecessor_id == node.memory_id
                    and edge.successor_id == invalidator_id
                    and edge.relation is VersionRelation.TEMP_INVALIDATES
                    for edge in graph.edges
                )
                if not matching_edge:
                    issues.append(
                        self._error(
                            "MISSING_TEMP_EDGE",
                            f"node {node.memory_id} lists {invalidator_id} without TEMP_INVALIDATES edge",
                            [node.memory_id, invalidator_id],
                        )
                    )

        if observations_by_id is not None:
            for node in graph.nodes:
                observation = observations_by_id.get(node.memory_id)
                if observation is None:
                    issues.append(
                        self._error(
                            "MISSING_OBSERVATION",
                            f"node {node.memory_id} has no StateObservation",
                            [node.memory_id],
                        )
                    )
                    continue
                if observation.state_key != node.state_key:
                    issues.append(
                        self._error(
                            "STATE_KEY_MISMATCH",
                            f"node and observation StateKey differ for {node.memory_id}",
                            [node.memory_id],
                        )
                    )

        current_by_key: dict[object, list[str]] = defaultdict(list)
        for node in graph.nodes:
            if node.computed_status is ComputedStatus.CURRENT and node.valid_to is None:
                current_by_key[node.state_key].append(node.memory_id)
        for state_key, memory_ids in current_by_key.items():
            if len(memory_ids) > 1:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="MULTIPLE_OPEN_CURRENT",
                        message=f"multiple open CURRENT nodes for {state_key}",
                        memory_ids=sorted(memory_ids),
                    )
                )

        return ValidationReport(issues=issues)

    @staticmethod
    def _has_cycle(graph: VersionGraph) -> bool:
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in graph.edges:
            adjacency[edge.predecessor_id].append(edge.successor_id)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            for successor in adjacency[node_id]:
                if visit(successor):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return any(visit(node.memory_id) for node in graph.nodes)

    @staticmethod
    def _error(code: str, message: str, memory_ids: list[str]) -> ValidationIssue:
        return ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=code,
            message=message,
            memory_ids=memory_ids,
        )
