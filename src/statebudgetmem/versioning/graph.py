from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from copy import deepcopy

from statebudgetmem.versioning.exceptions import DuplicateNodeError
from statebudgetmem.versioning.models import VersionEdge, VersionNode
from statebudgetmem.versioning.operations import VersionRelation


class VersionGraph:
    """In-memory directed acyclic graph of computed state versions."""

    def __init__(
        self,
        *,
        nodes: Iterable[VersionNode] = (),
        edges: Iterable[VersionEdge] = (),
    ) -> None:
        self._nodes: dict[str, VersionNode] = {}
        self._edges: list[VersionEdge] = []
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    @property
    def nodes(self) -> tuple[VersionNode, ...]:
        return tuple(self._nodes[memory_id] for memory_id in sorted(self._nodes))

    @property
    def edges(self) -> tuple[VersionEdge, ...]:
        return tuple(self._edges)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self._nodes

    def clone(self) -> "VersionGraph":
        return VersionGraph(
            nodes=(node.model_copy(deep=True) for node in self.nodes),
            edges=(edge.model_copy(deep=True) for edge in self.edges),
        )

    def replace_with(self, other: "VersionGraph") -> None:
        self._nodes = {
            node.memory_id: node.model_copy(deep=True) for node in other.nodes
        }
        self._edges = [edge.model_copy(deep=True) for edge in other.edges]

    def add_node(self, node: VersionNode) -> None:
        if node.memory_id in self._nodes:
            raise DuplicateNodeError(f"duplicate version node: {node.memory_id}")
        self._nodes[node.memory_id] = node.model_copy(deep=True)

    def update_node(self, node: VersionNode) -> None:
        if node.memory_id not in self._nodes:
            raise KeyError(f"unknown version node: {node.memory_id}")
        self._nodes[node.memory_id] = node.model_copy(deep=True)

    def get_node(self, memory_id: str) -> VersionNode:
        try:
            return self._nodes[memory_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"unknown version node: {memory_id}") from exc

    def add_edge(self, edge: VersionEdge) -> None:
        if edge.predecessor_id not in self._nodes:
            raise KeyError(f"unknown predecessor node: {edge.predecessor_id}")
        if edge.successor_id not in self._nodes:
            raise KeyError(f"unknown successor node: {edge.successor_id}")
        signature = (edge.predecessor_id, edge.successor_id, edge.relation)
        if any(
            (item.predecessor_id, item.successor_id, item.relation) == signature
            for item in self._edges
        ):
            return
        self._edges.append(edge.model_copy(deep=True))

    def incoming_edges(
        self,
        memory_id: str,
        relation: VersionRelation | None = None,
    ) -> tuple[VersionEdge, ...]:
        return tuple(
            edge.model_copy(deep=True)
            for edge in self._edges
            if edge.successor_id == memory_id
            and (relation is None or edge.relation is relation)
        )

    def outgoing_edges(
        self,
        memory_id: str,
        relation: VersionRelation | None = None,
    ) -> tuple[VersionEdge, ...]:
        return tuple(
            edge.model_copy(deep=True)
            for edge in self._edges
            if edge.predecessor_id == memory_id
            and (relation is None or edge.relation is relation)
        )

    def predecessors(
        self,
        memory_id: str,
        relation: VersionRelation | None = None,
    ) -> tuple[VersionNode, ...]:
        return tuple(
            self.get_node(edge.predecessor_id)
            for edge in self.incoming_edges(memory_id, relation)
        )

    def successors(
        self,
        memory_id: str,
        relation: VersionRelation | None = None,
    ) -> tuple[VersionNode, ...]:
        return tuple(
            self.get_node(edge.successor_id)
            for edge in self.outgoing_edges(memory_id, relation)
        )

    def has_path(self, source_id: str, target_id: str) -> bool:
        if source_id == target_id:
            return True
        queue: deque[str] = deque([source_id])
        visited: set[str] = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for edge in self.outgoing_edges(current):
                if edge.successor_id == target_id:
                    return True
                queue.append(edge.successor_id)
        return False

    def lineage(self, memory_id: str) -> tuple[VersionNode, ...]:
        if memory_id not in self._nodes:
            raise KeyError(f"unknown version node: {memory_id}")
        related: set[str] = {memory_id}
        queue: deque[str] = deque([memory_id])
        while queue:
            current = queue.popleft()
            neighbor_ids = [
                *(edge.predecessor_id for edge in self.incoming_edges(current)),
                *(edge.successor_id for edge in self.outgoing_edges(current)),
            ]
            for neighbor_id in neighbor_ids:
                if neighbor_id not in related:
                    related.add(neighbor_id)
                    queue.append(neighbor_id)
        return tuple(self.get_node(item) for item in sorted(related))

    def model_dump(self) -> dict[str, object]:
        return {
            "nodes": [node.model_dump(mode="json") for node in self.nodes],
            "edges": [edge.model_dump(mode="json") for edge in self.edges],
        }

    @classmethod
    def model_validate(cls, payload: dict[str, object]) -> "VersionGraph":
        raw_nodes = payload.get("nodes", [])
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError("graph payload requires list-valued nodes and edges")
        return cls(
            nodes=(VersionNode.model_validate(item) for item in raw_nodes),
            edges=(VersionEdge.model_validate(item) for item in raw_edges),
        )

    def __deepcopy__(self, memo: dict[int, object]) -> "VersionGraph":
        del memo
        return self.clone()
