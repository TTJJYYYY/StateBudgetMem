from __future__ import annotations

from statebudgetmem.versioning import (
    ComputedStatus,
    StateKey,
    VersionEdge,
    VersionGraph,
    VersionGraphValidator,
    VersionNode,
    VersionRelation,
)


def _node(memory_id: str) -> VersionNode:
    return VersionNode(
        memory_id=memory_id,
        state_key=StateKey(subject="user", attribute="location"),
        computed_status=ComputedStatus.HISTORICAL,
        valid_from="2026-01-01",
    )


def _edge(left: str, right: str) -> VersionEdge:
    return VersionEdge(
        predecessor_id=left,
        successor_id=right,
        relation=VersionRelation.SUPERSEDES,
        effective_from="2026-02-01",
        confidence=1.0,
        reason="test",
    )


def test_graph_clone_and_serialization_are_independent() -> None:
    graph = VersionGraph(nodes=[_node("a"), _node("b")], edges=[_edge("a", "b")])
    clone = graph.clone()
    changed = clone.get_node("b")
    changed.computed_status = ComputedStatus.CURRENT
    clone.update_node(changed)
    assert graph.get_node("b").computed_status is ComputedStatus.HISTORICAL
    restored = VersionGraph.model_validate(graph.model_dump())
    assert restored.model_dump() == graph.model_dump()


def test_validator_detects_cycle() -> None:
    graph = VersionGraph(nodes=[_node("a"), _node("b")])
    graph.add_edge(_edge("a", "b"))
    graph.add_edge(_edge("b", "a"))
    report = VersionGraphValidator().validate(graph)
    assert not report.is_valid
    assert any(issue.code == "GRAPH_CYCLE" for issue in report.errors)
