from __future__ import annotations

import pytest

from statebudgetmem.versioning import (
    ComputedStatus,
    DuplicateObservationError,
    StateKey,
    StateObservation,
    UpdateOperation,
    VersionRelation,
    VersioningEngine,
)


def _obs(
    memory_id: str,
    value: str,
    event_time: str,
    *,
    valid_to: str | None = None,
    dimensions: dict[str, str] | None = None,
    metadata: dict[str, object] | None = None,
) -> StateObservation:
    return StateObservation(
        memory_id=memory_id,
        state_key=StateKey(
            subject="user",
            attribute="exercise",
            dimensions=dimensions or {},
        ),
        value=value,
        text=value,
        event_time=event_time,
        valid_to=valid_to,
        confidence=1.0,
        metadata=metadata or {},
    )


def test_engine_add_noop_and_supersede() -> None:
    engine = VersioningEngine()
    first = engine.ingest_observation(_obs("m1", "run", "2026-01-01"))
    duplicate = engine.ingest_observation(_obs("m2", " run ", "2026-02-01"))
    changed = engine.ingest_observation(_obs("m3", "swim", "2026-03-01"))

    assert first.decision.operation is UpdateOperation.ADD
    assert duplicate.decision.operation is UpdateOperation.NOOP
    assert duplicate.skipped
    assert changed.decision.operation is UpdateOperation.SUPERSEDE
    assert changed.decision.target_memory_ids == ["m1"]
    assert engine.graph.get_node("m1").computed_status is ComputedStatus.HISTORICAL
    assert engine.graph.get_node("m3").computed_status is ComputedStatus.CURRENT
    assert any(
        edge.predecessor_id == "m1"
        and edge.successor_id == "m3"
        and edge.relation is VersionRelation.SUPERSEDES
        for edge in engine.graph.edges
    )


def test_temporary_invalidation_expires_and_base_state_resumes() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("base", "run", "2026-01-01"))
    result = engine.ingest_observation(
        _obs("injury", "cannot_run", "2026-02-01", valid_to="2026-02-10")
    )
    key = StateKey(subject="user", attribute="exercise")

    assert result.decision.operation is UpdateOperation.TEMP_INVALIDATE
    assert [item.memory_id for item in engine.resolve_at(key, "2026-02-05")] == ["injury"]
    assert [item.memory_id for item in engine.resolve_at(key, "2026-02-10")] == ["base"]


def test_restore_creates_new_version_and_closes_temporary_state() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("base", "run", "2026-01-01"))
    engine.ingest_observation(
        _obs("injury", "cannot_run", "2026-02-01", valid_to="2026-03-01")
    )
    restored = engine.ingest_observation(_obs("recovered", "run", "2026-02-15"))
    key = StateKey(subject="user", attribute="exercise")

    assert restored.decision.operation is UpdateOperation.RESTORE
    assert restored.decision.metadata["restore_source_ids"] == ["base"]
    assert restored.decision.metadata["temporary_target_ids"] == ["injury"]
    assert [item.memory_id for item in engine.resolve_at(key, "2026-02-20")] == ["recovered"]
    relations = {(edge.predecessor_id, edge.successor_id, edge.relation) for edge in engine.graph.edges}
    assert ("base", "recovered", VersionRelation.RESTORES) in relations
    assert ("injury", "recovered", VersionRelation.SUPERSEDES) in relations


def test_same_value_after_automatic_expiry_is_noop() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("base", "run", "2026-01-01"))
    engine.ingest_observation(
        _obs("pause", "cannot_run", "2026-02-01", valid_to="2026-02-10")
    )
    result = engine.ingest_observation(_obs("confirm", "run", "2026-02-20"))
    assert result.decision.operation is UpdateOperation.NOOP
    assert result.decision.target_memory_ids == ["base"]


def test_explicit_merge_replaces_exact_state() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("m1", "light", "2026-01-01"))
    result = engine.ingest_observation(
        _obs(
            "m2",
            "light_and_low_oil",
            "2026-02-01",
            metadata={"merge_request": True},
        )
    )
    assert result.decision.operation is UpdateOperation.MERGE
    assert engine.graph.get_node("m1").computed_status is ComputedStatus.HISTORICAL
    assert result.created_edges[0].relation is VersionRelation.MERGES_INTO


def test_explicit_delete_removes_current_state() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("m1", "run", "2026-01-01"))
    result = engine.ingest_observation(
        _obs("delete_event", "delete", "2026-02-01", metadata={"delete_request": True})
    )
    key = StateKey(subject="user", attribute="exercise")
    assert result.decision.operation is UpdateOperation.DELETE
    assert engine.graph.get_node("m1").computed_status is ComputedStatus.DELETED
    assert engine.resolve_current(key) == ()


def test_different_scope_is_added_without_destroying_base_scope() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("base", "run", "2026-01-01"))
    scoped = engine.ingest_observation(
        _obs(
            "rain",
            "indoor_run",
            "2026-02-01",
            dimensions={"weather": "rainy"},
        )
    )
    assert scoped.decision.operation is UpdateOperation.ADD
    assert len(engine.graph.nodes) == 2


def test_history_and_point_in_time_resolution() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("m1", "run", "2026-01-01"))
    engine.ingest_observation(_obs("m2", "swim", "2026-02-01"))
    key = StateKey(subject="user", attribute="exercise")
    assert [item.memory_id for item in engine.resolve_at(key, "2026-01-15")] == ["m1"]
    assert [item.memory_id for item in engine.resolve_at(key, "2026-02-15")] == ["m2"]
    assert [item.memory_id for item in engine.history(key)] == ["m1", "m2"]


def test_snapshot_roundtrip() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("m1", "run", "2026-01-01"))
    engine.ingest_observation(_obs("m2", "swim", "2026-02-01"))
    restored = VersioningEngine.from_snapshot(engine.snapshot())
    assert restored.snapshot() == engine.snapshot()
    assert restored.validate().is_valid


def test_ingest_is_idempotent_for_same_memory_id_and_content() -> None:
    engine = VersioningEngine()
    observation = _obs("m1", "run", "2026-01-01")
    engine.ingest_observation(observation)
    replay = engine.ingest_observation(observation)
    assert replay.decision.operation is UpdateOperation.NOOP
    assert replay.decision.metadata["idempotent_replay"] is True
    assert len(engine.graph.nodes) == 1


def test_reusing_memory_id_for_different_content_is_rejected() -> None:
    engine = VersioningEngine()
    engine.ingest_observation(_obs("m1", "run", "2026-01-01"))
    with pytest.raises(DuplicateObservationError):
        engine.ingest_observation(_obs("m1", "swim", "2026-01-01"))
