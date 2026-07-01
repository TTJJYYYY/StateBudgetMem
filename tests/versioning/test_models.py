from __future__ import annotations

import pytest
from pydantic import ValidationError

from statebudgetmem.versioning import (
    ComputedStatus,
    StateKey,
    UpdateDecision,
    UpdateOperation,
    VersionEdge,
    VersionNode,
    VersionRelation,
)


def test_update_operations_exist() -> None:
    operation_values = {operation.value for operation in UpdateOperation}
    assert len(operation_values) == 7
    assert operation_values == {
        "ADD",
        "MERGE",
        "SUPERSEDE",
        "TEMP_INVALIDATE",
        "RESTORE",
        "DELETE",
        "NOOP",
    }


def test_version_relations_exist() -> None:
    relation_values = {relation.value for relation in VersionRelation}
    assert "RESTORES" in relation_values
    assert relation_values == {
        "SUPERSEDES",
        "TEMP_INVALIDATES",
        "MERGES_INTO",
        "RESTORES",
        "DELETES",
    }


def test_restore_operation_and_restores_relation_are_distinct() -> None:
    decision_payload = _decision_payload()
    decision_payload["operation"] = "RESTORE"
    decision_payload["reason"] = "recover long-term state after temporary invalidation"
    decision = UpdateDecision.model_validate(decision_payload)
    edge = VersionEdge(
        predecessor_id="long-term-state",
        successor_id="recovered-state",
        relation=VersionRelation.RESTORES,
        effective_from="2026-06-30",
        confidence=0.8,
        reason="new version restores from old long-term state",
        metadata={},
    )

    assert decision.operation is UpdateOperation.RESTORE
    assert edge.relation is VersionRelation.RESTORES
    assert decision.operation.value != edge.relation.value


def test_state_key_distinguishes_subject_attribute_and_condition() -> None:
    base = StateKey(subject="user", attribute="diet")
    same = StateKey(subject="user", attribute="diet")
    different_subject = StateKey(subject="assistant", attribute="diet")
    different_attribute = StateKey(subject="user", attribute="location")
    different_condition = StateKey(subject="user", attribute="diet", condition="medical")

    assert base == same
    assert len({base, same, different_subject, different_attribute, different_condition}) == 4
    assert str(base) == "user:diet"
    assert str(different_condition) == "user:diet[medical]"


def test_update_decision_confidence_must_be_between_zero_and_one() -> None:
    payload = _decision_payload()
    payload["confidence"] = 1.01

    with pytest.raises(ValidationError):
        UpdateDecision.model_validate(payload)


def test_version_node_rejects_valid_to_before_valid_from() -> None:
    with pytest.raises(ValidationError, match="valid_to"):
        VersionNode(
            memory_id="M1",
            state_key=StateKey(subject="user", attribute="diet"),
            computed_status=ComputedStatus.CURRENT,
            valid_from="2026-06-30",
            valid_to="2026-06-29",
            invalidated_by=[],
            metadata={},
        )


def test_version_edge_rejects_self_edge() -> None:
    with pytest.raises(ValidationError, match="predecessor_id"):
        VersionEdge(
            predecessor_id="M1",
            successor_id="M1",
            relation=VersionRelation.SUPERSEDES,
            effective_from="2026-06-30",
            confidence=0.8,
            reason="same endpoint is invalid",
            metadata={},
        )


def test_version_edge_uses_old_to_new_field_names() -> None:
    edge = VersionEdge(
        predecessor_id="old-memory",
        successor_id="new-memory",
        relation=VersionRelation.SUPERSEDES,
        effective_from="2026-06-30",
        confidence=0.9,
        reason="new memory supersedes old memory",
        metadata={},
    )

    assert edge.predecessor_id == "old-memory"
    assert edge.successor_id == "new-memory"


def test_models_serialize_and_deserialize() -> None:
    state_key = StateKey(subject="user", attribute="diet", condition="health")
    decision = UpdateDecision.model_validate(_decision_payload())
    node = VersionNode(
        memory_id="M1",
        state_key=state_key,
        computed_status=ComputedStatus.TEMP_INVALIDATED,
        valid_from="2026-06-01",
        valid_to="2026-07-01",
        invalidated_by=["M2"],
        metadata={"source": "test"},
    )
    edge = VersionEdge(
        predecessor_id="M1",
        successor_id="M2",
        relation=VersionRelation.TEMP_INVALIDATES,
        effective_from="2026-06-30",
        confidence=0.75,
        reason="temporary medical restriction",
        metadata={"source": "test"},
    )

    assert StateKey.model_validate_json(state_key.model_dump_json()) == state_key
    assert UpdateDecision.model_validate_json(decision.model_dump_json()) == decision
    assert VersionNode.model_validate_json(node.model_dump_json()) == node
    assert VersionEdge.model_validate_json(edge.model_dump_json()) == edge


def _decision_payload() -> dict[str, object]:
    return {
        "new_memory_id": "M2",
        "operation": "SUPERSEDE",
        "target_memory_ids": ["M1"],
        "state_key": {
            "subject": "user",
            "attribute": "diet",
            "condition": None,
        },
        "confidence": 0.8,
        "reason": "newer preference update",
        "evidence": ["M1", "M2"],
        "requires_review": False,
        "metadata": {},
    }
