from __future__ import annotations

from datetime import date

from statebudgetmem.schemas import MemoryRecord, MemoryStatus, QueryRecord, QueryType
from statebudgetmem.versioning.operations import VersionRelation
from statebudgetmem.views import (
    CurrentOnlyMemoryMethod,
    DualViewMemoryMethod,
    HistoryOnlyMemoryMethod,
    RecordViewManager,
)


def _memory(
    memory_id: str,
    *,
    attribute: str,
    value: str,
    text: str,
    event_time: date,
    valid_from: date | None = None,
    valid_to: date | None = None,
    supersedes: list[str] | None = None,
    temporarily_invalidates: list[str] | None = None,
    token_cost: int = 10,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject="user",
        attribute=attribute,
        value=value,
        text=text,
        event_time=event_time,
        valid_from=valid_from or event_time,
        valid_to=valid_to,
        status=MemoryStatus.CURRENT,
        memory_type="test",
        importance=0.7,
        confidence=0.9,
        token_cost=token_cost,
        supersedes=supersedes or [],
        temporarily_invalidates=temporarily_invalidates or [],
    )


def _query(
    query_id: str,
    *,
    text: str,
    query_type: QueryType,
    reference_time: date,
) -> QueryRecord:
    return QueryRecord(
        query_id=query_id,
        text=text,
        query_type=query_type,
        reference_time=reference_time,
    )


def _commute_memories() -> list[MemoryRecord]:
    return [
        _memory(
            "m1",
            attribute="weekday_commute",
            value="drive",
            text="一月至三月，用户工作日开车上班。",
            event_time=date(2026, 1, 1),
            valid_to=date(2026, 4, 1),
        ),
        _memory(
            "m2",
            attribute="weekday_commute",
            value="metro",
            text="四月开始，用户工作日改为坐地铁上班。",
            event_time=date(2026, 4, 1),
            supersedes=["m1"],
        ),
    ]


def _temporary_memories() -> list[MemoryRecord]:
    return [
        _memory(
            "base",
            attribute="diet.preference",
            value="vegetarian",
            text="用户长期保持素食。",
            event_time=date(2026, 1, 1),
        ),
        _memory(
            "temporary",
            attribute="meal_permission",
            value="fast_for_exam",
            text="六月一日至六月十日检查前需要暂时空腹。",
            event_time=date(2026, 6, 1),
            valid_to=date(2026, 6, 10),
            temporarily_invalidates=["base"],
        ),
    ]


def test_record_view_manager_builds_latest_current_and_full_history() -> None:
    manager = RecordViewManager()
    manager.ingest(_commute_memories())

    assert [memory.memory_id for memory in manager.current_records()] == ["m2"]
    assert [memory.memory_id for memory in manager.history_records()] == ["m1", "m2"]


def test_general_query_returns_no_personal_memory() -> None:
    query = _query(
        "q_general",
        text="什么是公共交通？",
        query_type=QueryType.GENERAL,
        reference_time=date(2026, 6, 1),
    )
    method = DualViewMemoryMethod()
    method.ingest(_commute_memories())

    result = method.retrieve(query, top_k=3)

    assert result.retrieved_memories == []
    assert method.manager.route(query).selected_views == []


def test_current_query_uses_query_reference_time_and_restores_expired_state() -> None:
    method = CurrentOnlyMemoryMethod()
    method.ingest(_temporary_memories())

    during = method.retrieve(
        _query(
            "q_during",
            text="检查期间我现在能吃什么？",
            query_type=QueryType.CURRENT,
            reference_time=date(2026, 6, 5),
        ),
        top_k=3,
    )
    after = method.retrieve(
        _query(
            "q_after",
            text="检查结束后我现在的饮食状态是什么？",
            query_type=QueryType.CURRENT,
            reference_time=date(2026, 6, 20),
        ),
        top_k=3,
    )

    assert [item.memory_id for item in during.retrieved_memories] == ["temporary"]
    assert [item.memory_id for item in after.retrieved_memories] == ["base"]


def test_historical_query_uses_point_in_time_snapshot() -> None:
    method = HistoryOnlyMemoryMethod()
    method.ingest(_commute_memories())

    result = method.retrieve(
        _query(
            "q_history",
            text="二月份我怎么上班？",
            query_type=QueryType.HISTORICAL,
            reference_time=date(2026, 2, 15),
        ),
        top_k=3,
    )

    assert [item.memory_id for item in result.retrieved_memories] == ["m1"]


def test_change_query_keeps_current_and_full_version_history() -> None:
    method = DualViewMemoryMethod()
    method.ingest(_commute_memories())

    result = method.retrieve(
        _query(
            "q_change",
            text="我的通勤方式发生了什么变化？",
            query_type=QueryType.CHANGE,
            reference_time=date(2026, 6, 1),
        ),
        top_k=3,
    )

    assert {item.memory_id for item in result.retrieved_memories} == {"m1", "m2"}
    assert all(
        item.metadata.get("ranking_space") == "shared_tfidf_candidate_pool"
        for item in result.retrieved_memories
    )


def test_explicit_relation_fields_create_version_edges() -> None:
    manager = RecordViewManager()
    manager.ingest(_temporary_memories())

    edges = manager.version_manager.graph.edges
    assert any(
        edge.predecessor_id == "base"
        and edge.successor_id == "temporary"
        and edge.relation is VersionRelation.TEMP_INVALIDATES
        for edge in edges
    )


def test_supersedes_field_creates_version_edge() -> None:
    manager = RecordViewManager()
    manager.ingest(_commute_memories())

    assert any(
        edge.predecessor_id == "m1"
        and edge.successor_id == "m2"
        and edge.relation is VersionRelation.SUPERSEDES
        for edge in manager.version_manager.graph.edges
    )


def test_token_budget_is_never_exceeded() -> None:
    query = _query(
        "q_budget",
        text="我的通勤方式发生了什么变化？",
        query_type=QueryType.CHANGE,
        reference_time=date(2026, 6, 1),
    )
    method = DualViewMemoryMethod()
    method.ingest(_commute_memories())

    result = method.retrieve(query, top_k=3, token_budget=10)

    assert result.total_token_cost <= 10
    assert len(result.retrieved_memories) == 1
