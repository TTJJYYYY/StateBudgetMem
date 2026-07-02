from __future__ import annotations

from datetime import date

from statebudgetmem.schemas import MemoryRecord, MemoryStatus, QueryRecord, QueryType
from statebudgetmem.views import CurrentOnlyMemoryMethod, DualViewMemoryMethod, RecordViewManager


def _memories() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            memory_id="m1",
            subject="user",
            attribute="weekday_commute",
            value="drive",
            text="一月至三月，用户工作日开车上班。",
            event_time=date(2026, 1, 1),
            valid_from=date(2026, 1, 1),
            valid_to=date(2026, 4, 1),
            status=MemoryStatus.HISTORICAL,
            memory_type="habit",
            importance=0.7,
            confidence=0.9,
            token_cost=10,
        ),
        MemoryRecord(
            memory_id="m2",
            subject="user",
            attribute="weekday_commute",
            value="metro",
            text="四月开始，用户工作日改为坐地铁上班。",
            event_time=date(2026, 4, 1),
            valid_from=date(2026, 4, 1),
            valid_to=None,
            status=MemoryStatus.CURRENT,
            memory_type="habit",
            importance=0.7,
            confidence=0.9,
            token_cost=10,
        ),
    ]


def test_record_view_manager_builds_current_and_history_views() -> None:
    manager = RecordViewManager()
    manager.ingest(_memories())

    assert [memory.memory_id for memory in manager.current_records()] == ["m2"]
    assert [memory.memory_id for memory in manager.history_records()] == ["m1", "m2"]


def test_current_only_method_filters_stale_versions() -> None:
    query = QueryRecord(
        query_id="q_current",
        text="我现在工作日怎么上班？",
        query_type=QueryType.CURRENT,
        reference_time=date(2026, 6, 1),
        gold_relevant_memory_ids=["m1", "m2"],
        gold_valid_memory_ids=["m2"],
        gold_stale_memory_ids=["m1"],
    )

    method = CurrentOnlyMemoryMethod()
    method.ingest(_memories())
    result = method.retrieve(query, top_k=2)

    assert [item.memory_id for item in result.retrieved_memories] == ["m2"]
    assert result.retrieved_memories[0].source_view == "current"


def test_dual_view_keeps_history_for_change_query() -> None:
    query = QueryRecord(
        query_id="q_change",
        text="我的工作日通勤方式发生了什么变化？",
        query_type=QueryType.CHANGE,
        reference_time=date(2026, 6, 1),
        gold_relevant_memory_ids=["m1", "m2"],
        gold_valid_memory_ids=["m1", "m2"],
        gold_stale_memory_ids=[],
    )

    method = DualViewMemoryMethod()
    method.ingest(_memories())
    result = method.retrieve(query, top_k=2)

    assert {item.memory_id for item in result.retrieved_memories} == {"m1", "m2"}
    assert {item.source_view for item in result.retrieved_memories} <= {"current", "history"}
