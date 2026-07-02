from __future__ import annotations

from statebudgetmem.interfaces import MemoryType, UpdateOperation
from statebudgetmem.preprocessing import MemoryPreprocessor, PreprocessConfig
from statebudgetmem.preprocessing.models import ParsedMemory, parse_timestamp
from statebudgetmem.preprocessing.record_adapter import (
    parsed_memories_to_records,
    parsed_memory_to_record,
)
from statebudgetmem.schemas import MemoryRecord, MemoryStatus
from statebudgetmem.versioning import StateKey, VersioningEngine
from statebudgetmem.views import DualViewMemoryMethod


def test_parsed_memory_to_record_preserves_structured_fields() -> None:
    parsed = ParsedMemory(
        content="用户当前居住地是北京",
        timestamp=parse_timestamp("2026-02-01"),
        memory_type=MemoryType.FACT,
        operation=UpdateOperation.SUPERSEDE,
        attribute="home_location",
        value="北京",
        previous_value="上海",
        evidence_span="我以前住上海，现在搬到北京了",
        tags=["profile", "location"],
        confidence=0.9,
    )

    record = parsed_memory_to_record(parsed)

    assert isinstance(record, MemoryRecord)
    assert record.subject == "user"
    assert record.attribute == "home_location"
    assert record.value == "北京"
    assert record.status == MemoryStatus.CURRENT
    assert record.memory_type == "fact"
    assert record.metadata["versioning_intent"] == "SUPERSEDE"
    assert record.metadata["preprocessing_previous_value"] == "上海"


def test_adapter_output_can_feed_versioning_engine() -> None:
    parsed_items = [
        ParsedMemory(
            content="用户当前居住地是上海",
            timestamp=parse_timestamp("2026-01-01"),
            memory_type=MemoryType.FACT,
            operation=UpdateOperation.ADD,
            attribute="home_location",
            value="上海",
        ),
        ParsedMemory(
            content="用户当前居住地是北京",
            timestamp=parse_timestamp("2026-02-01"),
            memory_type=MemoryType.FACT,
            operation=UpdateOperation.SUPERSEDE,
            attribute="home_location",
            value="北京",
            previous_value="上海",
        ),
    ]

    records = parsed_memories_to_records(parsed_items)

    engine = VersioningEngine()
    engine.ingest_many(records)

    current = engine.resolve_current(StateKey(subject="user", attribute="home_location"))
    assert [item.memory_id for item in current] == [records[1].memory_id]


def test_pipeline_can_feed_dual_view_method() -> None:
    preprocessor = MemoryPreprocessor(PreprocessConfig(parser_type="rule"))
    records = preprocessor.parse_messages_to_records(
        [
            ("user", "我现在住在上海。", "2026-01-01"),
            ("user", "我以前住上海，现在搬到北京了。", "2026-02-01"),
        ]
    )

    method = DualViewMemoryMethod()
    method.ingest(records)

    current_ids = [memory.memory_id for memory in method.manager.current_records()]
    history_ids = [memory.memory_id for memory in method.manager.history_records()]

    assert len(records) >= 2
    assert len(current_ids) == 1
    assert len(history_ids) >= 2
