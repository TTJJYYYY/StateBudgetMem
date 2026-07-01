"""Tests for MemoryViewManager."""

from datetime import date

import pytest

from statebudgetmem.core.online import ViewType
from statebudgetmem.schemas import MemoryRecord
from statebudgetmem.versioning.engine import VersioningEngine
from statebudgetmem.views.manager import MemoryViewManager


def _make_record(
    memory_id: str,
    subject: str = "user",
    attribute: str = "preference",
    value: str = "like_spicy",
    text: str = "likes spicy food",
    event_time: str = "2026-01-15",
    valid_from: str | None = None,
    valid_to: str | None = None,
    supersedes: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject=subject,
        attribute=attribute,
        value=value,
        text=text,
        event_time=date.fromisoformat(event_time),
        valid_from=date.fromisoformat(valid_from) if valid_from else None,
        valid_to=date.fromisoformat(valid_to) if valid_to else None,
        status="CURRENT",
        memory_type="preference",
        importance=0.8,
        confidence=0.9,
        token_cost=10,
        supersedes=supersedes or [],
    )


class TestMemoryViewManager:

    def test_empty_engine_returns_empty_current_view(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        assert vm.current_entries() == []

    def test_single_memory_appears_in_current(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1", value="like_spicy"))
        entries = vm.current_entries()
        assert len(entries) == 1
        assert entries[0].value == "like_spicy"

    def test_superseded_memory_not_in_current(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1", value="like_spicy", event_time="2026-01-15"))
        engine.ingest(_make_record("m2", value="like_light",
            text="prefers light food", event_time="2026-06-15", supersedes=["m1"]))
        values = {e.value for e in vm.current_entries()}
        assert "like_light" in values
        assert "like_spicy" not in values

    def test_superseded_memory_appears_in_all_history(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1", value="like_spicy", event_time="2026-01-15"))
        engine.ingest(_make_record("m2", value="like_light",
            text="prefers light food", event_time="2026-06-15", supersedes=["m1"]))
        values = {e.value for e in vm.all_history_entries()}
        assert "like_spicy" in values
        assert "like_light" in values

    def test_select_view_current(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1"))
        assert len(vm.select_view(ViewType.CURRENT)) == 1

    def test_select_view_none(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1"))
        assert vm.select_view(ViewType.NONE) == []

    def test_select_view_both(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1", value="old", event_time="2026-01-01"))
        engine.ingest(_make_record("m2", value="new", event_time="2026-06-01", supersedes=["m1"]))
        values = {e.value for e in vm.select_view(ViewType.BOTH)}
        assert "new" in values
        assert "old" in values

    def test_get_current_view_abc(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1"))
        result = vm.get_current_view()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_history_view_abc(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1"))
        result = vm.get_history_view()
        assert len(result) == 1

    def test_to_texts(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1", value="like_spicy", text="likes spicy food"))
        texts = vm.to_texts(vm.current_entries())
        assert len(texts) == 1
        assert "like_spicy" in texts[0]

    def test_deterministic(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1", value="a"))
        engine.ingest(_make_record("m2", value="b"))
        ids1 = [e.memory_id for e in vm.current_entries()]
        ids2 = [e.memory_id for e in vm.current_entries()]
        assert ids1 == ids2

    def test_lineage_entries(self):
        engine = VersioningEngine()
        vm = MemoryViewManager(engine)
        engine.ingest(_make_record("m1", value="old", event_time="2026-01-01"))
        engine.ingest(_make_record("m2", value="new", event_time="2026-06-01", supersedes=["m1"]))
        ids = {e.memory_id for e in vm.lineage_entries("m1")}
        assert "m1" in ids
        assert "m2" in ids
