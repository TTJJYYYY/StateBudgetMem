"""Tests for MemoryPipeline."""

from datetime import date

import pytest

from statebudgetmem.apps.pipeline import MemoryPipeline
from statebudgetmem.schemas import MemoryRecord


def _make_record(
    memory_id: str,
    subject: str = "user",
    attribute: str = "preference",
    value: str = "v",
    text: str = "some text",
    event_time: str = "2026-01-15",
    supersedes: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject=subject,
        attribute=attribute,
        value=value,
        text=text,
        event_time=date.fromisoformat(event_time),
        status="CURRENT",
        memory_type="preference",
        importance=0.5,
        confidence=0.5,
        token_cost=5,
        supersedes=supersedes or [],
    )


class TestMemoryPipeline:

    def test_ask_empty_engine(self):
        p = MemoryPipeline()
        result = p.ask("hello")
        assert result.query == "hello"
        assert result.query_type in ("CURRENT", "HISTORICAL", "CHANGE", "GENERAL")
        assert len(result.view_entries) == 0

    def test_ask_with_memories(self):
        p = MemoryPipeline()
        p.ingest_scenarios([_make_record("m1", value="spicy", text="likes spicy food")])
        result = p.ask("我现在适合吃什么？")
        assert len(result.view_entries) > 0
        assert result.query_type == "CURRENT"

    def test_ask_general_query(self):
        p = MemoryPipeline()
        p.ingest_scenarios([_make_record("m1", value="spicy")])
        result = p.ask("今天天气怎么样？")
        assert result.query_type == "GENERAL"
        assert result.view_type == "none"

    def test_pipeline_result_to_dict(self):
        p = MemoryPipeline()
        result = p.ask("test")
        d = result.to_dict()
        assert "query" in d
        assert "query_type" in d
        assert "view_type" in d
        assert "retrieved" in d

    def test_pipeline_result_summary(self):
        p = MemoryPipeline()
        result = p.ask("test")
        s = result.summary()
        assert "test" in s

    def test_ingest_scenarios_with_list(self):
        p = MemoryPipeline()
        count = p.ingest_scenarios([_make_record("m1"), _make_record("m2")])
        assert count == 2

    def test_top_k_limits_retrieval(self):
        p = MemoryPipeline(top_k=2)
        for i in range(5):
            p.ingest_scenarios([_make_record(f"m{i}", text=f"memory {i}")])
        result = p.ask("memory")
        assert len(result.retrieved) <= 2

    def test_deterministic(self):
        p = MemoryPipeline()
        p.ingest_scenarios([_make_record("m1", text="spicy food")])
        r1 = p.ask("spicy")
        r2 = p.ask("spicy")
        assert r1.to_dict() == r2.to_dict()

    def test_build_pipeline_rule_mode(self):
        from statebudgetmem.apps.pipeline import build_pipeline
        from statebudgetmem.routing import RuleBasedRouter
        p = build_pipeline()
        assert isinstance(p.router, RuleBasedRouter)
