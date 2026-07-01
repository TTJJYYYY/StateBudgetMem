"""Integration contract tests for the pipeline."""

from datetime import date

from statebudgetmem.apps.pipeline import build_pipeline, MemoryPipeline
from statebudgetmem.schemas import MemoryRecord


def test_pipeline_general_to_none_view():
    p = build_pipeline()
    result = p.ask("光合作用的方程式是什么?")
    assert result.view_type == "none"


def test_pipeline_current_to_current_view():
    p = build_pipeline()
    mem = MemoryRecord(
        memory_id="test1", subject="user", attribute="food",
        value="spicy", text="likes spicy", event_time=date(2026, 1, 1),
        status="CURRENT", memory_type="preference",
        importance=0.5, confidence=0.5, token_cost=5,
    )
    p.ingest_scenarios([mem])
    result = p.ask("我现在喜欢吃什么？")
    assert result.view_type == "current"


def test_pipeline_result_json_serializable():
    p = build_pipeline()
    import json
    mem = MemoryRecord(
        memory_id="test1", subject="user", attribute="food",
        value="spicy", text="likes spicy", event_time=date(2026, 1, 1),
        status="CURRENT", memory_type="preference",
        importance=0.5, confidence=0.5, token_cost=5,
    )
    p.ingest_scenarios([mem])
    d = p.ask("吃什么？").to_dict()
    json.dumps(d)
