from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from statebudgetmem.schemas import MemoryStatus
from statebudgetmem.versioning import UpdateOperation, VersioningEngine


_TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "versioning"
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "memora_adapter", _TOOL_DIR / "memora_adapter.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
SemanticMemoraVersioningAdapter = _MODULE.SemanticMemoraVersioningAdapter
build_full_capability_records = _MODULE.build_full_capability_records


def _record(memory_id: str, value: str, event_time: str):
    from statebudgetmem.schemas import MemoryRecord
    from datetime import date

    return MemoryRecord(
        memory_id=memory_id,
        subject="memora:test",
        attribute="todo_list",
        value=value,
        text=value,
        event_time=date.fromisoformat(event_time),
        valid_from=date.fromisoformat(event_time),
        valid_to=None,
        status=MemoryStatus.CURRENT,
        memory_type="activity_memory",
        importance=0.5,
        confidence=1.0,
        token_cost=1,
        dimensions={"item_key": "task-1"},
        metadata={"memora_operation": "update", "operation_details": {}},
    )


def test_semantic_adapter_infers_merge_for_strict_payload_extension() -> None:
    adapter = SemanticMemoraVersioningAdapter()
    base = _record("m1", '{"description":"write report"}', "2026-01-01")
    extended = _record(
        "m2",
        '{"description":"write report","status":"started"}',
        "2026-01-02",
    )
    refined = adapter.refine_records([base, extended])
    assert refined[0].metadata["versioning_intent"] == "ADD"
    assert refined[1].metadata["versioning_intent"] == "MERGE"
    assert refined[1].metadata["versioning_target_ids"] == ["m1"]


def test_controlled_suite_covers_every_canonical_operation() -> None:
    engine = VersioningEngine()
    operations = []
    for record in build_full_capability_records():
        operations.append(engine.ingest(record).results[0].decision.operation)

    assert operations == [
        UpdateOperation.ADD,
        UpdateOperation.NOOP,
        UpdateOperation.MERGE,
        UpdateOperation.SUPERSEDE,
        UpdateOperation.TEMP_INVALIDATE,
        UpdateOperation.RESTORE,
        UpdateOperation.DELETE,
    ]
    assert engine.validate().is_valid
