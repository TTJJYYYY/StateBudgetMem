from __future__ import annotations

import json
import inspect
import time
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")

from statebudgetmem.baselines.memorybank.statebudgetmem_adapter import (
    StateBudgetMemDenseMethod,
    StateBudgetMemMode,
)
from statebudgetmem.core.experiment import (
    ExperimentConfig,
    MethodBuildContext,
    ResourceMetrics,
)
from statebudgetmem.core.method import MemoryMethod
from statebudgetmem.schemas.records import (
    MemoryRecord,
    MemoryStatus,
    QueryRecord,
    QueryType,
)
from statebudgetmem.schemas.results import MethodResult, RetrievedMemory
from statebudgetmem.unified_runner import _result_row, _write_jsonl


METHOD_NAMES = {
    StateBudgetMemMode.VERSIONING: "memorybank_versioning",
    StateBudgetMemMode.DUAL_VIEWS: "memorybank_dual_views",
    StateBudgetMemMode.RULE_ROUTING: "statebudgetmem_rule",
    StateBudgetMemMode.ORACLE_ROUTING: "statebudgetmem_oracle",
}


def _context() -> MethodBuildContext:
    return MethodBuildContext(
        experiment=ExperimentConfig(
            dataset_path=Path("fixture.jsonl"),
            embedding_backend="hash",
            embedding_model="deterministic_hash_embedding",
            top_k=2,
            candidate_k=3,
            reinforcement_enabled=False,
        ),
        work_dir=Path("results/test-statebudgetmem-dense"),
    )


def _method(
    mode: StateBudgetMemMode,
    *,
    router=None,
) -> StateBudgetMemDenseMethod:
    return StateBudgetMemDenseMethod(
        _context(),
        method_name=METHOD_NAMES[mode],
        mode=mode,
        router=router,
    )


def _memory(
    memory_id: str,
    *,
    value: str,
    text: str,
    event_time: date,
    valid_to: date | None = None,
    supersedes: list[str] | None = None,
    token_cost: int = 5,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject="user",
        attribute="weekday_commute",
        value=value,
        text=text,
        event_time=event_time,
        valid_from=event_time,
        valid_to=valid_to,
        status=MemoryStatus.CURRENT,
        memory_type="preference",
        importance=0.8,
        confidence=0.9,
        token_cost=token_cost,
        supersedes=supersedes or [],
    )


def _memories() -> list[MemoryRecord]:
    return [
        _memory(
            "m_drive",
            value="drive",
            text="一月至三月，用户工作日开车上班。",
            event_time=date(2026, 1, 1),
            valid_to=date(2026, 4, 1),
        ),
        _memory(
            "m_metro",
            value="metro",
            text="四月开始，用户工作日改为坐地铁上班。",
            event_time=date(2026, 4, 1),
            supersedes=["m_drive"],
        ),
    ]


def _query(
    text: str,
    query_type: QueryType,
    reference_time: date = date(2026, 6, 1),
    **gold,
) -> QueryRecord:
    return QueryRecord(
        query_id="q1",
        text=text,
        query_type=query_type,
        reference_time=reference_time,
        **gold,
    )


def _fake_result(query: QueryRecord, memory_id: str = "m_drive") -> MethodResult:
    return MethodResult(
        method_name="memorybank_core",
        query_id=query.query_id,
        retrieved_memories=[
            RetrievedMemory(
                memory_id=memory_id,
                score=0.75,
                rank=1,
                token_cost=5,
                source_view="flat",
                metadata={"core": "kept"},
            )
        ],
        total_token_cost=5,
        latency_ms=0.1,
        metadata={"candidate_k": 3, "base": "kept"},
    )


def _install_fake_dense(monkeypatch, method, *, sleep_seconds: float = 0.0):
    calls = []

    def fake(query, *, allowed_memory_ids, top_k, token_budget, mutate):
        calls.append(
            {
                "query": query,
                "allowed_memory_ids": allowed_memory_ids,
                "top_k": top_k,
                "token_budget": token_budget,
                "mutate": mutate,
            }
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)
        return _fake_result(query, sorted(allowed_memory_ids)[0])

    monkeypatch.setattr(method, "_retrieve_from_memorybank", fake)
    return calls


def _unified_result_row(
    query: QueryRecord, result: MethodResult
) -> dict[str, object]:
    return _result_row(
        "test-run",
        "test-scenario",
        query,
        result,
        _context().experiment,
        ResourceMetrics(
            retrieval_latency_ms=result.latency_ms,
            total_token_cost=result.total_token_cost,
        ),
    )


def test_method_name_must_match_mode() -> None:
    with pytest.raises(ValueError, match="does not match mode"):
        StateBudgetMemDenseMethod(
            _context(),
            method_name="statebudgetmem_oracle",
            mode=StateBudgetMemMode.VERSIONING,
        )


def test_public_package_exports_dense_adapter_types() -> None:
    from statebudgetmem.baselines.memorybank import (
        MemoryBankMethod as ExportedMemoryBankMethod,
        StateBudgetMemDenseMethod as ExportedMethod,
        StateBudgetMemMode as ExportedMode,
    )
    from statebudgetmem.baselines.memorybank.adapter import MemoryBankMethod

    assert ExportedMemoryBankMethod is MemoryBankMethod
    assert ExportedMethod is StateBudgetMemDenseMethod
    assert ExportedMode is StateBudgetMemMode


def test_public_method_signatures_match_frozen_protocol() -> None:
    for method_name in ("reset", "ingest", "retrieve"):
        assert inspect.signature(
            getattr(StateBudgetMemDenseMethod, method_name)
        ) == inspect.signature(getattr(MemoryMethod, method_name))


def test_reset_clears_bank_views_and_reuses_embedding() -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING)
    encoder = method.embedding_model
    method.ingest(_memories())
    method._resolve_eligibility(_query("我现在怎么上班？", QueryType.CURRENT))

    method.reset()

    assert method.embedding_model is encoder
    assert method.bank.memories == []
    assert method.bank.memories_by_id == {}
    assert method.bank.index is None
    assert method.bank.faiss_id_to_mid == {}
    assert method.bank.next_faiss_id == 0
    assert method.bank.access_count == 0
    assert method.view_manager.memories == ()
    assert method.view_manager.memories_by_id == {}
    assert method.view_manager.version_manager.graph.nodes == ()
    assert method.last_eligibility_decision is None


def test_ingest_preserves_and_aligns_all_memory_ids() -> None:
    method = _method(StateBudgetMemMode.VERSIONING)
    records = _memories()
    original_ids = [record.memory_id for record in records]

    method.ingest(records)

    assert [record.memory_id for record in records] == original_ids
    assert set(original_ids) == set(method.bank.memories_by_id)
    assert set(original_ids) == set(method.view_manager.memories_by_id)


def test_duplicate_ids_fail_before_any_write_and_leave_no_partial_state() -> None:
    method = _method(StateBudgetMemMode.VERSIONING)
    method.ingest(_memories())
    method._resolve_eligibility(_query("current commute", QueryType.CURRENT))
    duplicate = _memory(
        "duplicate",
        value="walk",
        text="用户步行上班。",
        event_time=date(2026, 1, 1),
    )

    with pytest.raises(ValueError, match="duplicate.*duplicate"):
        method.ingest([duplicate, duplicate.model_copy()])

    assert method.bank.memories_by_id == {}
    assert method.view_manager.memories_by_id == {}
    assert method._input_memory_ids == frozenset()
    assert method.last_eligibility_decision is None

    method.ingest(_memories())
    assert set(method.bank.memories_by_id) == {"m_drive", "m_metro"}


def test_view_ingest_failure_rolls_back_all_state_and_preserves_exception(
    monkeypatch,
) -> None:
    method = _method(StateBudgetMemMode.VERSIONING)
    method.ingest(_memories())
    method._resolve_eligibility(_query("current commute", QueryType.CURRENT))
    original_ingest = method.view_manager.ingest
    original_error = RuntimeError("injected view ingest failure")

    def fail_view_ingest(_records) -> None:
        raise original_error

    monkeypatch.setattr(method.view_manager, "ingest", fail_view_ingest)

    with pytest.raises(RuntimeError, match="injected view ingest failure") as caught:
        method.ingest(_memories())

    assert caught.value is original_error
    assert method.bank.memories_by_id == {}
    assert method.bank.index is None
    assert method.view_manager.memories_by_id == {}
    assert method.view_manager.version_manager.graph.nodes == ()
    assert method._input_memory_ids == frozenset()
    assert method.last_eligibility_decision is None

    monkeypatch.setattr(method.view_manager, "ingest", original_ingest)
    method.ingest(_memories())
    assert set(method.bank.memories_by_id) == {"m_drive", "m_metro"}


class _ExplodingRouter:
    def classify(self, _query):
        raise AssertionError("router must not be called")


@pytest.mark.parametrize(
    "mode",
    [
        StateBudgetMemMode.VERSIONING,
        StateBudgetMemMode.DUAL_VIEWS,
        StateBudgetMemMode.ORACLE_ROUTING,
    ],
)
def test_non_rule_modes_do_not_call_rule_router(mode) -> None:
    method = _method(mode, router=_ExplodingRouter())
    method.ingest(_memories())
    method._resolve_eligibility(_query("我现在怎么上班？", QueryType.CURRENT))


@pytest.mark.parametrize(
    "query_type",
    [
        QueryType.CURRENT,
        QueryType.HISTORICAL,
        QueryType.CHANGE,
        QueryType.GENERAL,
    ],
)
def test_versioning_eligibility_ignores_query_type(query_type) -> None:
    method = _method(StateBudgetMemMode.VERSIONING, router=_ExplodingRouter())
    method.ingest(_memories())

    decision = method._resolve_eligibility(
        _query("same query", query_type, reference_time=date(2026, 6, 1))
    )

    assert decision.eligible_memory_ids == frozenset({"m_metro"})
    assert decision.effective_query_type is QueryType.CURRENT
    assert decision.predicted_query_type is None
    assert decision.source_view == "current"
    assert decision.router_source == "none"
    assert decision.selection_policy == "current_only_no_router"


def test_versioning_uses_current_snapshot_for_every_personal_query(monkeypatch) -> None:
    method = _method(StateBudgetMemMode.VERSIONING)
    method.ingest(_memories())
    query = _query(
        "以前怎么上班？",
        QueryType.HISTORICAL,
        reference_time=date(2026, 2, 1),
    )
    calls = _install_fake_dense(monkeypatch, method)

    result = method.retrieve(query, top_k=2, token_budget=9, mutate=True)

    assert calls[0]["allowed_memory_ids"] == {"m_drive"}
    assert calls[0]["top_k"] == 2
    assert calls[0]["token_budget"] == 9
    assert calls[0]["mutate"] is True
    assert result.predicted_query_type is None
    assert result.metadata["effective_query_type"] == "CURRENT"
    assert result.metadata["router_source"] == "none"
    assert result.metadata["source_view"] == "current"
    assert result.metadata["selection_policy"] == "current_only_no_router"


@pytest.mark.parametrize(
    "query_type",
    [
        QueryType.CURRENT,
        QueryType.HISTORICAL,
        QueryType.CHANGE,
        QueryType.GENERAL,
    ],
)
def test_dual_views_eligibility_ignores_query_type(query_type) -> None:
    method = _method(StateBudgetMemMode.DUAL_VIEWS, router=_ExplodingRouter())
    method.ingest(_memories())

    decision = method._resolve_eligibility(
        _query("same query", query_type, reference_time=date(2026, 6, 1))
    )

    assert decision.eligible_memory_ids == frozenset({"m_drive", "m_metro"})
    assert decision.effective_query_type is QueryType.CHANGE
    assert decision.predicted_query_type is None
    assert decision.source_view == "current_and_history"
    assert decision.router_source == "none"
    assert decision.selection_policy == "current_and_history_no_router"


def test_dual_views_uses_current_and_full_history_without_gold(monkeypatch) -> None:
    method = _method(StateBudgetMemMode.DUAL_VIEWS)
    method.ingest(_memories())
    query = _query(
        "我现在怎么上班？",
        QueryType.CURRENT,
        gold_relevant_memory_ids=["wrong"],
        gold_valid_memory_ids=["wrong"],
        gold_stale_memory_ids=["wrong"],
    )
    calls = _install_fake_dense(monkeypatch, method)

    result = method.retrieve(query, top_k=2)

    assert calls[0]["allowed_memory_ids"] == {"m_drive", "m_metro"}
    assert result.predicted_query_type is None
    assert result.metadata["effective_query_type"] == "CHANGE"
    assert result.metadata["router_source"] == "none"
    assert result.metadata["source_view"] == "current_and_history"
    assert (
        result.metadata["selection_policy"] == "current_and_history_no_router"
    )


def test_eligibility_and_dual_view_order_are_deterministic() -> None:
    method = _method(StateBudgetMemMode.DUAL_VIEWS)
    method.ingest(list(reversed(_memories())))
    query = _query("current commute", QueryType.CURRENT)

    first = method._resolve_eligibility(query)
    first_order = [
        record.memory_id for record in method._current_and_history_records(query)
    ]
    second = method._resolve_eligibility(query)
    second_order = [
        record.memory_id for record in method._current_and_history_records(query)
    ]

    assert first == second
    assert first.eligible_memory_ids == frozenset({"m_drive", "m_metro"})
    assert first_order == second_order == ["m_drive", "m_metro"]
    assert first.selection_policy == second.selection_policy
    assert first.selection_policy == "current_and_history_no_router"
    assert dict(first.metadata) == dict(second.metadata) == {"mode": "dual_views"}


@pytest.mark.parametrize(
    "text, annotated_type, reference_time, expected_type, expected_ids, source_view",
    [
        (
            "我现在怎么上班？",
            QueryType.HISTORICAL,
            date(2026, 6, 1),
            QueryType.CURRENT,
            {"m_metro"},
            "current",
        ),
        (
            "我以前怎么上班？",
            QueryType.CURRENT,
            date(2026, 2, 1),
            QueryType.HISTORICAL,
            {"m_drive"},
            "history",
        ),
        (
            "我的通勤方式发生了什么变化？",
            QueryType.CURRENT,
            date(2026, 6, 1),
            QueryType.CHANGE,
            {"m_drive", "m_metro"},
            "current_and_history",
        ),
    ],
)
def test_rule_routing_selects_view_from_text_not_annotation(
    monkeypatch,
    text,
    annotated_type,
    reference_time,
    expected_type,
    expected_ids,
    source_view,
) -> None:
    method = _method(StateBudgetMemMode.RULE_ROUTING)
    method.ingest(_memories())
    query = _query(text, annotated_type, reference_time)
    calls = _install_fake_dense(monkeypatch, method)

    result = method.retrieve(query, top_k=3)

    assert calls[0]["allowed_memory_ids"] == expected_ids
    assert result.predicted_query_type is expected_type
    assert result.metadata["router_source"] == "rule"
    assert result.metadata["source_view"] == source_view
    assert result.metadata["selection_policy"] == "rule_routed"


def test_rule_general_skips_dense_retrieval(monkeypatch) -> None:
    method = _method(StateBudgetMemMode.RULE_ROUTING)
    method.ingest(_memories())
    called = False

    def fail(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("dense retrieval must be skipped")

    monkeypatch.setattr(method, "_retrieve_from_memorybank", fail)
    result = method.retrieve(
        _query("法国的首都是什么？", QueryType.CURRENT),
        top_k=2,
    )

    assert called is False
    assert result.retrieved_memories == []
    assert result.predicted_query_type is QueryType.GENERAL
    assert result.total_token_cost == 0
    assert result.metadata["source_view"] == "none"
    assert result.metadata["skipped_dense_retrieval"] is True
    assert result.metadata["selection_policy"] == "rule_routed"


@pytest.mark.parametrize(
    "query_type, reference_time, expected_ids",
    [
        (QueryType.CURRENT, date(2026, 6, 1), {"m_metro"}),
        (QueryType.HISTORICAL, date(2026, 2, 1), {"m_drive"}),
        (QueryType.CHANGE, date(2026, 6, 1), {"m_drive", "m_metro"}),
    ],
)
def test_oracle_uses_only_query_type_and_not_gold(
    monkeypatch, query_type, reference_time, expected_ids
) -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING, router=_ExplodingRouter())
    method.ingest(_memories())
    query = _query(
        "annotation controls this query",
        query_type,
        reference_time,
        gold_relevant_memory_ids=["wrong"],
        gold_valid_memory_ids=["wrong"],
        gold_stale_memory_ids=["wrong"],
    )
    calls = _install_fake_dense(monkeypatch, method)

    result = method.retrieve(query, top_k=2)

    assert calls[0]["allowed_memory_ids"] == expected_ids
    assert result.predicted_query_type is query_type
    assert result.metadata["router_source"] == "oracle_query_type"
    assert result.metadata["selection_policy"] == "oracle_routed"


def test_oracle_eligibility_ignores_all_gold_memory_ids() -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING, router=_ExplodingRouter())
    method.ingest(_memories())
    plain = _query("same query", QueryType.CHANGE)
    annotated = _query(
        "same query",
        QueryType.CHANGE,
        gold_relevant_memory_ids=["not-relevant"],
        gold_valid_memory_ids=["not-valid"],
        gold_stale_memory_ids=["not-stale"],
    )

    plain_decision = method._resolve_eligibility(plain)
    annotated_decision = method._resolve_eligibility(annotated)

    assert plain_decision.eligible_memory_ids == annotated_decision.eligible_memory_ids
    assert plain_decision.predicted_query_type is QueryType.CHANGE
    assert annotated_decision.predicted_query_type is QueryType.CHANGE
    assert annotated_decision.selection_policy == "oracle_routed"


def test_oracle_general_returns_valid_empty_result_without_dense(monkeypatch) -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING)
    method.ingest(_memories())
    monkeypatch.setattr(
        method,
        "_retrieve_from_memorybank",
        lambda *_args, **_kwargs: pytest.fail("dense retrieval must be skipped"),
    )

    result = method.retrieve(_query("anything", QueryType.GENERAL), top_k=1)

    assert result.method_name == "statebudgetmem_oracle"
    assert result.query_id == "q1"
    assert result.retrieved_memories == []
    assert result.predicted_query_type is QueryType.GENERAL
    assert result.total_token_cost == 0
    assert result.latency_ms >= 0
    assert result.metadata["eligible_memory_count"] == 0
    assert result.metadata["selection_policy"] == "oracle_routed"


def test_missing_scoped_interface_raises_for_nonempty_eligibility() -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING)
    method.ingest(_memories())

    with pytest.raises(
        RuntimeError, match="scoped MemoryBank retrieval interface is not available"
    ):
        method.retrieve(_query("anything", QueryType.CURRENT), top_k=1)


@pytest.mark.parametrize("mutate", [False, True])
def test_dense_connection_receives_mutation_and_budget(monkeypatch, mutate) -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING)
    method.ingest(_memories())
    calls = _install_fake_dense(monkeypatch, method)

    method.retrieve(
        _query("anything", QueryType.CURRENT),
        top_k=7,
        token_budget=13,
        mutate=mutate,
    )

    assert calls[0]["top_k"] == 7
    assert calls[0]["token_budget"] == 13
    assert calls[0]["mutate"] is mutate


def test_result_preserves_base_items_scores_ranks_tokens_and_metadata(monkeypatch) -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING)
    method.ingest(_memories())
    calls = _install_fake_dense(monkeypatch, method)

    result = method.retrieve(_query("anything", QueryType.CURRENT), top_k=1)

    assert calls
    assert result.retrieved_memories[0].score == 0.75
    assert result.retrieved_memories[0].rank == 1
    assert result.retrieved_memories[0].token_cost == 5
    assert result.retrieved_memories[0].metadata == {"core": "kept"}
    assert result.total_token_cost == 5
    assert result.metadata["base"] == "kept"
    assert result.metadata["candidate_k"] == 3
    assert {
        "statebudgetmem_mode",
        "effective_query_type",
        "router_source",
        "source_view",
        "eligible_memory_count",
        "base_method_name",
        "selection_policy",
    } <= set(result.metadata)


@pytest.mark.parametrize(
    "mode, text, annotated_type, expected_prediction",
    [
        (StateBudgetMemMode.VERSIONING, "anything", QueryType.GENERAL, None),
        (StateBudgetMemMode.DUAL_VIEWS, "anything", QueryType.GENERAL, None),
        (
            StateBudgetMemMode.RULE_ROUTING,
            "我现在怎么上班？",
            QueryType.HISTORICAL,
            QueryType.CURRENT,
        ),
        (
            StateBudgetMemMode.ORACLE_ROUTING,
            "anything",
            QueryType.HISTORICAL,
            QueryType.HISTORICAL,
        ),
    ],
)
def test_predicted_query_type_reflects_only_actual_routing(
    monkeypatch, mode, text, annotated_type, expected_prediction
) -> None:
    method = _method(mode)
    method.ingest(_memories())
    _install_fake_dense(monkeypatch, method)

    result = method.retrieve(_query(text, annotated_type), top_k=1)

    assert result.predicted_query_type is expected_prediction


def test_unified_runner_serializes_empty_and_fake_dense_results(
    tmp_path, monkeypatch
) -> None:
    versioning = _method(StateBudgetMemMode.VERSIONING)
    versioning.ingest(_memories())
    _install_fake_dense(monkeypatch, versioning)
    versioning_query = _query("same query", QueryType.GENERAL)
    versioning_result = versioning.retrieve(versioning_query, top_k=1)

    dual_views = _method(StateBudgetMemMode.DUAL_VIEWS)
    dual_views.ingest(_memories())
    _install_fake_dense(monkeypatch, dual_views)
    dual_query = _query("same query", QueryType.GENERAL)
    dual_result = dual_views.retrieve(dual_query, top_k=1)

    rule = _method(StateBudgetMemMode.RULE_ROUTING)
    rule.ingest(_memories())
    rule_query = _query("法国的首都是什么？", QueryType.CURRENT)
    rule_result = rule.retrieve(rule_query, top_k=1)

    oracle = _method(StateBudgetMemMode.ORACLE_ROUTING)
    oracle.ingest(_memories())
    oracle_query = _query("general question", QueryType.GENERAL)
    oracle_result = oracle.retrieve(oracle_query, top_k=1)

    output_path = tmp_path / "raw.jsonl"
    _write_jsonl(
        output_path,
        [
            _unified_result_row(versioning_query, versioning_result),
            _unified_result_row(dual_query, dual_result),
            _unified_result_row(rule_query, rule_result),
            _unified_result_row(oracle_query, oracle_result),
        ],
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 4
    for row in rows:
        metadata = row["method_metadata"]
        assert isinstance(metadata, dict)
        assert isinstance(metadata["statebudgetmem_mode"], str)
        assert isinstance(metadata["effective_query_type"], str)
        assert isinstance(metadata["selection_policy"], str)
        assert not any("gold" in key for key in metadata)
    assert rows[0]["retrieved_memory_ids"] == ["m_metro"]
    assert rows[0]["predicted_query_type"] is None
    assert rows[1]["retrieved_memory_ids"] == ["m_drive"]
    assert rows[1]["predicted_query_type"] is None
    assert rows[2]["retrieved_memory_ids"] == []
    assert rows[2]["predicted_query_type"] == "GENERAL"
    assert rows[3]["retrieved_memory_ids"] == []
    assert rows[3]["predicted_query_type"] == "GENERAL"


def test_adapter_latency_covers_dense_call(monkeypatch) -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING)
    method.ingest(_memories())
    _install_fake_dense(monkeypatch, method, sleep_seconds=0.01)

    result = method.retrieve(_query("anything", QueryType.CURRENT), top_k=1)

    assert result.latency_ms >= 8.0


def test_retrieve_does_not_modify_original_query(monkeypatch) -> None:
    method = _method(StateBudgetMemMode.RULE_ROUTING)
    method.ingest(_memories())
    query = _query("我以前怎么上班？", QueryType.CURRENT, date(2026, 2, 1))
    before = query.model_dump(mode="json")
    _install_fake_dense(monkeypatch, method)

    method.retrieve(query, top_k=1)

    assert query.model_dump(mode="json") == before


def test_adapter_instances_do_not_share_mutable_memorybank_state() -> None:
    first = _method(StateBudgetMemMode.VERSIONING)
    second = _method(StateBudgetMemMode.VERSIONING)
    first.ingest(_memories())

    assert first.base_method is not second.base_method
    assert first.bank is not second.bank
    assert first.view_manager is not second.view_manager
    assert set(first.bank.memories_by_id) == {"m_drive", "m_metro"}
    assert second.bank.memories_by_id == {}


@pytest.mark.parametrize(
    "top_k, token_budget, match",
    [(0, None, "top_k"), (-1, None, "top_k"), (1, -1, "token_budget")],
)
def test_retrieve_validates_limits(top_k, token_budget, match) -> None:
    method = _method(StateBudgetMemMode.ORACLE_ROUTING)
    with pytest.raises(ValueError, match=match):
        method.retrieve(
            _query("anything", QueryType.GENERAL),
            top_k=top_k,
            token_budget=token_budget,
        )
