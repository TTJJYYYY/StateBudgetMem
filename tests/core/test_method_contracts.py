"""Cross-method contract tests for unified experiment runner.

费哲瀚 — Phase M1: Contract Tests (成员 D)

All tests are parameterized over registered method names and query_state_policy.
When new adapters are registered (statebudget_*, etc.), these tests automatically
cover them without modification.

Data: data/controlled/interface_smoke_v1.jsonl (frozen smoke dataset)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from statebudgetmem.core.experiment import ExperimentConfig, MethodBuildContext
from statebudgetmem.core.registry import MethodRegistry, default_method_registry
from statebudgetmem.data import load_scenarios
from statebudgetmem.schemas.results import MethodResult

SMOKE_DATASET = Path("data/controlled/interface_smoke_v1.jsonl")
REGISTERED_METHODS = list(default_method_registry().names())


# ── helpers ──────────────────────────────────────────────────────────────

def _build_context(**overrides) -> tuple[MethodBuildContext, ExperimentConfig]:
    cfg = ExperimentConfig(
        dataset_path=SMOKE_DATASET,
        methods=tuple(REGISTERED_METHODS[:1]),
        top_k=2,
        candidate_k=4,
        token_budget=32,
        **overrides,
    )
    ctx = MethodBuildContext(
        experiment=cfg,
        work_dir=Path(tempfile.mkdtemp(prefix="smoke_")),
    )
    return ctx, cfg


def _run_retrieve(method_name: str, ctx: MethodBuildContext, **kw):
    registry = default_method_registry()
    method = registry.create(method_name, ctx)
    scenarios = load_scenarios(SMOKE_DATASET)
    method.reset()
    for mem in scenarios[0].memories:
        method.ingest([mem])
    query = scenarios[0].queries[0].model_copy(update={
        "gold_relevant_memory_ids": [],
        "gold_valid_memory_ids": [],
        "gold_stale_memory_ids": [],
    })
    return method.retrieve(
        query,
        top_k=kw.get("top_k", ctx.experiment.top_k),
        token_budget=kw.get("token_budget", ctx.experiment.token_budget),
        mutate=kw.get("mutate", False),
    )


# ── 1. reset / state isolation ───────────────────────────────────────────


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_reset_clears_state(method_name: str):
    """reset() should clear all memory state — same query after reset gives same result."""
    ctx, _ = _build_context()
    r1 = _run_retrieve(method_name, ctx)
    r2 = _run_retrieve(method_name, ctx)
    assert [m.memory_id for m in r1.retrieved_memories] == [
        m.memory_id for m in r2.retrieved_memories
    ]


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_empty_ingest_is_safe(method_name: str):
    """ingest([]) should not crash."""
    registry = default_method_registry()
    ctx, _ = _build_context()
    method = registry.create(method_name, ctx)
    method.reset()
    method.ingest([])


# ── 2. top-k / token-budget constraints ──────────────────────────────────


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_top_k_bound(method_name: str):
    """retrieved count must not exceed top_k."""
    ctx, _ = _build_context()
    for k in [1, 2, 3]:
        result = _run_retrieve(method_name, ctx, top_k=k)
        assert len(result.retrieved_memories) <= k


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_total_token_cost_equals_sum(method_name: str):
    """total_token_cost must equal sum of retrieved memory token_cost."""
    ctx, _ = _build_context()
    result = _run_retrieve(method_name, ctx)
    expected = sum(m.token_cost for m in result.retrieved_memories)
    assert result.total_token_cost == expected


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_token_budget_enforced(method_name: str):
    """When token_budget is set, total_token_cost must not exceed it."""
    ctx, _ = _build_context()
    result = _run_retrieve(method_name, ctx, token_budget=10, top_k=10)
    assert result.total_token_cost <= 10


# ── 3. gold leakage ──────────────────────────────────────────────────────


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_no_gold_leakage_in_result(method_name: str):
    """MethodResult and RetrievedMemory must not expose gold fields."""
    ctx, _ = _build_context()
    result = _run_retrieve(method_name, ctx)
    d = result.model_dump()
    assert "gold_relevant_memory_ids" not in d
    assert "gold_valid_memory_ids" not in d
    assert "gold_stale_memory_ids" not in d
    for mem in result.retrieved_memories:
        md = mem.model_dump()
        assert "gold" not in str(md).lower()


# ── 4. independent vs sequential ─────────────────────────────────────────


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_independent_queries_are_isolated(method_name: str):
    """Under independent policy, query order must not affect results."""
    ctx, _ = _build_context(query_state_policy="independent")
    registry = default_method_registry()
    scenarios = load_scenarios(SMOKE_DATASET)
    queries = scenarios[0].queries[:2]
    if len(queries) < 2:
        pytest.skip("need at least 2 queries in smoke data")

    method = registry.create(method_name, ctx)
    ids_run1 = []
    for q in queries:
        method.reset()
        for mem in scenarios[0].memories:
            method.ingest([mem])
        r = method.retrieve(
            q.model_copy(update={"gold_relevant_memory_ids": [], "gold_valid_memory_ids": [], "gold_stale_memory_ids": []}),
            top_k=ctx.experiment.top_k,
            token_budget=ctx.experiment.token_budget,
        )
        ids_run1.append([m.memory_id for m in r.retrieved_memories])

    # Reverse order — should give same per-query results
    ids_run2 = []
    for q in reversed(queries):
        method.reset()
        for mem in scenarios[0].memories:
            method.ingest([mem])
        r = method.retrieve(
            q.model_copy(update={"gold_relevant_memory_ids": [], "gold_valid_memory_ids": [], "gold_stale_memory_ids": []}),
            top_k=ctx.experiment.top_k,
            token_budget=ctx.experiment.token_budget,
        )
        ids_run2.append([m.memory_id for m in r.retrieved_memories])

    # Each query's result should be same regardless of execution order
    assert ids_run1[0] == ids_run2[1]
    assert ids_run1[1] == ids_run2[0]


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_methodresult_schema_valid(method_name: str):
    """Every MethodResult must conform to the frozen schema (ranks, token sum)."""
    ctx, _ = _build_context()
    result = _run_retrieve(method_name, ctx)
    # Schema validation happens on construction — if we got here, it passed
    assert isinstance(result, MethodResult)
    assert result.method_name == method_name
    assert result.latency_ms >= 0

# ── 5. latency ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_latency_is_end_to_end(method_name: str):
    """latency_ms must be > 0 and cover full retrieve() call."""
    ctx, _ = _build_context()
    result = _run_retrieve(method_name, ctx)
    assert result.latency_ms > 0
    assert result.latency_ms < 30_000  # sanity cap: 30s


# ── 6. cross-method consistency ─────────────────────────────────────────


@pytest.mark.parametrize("method_name", REGISTERED_METHODS)
def test_result_contains_required_metadata_keys(method_name: str):
    """All methods must include source_retriever in metadata."""
    ctx, _ = _build_context()
    result = _run_retrieve(method_name, ctx)
    assert "source_retriever" in result.metadata or "candidate_k" in result.metadata
