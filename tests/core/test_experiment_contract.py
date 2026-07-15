from __future__ import annotations

from pathlib import Path

import pytest

from statebudgetmem.core.experiment import (
    ExperimentConfig,
    MethodBuildContext,
    ResourceMetrics,
)
from statebudgetmem.core.registry import MethodRegistry, default_method_registry


def test_experiment_config_freezes_minimum_comparison_fields() -> None:
    config = ExperimentConfig(
        dataset_path=Path("data/controlled/interface_smoke_v1.jsonl"),
        methods=("tfidf_topk",),
        top_k=2,
        token_budget=32,
        random_seed=42,
    )
    assert config.candidate_k == 20
    assert config.forgetting_enabled is True
    assert config.forgetting_threshold == 0.3
    assert config.exclude_forgotten is False
    assert config.reinforcement_enabled is False
    assert config.query_state_policy == "independent"
    assert config.embedding_backend == "method_default"
    assert config.token_counter_name == "memory_record_token_cost"


def test_experiment_config_rejects_empty_duplicate_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ExperimentConfig(dataset_path=Path("fixture.jsonl"), methods=())
    with pytest.raises(ValueError, match="duplicates"):
        ExperimentConfig(dataset_path=Path("fixture.jsonl"), methods=("a", "a"))
    with pytest.raises(ValueError):
        ExperimentConfig(dataset_path=Path("fixture.jsonl"), top_k=0)
    with pytest.raises(ValueError, match="candidate_k"):
        ExperimentConfig(dataset_path=Path("fixture.jsonl"), top_k=5, candidate_k=4)
    with pytest.raises(ValueError):
        ExperimentConfig(dataset_path=Path("fixture.jsonl"), forgetting_threshold=1.1)
    with pytest.raises(ValueError):
        ExperimentConfig(dataset_path=Path("fixture.jsonl"), query_state_policy="mixed")
    with pytest.raises(ValueError):
        ResourceMetrics(total_token_cost=-1)


def test_default_registry_exposes_all_dense_method_variants() -> None:
    registry = default_method_registry()
    context = MethodBuildContext(
        experiment=ExperimentConfig(
            dataset_path=Path("fixture.jsonl"),
            embedding_backend="hash",
            embedding_model="deterministic_hash_embedding",
        ),
        work_dir=Path("results/test"),
    )
    assert registry.names() == (
        "memorybank_core",
        "memorybank_dual_views",
        "memorybank_versioning",
        "statebudgetmem_oracle",
        "statebudgetmem_rule",
        "tfidf_topk",
    )
    assert registry.create("tfidf_topk", context).name == "tfidf_topk"
    assert registry.create("memorybank_core", context).name == "memorybank_core"
    with pytest.raises(ValueError, match="available methods") as caught:
        registry.create("not_registered", context)
    for method_name in registry.names():
        assert method_name in str(caught.value)


@pytest.mark.parametrize(
    "method_name, mode_name",
    [
        ("memorybank_versioning", "VERSIONING"),
        ("memorybank_dual_views", "DUAL_VIEWS"),
        ("statebudgetmem_rule", "RULE_ROUTING"),
        ("statebudgetmem_oracle", "ORACLE_ROUTING"),
    ],
)
def test_default_registry_creates_statebudgetmem_dense_variants(
    method_name: str, mode_name: str
) -> None:
    from statebudgetmem.baselines.memorybank.statebudgetmem_adapter import (
        StateBudgetMemDenseMethod,
        StateBudgetMemMode,
    )

    registry = default_method_registry()
    context = MethodBuildContext(
        experiment=ExperimentConfig(
            dataset_path=Path("fixture.jsonl"),
            embedding_backend="hash",
            embedding_model="deterministic_hash_embedding",
        ),
        work_dir=Path("results/test"),
    )

    method = registry.create(method_name, context)

    assert isinstance(method, StateBudgetMemDenseMethod)
    assert method.name == method_name
    assert method.mode is StateBudgetMemMode[mode_name]
    assert method.base_method.name == "memorybank_core"
    assert method.embedding_model is method.base_method.embedding_model


def test_default_registry_creates_isolated_dense_method_instances() -> None:
    registry = default_method_registry()
    context = MethodBuildContext(
        experiment=ExperimentConfig(
            dataset_path=Path("fixture.jsonl"),
            embedding_backend="hash",
            embedding_model="deterministic_hash_embedding",
        ),
        work_dir=Path("results/test"),
    )

    first = registry.create("memorybank_versioning", context)
    second = registry.create("memorybank_versioning", context)

    assert first is not second
    assert first.base_method is not second.base_method
    assert first.bank is not second.bank
    assert first.view_manager is not second.view_manager


def test_factory_receives_complete_method_build_context() -> None:
    registry = MethodRegistry()
    received = []
    context = MethodBuildContext(
        experiment=ExperimentConfig(
            dataset_path=Path("fixture.jsonl"),
            embedding_backend="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            candidate_k=12,
            reinforcement_enabled=True,
            query_state_policy="sequential",
        ),
        work_dir=Path("results/method"),
    )
    registry.register(
        "example",
        lambda actual: received.append(actual)
        or default_method_registry().create("tfidf_topk", actual),
    )

    registry.create("example", context)

    assert received == [context]
    assert received[0].experiment.candidate_k == 12
    assert received[0].experiment.reinforcement_enabled is True


def test_registry_rejects_duplicate_registration() -> None:
    registry = MethodRegistry()
    factory = lambda context: default_method_registry().create("tfidf_topk", context)
    registry.register("example", factory)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("example", factory)


@pytest.mark.parametrize("policy", ["independent", "sequential"])
def test_query_state_policy_accepts_frozen_values(policy: str) -> None:
    config = ExperimentConfig(
        dataset_path=Path("fixture.jsonl"), query_state_policy=policy
    )
    assert config.query_state_policy == policy
