from __future__ import annotations

from pathlib import Path
from typing import Any

from statebudgetmem.core.registry import default_method_registry
from statebudgetmem.data.validation import (
    validate_dataset_manifest,
)
from statebudgetmem.unified_runner import (
    load_experiment_config,
)


CONFIG_DIR = Path("configs/fair_experiments")

CONFIG_PATHS = {
    "tfidf_topk": (
        CONFIG_DIR / "m1_tfidf_topk.yaml"
    ),
    "memorybank_core": (
        CONFIG_DIR / "m1_memorybank_core.yaml"
    ),
    "memorybank_versioning": (
        CONFIG_DIR / "m1_memorybank_versioning.yaml"
    ),
    "memorybank_dual_views": (
        CONFIG_DIR / "m1_memorybank_dual_views.yaml"
    ),
    "statebudgetmem_rule": (
        CONFIG_DIR / "m1_statebudgetmem_rule.yaml"
    ),
    "statebudgetmem_oracle": (
        CONFIG_DIR / "m1_statebudgetmem_oracle.yaml"
    ),
}

DENSE_METHODS = {
    "memorybank_core",
    "memorybank_versioning",
    "memorybank_dual_views",
    "statebudgetmem_rule",
    "statebudgetmem_oracle",
}

DATASET_PATH = Path(
    "data/controlled/temporal_challenge_v1.jsonl"
)

MANIFEST_PATH = Path(
    "data/controlled/manifests/"
    "temporal_challenge_v1.manifest.json"
)

# 这些字段在 M1 主实验中必须对所有方法完全一致。
SHARED_FAIR_FIELDS = (
    "dataset_path",
    "top_k",
    "candidate_k",
    "token_budget",
    "random_seed",
    "repeat",
    "forgetting_enabled",
    "forgetting_threshold",
    "exclude_forgotten",
    "reinforcement_enabled",
    "query_state_policy",
    "token_counter_name",
)


def _load_configs() -> dict[str, Any]:
    """Load every M1 main configuration through the unified Runner."""

    return {
        method_name: load_experiment_config(config_path)
        for method_name, config_path in CONFIG_PATHS.items()
    }


def test_formal_dataset_manifest_matches() -> None:
    audit = validate_dataset_manifest(
        DATASET_PATH,
        MANIFEST_PATH,
    )

    assert audit["scenario_count"] == 32
    assert audit["memory_count"] == 193
    assert audit["query_count"] == 96

    assert audit["query_type_counts"] == {
        "CURRENT": 32,
        "HISTORICAL": 32,
        "CHANGE": 32,
        "GENERAL": 0,
    }

    assert audit["sha256"] == (
        "f93331a2d93588fa8931efb4484fce577f5a5c9c4e679c51"
        "d4bb0192af6c8dd9"
    )


def test_m1_config_method_names_are_correct() -> None:
    configs = _load_configs()

    assert set(configs) == {
        "tfidf_topk",
        "memorybank_core",
        "memorybank_versioning",
        "memorybank_dual_views",
        "statebudgetmem_rule",
        "statebudgetmem_oracle",
    }

    for expected_method, config in configs.items():
        assert config.methods == (expected_method,)
        assert config.dataset_path == DATASET_PATH


def test_all_m1_methods_are_registered() -> None:
    registered_methods = set(
        default_method_registry().names()
    )

    assert set(CONFIG_PATHS).issubset(
        registered_methods
    )


def test_m1_configs_share_fair_parameters() -> None:
    configs = _load_configs()

    reference_method = "tfidf_topk"
    reference_config = configs[reference_method]

    for method_name, config in configs.items():
        for field_name in SHARED_FAIR_FIELDS:
            actual_value = getattr(
                config,
                field_name,
            )
            reference_value = getattr(
                reference_config,
                field_name,
            )

            assert actual_value == reference_value, (
                f"{method_name} changes shared fair field "
                f"{field_name!r}: "
                f"{actual_value!r} != "
                f"{reference_value!r}"
            )


def test_m1_retrieval_limits_are_valid() -> None:
    configs = _load_configs()

    for method_name, config in configs.items():
        assert config.top_k == 3, method_name
        assert config.candidate_k == 20, method_name
        assert config.token_budget == 32, method_name

        assert config.top_k > 0, method_name
        assert (
            config.candidate_k >= config.top_k
        ), method_name
        assert config.token_budget > 0, method_name

        assert config.random_seed == 42, method_name
        assert config.repeat == 1, method_name

        assert (
            config.query_state_policy
            == "independent"
        ), method_name

        assert (
            config.reinforcement_enabled is False
        ), method_name


def test_all_dense_methods_use_frozen_backend() -> None:
    configs = _load_configs()

    for method_name in DENSE_METHODS:
        config = configs[method_name]

        assert config.embedding_backend == "hash", (
            f"{method_name} must use the shared hash backend"
        )

        assert (
            config.embedding_model
            == "deterministic_hash_embedding"
        ), (
            f"{method_name} must use the shared "
            "deterministic embedding model"
        )


def test_tfidf_does_not_claim_dense_backend() -> None:
    config = _load_configs()["tfidf_topk"]

    assert (
        config.embedding_backend
        == "method_default"
    )
    assert (
        config.embedding_model
        == "method_default"
    )


def test_result_directories_are_unique_and_relative() -> None:
    configs = _load_configs()

    result_dirs = [
        config.results_dir
        for config in configs.values()
    ]

    assert len(result_dirs) == len(set(result_dirs))

    for method_name, config in configs.items():
        result_dir = config.results_dir

        assert not result_dir.is_absolute(), method_name

        assert result_dir.as_posix() == (
            "results/fair_experiments/m1/"
            f"{method_name}"
        )