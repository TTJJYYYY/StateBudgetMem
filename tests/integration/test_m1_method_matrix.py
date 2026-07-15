from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Any

from statebudgetmem.core.registry import (
    default_method_registry,
)
from statebudgetmem.data.validation import (
    validate_dataset_manifest,
)
from statebudgetmem.unified_runner import (
    load_experiment_config,
)


MATRIX_PATH = Path(
    "configs/fair_experiments/m1_method_matrix.json"
)

EXPECTED_METHOD_IDS = {
    "tfidf_topk",
    "memorybank_core",
    "memorybank_versioning",
    "memorybank_dual_views",
    "statebudgetmem_rule_routing",
    "statebudgetmem_oracle_routing",
}

EXPECTED_REGISTRY_NAMES = {
    "tfidf_topk",
    "memorybank_core",
    "memorybank_versioning",
    "memorybank_dual_views",
    "statebudgetmem_rule",
    "statebudgetmem_oracle",
}

EXPECTED_DENSE_REGISTRY_NAMES = {
    "memorybank_core",
    "memorybank_versioning",
    "memorybank_dual_views",
    "statebudgetmem_rule",
    "statebudgetmem_oracle",
}

SHARED_CONFIG_FIELDS = (
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


def _load_matrix() -> dict[str, Any]:
    """Load the machine-readable M1 method matrix."""

    assert MATRIX_PATH.is_file(), (
        f"missing method matrix: {MATRIX_PATH}"
    )

    with MATRIX_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        matrix = json.load(handle)

    assert isinstance(matrix, dict)
    assert isinstance(matrix.get("dataset"), dict)
    assert isinstance(
        matrix.get("shared_parameters"),
        dict,
    )
    assert isinstance(matrix.get("methods"), list)
    assert isinstance(matrix.get("ablations"), dict)
    assert isinstance(matrix.get("oracle_policy"), dict)

    return matrix


def _normalise_expected_value(
    actual_value: Any,
    expected_value: Any,
) -> Any:
    """Convert JSON values to ExperimentConfig field types."""

    if isinstance(actual_value, Path):
        assert isinstance(expected_value, str)
        return Path(expected_value)

    if isinstance(actual_value, tuple):
        if isinstance(expected_value, list):
            return tuple(expected_value)

        if isinstance(expected_value, str):
            return (expected_value,)

    return expected_value


def test_m1_method_matrix_contains_six_unique_methods() -> None:
    matrix = _load_matrix()
    methods = matrix["methods"]

    method_ids = {
        method["method_id"]
        for method in methods
    }

    registry_names = {
        method["registry_name"]
        for method in methods
    }

    assert len(methods) == 6
    assert method_ids == EXPECTED_METHOD_IDS
    assert registry_names == EXPECTED_REGISTRY_NAMES

    assert len(method_ids) == len(methods)
    assert len(registry_names) == len(methods)


def test_all_methods_are_ready_and_registered() -> None:
    matrix = _load_matrix()

    registered_methods = set(
        default_method_registry().names()
    )

    for method in matrix["methods"]:
        assert method["status"] == "ready"

        registry_name = method["registry_name"]

        assert isinstance(registry_name, str)
        assert registry_name
        assert registry_name in registered_methods

        assert method["main_config"] is not None
        assert method["ablation_config_glob"] is not None

    assert EXPECTED_REGISTRY_NAMES.issubset(
        registered_methods
    )


def test_ready_methods_have_loadable_main_configs() -> None:
    matrix = _load_matrix()
    shared_parameters = matrix["shared_parameters"]

    for method in matrix["methods"]:
        config_path = Path(method["main_config"])

        assert config_path.is_file(), (
            f"missing main config: {config_path}"
        )

        config = load_experiment_config(config_path)

        assert config.methods == (
            method["registry_name"],
        )

        for field_name in SHARED_CONFIG_FIELDS:
            actual_value = getattr(
                config,
                field_name,
            )

            expected_value = _normalise_expected_value(
                actual_value,
                shared_parameters[field_name],
            )

            assert actual_value == expected_value, (
                f"{method['method_id']} changes "
                f"shared field {field_name!r}: "
                f"{actual_value!r} != "
                f"{expected_value!r}"
            )


def test_dense_methods_use_shared_backend() -> None:
    matrix = _load_matrix()
    dense_backend = matrix["dense_backend"]

    dense_methods = [
        method
        for method in matrix["methods"]
        if method["backend_family"] == "dense"
    ]

    assert len(dense_methods) == 5

    assert {
        method["registry_name"]
        for method in dense_methods
    } == EXPECTED_DENSE_REGISTRY_NAMES

    for method in dense_methods:
        config = load_experiment_config(
            Path(method["main_config"])
        )

        assert (
            config.embedding_backend
            == dense_backend["embedding_backend"]
        )

        assert (
            config.embedding_model
            == dense_backend["embedding_model"]
        )

    lexical_methods = [
        method
        for method in matrix["methods"]
        if method["backend_family"] == "lexical"
    ]

    assert len(lexical_methods) == 1
    assert (
        lexical_methods[0]["registry_name"]
        == "tfidf_topk"
    )

    tfidf_config = load_experiment_config(
        Path(lexical_methods[0]["main_config"])
    )

    assert (
        tfidf_config.embedding_backend
        == "method_default"
    )
    assert (
        tfidf_config.embedding_model
        == "method_default"
    )


def test_all_methods_have_six_ablation_configs() -> None:
    matrix = _load_matrix()

    all_ablation_paths: list[Path] = []

    for method in matrix["methods"]:
        config_glob = method["ablation_config_glob"]

        assert isinstance(config_glob, str)
        assert config_glob

        matching_paths = sorted(
            Path(path)
            for path in glob(config_glob)
        )

        assert len(matching_paths) == 6, (
            f"{method['method_id']} should have "
            "six generated ablation configs, "
            f"got {len(matching_paths)}"
        )

        for config_path in matching_paths:
            assert config_path.is_file()

            config = load_experiment_config(
                config_path
            )

            assert config.methods == (
                method["registry_name"],
            )

        all_ablation_paths.extend(matching_paths)

    assert len(all_ablation_paths) == 36
    assert len(all_ablation_paths) == len(
        set(all_ablation_paths)
    )


def test_ablation_plan_matches_generated_matrix() -> None:
    matrix = _load_matrix()
    ablations = matrix["ablations"]

    assert ablations["mode"] == "single_variable"

    assert ablations["baseline"] == {
        "top_k": 3,
        "candidate_k": 20,
        "token_budget": 32,
    }

    assert ablations["sweeps"] == {
        "top_k": [1, 3, 5],
        "candidate_k": [5, 20, 40],
        "token_budget": [16, 32, 64],
    }

    assert ablations[
        "generated_nonbaseline_values"
    ] == {
        "top_k": [1, 5],
        "candidate_k": [5, 40],
        "token_budget": [16, 64],
    }

    assert ablations["method_count"] == 6
    assert ablations["generated_config_count"] == 36


def test_dataset_manifest_matches_matrix() -> None:
    matrix = _load_matrix()
    dataset = matrix["dataset"]

    audit = validate_dataset_manifest(
        dataset["path"],
        dataset["manifest_path"],
    )

    assert audit["dataset_path"] == dataset["path"]
    assert audit["sha256"] == dataset["sha256"]

    assert audit["scenario_count"] == 32
    assert audit["memory_count"] == 193
    assert audit["query_count"] == 96


def test_oracle_policy_prevents_gold_leakage() -> None:
    matrix = _load_matrix()
    oracle_policy = matrix["oracle_policy"]

    assert oracle_policy["allowed_fields"] == [
        "query_type"
    ]

    forbidden_fields = set(
        oracle_policy["forbidden_fields"]
    )

    assert {
        "gold_memory_ids",
        "gold_answer",
        "expected_memory_ids",
        "target_memory_ids",
    }.issubset(forbidden_fields)

    assert "query_type" not in forbidden_fields

    oracle_method = next(
        method
        for method in matrix["methods"]
        if method["registry_name"]
        == "statebudgetmem_oracle"
    )

    assert oracle_method["status"] == "ready"
    assert (
        oracle_method["backend_family"]
        == "dense"
    )