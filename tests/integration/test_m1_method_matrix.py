from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Any

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
    """Load and validate the root structure of the M1 method matrix."""

    assert MATRIX_PATH.is_file(), (
        f"missing method matrix: {MATRIX_PATH}"
    )

    with MATRIX_PATH.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)

    assert isinstance(matrix, dict)
    assert isinstance(matrix.get("methods"), list)
    assert isinstance(matrix.get("shared_parameters"), dict)

    return matrix


def _normalise_expected_value(
    actual_value: Any,
    expected_value: Any,
) -> Any:
    """Convert JSON values to the types used by ExperimentConfig."""

    if isinstance(actual_value, Path):
        assert isinstance(expected_value, str)
        return Path(expected_value)

    if isinstance(actual_value, tuple):
        if isinstance(expected_value, list):
            return tuple(expected_value)

        if isinstance(expected_value, str):
            return (expected_value,)

    return expected_value


def test_m1_minimum_method_matrix_is_complete() -> None:
    matrix = _load_matrix()

    methods = matrix["methods"]
    method_ids = {
        method["method_id"]
        for method in methods
    }

    assert method_ids == EXPECTED_METHOD_IDS
    assert len(methods) == 6
    assert len(method_ids) == len(methods)


def test_ready_methods_have_loadable_configs() -> None:
    matrix = _load_matrix()
    shared = matrix["shared_parameters"]

    ready_methods = [
        method
        for method in matrix["methods"]
        if method["status"] == "ready"
    ]

    assert {
        method["method_id"]
        for method in ready_methods
    } == {
        "tfidf_topk",
        "memorybank_core",
    }

    for method in ready_methods:
        assert method["registry_name"] is not None
        assert method["main_config"] is not None

        config_path = Path(method["main_config"])
        assert config_path.is_file()

        config = load_experiment_config(config_path)

        assert config.methods == (
            method["registry_name"],
        )

        for field_name in SHARED_CONFIG_FIELDS:
            actual_value = getattr(config, field_name)
            expected_value = _normalise_expected_value(
                actual_value,
                shared[field_name],
            )

            assert actual_value == expected_value, (
                f"{method['method_id']} changes "
                f"shared field {field_name!r}: "
                f"{actual_value!r} != {expected_value!r}"
            )


def test_pending_methods_do_not_invent_registry_names() -> None:
    matrix = _load_matrix()

    pending_methods = [
        method
        for method in matrix["methods"]
        if method["status"] == "pending"
    ]

    assert len(pending_methods) == 4

    for method in pending_methods:
        assert method["backend_family"] == "dense"
        assert method["registry_name"] is None
        assert method["main_config"] is None
        assert method["ablation_config_glob"] is None


def test_ready_dense_methods_use_frozen_backend() -> None:
    matrix = _load_matrix()
    dense_backend = matrix["dense_backend"]

    ready_dense_methods = [
        method
        for method in matrix["methods"]
        if (
            method["status"] == "ready"
            and method["backend_family"] == "dense"
        )
    ]

    assert {
        method["method_id"]
        for method in ready_dense_methods
    } == {
        "memorybank_core",
    }

    for method in ready_dense_methods:
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

    tfidf_method = next(
        method
        for method in matrix["methods"]
        if method["method_id"] == "tfidf_topk"
    )

    tfidf_config = load_experiment_config(
        Path(tfidf_method["main_config"])
    )

    assert tfidf_config.embedding_backend == "method_default"
    assert tfidf_config.embedding_model == "method_default"


def test_ablation_plan_and_generated_files_match() -> None:
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

    assert ablations["generated_nonbaseline_values"] == {
        "top_k": [1, 5],
        "candidate_k": [5, 40],
        "token_budget": [16, 64],
    }

    for method in matrix["methods"]:
        if method["status"] != "ready":
            continue

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

            config = load_experiment_config(config_path)

            assert config.methods == (
                method["registry_name"],
            )


def test_dataset_and_oracle_policy_are_valid() -> None:
    matrix = _load_matrix()

    dataset = matrix["dataset"]

    audit = validate_dataset_manifest(
        dataset["path"],
        dataset["manifest_path"],
    )

    assert audit["sha256"] == dataset["sha256"]

    assert (
        audit["dataset_path"]
        == dataset["path"]
    )

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