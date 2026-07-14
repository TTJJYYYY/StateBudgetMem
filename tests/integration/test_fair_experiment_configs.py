from __future__ import annotations

from pathlib import Path

from statebudgetmem.data.validation import validate_dataset_manifest
from statebudgetmem.unified_runner import load_experiment_config


CONFIG_DIR = Path("configs/fair_experiments")

CONFIG_PATHS = {
    "tfidf_topk": CONFIG_DIR / "m1_tfidf_topk.yaml",
    "memorybank_core": CONFIG_DIR / "m1_memorybank_core.yaml",
}

DATASET_PATH = Path(
    "data/controlled/temporal_challenge_v1.jsonl"
)
MANIFEST_PATH = Path(
    "data/controlled/manifests/"
    "temporal_challenge_v1.manifest.json"
)

# 正式公平实验中，各方法不得修改这些字段。
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


def _load_configs():
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


def test_m1_config_method_names_are_correct() -> None:
    configs = _load_configs()

    for expected_method, config in configs.items():
        assert config.methods == (expected_method,)
        assert config.dataset_path == DATASET_PATH


def test_m1_configs_share_fair_parameters() -> None:
    configs = _load_configs()

    reference_method = "tfidf_topk"
    reference_config = configs[reference_method]

    for method_name, config in configs.items():
        for field_name in SHARED_FAIR_FIELDS:
            actual_value = getattr(config, field_name)
            reference_value = getattr(
                reference_config,
                field_name,
            )

            assert actual_value == reference_value, (
                f"{method_name} changes shared fair field "
                f"{field_name!r}: "
                f"{actual_value!r} != {reference_value!r}"
            )


def test_m1_retrieval_limits_are_valid() -> None:
    configs = _load_configs()

    for method_name, config in configs.items():
        assert config.top_k > 0, method_name
        assert config.candidate_k >= config.top_k, method_name
        assert config.token_budget > 0, method_name

        assert config.random_seed == 42, method_name
        assert config.repeat == 1, method_name

        assert (
            config.query_state_policy == "independent"
        ), method_name
        assert (
            config.reinforcement_enabled is False
        ), method_name


def test_memorybank_uses_frozen_dense_backend() -> None:
    config = load_experiment_config(
        CONFIG_PATHS["memorybank_core"]
    )

    assert config.embedding_backend == "hash"
    assert (
        config.embedding_model
        == "deterministic_hash_embedding"
    )


def test_tfidf_does_not_claim_dense_backend() -> None:
    config = load_experiment_config(
        CONFIG_PATHS["tfidf_topk"]
    )

    assert config.embedding_backend == "method_default"
    assert config.embedding_model == "method_default"


def test_result_directories_are_unique_and_relative() -> None:
    configs = _load_configs()

    result_dirs = [
        config.results_dir
        for config in configs.values()
    ]

    assert len(result_dirs) == len(set(result_dirs))

    for result_dir in result_dirs:
        assert not result_dir.is_absolute()
        assert result_dir.as_posix().startswith(
            "results/fair_experiments/m1/"
        )