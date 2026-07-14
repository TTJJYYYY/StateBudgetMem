from __future__ import annotations

from pathlib import Path

from statebudgetmem.unified_runner import load_experiment_config


CONFIG_ROOT = Path("configs/fair_experiments")
ABLATION_DIR = CONFIG_ROOT / "ablations"

BASE_CONFIG_PATHS = {
    "tfidf_topk": CONFIG_ROOT / "m1_tfidf_topk.yaml",
    "memorybank_core": CONFIG_ROOT / "m1_memorybank_core.yaml",
}

ABLATION_SWEEPS = {
    "top_k": (1, 5),
    "candidate_k": (5, 40),
    "token_budget": (16, 64),
}

SHARED_FIELDS = (
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


def _config_path(
    method_name: str,
    parameter_name: str,
    value: int,
) -> Path:
    return ABLATION_DIR / (
        f"m1_{method_name}_{parameter_name}_{value}.yaml"
    )


def _expected_entries():
    for method_name in BASE_CONFIG_PATHS:
        for parameter_name, values in ABLATION_SWEEPS.items():
            for value in values:
                yield (
                    method_name,
                    parameter_name,
                    value,
                    _config_path(
                        method_name,
                        parameter_name,
                        value,
                    ),
                )


def test_ablation_file_set_is_complete() -> None:
    expected_paths = {
        config_path
        for _, _, _, config_path in _expected_entries()
    }

    actual_paths = set(
        ABLATION_DIR.glob("m1_*.yaml")
    )

    assert actual_paths == expected_paths


def test_all_ablation_configs_load() -> None:
    for (
        method_name,
        parameter_name,
        value,
        config_path,
    ) in _expected_entries():
        config = load_experiment_config(config_path)

        assert config.methods == (method_name,)
        assert getattr(config, parameter_name) == value


def test_each_ablation_changes_only_one_primary_parameter() -> None:
    for (
        method_name,
        parameter_name,
        value,
        config_path,
    ) in _expected_entries():
        base = load_experiment_config(
            BASE_CONFIG_PATHS[method_name]
        )
        variant = load_experiment_config(config_path)

        for field_name in SHARED_FIELDS:
            actual_value = getattr(variant, field_name)
            base_value = getattr(base, field_name)

            if field_name == parameter_name:
                assert actual_value == value
                assert actual_value != base_value
            else:
                assert actual_value == base_value, (
                    f"{config_path} unexpectedly changes "
                    f"{field_name!r}: "
                    f"{actual_value!r} != {base_value!r}"
                )

        assert variant.methods == base.methods
        assert (
            variant.embedding_backend
            == base.embedding_backend
        )
        assert (
            variant.embedding_model
            == base.embedding_model
        )


def test_ablation_retrieval_limits_are_valid() -> None:
    for _, _, _, config_path in _expected_entries():
        config = load_experiment_config(config_path)

        assert config.top_k > 0
        assert config.candidate_k >= config.top_k
        assert config.token_budget > 0

        assert config.random_seed == 42
        assert config.repeat == 1
        assert config.query_state_policy == "independent"
        assert config.reinforcement_enabled is False


def test_paired_methods_share_fair_parameters() -> None:
    for parameter_name, values in ABLATION_SWEEPS.items():
        for value in values:
            tfidf_config = load_experiment_config(
                _config_path(
                    "tfidf_topk",
                    parameter_name,
                    value,
                )
            )
            memorybank_config = load_experiment_config(
                _config_path(
                    "memorybank_core",
                    parameter_name,
                    value,
                )
            )

            for field_name in SHARED_FIELDS:
                assert getattr(
                    tfidf_config,
                    field_name,
                ) == getattr(
                    memorybank_config,
                    field_name,
                ), (
                    f"{parameter_name}={value} differs "
                    f"between methods in field "
                    f"{field_name!r}"
                )


def test_ablation_result_directories_are_unique() -> None:
    result_dirs = []

    for (
        method_name,
        parameter_name,
        value,
        config_path,
    ) in _expected_entries():
        config = load_experiment_config(config_path)
        result_dir = config.results_dir

        result_dirs.append(result_dir)

        assert not result_dir.is_absolute()
        assert result_dir.as_posix() == (
            "results/fair_experiments/m1/"
            f"ablations/{parameter_name}/{value}/"
            f"{method_name}"
        )

    assert len(result_dirs) == len(set(result_dirs))