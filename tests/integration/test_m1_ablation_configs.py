from __future__ import annotations

from pathlib import Path
from typing import Iterator

from statebudgetmem.unified_runner import (
    load_experiment_config,
)


CONFIG_ROOT = Path("configs/fair_experiments")
ABLATION_DIR = CONFIG_ROOT / "ablations"

BASE_CONFIG_PATHS = {
    "tfidf_topk": (
        CONFIG_ROOT / "m1_tfidf_topk.yaml"
    ),
    "memorybank_core": (
        CONFIG_ROOT / "m1_memorybank_core.yaml"
    ),
    "memorybank_versioning": (
        CONFIG_ROOT / "m1_memorybank_versioning.yaml"
    ),
    "memorybank_dual_views": (
        CONFIG_ROOT / "m1_memorybank_dual_views.yaml"
    ),
    "statebudgetmem_rule": (
        CONFIG_ROOT / "m1_statebudgetmem_rule.yaml"
    ),
    "statebudgetmem_oracle": (
        CONFIG_ROOT / "m1_statebudgetmem_oracle.yaml"
    ),
}

DENSE_METHODS = {
    "memorybank_core",
    "memorybank_versioning",
    "memorybank_dual_views",
    "statebudgetmem_rule",
    "statebudgetmem_oracle",
}

# 主实验基线值已经由各方法的主配置表示。
# 消融目录只保存非基线值。
ABLATION_SWEEPS = {
    "top_k": (1, 5),
    "candidate_k": (5, 40),
    "token_budget": (16, 64),
}

# 同一消融条件下，六种方法必须保持一致的字段。
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


def _config_path(
    method_name: str,
    parameter_name: str,
    value: int,
) -> Path:
    """Return the expected path of one generated config."""

    return ABLATION_DIR / (
        f"m1_{method_name}_"
        f"{parameter_name}_{value}.yaml"
    )


def _expected_entries() -> Iterator[
    tuple[str, str, int, Path]
]:
    """Yield every method/parameter/value/config combination."""

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

    assert len(expected_paths) == 36
    assert actual_paths == expected_paths


def test_all_ablation_configs_load() -> None:
    for (
        method_name,
        parameter_name,
        value,
        config_path,
    ) in _expected_entries():
        assert config_path.is_file(), (
            f"missing generated config: {config_path}"
        )

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

        for field_name in SHARED_FAIR_FIELDS:
            base_value = getattr(base, field_name)
            variant_value = getattr(
                variant,
                field_name,
            )

            if field_name == parameter_name:
                assert variant_value == value
                assert variant_value != base_value
            else:
                assert variant_value == base_value, (
                    f"{config_path} unexpectedly changes "
                    f"{field_name!r}: "
                    f"{variant_value!r} != {base_value!r}"
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
    for (
        method_name,
        _,
        _,
        config_path,
    ) in _expected_entries():
        config = load_experiment_config(config_path)

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


def test_all_methods_share_each_ablation_condition() -> None:
    """Ensure no method receives a private experimental advantage."""

    reference_method = "tfidf_topk"

    for parameter_name, values in ABLATION_SWEEPS.items():
        for value in values:
            reference_config = load_experiment_config(
                _config_path(
                    reference_method,
                    parameter_name,
                    value,
                )
            )

            for method_name in BASE_CONFIG_PATHS:
                config = load_experiment_config(
                    _config_path(
                        method_name,
                        parameter_name,
                        value,
                    )
                )

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
                        f"{method_name} differs under "
                        f"{parameter_name}={value} in "
                        f"field {field_name!r}: "
                        f"{actual_value!r} != "
                        f"{reference_value!r}"
                    )


def test_all_dense_ablation_configs_use_frozen_backend() -> None:
    for (
        method_name,
        _,
        _,
        config_path,
    ) in _expected_entries():
        config = load_experiment_config(config_path)

        if method_name in DENSE_METHODS:
            assert config.embedding_backend == "hash"
            assert (
                config.embedding_model
                == "deterministic_hash_embedding"
            )
        else:
            assert method_name == "tfidf_topk"
            assert (
                config.embedding_backend
                == "method_default"
            )
            assert (
                config.embedding_model
                == "method_default"
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
            f"ablations/{parameter_name}/"
            f"{value}/{method_name}"
        )

    assert len(result_dirs) == 36
    assert len(result_dirs) == len(set(result_dirs))