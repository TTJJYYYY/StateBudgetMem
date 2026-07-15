from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIGS = {
    "tfidf_topk": (
        PROJECT_ROOT
        / "configs/fair_experiments/m1_tfidf_topk.yaml"
    ),
    "memorybank_core": (
        PROJECT_ROOT
        / "configs/fair_experiments/m1_memorybank_core.yaml"
    ),
    "memorybank_versioning": (
        PROJECT_ROOT
        / "configs/fair_experiments/"
        "m1_memorybank_versioning.yaml"
    ),
    "memorybank_dual_views": (
        PROJECT_ROOT
        / "configs/fair_experiments/"
        "m1_memorybank_dual_views.yaml"
    ),
    "statebudgetmem_rule": (
        PROJECT_ROOT
        / "configs/fair_experiments/"
        "m1_statebudgetmem_rule.yaml"
    ),
    "statebudgetmem_oracle": (
        PROJECT_ROOT
        / "configs/fair_experiments/"
        "m1_statebudgetmem_oracle.yaml"
    ),
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "configs/fair_experiments/ablations"
)

# M1 主实验基线：
# top_k=3, candidate_k=20, token_budget=32
#
# 生成器只创建非基线值，主实验配置本身代表基线值。
ABLATION_SWEEPS = {
    "top_k": (1, 5),
    "candidate_k": (5, 40),
    "token_budget": (16, 64),
}

# 保持生成文件字段顺序统一，方便人工审查和版本比较。
CONFIG_FIELD_ORDER = (
    "dataset_path",
    "results_dir",
    "methods",
    "top_k",
    "candidate_k",
    "token_budget",
    "random_seed",
    "repeat",
    "embedding_backend",
    "embedding_model",
    "forgetting_enabled",
    "forgetting_threshold",
    "exclude_forgotten",
    "reinforcement_enabled",
    "query_state_policy",
    "token_counter_name",
)

INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+\.\d*|\d*\.\d+)"
    r"(?:[eE][+-]?\d+)?$"
)


def _parse_scalar(raw_value: str) -> Any:
    """Parse scalar values used by the flat experiment YAML files."""

    value = raw_value.strip()

    if not value:
        return ""

    lowered = value.lower()

    if lowered == "true":
        return True

    if lowered == "false":
        return False

    if lowered in {"null", "none", "~"}:
        return None

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {'"', "'"}
    ):
        if value[0] == '"':
            return json.loads(value)

        return value[1:-1].replace("''", "'")

    if INTEGER_PATTERN.fullmatch(value):
        return int(value)

    if FLOAT_PATTERN.fullmatch(value):
        return float(value)

    return value


def _load_flat_yaml(path: Path) -> dict[str, Any]:
    """Read the project's flat key-value YAML configuration."""

    if not path.is_file():
        raise FileNotFoundError(
            f"config not found: {path}"
        )

    config: dict[str, Any] = {}

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):
        stripped = raw_line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if ":" not in raw_line:
            raise ValueError(
                f"{path}:{line_number}: "
                "expected 'key: value'"
            )

        key, raw_value = raw_line.split(":", 1)
        key = key.strip()

        if not key:
            raise ValueError(
                f"{path}:{line_number}: "
                "empty configuration key"
            )

        if key in config:
            raise ValueError(
                f"{path}:{line_number}: "
                f"duplicate key {key!r}"
            )

        config[key] = _parse_scalar(raw_value)

    missing_fields = [
        field_name
        for field_name in CONFIG_FIELD_ORDER
        if field_name not in config
    ]

    if missing_fields:
        raise ValueError(
            f"{path}: missing required fields: "
            f"{missing_fields}"
        )

    unexpected_fields = sorted(
        set(config) - set(CONFIG_FIELD_ORDER)
    )

    if unexpected_fields:
        raise ValueError(
            f"{path}: unexpected fields: "
            f"{unexpected_fields}"
        )

    return config


def _serialise_scalar(value: Any) -> str:
    """Serialise a scalar using syntax valid in YAML."""

    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return repr(value)

    if isinstance(value, str):
        # JSON string syntax is also valid YAML syntax.
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    raise TypeError(
        "unsupported config value type: "
        f"{type(value).__name__}"
    )


def _write_flat_yaml(
    path: Path,
    config: dict[str, Any],
) -> None:
    """Write one generated experiment configuration."""

    missing_fields = [
        field_name
        for field_name in CONFIG_FIELD_ORDER
        if field_name not in config
    ]

    if missing_fields:
        raise ValueError(
            "cannot write config with missing fields: "
            f"{missing_fields}"
        )

    lines = [
        (
            "# Generated by "
            "scripts/generate_m1_ablation_configs.py."
        ),
        (
            "# Do not edit manually; "
            "update the generator or base config instead."
        ),
    ]

    for field_name in CONFIG_FIELD_ORDER:
        lines.append(
            f"{field_name}: "
            f"{_serialise_scalar(config[field_name])}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _output_filename(
    method_name: str,
    parameter_name: str,
    value: int,
) -> str:
    """Build a stable filename for one ablation config."""

    return (
        f"m1_{method_name}_"
        f"{parameter_name}_{value}.yaml"
    )


def build_expected_configs() -> dict[
    Path,
    dict[str, Any],
]:
    """Build the complete M1 single-variable ablation matrix."""

    expected: dict[
        Path,
        dict[str, Any],
    ] = {}

    for method_name, base_path in BASE_CONFIGS.items():
        base_config = _load_flat_yaml(base_path)

        if base_config["methods"] != method_name:
            raise ValueError(
                "base config method mismatch in "
                f"{base_path}: expected "
                f"{method_name!r}, got "
                f"{base_config['methods']!r}"
            )

        for (
            parameter_name,
            values,
        ) in ABLATION_SWEEPS.items():
            for value in values:
                config = deepcopy(base_config)

                # 单变量消融：只改变当前实验变量。
                config[parameter_name] = value

                # results_dir 变化仅用于隔离输出，
                # 不属于算法变量。
                config["results_dir"] = (
                    "results/fair_experiments/m1/"
                    f"ablations/{parameter_name}/"
                    f"{value}/{method_name}"
                )

                output_path = (
                    OUTPUT_DIR
                    / _output_filename(
                        method_name,
                        parameter_name,
                        value,
                    )
                )

                if output_path in expected:
                    raise ValueError(
                        "duplicate generated path: "
                        f"{output_path}"
                    )

                expected[output_path] = config

    return expected


def write_configs() -> int:
    """Generate all expected M1 ablation YAML files."""

    expected = build_expected_configs()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_paths = set(expected)

    # 删除旧生成器留下、但已不属于当前矩阵的配置。
    for existing_path in OUTPUT_DIR.glob(
        "m1_*.yaml"
    ):
        if existing_path not in expected_paths:
            existing_path.unlink()

    for output_path, config in expected.items():
        _write_flat_yaml(
            output_path,
            config,
        )

    relative_output_dir = OUTPUT_DIR.relative_to(
        PROJECT_ROOT
    )

    print(
        f"Generated {len(expected)} "
        "M1 ablation configs in "
        f"{relative_output_dir}"
    )

    return 0


def check_configs() -> int:
    """Verify generated files are complete and unchanged."""

    expected = build_expected_configs()
    expected_paths = set(expected)

    actual_paths = set(
        OUTPUT_DIR.glob("m1_*.yaml")
    )

    errors: list[str] = []

    for path in sorted(
        expected_paths - actual_paths
    ):
        errors.append(
            "missing config: "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

    for path in sorted(
        actual_paths - expected_paths
    ):
        errors.append(
            "unexpected config: "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

    for path, expected_config in expected.items():
        if not path.is_file():
            continue

        try:
            actual_config = _load_flat_yaml(path)
        except (OSError, ValueError) as exc:
            errors.append(
                "invalid config "
                f"{path.relative_to(PROJECT_ROOT)}: "
                f"{exc}"
            )
            continue

        changed_fields = [
            field_name
            for field_name in CONFIG_FIELD_ORDER
            if actual_config.get(field_name)
            != expected_config.get(field_name)
        ]

        if changed_fields:
            errors.append(
                "stale or manually modified config: "
                f"{path.relative_to(PROJECT_ROOT)}; "
                f"changed fields: {changed_fields}"
            )

    if errors:
        print(
            "Ablation configuration check failed:"
        )

        for error in errors:
            print(f"- {error}")

        return 1

    print(
        "Ablation configuration check passed: "
        f"{len(expected)} configs"
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify M1 single-variable "
            "ablation configurations."
        )
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify that generated configurations "
            "are complete and unchanged."
        ),
    )

    args = parser.parse_args()

    if args.check:
        return check_configs()

    return write_configs()


if __name__ == "__main__":
    raise SystemExit(main())