from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


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
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "configs/fair_experiments/ablations"
)

# 主实验基线为：
# top_k=3, candidate_k=20, token_budget=32。
# 这里只生成非基线值。
ABLATION_SWEEPS = {
    "top_k": (1, 5),
    "candidate_k": (5, 40),
    "token_budget": (16, 64),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"config root must be a mapping: {path}"
        )

    return data


def _output_filename(
    method_name: str,
    parameter_name: str,
    value: int,
) -> str:
    return (
        f"m1_{method_name}_"
        f"{parameter_name}_{value}.yaml"
    )


def build_expected_configs() -> dict[Path, dict[str, Any]]:
    expected: dict[Path, dict[str, Any]] = {}

    for method_name, base_path in BASE_CONFIGS.items():
        base_config = _load_yaml(base_path)

        for parameter_name, values in ABLATION_SWEEPS.items():
            for value in values:
                config = deepcopy(base_config)
                config[parameter_name] = value
                config["results_dir"] = (
                    "results/fair_experiments/m1/"
                    f"ablations/{parameter_name}/{value}/"
                    f"{method_name}"
                )

                output_path = OUTPUT_DIR / _output_filename(
                    method_name,
                    parameter_name,
                    value,
                )
                expected[output_path] = config

    return expected


def write_configs() -> int:
    expected = build_expected_configs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_paths = set(expected)

    # 只清理本脚本生成模式对应的旧 YAML。
    for existing_path in OUTPUT_DIR.glob("m1_*.yaml"):
        if existing_path not in expected_paths:
            existing_path.unlink()

    for output_path, config in expected.items():
        with output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            yaml.safe_dump(
                config,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )

    print(
        f"Generated {len(expected)} M1 ablation configs "
        f"in {OUTPUT_DIR.relative_to(PROJECT_ROOT)}"
    )
    return 0


def check_configs() -> int:
    expected = build_expected_configs()
    expected_paths = set(expected)
    actual_paths = set(OUTPUT_DIR.glob("m1_*.yaml"))

    missing_paths = sorted(expected_paths - actual_paths)
    unexpected_paths = sorted(actual_paths - expected_paths)

    errors: list[str] = []

    for path in missing_paths:
        errors.append(
            f"missing config: {path.relative_to(PROJECT_ROOT)}"
        )

    for path in unexpected_paths:
        errors.append(
            f"unexpected config: {path.relative_to(PROJECT_ROOT)}"
        )

    for path, expected_config in expected.items():
        if not path.is_file():
            continue

        actual_config = _load_yaml(path)

        if actual_config != expected_config:
            errors.append(
                "stale or manually modified config: "
                f"{path.relative_to(PROJECT_ROOT)}"
            )

    if errors:
        print("Ablation configuration check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Ablation configuration check passed: "
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
            "Check that generated configs are complete and "
            "match their base configurations."
        ),
    )
    args = parser.parse_args()

    if args.check:
        return check_configs()

    return write_configs()


if __name__ == "__main__":
    raise SystemExit(main())
