from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from statebudgetmem.data.io import load_scenarios
from statebudgetmem.schemas import MemoryStatus, QueryType, Scenario


def _find_duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def calculate_sha256(path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")

    digest = sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_relation_targets(scenario: Scenario) -> None:
    """Ensure all version-relation targets exist in the same scenario."""

    memory_ids = {memory.memory_id for memory in scenario.memories}

    for memory in scenario.memories:
        relation_fields = {
            "supersedes": memory.supersedes,
            "temporarily_invalidates": memory.temporarily_invalidates,
        }

        for relation_name, target_ids in relation_fields.items():
            unknown_ids = sorted(
                target_id
                for target_id in target_ids
                if target_id not in memory_ids
            )

            if unknown_ids:
                raise ValueError(
                    f"scenario {scenario.scenario_id}, memory "
                    f"{memory.memory_id}: {relation_name} references "
                    f"unknown memory ids: {unknown_ids}"
                )

            if memory.memory_id in target_ids:
                raise ValueError(
                    f"scenario {scenario.scenario_id}, memory "
                    f"{memory.memory_id}: {relation_name} cannot "
                    "reference the memory itself"
                )


def audit_dataset(path: str | Path) -> dict[str, Any]:
    """Load and audit a controlled Scenario JSONL dataset.

    The shared Pydantic schemas perform record-level validation.
    This function adds dataset-level validation and statistics.
    """

    dataset_path = Path(path)
    scenarios = load_scenarios(dataset_path)

    if not scenarios:
        raise ValueError(f"dataset contains no scenarios: {dataset_path}")

    scenario_ids: list[str] = []
    memory_ids: list[str] = []
    query_ids: list[str] = []

    query_type_counts: Counter[str] = Counter()
    memory_status_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()

    for scenario in scenarios:
        scenario_ids.append(scenario.scenario_id)
        _validate_relation_targets(scenario)

        for memory in scenario.memories:
            memory_ids.append(memory.memory_id)
            memory_status_counts[memory.status.value] += 1

            relation_counts["supersedes_edges"] += len(
                memory.supersedes
            )
            relation_counts["temporarily_invalidates_edges"] += len(
                memory.temporarily_invalidates
            )

        for query in scenario.queries:
            query_ids.append(query.query_id)
            query_type_counts[query.query_type.value] += 1

    duplicate_scenario_ids = _find_duplicates(scenario_ids)
    duplicate_memory_ids = _find_duplicates(memory_ids)
    duplicate_query_ids = _find_duplicates(query_ids)

    if duplicate_scenario_ids:
        raise ValueError(
            "duplicate scenario_id values across dataset: "
            f"{duplicate_scenario_ids}"
        )

    if duplicate_memory_ids:
        raise ValueError(
            "duplicate memory_id values across dataset: "
            f"{duplicate_memory_ids}"
        )

    if duplicate_query_ids:
        raise ValueError(
            "duplicate query_id values across dataset: "
            f"{duplicate_query_ids}"
        )

    # Include zero-count enum values so coverage is always explicit.
    complete_query_type_counts = {
        query_type.value: query_type_counts.get(query_type.value, 0)
        for query_type in QueryType
    }
    complete_memory_status_counts = {
        status.value: memory_status_counts.get(status.value, 0)
        for status in MemoryStatus
    }

    return {
        "dataset_path": dataset_path.as_posix(),
        "sha256": calculate_sha256(dataset_path),
        "scenario_count": len(scenarios),
        "memory_count": len(memory_ids),
        "query_count": len(query_ids),
        "query_type_counts": complete_query_type_counts,
        "memory_status_counts": complete_memory_status_counts,
        "relation_counts": {
            "supersedes_edges": relation_counts.get(
                "supersedes_edges", 0
            ),
            "temporarily_invalidates_edges": relation_counts.get(
                "temporarily_invalidates_edges", 0
            ),
        },
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a dataset manifest JSON file."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path}"
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid manifest JSON at {manifest_path}: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a JSON object")

    return manifest


def validate_dataset_manifest(
    dataset_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate that a manifest exactly describes a dataset."""

    audit = audit_dataset(dataset_path)
    manifest = load_manifest(manifest_path)

    manifest_dataset_path = manifest.get("dataset_path")
    if not isinstance(manifest_dataset_path, str):
        raise ValueError(
            "manifest field 'dataset_path' must be a string"
        )

    if Path(manifest_dataset_path).is_absolute():
        raise ValueError(
            "manifest dataset_path must be repository-relative, "
            f"got absolute path: {manifest_dataset_path}"
        )

    expected_fields = {
        "dataset_path": audit["dataset_path"],
        "sha256": audit["sha256"],
        "scenario_count": audit["scenario_count"],
        "memory_count": audit["memory_count"],
        "query_count": audit["query_count"],
        "query_type_counts": audit["query_type_counts"],
        "memory_status_counts": audit["memory_status_counts"],
        "relation_counts": audit["relation_counts"],
    }

    mismatches: list[str] = []

    for field_name, expected_value in expected_fields.items():
        actual_value = manifest.get(field_name)

        if actual_value != expected_value:
            mismatches.append(
                f"{field_name}: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )

    if manifest.get("frozen") is not True:
        mismatches.append(
            "frozen: expected True for a versioned experiment dataset"
        )

    if mismatches:
        details = "\n- ".join(mismatches)
        raise ValueError(
            "dataset manifest does not match dataset:\n- "
            f"{details}"
        )

    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a controlled StateBudgetMem dataset and "
            "its manifest."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Repository-relative path to the JSONL dataset.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Repository-relative path to the manifest JSON.",
    )
    args = parser.parse_args()

    audit = validate_dataset_manifest(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
    )

    print("Dataset validation passed.")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())