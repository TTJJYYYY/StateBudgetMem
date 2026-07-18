from __future__ import annotations

import json
from pathlib import Path

import pytest

from statebudgetmem.data.validation import (
    audit_dataset,
    validate_dataset_manifest,
)


DATASET_PATH = Path(
    "data/controlled/temporal_challenge_v1.jsonl"
)
MANIFEST_PATH = Path(
    "data/controlled/manifests/"
    "temporal_challenge_v1.manifest.json"
)


def test_temporal_challenge_v1_audit() -> None:
    audit = audit_dataset(DATASET_PATH)

    assert audit["scenario_count"] == 32
    assert audit["memory_count"] == 193
    assert audit["query_count"] == 96

    assert audit["query_type_counts"] == {
        "CURRENT": 32,
        "HISTORICAL": 32,
        "CHANGE": 32,
        "GENERAL": 0,
    }

    assert audit["memory_status_counts"] == {
        "CURRENT": 147,
        "HISTORICAL": 38,
        "INVALIDATED": 4,
        "UNKNOWN": 4,
    }

    assert audit["relation_counts"] == {
        "supersedes_edges": 25,
        "temporarily_invalidates_edges": 11,
    }

    assert audit["sha256"] == (
        "f93331a2d93588fa8931efb4484fce577f5a5c9c4e679c51"
        "d4bb0192af6c8dd9"
    )


def test_temporal_challenge_v1_manifest_matches() -> None:
    audit = validate_dataset_manifest(
        DATASET_PATH,
        MANIFEST_PATH,
    )

    assert audit["scenario_count"] == 32
    assert audit["query_type_counts"]["GENERAL"] == 0


def test_checksum_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )
    manifest["sha256"] = "0" * 64

    invalid_manifest_path = tmp_path / "invalid_manifest.json"
    invalid_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="sha256",
    ):
        validate_dataset_manifest(
            DATASET_PATH,
            invalid_manifest_path,
        )


def test_absolute_dataset_path_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )
    manifest["dataset_path"] = str(DATASET_PATH.resolve())

    invalid_manifest_path = tmp_path / "absolute_path.json"
    invalid_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="repository-relative",
    ):
        validate_dataset_manifest(
            DATASET_PATH,
            invalid_manifest_path,
        )