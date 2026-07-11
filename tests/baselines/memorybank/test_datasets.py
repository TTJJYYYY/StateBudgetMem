from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from statebudgetmem.baselines.memorybank.datasets import (
    REPRODUCTION_QUESTION_TYPES,
    REQUIRED_PER_USER_QUESTION_TYPES,
    build_reproduction_memory_catalog,
    load_reproduction_dataset,
    reproduction_dataset_stats,
    validate_reproduction_probes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "data" / "memorybank_reproduction"


def _load_formal_dataset():
    return load_reproduction_dataset(DATASET_DIR)


def test_formal_reproduction_dataset_meets_phase1_scale() -> None:
    users, probes = _load_formal_dataset()
    stats = reproduction_dataset_stats(users, probes)

    assert stats["user_count"] == 5
    assert stats["user_day_count"] == 35
    assert stats["probe_count"] == 50
    assert stats["probes_per_user"] == {
        "user_001": 10,
        "user_002": 10,
        "user_003": 10,
        "user_004": 10,
        "user_005": 10,
    }


def test_all_probe_ids_and_gold_memory_ids_are_valid() -> None:
    users, probes = _load_formal_dataset()
    catalog = build_reproduction_memory_catalog(users)

    query_ids = [probe.query_id for probe in probes]
    assert len(query_ids) == len(set(query_ids))

    for probe in probes:
        assert probe.user_id in {user.user_id for user in users}
        assert probe.question_type in REPRODUCTION_QUESTION_TYPES
        assert probe.reference_answer
        assert probe.expected_keywords
        for memory_id in probe.gold_memory_ids:
            assert memory_id in catalog
            assert catalog[memory_id]["user_id"] == probe.user_id


def test_question_types_have_required_global_and_per_user_coverage() -> None:
    users, probes = _load_formal_dataset()

    assert {probe.question_type for probe in probes} == set(
        REPRODUCTION_QUESTION_TYPES
    )

    for user in users:
        user_types = {
            probe.question_type
            for probe in probes
            if probe.user_id == user.user_id
        }
        assert REQUIRED_PER_USER_QUESTION_TYPES <= user_types

    counts = Counter(probe.question_type for probe in probes)
    assert counts["negative_memory"] == 5
    assert counts["temporal_memory"] >= 5
    assert counts["user_portrait"] >= 5


def test_negative_questions_have_no_positive_gold_and_specific_keywords() -> None:
    _, probes = _load_formal_dataset()
    negative_probes = [
        probe for probe in probes if probe.question_type == "negative_memory"
    ]

    assert len(negative_probes) == 5
    for probe in negative_probes:
        assert probe.gold_memory_ids == []
        assert len(probe.expected_keywords) >= 2
        assert any(keyword.casefold() == "no" for keyword in probe.expected_keywords)
        assert any(keyword.casefold() != "no" for keyword in probe.expected_keywords)


def test_expected_keywords_are_supported_by_reference_answers() -> None:
    _, probes = _load_formal_dataset()

    for probe in probes:
        reference = probe.reference_answer.casefold()
        for keyword in probe.expected_keywords:
            assert keyword.casefold() in reference


def test_validator_rejects_unknown_gold_memory_id() -> None:
    users, probes = _load_formal_dataset()
    invalid = [
        replace(probes[0], gold_memory_ids=["user_001_missing_memory"]),
        *probes[1:],
    ]

    with pytest.raises(ValueError, match="unknown gold memory ID"):
        validate_reproduction_probes(users, invalid)


def test_validator_rejects_duplicate_query_id() -> None:
    users, probes = _load_formal_dataset()
    invalid = [probes[0], replace(probes[1], query_id=probes[0].query_id), *probes[2:]]

    with pytest.raises(ValueError, match="Duplicate query_id"):
        validate_reproduction_probes(users, invalid)


def test_validator_rejects_negative_probe_with_positive_evidence() -> None:
    users, probes = _load_formal_dataset()
    index = next(
        i for i, probe in enumerate(probes) if probe.question_type == "negative_memory"
    )
    invalid_probe = replace(
        probes[index],
        gold_memory_ids=[f"{probes[index].user_id}_day01_dialog01"],
    )
    invalid = [*probes[:index], invalid_probe, *probes[index + 1 :]]

    with pytest.raises(ValueError, match="must have empty gold_memory_ids"):
        validate_reproduction_probes(users, invalid)


def test_validator_rejects_missing_per_user_temporal_coverage() -> None:
    users, probes = _load_formal_dataset()
    invalid = [
        probe
        for probe in probes
        if not (
            probe.user_id == "user_003"
            and probe.question_type == "temporal_memory"
        )
    ]

    with pytest.raises(ValueError, match="Per-user B3 question coverage"):
        validate_reproduction_probes(users, invalid)


def test_validator_rejects_keywords_not_grounded_in_reference_answer() -> None:
    users, probes = _load_formal_dataset()
    invalid = [
        replace(probes[0], expected_keywords=["Java"]),
        *probes[1:],
    ]

    with pytest.raises(ValueError, match="not supported by reference_answer"):
        validate_reproduction_probes(users, invalid)


def test_validator_rejects_negative_claim_present_in_memory() -> None:
    users, probes = _load_formal_dataset()
    index = next(
        i for i, probe in enumerate(probes) if probe.query_id == "q009"
    )
    invalid_probe = replace(
        probes[index],
        reference_answer="No, I did not mention Python.",
        expected_keywords=["No", "Python"],
    )
    invalid = [*probes[:index], invalid_probe, *probes[index + 1 :]]

    with pytest.raises(ValueError, match="contradicted by stored memory text"):
        validate_reproduction_probes(users, invalid)