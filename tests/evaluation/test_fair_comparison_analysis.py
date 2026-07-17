from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from statebudgetmem.evaluation.fair_comparison_analysis import (
    EXPECTED_METHODS,
    analyze_fair_comparison,
    build_grouped_rows,
    validate_records,
)


def test_grouped_rows_include_empty_general_placeholder() -> None:
    records = _records()
    rows = build_grouped_rows(records)

    assert len(rows) == len(EXPECTED_METHODS) * 4
    general_rows = [
        row for row in rows if row["query_type"] == "GENERAL"
    ]
    assert len(general_rows) == len(EXPECTED_METHODS)
    assert all(row["query_count"] == 0 for row in general_rows)
    assert all(
        row["mean_recall_at_k"] is None
        for row in general_rows
    )


def test_validate_records_rejects_query_set_mismatch() -> None:
    records = _records()
    records = [
        record
        for record in records
        if not (
            record["method"] == "tfidf_topk"
            and record["query_id"] == "Q_CHANGE"
        )
    ]

    with pytest.raises(ValueError, match="query set mismatch|expected 3"):
        validate_records(
            records,
            expected_queries_per_method=3,
        )


def test_analysis_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "per_query_results.jsonl"
    with input_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for record in _records():
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    output_dir = tmp_path / "analysis"
    metadata = analyze_fair_comparison(
        input_path,
        output_dir,
        expected_queries_per_method=3,
    )

    expected = {
        "by_query_type.csv",
        "by_query_type.json",
        "routing_confusion.csv",
        "dual_views_equivalence.csv",
        "rule_error_cases.jsonl",
        "analysis_metadata.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert metadata["total_record_count"] == 18
    assert (
        metadata["dual_views_equivalence"][
            "retrieved_ids_equivalence_rate"
        ]
        == 1.0
    )
    assert (
        metadata["routing"]["overall_routing_accuracy"]
        == pytest.approx(1 / 3)
    )

    with (
        output_dir / "routing_confusion.csv"
    ).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    historical = next(
        row
        for row in rows
        if row["gold_query_type"] == "HISTORICAL"
    )
    assert historical["GENERAL"] == "1"

    error_cases = [
        json.loads(line)
        for line in (
            output_dir / "rule_error_cases.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    patterns = {
        pattern
        for case in error_cases
        for pattern in case["error_patterns"]
    }
    assert "fallback_to_general" in patterns
    assert "change_pattern_too_literal" in patterns
    assert "correct_route_but_dense_miss" in patterns


def _records() -> list[dict[str, object]]:
    query_specs = (
        ("Q_CURRENT", "CURRENT", "我现在怎么通勤？"),
        ("Q_HISTORICAL", "HISTORICAL", "我以前怎么通勤？"),
        ("Q_CHANGE", "CHANGE", "我从以前到现在怎么变了？"),
    )

    records: list[dict[str, object]] = []
    for method in EXPECTED_METHODS:
        for query_id, query_type, query_text in query_specs:
            predicted: str | None = None
            recall = 0.5
            valid_recall = 0.5
            eligible = 2
            retrieved = [f"{query_id}_M1"]
            source_view = "flat"
            selection_policy = None

            if method == "statebudgetmem_oracle":
                predicted = query_type
                recall = 1.0
                valid_recall = 1.0
                eligible = 2
                source_view = query_type.lower()
            elif method == "statebudgetmem_rule":
                if query_type == "CURRENT":
                    predicted = "CURRENT"
                    eligible = 1
                elif query_type == "HISTORICAL":
                    predicted = "GENERAL"
                    recall = 0.0
                    valid_recall = 0.0
                    eligible = 0
                    retrieved = []
                else:
                    predicted = "CURRENT"
                    recall = 0.0
                    valid_recall = 0.0
                    eligible = 1
                source_view = (
                    predicted.lower()
                    if predicted is not None
                    else None
                )
                selection_policy = "rule_routed"
            elif method == "memorybank_dual_views":
                source_view = "current_and_history"
                selection_policy = (
                    "current_and_history_no_router"
                )

            records.append(
                {
                    "method": method,
                    "run_id": f"run-{method}",
                    "scenario_id": "S1",
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_type": query_type,
                    "predicted_query_type": predicted,
                    "repeat_index": 0,
                    "source_view": source_view,
                    "selection_policy": selection_policy,
                    "eligible_memory_count": eligible,
                    "candidate_k": 20,
                    "retrieved_memory_ids": retrieved,
                    "retrieved_scores": [0.8] if retrieved else [],
                    "recall_at_k": recall,
                    "valid_recall_at_k": valid_recall,
                    "stale_retrieval_rate": 0.0,
                    "total_token_cost": 10 if retrieved else 0,
                    "ingest_latency_ms": 1.0,
                    "retrieval_latency_ms": 2.0,
                    "top_k": 3,
                    "token_budget": 256,
                    "random_seed": 42,
                    "status": "success",
                    "error": None,
                }
            )
    return records
