from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

EXPECTED_METHODS = (
    "tfidf_topk",
    "memorybank_core",
    "memorybank_versioning",
    "memorybank_dual_views",
    "statebudgetmem_rule",
    "statebudgetmem_oracle",
)

QUERY_TYPES = ("CURRENT", "HISTORICAL", "CHANGE", "GENERAL")
PREDICTED_QUERY_TYPES = (*QUERY_TYPES, "NONE")

GROUPED_FIELDS = (
    "method",
    "query_type",
    "query_count",
    "mean_recall_at_k",
    "mean_valid_recall_at_k",
    "mean_stale_retrieval_rate",
    "empty_retrieval_rate",
    "no_retrieval_rate",
    "mean_retrieved_count",
    "mean_total_token_cost",
    "mean_ingest_latency_ms",
    "mean_retrieval_latency_ms",
)

DUAL_FIELDS = (
    "scenario_id",
    "query_id",
    "query_type",
    "repeat_index",
    "same_retrieved_ids",
    "same_scores",
    "same_recall",
    "same_valid_recall",
    "same_stale_rate",
    "same_all",
    "core_retrieved_ids",
    "dual_retrieved_ids",
    "core_eligible_memory_count",
    "dual_eligible_memory_count",
    "dual_source_view",
    "dual_selection_policy",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"input JSONL not found: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {input_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"expected JSON object at {input_path}:{line_number}"
                )
            records.append(payload)

    if not records:
        raise ValueError(f"input JSONL is empty: {input_path}")
    return records


def validate_records(
    records: list[dict[str, Any]],
    *,
    expected_methods: Iterable[str] = EXPECTED_METHODS,
    expected_queries_per_method: int | None = None,
) -> dict[str, Any]:
    methods = tuple(expected_methods)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, record in enumerate(records):
        method = record.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError(f"record {index} has no valid method")
        by_method[method].append(record)

        if record.get("status") != "success":
            raise ValueError(
                f"{method}/{record.get('query_id')}: "
                f"status={record.get('status')!r}, error={record.get('error')!r}"
            )

    actual_methods = set(by_method)
    expected_set = set(methods)
    if actual_methods != expected_set:
        raise ValueError(
            "method set mismatch: "
            f"missing={sorted(expected_set - actual_methods)}, "
            f"extra={sorted(actual_methods - expected_set)}"
        )

    query_keys_by_method: dict[str, set[tuple[str, int]]] = {}
    query_types_by_key: dict[tuple[str, int], str] = {}

    for method in methods:
        method_records = by_method[method]
        if (
            expected_queries_per_method is not None
            and len(method_records) != expected_queries_per_method
        ):
            raise ValueError(
                f"{method}: expected {expected_queries_per_method} records, "
                f"got {len(method_records)}"
            )

        keys: set[tuple[str, int]] = set()
        for record in method_records:
            query_id = record.get("query_id")
            repeat_index = int(record.get("repeat_index", 0))
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"{method}: record has no valid query_id")

            key = (query_id, repeat_index)
            if key in keys:
                raise ValueError(f"{method}: duplicate query key {key}")
            keys.add(key)

            query_type = record.get("query_type")
            if query_type not in QUERY_TYPES:
                raise ValueError(
                    f"{method}/{query_id}: invalid query_type={query_type!r}"
                )

            previous_type = query_types_by_key.setdefault(key, query_type)
            if previous_type != query_type:
                raise ValueError(
                    f"{key}: inconsistent query types "
                    f"{previous_type!r} and {query_type!r}"
                )

        query_keys_by_method[method] = keys

    reference_method = methods[0]
    reference_keys = query_keys_by_method[reference_method]
    for method, keys in query_keys_by_method.items():
        if keys != reference_keys:
            raise ValueError(
                f"{method}: query set mismatch; "
                f"missing={sorted(reference_keys - keys)[:5]}, "
                f"extra={sorted(keys - reference_keys)[:5]}"
            )

    common_fields: dict[str, list[Any]] = {}
    for field in ("top_k", "token_budget", "random_seed"):
        values = sorted(
            {record.get(field) for record in records},
            key=lambda value: (value is None, str(value)),
        )
        common_fields[field] = values
        if len(values) != 1:
            raise ValueError(f"inconsistent {field}: {values}")

    candidate_k_values = sorted(
        {
            record.get("candidate_k")
            for record in records
            if record.get("candidate_k") is not None
        }
    )

    query_type_counts = Counter(query_types_by_key.values())
    return {
        "method_count": len(methods),
        "methods": list(methods),
        "query_count_per_method": len(reference_keys),
        "total_record_count": len(records),
        "query_type_counts": {
            query_type: query_type_counts.get(query_type, 0)
            for query_type in QUERY_TYPES
        },
        "top_k_values": common_fields["top_k"],
        "token_budget_values": common_fields["token_budget"],
        "random_seed_values": common_fields["random_seed"],
        "candidate_k_values_non_null": candidate_k_values,
        "fairness_validation_passed": True,
    }


def build_grouped_rows(
    records: list[dict[str, Any]],
    *,
    expected_methods: Iterable[str] = EXPECTED_METHODS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in expected_methods:
        method_records = [
            record for record in records if record["method"] == method
        ]
        for query_type in QUERY_TYPES:
            selected = [
                record
                for record in method_records
                if record["query_type"] == query_type
            ]
            query_count = len(selected)

            if query_count == 0:
                rows.append(
                    {
                        "method": method,
                        "query_type": query_type,
                        "query_count": 0,
                        "mean_recall_at_k": None,
                        "mean_valid_recall_at_k": None,
                        "mean_stale_retrieval_rate": None,
                        "empty_retrieval_rate": None,
                        "no_retrieval_rate": None,
                        "mean_retrieved_count": None,
                        "mean_total_token_cost": None,
                        "mean_ingest_latency_ms": None,
                        "mean_retrieval_latency_ms": None,
                    }
                )
                continue

            retrieved_counts = [
                len(record.get("retrieved_memory_ids") or [])
                for record in selected
            ]
            empty_rate = mean(count == 0 for count in retrieved_counts)
            rows.append(
                {
                    "method": method,
                    "query_type": query_type,
                    "query_count": query_count,
                    "mean_recall_at_k": _mean_field(selected, "recall_at_k"),
                    "mean_valid_recall_at_k": _mean_field(
                        selected, "valid_recall_at_k"
                    ),
                    "mean_stale_retrieval_rate": _mean_field(
                        selected, "stale_retrieval_rate"
                    ),
                    "empty_retrieval_rate": empty_rate,
                    "no_retrieval_rate": (
                        empty_rate if query_type == "GENERAL" else None
                    ),
                    "mean_retrieved_count": mean(retrieved_counts),
                    "mean_total_token_cost": _mean_field(
                        selected, "total_token_cost"
                    ),
                    "mean_ingest_latency_ms": _mean_field(
                        selected, "ingest_latency_ms"
                    ),
                    "mean_retrieval_latency_ms": _mean_field(
                        selected, "retrieval_latency_ms"
                    ),
                }
            )
    return rows


def build_routing_confusion(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rule_records = [
        record
        for record in records
        if record["method"] == "statebudgetmem_rule"
    ]
    counts: dict[str, Counter[str]] = {
        query_type: Counter() for query_type in QUERY_TYPES
    }

    correct = 0
    correct_by_type = Counter()
    total_by_type = Counter()

    for record in rule_records:
        gold = record["query_type"]
        predicted = record.get("predicted_query_type") or "NONE"
        if predicted not in PREDICTED_QUERY_TYPES:
            predicted = "NONE"

        counts[gold][predicted] += 1
        total_by_type[gold] += 1
        if predicted == gold:
            correct += 1
            correct_by_type[gold] += 1

    matrix_rows: list[dict[str, Any]] = []
    for gold in QUERY_TYPES:
        row: dict[str, Any] = {"gold_query_type": gold}
        for predicted in PREDICTED_QUERY_TYPES:
            row[predicted] = counts[gold][predicted]
        row["total"] = sum(counts[gold].values())
        matrix_rows.append(row)

    total_row: dict[str, Any] = {"gold_query_type": "TOTAL"}
    for predicted in PREDICTED_QUERY_TYPES:
        total_row[predicted] = sum(
            counts[gold][predicted] for gold in QUERY_TYPES
        )
    total_row["total"] = len(rule_records)
    matrix_rows.append(total_row)

    accuracy_by_type = {
        query_type: (
            correct_by_type[query_type] / total_by_type[query_type]
            if total_by_type[query_type]
            else None
        )
        for query_type in QUERY_TYPES
    }
    summary = {
        "query_count": len(rule_records),
        "overall_routing_accuracy": (
            correct / len(rule_records) if rule_records else None
        ),
        "routing_accuracy_by_query_type": accuracy_by_type,
    }
    return matrix_rows, summary


def build_dual_equivalence(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = _index_method(records, "memorybank_core")
    dual = _index_method(records, "memorybank_dual_views")

    if core.keys() != dual.keys():
        raise ValueError("core and dual views query sets do not match")

    rows: list[dict[str, Any]] = []
    for key in sorted(core):
        core_record = core[key]
        dual_record = dual[key]

        same_ids = (
            core_record.get("retrieved_memory_ids") or []
        ) == (dual_record.get("retrieved_memory_ids") or [])
        same_scores = _float_lists_close(
            core_record.get("retrieved_scores") or [],
            dual_record.get("retrieved_scores") or [],
        )
        same_recall = _numbers_close(
            core_record.get("recall_at_k"),
            dual_record.get("recall_at_k"),
        )
        same_valid_recall = _numbers_close(
            core_record.get("valid_recall_at_k"),
            dual_record.get("valid_recall_at_k"),
        )
        same_stale = _numbers_close(
            core_record.get("stale_retrieval_rate"),
            dual_record.get("stale_retrieval_rate"),
        )

        rows.append(
            {
                "scenario_id": core_record.get("scenario_id"),
                "query_id": core_record["query_id"],
                "query_type": core_record["query_type"],
                "repeat_index": core_record.get("repeat_index", 0),
                "same_retrieved_ids": same_ids,
                "same_scores": same_scores,
                "same_recall": same_recall,
                "same_valid_recall": same_valid_recall,
                "same_stale_rate": same_stale,
                "same_all": (
                    same_ids
                    and same_scores
                    and same_recall
                    and same_valid_recall
                    and same_stale
                ),
                "core_retrieved_ids": core_record.get(
                    "retrieved_memory_ids", []
                ),
                "dual_retrieved_ids": dual_record.get(
                    "retrieved_memory_ids", []
                ),
                "core_eligible_memory_count": core_record.get(
                    "eligible_memory_count"
                ),
                "dual_eligible_memory_count": dual_record.get(
                    "eligible_memory_count"
                ),
                "dual_source_view": dual_record.get("source_view"),
                "dual_selection_policy": dual_record.get(
                    "selection_policy"
                ),
            }
        )

    summary = {
        "query_count": len(rows),
        "retrieved_ids_equivalence_rate": _bool_rate(
            rows, "same_retrieved_ids"
        ),
        "scores_equivalence_rate": _bool_rate(rows, "same_scores"),
        "recall_equivalence_rate": _bool_rate(rows, "same_recall"),
        "valid_recall_equivalence_rate": _bool_rate(
            rows, "same_valid_recall"
        ),
        "stale_rate_equivalence_rate": _bool_rate(
            rows, "same_stale_rate"
        ),
        "all_equivalence_rate": _bool_rate(rows, "same_all"),
        "all_equivalence_rate_by_query_type": {
            query_type: _bool_rate(
                [row for row in rows if row["query_type"] == query_type],
                "same_all",
            )
            for query_type in QUERY_TYPES
        },
    }
    return rows, summary


def build_rule_error_cases(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rule = _index_method(records, "statebudgetmem_rule")
    oracle = _index_method(records, "statebudgetmem_oracle")

    if rule.keys() != oracle.keys():
        raise ValueError("rule and oracle query sets do not match")

    cases: list[dict[str, Any]] = []
    pattern_counts: Counter[str] = Counter()

    for key in sorted(rule):
        rule_record = rule[key]
        oracle_record = oracle[key]

        recall_gap = float(oracle_record["recall_at_k"]) - float(
            rule_record["recall_at_k"]
        )
        valid_recall_gap = float(
            oracle_record["valid_recall_at_k"]
        ) - float(rule_record["valid_recall_at_k"])

        gold = rule_record["query_type"]
        predicted = rule_record.get("predicted_query_type")
        route_correct = predicted == gold

        if (
            recall_gap <= 1e-12
            and valid_recall_gap <= 1e-12
            and route_correct
        ):
            continue

        patterns = _classify_rule_error(
            rule_record,
            oracle_record,
            recall_gap=recall_gap,
            valid_recall_gap=valid_recall_gap,
        )
        for pattern in patterns:
            pattern_counts[pattern] += 1

        cases.append(
            {
                "scenario_id": rule_record.get("scenario_id"),
                "query_id": rule_record["query_id"],
                "query_text": rule_record.get("query_text"),
                "query_type": gold,
                "predicted_query_type": predicted,
                "route_correct": route_correct,
                "primary_error_pattern": patterns[0],
                "error_patterns": patterns,
                "rule_retrieved_memory_ids": rule_record.get(
                    "retrieved_memory_ids", []
                ),
                "oracle_retrieved_memory_ids": oracle_record.get(
                    "retrieved_memory_ids", []
                ),
                "rule_recall_at_k": rule_record.get("recall_at_k"),
                "oracle_recall_at_k": oracle_record.get("recall_at_k"),
                "recall_gap": recall_gap,
                "rule_valid_recall_at_k": rule_record.get(
                    "valid_recall_at_k"
                ),
                "oracle_valid_recall_at_k": oracle_record.get(
                    "valid_recall_at_k"
                ),
                "valid_recall_gap": valid_recall_gap,
                "rule_stale_retrieval_rate": rule_record.get(
                    "stale_retrieval_rate"
                ),
                "oracle_stale_retrieval_rate": oracle_record.get(
                    "stale_retrieval_rate"
                ),
                "rule_total_token_cost": rule_record.get(
                    "total_token_cost"
                ),
                "oracle_total_token_cost": oracle_record.get(
                    "total_token_cost"
                ),
                "rule_eligible_memory_count": rule_record.get(
                    "eligible_memory_count"
                ),
                "oracle_eligible_memory_count": oracle_record.get(
                    "eligible_memory_count"
                ),
                "rule_source_view": rule_record.get("source_view"),
                "oracle_source_view": oracle_record.get("source_view"),
                "rule_selection_policy": rule_record.get(
                    "selection_policy"
                ),
            }
        )

    cases.sort(
        key=lambda case: (
            -float(case["valid_recall_gap"]),
            -float(case["recall_gap"]),
            case["query_id"],
        )
    )
    summary = {
        "error_case_count": len(cases),
        "error_pattern_counts": dict(pattern_counts.most_common()),
        "mean_recall_gap_in_error_cases": (
            mean(float(case["recall_gap"]) for case in cases)
            if cases
            else 0.0
        ),
        "mean_valid_recall_gap_in_error_cases": (
            mean(float(case["valid_recall_gap"]) for case in cases)
            if cases
            else 0.0
        ),
    }
    return cases, summary


def analyze_fair_comparison(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    expected_methods: Iterable[str] = EXPECTED_METHODS,
    expected_queries_per_method: int | None = 96,
) -> dict[str, Any]:
    records = read_jsonl(input_path)
    validation = validate_records(
        records,
        expected_methods=expected_methods,
        expected_queries_per_method=expected_queries_per_method,
    )
    grouped_rows = build_grouped_rows(
        records, expected_methods=expected_methods
    )
    confusion_rows, routing_summary = build_routing_confusion(records)
    dual_rows, dual_summary = build_dual_equivalence(records)
    rule_error_cases, rule_error_summary = build_rule_error_cases(records)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    _write_csv(target / "by_query_type.csv", grouped_rows, GROUPED_FIELDS)
    _write_json(target / "by_query_type.json", grouped_rows)
    _write_csv(
        target / "routing_confusion.csv",
        confusion_rows,
        ("gold_query_type", *PREDICTED_QUERY_TYPES, "total"),
    )
    _write_csv(
        target / "dual_views_equivalence.csv",
        dual_rows,
        DUAL_FIELDS,
    )
    _write_jsonl(target / "rule_error_cases.jsonl", rule_error_cases)

    metadata = {
        **validation,
        "input_path": str(Path(input_path)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "general_query_note": (
            "GENERAL has no samples in temporal_challenge_v1; "
            "its grouped metrics are null and no performance "
            "conclusion should be drawn."
        ),
        "routing": routing_summary,
        "dual_views_equivalence": dual_summary,
        "rule_error_analysis": rule_error_summary,
        "outputs": {
            "by_query_type_csv": str(target / "by_query_type.csv"),
            "by_query_type_json": str(target / "by_query_type.json"),
            "routing_confusion_csv": str(
                target / "routing_confusion.csv"
            ),
            "dual_views_equivalence_csv": str(
                target / "dual_views_equivalence.csv"
            ),
            "rule_error_cases_jsonl": str(
                target / "rule_error_cases.jsonl"
            ),
        },
    }
    _write_json(target / "analysis_metadata.json", metadata)
    return metadata


def _classify_rule_error(
    rule_record: dict[str, Any],
    oracle_record: dict[str, Any],
    *,
    recall_gap: float,
    valid_recall_gap: float,
) -> list[str]:
    gold = rule_record["query_type"]
    predicted = rule_record.get("predicted_query_type")
    text = str(rule_record.get("query_text") or "")
    patterns: list[str] = []

    if predicted is None:
        patterns.append("missing_prediction")
    elif predicted == "GENERAL" and gold != "GENERAL":
        if "加" in text or "减" in text:
            patterns.append("general_keyword_false_positive")
        else:
            patterns.append("fallback_to_general")
    elif predicted != gold:
        if (
            gold == "CHANGE"
            and predicted != "CHANGE"
            and (
                ("从" in text and "到" in text)
                or ("还" in text and "吗" in text)
            )
        ):
            patterns.append("change_pattern_too_literal")
        patterns.append("wrong_temporal_view")

    rule_eligible = rule_record.get("eligible_memory_count")
    oracle_eligible = oracle_record.get("eligible_memory_count")
    if (
        recall_gap > 1e-12 or valid_recall_gap > 1e-12
    ) and _less_than(rule_eligible, oracle_eligible):
        patterns.append("eligibility_filter_overpruning")

    if (
        predicted == gold
        and (recall_gap > 1e-12 or valid_recall_gap > 1e-12)
    ):
        patterns.append("correct_route_but_dense_miss")

    if not patterns:
        patterns.append("retrieval_gap_unclassified")
    return list(dict.fromkeys(patterns))


def _index_method(
    records: list[dict[str, Any]], method: str
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (
            record["query_id"],
            int(record.get("repeat_index", 0)),
        ): record
        for record in records
        if record["method"] == method
    }


def _mean_field(
    records: list[dict[str, Any]], field: str
) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return mean(values) if values else None


def _bool_rate(
    rows: list[dict[str, Any]], field: str
) -> float | None:
    return mean(bool(row[field]) for row in rows) if rows else None


def _numbers_close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _float_lists_close(left: list[Any], right: list[Any]) -> bool:
    return len(left) == len(right) and all(
        _numbers_close(a, b) for a, b in zip(left, right)
    )


def _less_than(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return float(left) < float(right)
    except (TypeError, ValueError):
        return False


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Iterable[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )