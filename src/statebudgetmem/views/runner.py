from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from statebudgetmem.data import load_scenarios
from statebudgetmem.evaluation.metrics import (
    recall_at_k,
    stale_retrieval_rate,
    valid_recall_at_k,
)
from statebudgetmem.interfaces import MemoryMethod
from statebudgetmem.views.methods import (
    CurrentOnlyMemoryMethod,
    DualViewMemoryMethod,
    FlatViewMemoryMethod,
    HistoryOnlyMemoryMethod,
)


@dataclass(frozen=True)
class ViewsExperimentConfig:
    dataset_path: Path
    top_k: int
    random_seed: int
    results_dir: Path
    methods: tuple[str, ...] = ("flat", "current", "dual")
    token_budget: int | None = None


def build_view_method(name: str) -> MemoryMethod:
    normalized = name.strip().lower().replace("-", "_")

    if normalized in {"flat", "views_flat", "unified", "unified_memory"}:
        return FlatViewMemoryMethod()

    if normalized in {"current", "current_only", "views_current_only"}:
        return CurrentOnlyMemoryMethod()

    if normalized in {"history", "history_only", "views_history_only"}:
        return HistoryOnlyMemoryMethod()

    if normalized in {"dual", "both", "views_dual"}:
        return DualViewMemoryMethod()

    raise ValueError(f"unsupported views method: {name}")


def run_views_experiment(config: ViewsExperimentConfig) -> dict[str, Any]:
    if config.top_k < 1:
        raise ValueError("top_k must be at least 1")

    scenarios = load_scenarios(config.dataset_path)

    run_id = _run_id(config)
    raw_dir = config.results_dir / "raw"
    summary_dir = config.results_dir / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for method_name in config.methods:
        method = build_view_method(method_name)

        for scenario in scenarios:
            method.reset()
            method.ingest(scenario.memories)

            for query in scenario.queries:
                result = method.retrieve(
                    query,
                    top_k=config.top_k,
                    token_budget=config.token_budget,
                )

                retrieved_ids = [
                    item.memory_id for item in result.retrieved_memories
                ]

                raw_rows.append(
                    {
                        "run_id": run_id,
                        "scenario_id": scenario.scenario_id,
                        "query_id": query.query_id,
                        "query_text": query.text,
                        "query_type": query.query_type.value,
                        "reference_time": query.reference_time.isoformat(),
                        "routing_mode": "oracle_query_type",
                        "method": result.method_name,
                        "retrieved_memory_ids": retrieved_ids,
                        "retrieved_scores": [
                            round(item.score, 12)
                            for item in result.retrieved_memories
                        ],
                        "source_views": [
                            item.source_view for item in result.retrieved_memories
                        ],
                        "retrieved_valid_flags": [
                            memory_id in query.gold_valid_memory_ids
                            for memory_id in retrieved_ids
                        ],
                        "retrieved_stale_flags": [
                            memory_id in query.gold_stale_memory_ids
                            for memory_id in retrieved_ids
                        ],
                        "recall_at_k": recall_at_k(
                            retrieved_ids,
                            query.gold_relevant_memory_ids,
                        ),
                        "valid_recall_at_k": valid_recall_at_k(
                            retrieved_ids,
                            query.gold_valid_memory_ids,
                        ),
                        "stale_retrieval_rate": stale_retrieval_rate(
                            retrieved_ids,
                            query.gold_stale_memory_ids,
                        ),
                        "total_token_cost": result.total_token_cost,
                        "retrieval_latency_ms": result.latency_ms,
                        "top_k": config.top_k,
                        "token_budget": config.token_budget,
                        "random_seed": config.random_seed,
                    }
                )

    raw_path = raw_dir / f"{run_id}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary_rows = _summaries(
        config,
        run_id,
        raw_rows,
        raw_path,
        time.perf_counter() - started,
    )

    json_path = summary_dir / f"{run_id}.json"
    csv_path = summary_dir / f"{run_id}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=False, indent=2, sort_keys=True)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "run_id": run_id,
        "raw_path": str(raw_path),
        "summary_json_path": str(json_path),
        "summary_csv_path": str(csv_path),
        "summary": summary_rows,
    }


def _summaries(
    config: ViewsExperimentConfig,
    run_id: str,
    raw_rows: list[dict[str, Any]],
    raw_path: Path,
    elapsed_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = sorted({row["method"] for row in raw_rows})
    query_types = sorted({row["query_type"] for row in raw_rows})

    for method in methods:
        method_rows = [row for row in raw_rows if row["method"] == method]
        rows.append(
            _summary_row(
                config,
                run_id,
                method_rows,
                method=method,
                query_type="ALL",
                raw_path=raw_path,
                elapsed_seconds=elapsed_seconds,
            )
        )

        for query_type in query_types:
            selected = [
                row for row in method_rows if row["query_type"] == query_type
            ]
            if not selected:
                continue
            rows.append(
                _summary_row(
                    config,
                    run_id,
                    selected,
                    method=method,
                    query_type=query_type,
                    raw_path=raw_path,
                    elapsed_seconds=elapsed_seconds,
                )
            )

    return rows


def _summary_row(
    config: ViewsExperimentConfig,
    run_id: str,
    selected: list[dict[str, Any]],
    *,
    method: str,
    query_type: str,
    raw_path: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "method": method,
        "query_type": query_type,
        "routing_mode": "oracle_query_type",
        "query_count": len(selected),
        "mean_recall_at_k": _mean(selected, "recall_at_k"),
        "mean_valid_recall_at_k": _mean(selected, "valid_recall_at_k"),
        "mean_stale_retrieval_rate": _mean(
            selected,
            "stale_retrieval_rate",
        ),
        "mean_token_cost": _mean(selected, "total_token_cost"),
        "mean_retrieval_latency_ms": _mean(
            selected,
            "retrieval_latency_ms",
        ),
        "top_k": config.top_k,
        "token_budget": config.token_budget,
        "dataset_path": str(config.dataset_path),
        "random_seed": config.random_seed,
        "results_raw_path": str(raw_path),
        "model_name": "offline_tfidf_cosine_with_versioned_views",
        "run_time_seconds": elapsed_seconds,
        "git_commit": _git_commit(),
        "hardware_info": platform.platform(),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0

    return mean(float(row[key]) for row in rows)


def _run_id(config: ViewsExperimentConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    method_part = "_".join(config.methods)
    return f"views_{method_part}_seed{config.random_seed}_k{config.top_k}_{timestamp}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    commit = result.stdout.strip()
    return commit or None
