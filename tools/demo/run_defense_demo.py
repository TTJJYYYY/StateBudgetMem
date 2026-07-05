#!/usr/bin/env python3
"""Unified offline defense/demo runner for StateBudgetMem.

The script intentionally reuses existing experiment entry points and writes a
single machine-readable summary for group meetings or thesis-defense demos.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statebudgetmem.baselines.tfidf import BaselineConfig, run_baseline
from statebudgetmem.routing import QueryRecord as RoutingQueryRecord
from statebudgetmem.routing import RuleBasedRouter
from statebudgetmem.schemas import MemoryRecord, MemoryStatus, QueryType
from statebudgetmem.versioning import StateKey, UpdateOperation, VersioningEngine
from statebudgetmem.views import ViewsExperimentConfig, run_views_experiment


@dataclass(frozen=True)
class DefenseDemoConfig:
    baseline_dataset: Path
    views_dataset: Path
    results_dir: Path
    top_k: int
    random_seed: int
    skip_baseline: bool = False
    skip_views: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the unified StateBudgetMem defense demo.",
    )
    parser.add_argument(
        "--baseline-dataset",
        type=Path,
        default=Path("data/controlled/baseline_scenarios.jsonl"),
    )
    parser.add_argument(
        "--views-dataset",
        type=Path,
        default=Path("data/controlled/temporal_challenge_v1.jsonl"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/defense_demo"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-views", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = DefenseDemoConfig(
        baseline_dataset=args.baseline_dataset,
        views_dataset=args.views_dataset,
        results_dir=args.results_dir,
        top_k=args.top_k,
        random_seed=args.seed,
        skip_baseline=args.skip_baseline,
        skip_views=args.skip_views,
    )
    summary = run_defense_demo(config)
    print_summary(summary)
    return 0


def run_defense_demo(config: DefenseDemoConfig) -> dict[str, Any]:
    if config.top_k < 1:
        raise ValueError("top_k must be at least 1")

    started = time.perf_counter()
    config.results_dir.mkdir(parents=True, exist_ok=True)

    baseline_result = None
    if not config.skip_baseline:
        baseline_result = _run_tfidf_baseline(config)

    versioning_result = _run_versioning_example()

    views_result = None
    if not config.skip_views:
        views_result = _run_views(config)

    routing_result = _run_routing_examples()

    run_id = _run_id()
    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_k": config.top_k,
        "random_seed": config.random_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "sections": {
            "tfidf_baseline": baseline_result,
            "versioning_example": versioning_result,
            "views_experiment": views_result,
            "routing_examples": routing_result,
        },
    }
    output_path = config.results_dir / f"{run_id}.json"
    latest_path = config.results_dir / "latest_summary.json"
    summary["summary_json_path"] = str(output_path)
    _write_json(output_path, summary)
    _write_json(latest_path, summary)
    return summary


def _run_tfidf_baseline(config: DefenseDemoConfig) -> dict[str, Any]:
    result = run_baseline(
        BaselineConfig(
            method="tfidf_topk",
            dataset_path=config.baseline_dataset,
            top_k=config.top_k,
            random_seed=config.random_seed,
            results_dir=Path("results"),
            config_path=Path("configs/baseline.yaml"),
        )
    )
    summary = result["summary"]
    return {
        "run_id": result["run_id"],
        "summary_json_path": result["summary_json_path"],
        "raw_path": result["raw_path"],
        "query_count": summary["query_count"],
        "recall_at_k": summary["mean_recall_at_k"],
        "valid_recall_at_k": summary["mean_valid_recall_at_k"],
        "stale_retrieval_rate": summary["mean_stale_retrieval_rate"],
        "mean_token_cost": summary["mean_token_cost"],
        "mean_retrieval_latency_ms": summary["mean_retrieval_latency_ms"],
    }


def _run_versioning_example() -> dict[str, Any]:
    engine = VersioningEngine()
    records = [
        _memory("diet_1", "no_spicy_food", "2026-01-01"),
        _memory("diet_2", "mild_spicy_food_is_ok", "2026-03-01"),
        _memory(
            "diet_3",
            "avoid_spicy_food_for_stomach",
            "2026-05-01",
            valid_to="2026-05-20",
        ),
        _memory("diet_4", "mild_spicy_food_is_ok", "2026-05-10"),
    ]

    operations: list[dict[str, Any]] = []
    for record in records:
        batch = engine.ingest(record)
        for result in batch.results:
            operations.append(
                {
                    "memory_id": result.decision.new_memory_id,
                    "operation": result.decision.operation.value,
                    "targets": result.decision.target_memory_ids,
                    "reason": result.decision.reason,
                }
            )

    key = StateKey(subject="user", attribute="diet.spice")
    current = engine.resolve_current(key)
    history = engine.history(key)
    validation = engine.validate()
    return {
        "operations": operations,
        "current_memory_ids": [item.memory_id for item in current],
        "history_memory_ids": [item.memory_id for item in history],
        "history_values": [item.value for item in history],
        "validation_is_valid": validation.is_valid,
        "operation_counts": _operation_counts(operations),
    }


def _memory(
    memory_id: str,
    value: str,
    event_time: str,
    *,
    valid_to: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        subject="user",
        attribute="diet.spice",
        value=value,
        text=value.replace("_", " "),
        event_time=event_time,
        valid_from=event_time,
        valid_to=valid_to,
        status=MemoryStatus.CURRENT,
        memory_type="state",
        importance=0.8,
        confidence=0.9,
        token_cost=8,
    )


def _operation_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
    counts = {operation.value: 0 for operation in UpdateOperation}
    for row in operations:
        counts[row["operation"]] = counts.get(row["operation"], 0) + 1
    return {key: value for key, value in counts.items() if value}


def _run_views(config: DefenseDemoConfig) -> dict[str, Any]:
    result = run_views_experiment(
        ViewsExperimentConfig(
            dataset_path=config.views_dataset,
            top_k=config.top_k,
            random_seed=config.random_seed,
            results_dir=Path("results/views"),
            methods=("flat", "current", "dual"),
        )
    )
    all_rows = [row for row in result["summary"] if row["query_type"] == "ALL"]
    comparison = {
        row["method"]: {
            "recall_at_k": row["mean_recall_at_k"],
            "valid_recall_at_k": row["mean_valid_recall_at_k"],
            "stale_retrieval_rate": row["mean_stale_retrieval_rate"],
            "mean_token_cost": row["mean_token_cost"],
        }
        for row in all_rows
    }
    return {
        "run_id": result["run_id"],
        "summary_json_path": result["summary_json_path"],
        "raw_path": result["raw_path"],
        "flat_current_dual": comparison,
    }


def _run_routing_examples() -> dict[str, Any]:
    router = RuleBasedRouter()
    examples = [
        ("current_food", "我现在适合吃什么?"),
        ("historical_food", "我以前是不是不吃辣?"),
        ("change_food", "我的饮食习惯是怎么变化的?"),
        ("general_food", "什么是低脂饮食?"),
    ]
    rows = []
    for example_id, text in examples:
        query = RoutingQueryRecord(text=text, reference_time="2026-07-05T10:00:00")
        query_type = router.classify(query)
        view_type = router.route(text, query_type)
        rows.append(
            {
                "example_id": example_id,
                "query": text,
                "query_type": query_type.value,
                "selected_view": view_type.value,
            }
        )
    return {
        "router": "RuleBasedRouter",
        "examples": rows,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\nStateBudgetMem defense demo")
    print("=" * 32)

    baseline = summary["sections"]["tfidf_baseline"]
    if baseline is None:
        print("\n[1] TF-IDF baseline: skipped")
    else:
        print("\n[1] TF-IDF baseline")
        print(f"  recall@k              : {baseline['recall_at_k']:.4f}")
        print(f"  valid recall@k        : {baseline['valid_recall_at_k']:.4f}")
        print(f"  stale retrieval rate  : {baseline['stale_retrieval_rate']:.4f}")
        print(f"  summary               : {baseline['summary_json_path']}")

    versioning = summary["sections"]["versioning_example"]
    print("\n[2] Versioning example")
    print(f"  operation counts      : {versioning['operation_counts']}")
    print(f"  current memory ids    : {versioning['current_memory_ids']}")
    print(f"  history memory ids    : {versioning['history_memory_ids']}")
    print(f"  validation is valid   : {versioning['validation_is_valid']}")

    views = summary["sections"]["views_experiment"]
    if views is None:
        print("\n[3] Views experiment: skipped")
    else:
        print("\n[3] Views experiment: flat/current/dual")
        for method, metrics in views["flat_current_dual"].items():
            print(
                "  "
                f"{method:<18} "
                f"recall@k={metrics['recall_at_k']:.4f} "
                f"valid={metrics['valid_recall_at_k']:.4f} "
                f"stale={metrics['stale_retrieval_rate']:.4f}"
            )
        print(f"  summary               : {views['summary_json_path']}")

    routing = summary["sections"]["routing_examples"]
    print("\n[4] Routing examples")
    for row in routing["examples"]:
        print(
            "  "
            f"{row['example_id']:<16} "
            f"{row['query_type']:<10} -> {row['selected_view']:<7} "
            f"{row['query']}"
        )

    print("\nUnified summary")
    print(f"  {summary['summary_json_path']}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"defense_demo_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
