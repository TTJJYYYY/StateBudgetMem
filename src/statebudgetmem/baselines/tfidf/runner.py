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
from statebudgetmem.evaluation import evaluate_retrieval
from statebudgetmem.baselines.tfidf.retriever import TfidfCosineRetriever


@dataclass(frozen=True)
class BaselineConfig:
    method: str
    dataset_path: Path
    top_k: int
    random_seed: int
    results_dir: Path
    config_path: Path


def run_baseline(config: BaselineConfig) -> dict[str, Any]:
    if config.method != "tfidf_topk":
        raise ValueError(f"unsupported method for Task001: {config.method}")
    if config.top_k < 1:
        raise ValueError("top_k must be at least 1")

    scenarios = load_scenarios(config.dataset_path)
    retriever = TfidfCosineRetriever()
    run_id = _run_id(config)
    raw_dir = config.results_dir / "raw"
    summary_dir = config.results_dir / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for scenario in scenarios:
        for query in scenario.queries:
            retrieval_started = time.perf_counter()
            retrieved = retriever.retrieve(query, scenario.memories, config.top_k)
            latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
            metrics = evaluate_retrieval(query, retrieved, latency_ms)
            retrieved_ids = [item.memory.memory_id for item in retrieved]
            row = {
                "run_id": run_id,
                "scenario_id": scenario.scenario_id,
                "query_id": query.query_id,
                "query_text": query.text,
                "query_type": query.query_type.value,
                "retrieved_memory_ids": retrieved_ids,
                "retrieved_scores": [round(item.score, 12) for item in retrieved],
                "retrieved_valid_flags": [
                    item.memory.memory_id in query.gold_valid_memory_ids for item in retrieved
                ],
                "retrieved_stale_flags": [
                    item.memory.memory_id in query.gold_stale_memory_ids for item in retrieved
                ],
                "recall_at_k": metrics["recall_at_k"],
                "valid_recall_at_k": metrics["valid_recall_at_k"],
                "stale_retrieval_rate": metrics["stale_retrieval_rate"],
                "average_token_cost": metrics["average_token_cost"],
                "retrieval_latency_ms": metrics["retrieval_latency_ms"],
                "method": config.method,
                "top_k": config.top_k,
                "random_seed": config.random_seed,
            }
            raw_rows.append(row)

    raw_path = raw_dir / f"{run_id}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = _summarize(config, run_id, raw_rows, raw_path, time.perf_counter() - started)
    json_path = summary_dir / f"{run_id}.json"
    csv_path = summary_dir / f"{run_id}.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return {
        "run_id": run_id,
        "raw_path": str(raw_path),
        "summary_json_path": str(json_path),
        "summary_csv_path": str(csv_path),
        "summary": summary,
    }


def _summarize(
    config: BaselineConfig,
    run_id: str,
    raw_rows: list[dict[str, Any]],
    raw_path: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    query_count = len(raw_rows)
    summary = {
        "run_id": run_id,
        "query_count": query_count,
        "mean_recall_at_k": _mean(raw_rows, "recall_at_k"),
        "mean_valid_recall_at_k": _mean(raw_rows, "valid_recall_at_k"),
        "mean_stale_retrieval_rate": _mean(raw_rows, "stale_retrieval_rate"),
        "mean_token_cost": _mean(raw_rows, "average_token_cost"),
        "mean_retrieval_latency_ms": _mean(raw_rows, "retrieval_latency_ms"),
        "method": config.method,
        "top_k": config.top_k,
        "dataset_path": str(config.dataset_path),
        "random_seed": config.random_seed,
        "config_path": str(config.config_path),
        "results_raw_path": str(raw_path),
        "model_name": "offline_tfidf_cosine",
        "run_time_seconds": elapsed_seconds,
        "git_commit": _git_commit(),
        "hardware_info": platform.platform(),
    }
    return summary


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return mean(float(row[key]) for row in rows)


def _run_id(config: BaselineConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.method}_seed{config.random_seed}_k{config.top_k}_{timestamp}"


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
