from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import yaml

from statebudgetmem.core.experiment import (
    ExperimentConfig,
    MethodBuildContext,
    ResourceMetrics,
    RunMetadata,
)
from statebudgetmem.core.registry import MethodRegistry, default_method_registry
from statebudgetmem.data import load_scenarios
from statebudgetmem.evaluation.metrics import recall_at_k, stale_retrieval_rate, valid_recall_at_k


def run_unified_experiment(
    config: ExperimentConfig,
    *,
    registry: MethodRegistry | None = None,
) -> dict[str, Any]:
    """Run registered methods on identical scenarios and write stable artifacts."""

    scenarios = load_scenarios(config.dataset_path)
    if not scenarios:
        raise ValueError("dataset must contain at least one scenario")
    random.seed(config.random_seed)

    run_id = _run_id(config)
    run_dir = config.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    method_registry = registry or default_method_registry()
    methods = [
        method_registry.create(
            name,
            MethodBuildContext(
                experiment=config,
                work_dir=run_dir / "methods" / name,
            ),
        )
        for name in config.methods
    ]
    _validate_method_names(config, methods)
    raw_path = run_dir / "raw.jsonl"
    summary_json_path = run_dir / "summary.json"
    summary_csv_path = run_dir / "summary.csv"
    environment_path = run_dir / "environment.json"

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for repeat_index in range(config.repeat):
        for method in methods:
            for scenario in scenarios:
                if config.query_state_policy == "sequential":
                    method.reset()
                    ingest_started = time.perf_counter()
                    method.ingest(scenario.memories)
                    shared_ingest_latency_ms = (
                        time.perf_counter() - ingest_started
                    ) * 1000.0
                for query in scenario.queries:
                    if config.query_state_policy == "independent":
                        method.reset()
                        ingest_started = time.perf_counter()
                        method.ingest(scenario.memories)
                        ingest_latency_ms = (
                            time.perf_counter() - ingest_started
                        ) * 1000.0
                    else:
                        ingest_latency_ms = shared_ingest_latency_ms
                    result = method.retrieve(
                        query.model_copy(
                            update={
                                "gold_relevant_memory_ids": [],
                                "gold_valid_memory_ids": [],
                                "gold_stale_memory_ids": [],
                            }
                        ),
                        top_k=config.top_k,
                        token_budget=config.token_budget,
                        mutate=config.reinforcement_enabled,
                    )
                    if result.query_id != query.query_id:
                        raise ValueError(
                            f"method {method.name} returned query_id {result.query_id!r} "
                            f"for {query.query_id!r}"
                        )
                    rows.append(
                        _result_row(
                            run_id,
                            scenario.scenario_id,
                            query,
                            result,
                            config,
                            ResourceMetrics(
                                ingest_latency_ms=ingest_latency_ms,
                                retrieval_latency_ms=result.latency_ms,
                                total_token_cost=result.total_token_cost,
                                repeat_index=repeat_index,
                            ),
                        )
                    )

    _write_jsonl(raw_path, rows)
    summary = _summarize(rows, config, time.perf_counter() - started)
    _write_json(summary_json_path, summary)
    _write_csv(summary_csv_path, summary["methods"])
    metadata = _run_metadata(run_id, config)
    _write_json(environment_path, metadata.model_dump(mode="json"))
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "raw_path": str(raw_path),
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "environment_path": str(environment_path),
        "summary": summary,
    }


def _validate_method_names(config: ExperimentConfig, methods: list[Any]) -> None:
    actual = tuple(method.name for method in methods)
    if actual != config.methods:
        raise ValueError(f"adapter names {actual!r} do not match configured names {config.methods!r}")


def _result_row(run_id, scenario_id, query, result, config, resources) -> dict[str, Any]:
    ids = [item.memory_id for item in result.retrieved_memories]
    return {
        "run_id": run_id,
        "method": result.method_name,
        "scenario_id": scenario_id,
        "query_id": query.query_id,
        "query_text": query.text,
        "query_type": query.query_type.value,
        "predicted_query_type": (
            result.predicted_query_type.value if result.predicted_query_type else None
        ),
        "retrieved_memory_ids": ids,
        "retrieved_scores": [item.score for item in result.retrieved_memories],
        "source_views": [item.source_view for item in result.retrieved_memories],
        "recall_at_k": recall_at_k(ids, query.gold_relevant_memory_ids),
        "valid_recall_at_k": valid_recall_at_k(ids, query.gold_valid_memory_ids),
        "stale_retrieval_rate": stale_retrieval_rate(ids, query.gold_stale_memory_ids),
        "total_token_cost": result.total_token_cost,
        "ingest_latency_ms": resources.ingest_latency_ms,
        "retrieval_latency_ms": resources.retrieval_latency_ms,
        "peak_rss_bytes": resources.peak_rss_bytes,
        "storage_bytes": resources.storage_bytes,
        "repeat_index": resources.repeat_index,
        "top_k": config.top_k,
        "token_budget": config.token_budget,
        "random_seed": config.random_seed,
        "reinforcement_enabled": config.reinforcement_enabled,
        "query_state_policy": config.query_state_policy,
        "status": "success",
        "error": None,
        "method_metadata": result.metadata,
    }


def _summarize(rows, config, elapsed_seconds) -> dict[str, Any]:
    method_rows = []
    for name in config.methods:
        selected = [row for row in rows if row["method"] == name]
        method_rows.append(
            {
                "method": name,
                "query_count": len(selected),
                "mean_recall_at_k": _mean(selected, "recall_at_k"),
                "mean_valid_recall_at_k": _mean(selected, "valid_recall_at_k"),
                "mean_stale_retrieval_rate": _mean(selected, "stale_retrieval_rate"),
                "mean_total_token_cost": _mean(selected, "total_token_cost"),
                "mean_ingest_latency_ms": _mean(selected, "ingest_latency_ms"),
                "mean_retrieval_latency_ms": _mean(selected, "retrieval_latency_ms"),
                "top_k": config.top_k,
                "candidate_k": config.candidate_k,
                "token_budget": config.token_budget,
                "random_seed": config.random_seed,
                "repeat": config.repeat,
                "forgetting_enabled": config.forgetting_enabled,
                "forgetting_threshold": config.forgetting_threshold,
                "exclude_forgotten": config.exclude_forgotten,
                "reinforcement_enabled": config.reinforcement_enabled,
                "query_state_policy": config.query_state_policy,
            }
        )
    return {
        "dataset_path": str(config.dataset_path),
        "elapsed_seconds": elapsed_seconds,
        "methods": method_rows,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def _run_metadata(run_id: str, config: ExperimentConfig) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        dataset_path=str(config.dataset_path),
        dataset_sha256=_sha256(config.dataset_path),
        config_path=str(config.config_path) if config.config_path else None,
        git_commit=_git_value(["rev-parse", "HEAD"]),
        dirty_worktree=_dirty_worktree(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        hardware={"machine": platform.machine(), "processor": platform.processor()},
        dependency_versions=_dependency_versions(("pydantic", "PyYAML")),
        embedding_backend=config.embedding_backend,
        embedding_model=config.embedding_model,
        token_counter_name=config.token_counter_name,
        random_seed=config.random_seed,
    )


def _dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _dirty_worktree() -> bool | None:
    value = _git_value(["status", "--porcelain"])
    return None if value is None else bool(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_id(config: ExperimentConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"unified_smoke_seed{config.random_seed}_{stamp}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a YAML mapping")
    raw_methods = payload.get("methods")
    if isinstance(raw_methods, str):
        payload["methods"] = tuple(
            item.strip() for item in raw_methods.split(",") if item.strip()
        )
    payload["config_path"] = config_path
    return ExperimentConfig.model_validate(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m statebudgetmem.unified_runner")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="data/controlled/interface_smoke_v1.jsonl")
    parser.add_argument("--results-dir", default="results/interface_smoke")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    if args.config:
        config = load_experiment_config(args.config)
    else:
        config = ExperimentConfig(
            dataset_path=Path(args.dataset),
            results_dir=Path(args.results_dir),
            methods=tuple(args.methods or ["tfidf_topk"]),
            top_k=args.top_k,
            token_budget=args.token_budget,
            random_seed=args.seed,
            repeat=args.repeat,
        )
    result = run_unified_experiment(
        config
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
