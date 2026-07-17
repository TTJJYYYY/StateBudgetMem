#!/usr/bin/env python3
"""Run a deterministic, local-only MemoryBank budget sweep.

The synthetic probes in this runner are intended for quality/resource trend
analysis. They are not a replacement for the project's 96-query fair
comparison and do not measure LLM answer quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statebudgetmem.baselines.memorybank import MemoryBank  # noqa: E402
from statebudgetmem.baselines.memorybank.metrics import (  # noqa: E402
    MemoryBankMetricSpec,
    evaluate_reproduction_row,
    summarize_metric_rows,
)


DATASET_SOURCE = "synthetic_memorybank_budget_probe"
EMBEDDING_BACKEND = "hash"
EMBEDDING_MODEL = "deterministic_hash_embedding"
TOKEN_METRIC = "deterministic_proxy_not_real_tokenizer"
TOKEN_WARNING = (
    "token_proxy is a deterministic local proxy, not a real tokenizer token count."
)
DEFAULT_RESULTS_ROOT = Path("results/budget_sweep")
DEFAULT_TOKEN_BUDGETS = (64, 128, 256, 512)
DEFAULT_TOP_K = (1, 3, 5)
DEFAULT_CANDIDATE_K = (5, 10, 20)
DEFAULT_MEMORY_COUNTS = (100, 500, 1000, 2000)
DEFAULT_FORGETTING_THRESHOLDS = (0.3,)
SEED_MEMORY_COUNT = 4

ROW_FIELDNAMES = (
    "run_id",
    "repeat_index",
    "query_id",
    "token_budget",
    "top_k",
    "candidate_k",
    "memory_count",
    "forgetting_threshold",
    "seed",
    "embedding_backend",
    "embedding_dim",
    "current_time",
    "retrieval_latency_ms",
    "selected_token_proxy",
    "selected_memory_count",
    "retrieved_count_before_budget",
    "candidate_count_before_forgetting",
    "candidate_count_after_forgetting",
    "memory_retrieval_accuracy",
    "answer_accuracy",
    "response_correctness",
    "stale_retrieval_rate",
    "relevant_loss_rate",
    "stale_retrieval_case_rate",
    "token_budget_used_ratio",
    "estimated_memory_storage_bytes",
    "estimated_faiss_index_bytes",
    "faiss_index_size",
    "retrieval_rss_before_bytes",
    "retrieval_rss_peak_bytes",
    "retrieval_rss_delta_bytes",
    "token_metric",
    "selected_memory_ids",
    "local_only",
    "cloud_api_used",
    "llm_called",
)

SUMMARY_FIELDNAMES = (
    "token_budget",
    "top_k",
    "candidate_k",
    "memory_count",
    "forgetting_threshold",
    "run_count",
    "query_count",
    "memory_retrieval_accuracy",
    "answer_accuracy",
    "response_correctness",
    "stale_retrieval_rate",
    "relevant_loss_rate",
    "stale_retrieval_case_rate",
    "mean_retrieval_latency_ms",
    "p95_retrieval_latency_ms",
    "mean_selected_token_proxy",
    "mean_token_budget_used_ratio",
    "mean_selected_memory_count",
    "mean_estimated_memory_storage_bytes",
    "max_estimated_memory_storage_bytes",
    "mean_estimated_faiss_index_bytes",
    "max_faiss_index_size",
    "mean_retrieval_rss_peak_bytes",
    "max_retrieval_rss_peak_bytes",
    "rss_available",
    "token_metric",
)


@dataclass(frozen=True)
class BudgetProbe:
    query_id: str
    query: str
    relevant_keywords: tuple[str, ...]
    answer_keywords: tuple[str, ...]
    stale_keywords: tuple[str, ...] = ()


PROBES = (
    BudgetProbe(
        query_id="q_book",
        query="What Python book did you recommend?",
        relevant_keywords=("python", "book"),
        answer_keywords=("python", "book"),
    ),
    BudgetProbe(
        query_id="q_food",
        query="What food should I avoid now?",
        relevant_keywords=("avoid", "spicy", "stomach"),
        answer_keywords=("avoid", "spicy"),
        stale_keywords=("used to like spicy hotpot", "old spicy preference"),
    ),
    BudgetProbe(
        query_id="q_hobby",
        query="What do you remember about my hobbies?",
        relevant_keywords=("basketball", "swimming"),
        answer_keywords=("basketball", "swimming"),
    ),
)


class HashEmbeddingModel:
    """Deterministic local encoder used by the synthetic sweep."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, text: str):
        import numpy as np

        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = digest[0] % self.dim
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[index] += sign
        if not vector.any():
            vector[0] = 1.0
        return vector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--token-budget",
        "--prompt-token-budget",
        dest="token_budget",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOKEN_BUDGETS),
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K))
    parser.add_argument(
        "--candidate-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_K),
    )
    parser.add_argument(
        "--memory-count",
        type=int,
        nargs="+",
        default=list(DEFAULT_MEMORY_COUNTS),
    )
    parser.add_argument(
        "--forgetting-threshold",
        type=float,
        nargs="+",
        default=list(DEFAULT_FORGETTING_THRESHOLDS),
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--current-time", default="2026-07-10 10:00")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run token_budget=64,128; top_k=1,3; candidate_k=5,20; "
            "memory_count=100; threshold=0.3; repeat=1."
        ),
    )
    return parser.parse_args(argv)


def apply_quick_grid(args: argparse.Namespace) -> None:
    args.token_budget = [64, 128]
    args.top_k = [1, 3]
    args.candidate_k = [5, 20]
    args.memory_count = [100]
    args.forgetting_threshold = [0.3]
    args.repeat = 1


def validate_args(args: argparse.Namespace) -> None:
    checks = (
        (args.token_budget, "token_budget", lambda value: value > 0),
        (args.top_k, "top_k", lambda value: value > 0),
        (args.candidate_k, "candidate_k", lambda value: value > 0),
        (
            args.memory_count,
            "memory_count",
            lambda value: value >= SEED_MEMORY_COUNT,
        ),
        (args.forgetting_threshold, "forgetting_threshold", lambda value: value >= 0),
    )
    for values, name, predicate in checks:
        if not values or any(not predicate(value) for value in values):
            requirement = (
                f">= {SEED_MEMORY_COUNT}"
                if name == "memory_count"
                else ">= 0"
                if name == "forgetting_threshold"
                else "> 0"
            )
            raise ValueError(f"{name} values must be {requirement}")
    if args.repeat <= 0:
        raise ValueError("repeat must be > 0")
    if args.embedding_dim <= 0:
        raise ValueError("embedding_dim must be > 0")


def invalid_combinations(args: argparse.Namespace) -> list[dict[str, int]]:
    return [
        {"top_k": top_k, "candidate_k": candidate_k}
        for top_k, candidate_k in product(args.top_k, args.candidate_k)
        if top_k > candidate_k
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        apply_quick_grid(args)
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "budget_sweep_%Y%m%dT%H%M%SZ"
    )
    output = args.results_root
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR", str((ROOT / ".tmp" / "matplotlib-budget-sweep").resolve())
    )

    paths = {
        "rows_csv": output / "budget_sweep_rows.csv",
        "rows_json": output / "budget_sweep_rows.json",
        "summary_csv": output / "budget_sweep_summary.csv",
        "summary_json": output / "budget_sweep_summary.json",
        "resource_json": output / "resource_metrics.json",
        "manifest": output / "manifest.json",
    }
    git_commit, git_error = get_git_commit()
    process, rss_error = get_rss_process()
    rss_before = read_rss(process)
    tracemalloc.start()
    started = time.perf_counter()
    rows = run_budget_sweep(args, run_id=run_id, process=process)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _, peak_tracemalloc_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = read_rss(process)

    configuration_rows = aggregate_configurations(rows)
    skipped = invalid_combinations(args)
    _write_csv(paths["rows_csv"], rows, ROW_FIELDNAMES)
    _write_json(paths["rows_json"], rows)
    _write_csv(paths["summary_csv"], configuration_rows, SUMMARY_FIELDNAMES)
    figure_paths = generate_figures(configuration_rows, figures_dir)

    output_paths = {key: repository_relative(path) for key, path in paths.items()}
    output_paths["figures"] = [
        repository_relative(path) for path in figure_paths
    ]
    summary = build_budget_summary(
        rows=rows,
        configuration_rows=configuration_rows,
        args=args,
        run_id=run_id,
        git_commit=git_commit,
        git_error=git_error,
        skipped_invalid_combinations=skipped,
        elapsed_ms=elapsed_ms,
        output_paths=output_paths,
    )
    _write_json(paths["summary_json"], summary)

    resources = build_budget_resources(
        rows=rows,
        args=args,
        run_id=run_id,
        elapsed_ms=elapsed_ms,
        peak_tracemalloc_bytes=peak_tracemalloc_bytes,
        rss_before_bytes=rss_before,
        rss_after_bytes=rss_after,
        rss_error=rss_error,
        artifact_paths=[
            paths["rows_csv"],
            paths["rows_json"],
            paths["summary_csv"],
            paths["summary_json"],
            *figure_paths,
        ],
    )
    _write_json_stable_size(paths["resource_json"], resources)

    manifest_artifacts = [
        paths["rows_csv"],
        paths["rows_json"],
        paths["summary_csv"],
        paths["summary_json"],
        paths["resource_json"],
        *figure_paths,
    ]
    manifest = build_manifest(
        args=args,
        run_id=run_id,
        git_commit=git_commit,
        git_error=git_error,
        rows=rows,
        configuration_rows=configuration_rows,
        skipped_invalid_combinations=skipped,
        artifacts=manifest_artifacts,
    )
    _write_json(paths["manifest"], manifest)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "row_count": len(rows),
                "configuration_count": len(configuration_rows),
                "elapsed_ms": elapsed_ms,
                "results_root": repository_relative(output),
                "skipped_invalid_combinations": skipped,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_budget_sweep(
    args: argparse.Namespace,
    run_id: str,
    process: Any | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for memory_count, threshold, repeat_index in product(
        args.memory_count,
        args.forgetting_threshold,
        range(args.repeat),
    ):
        memory_bank = build_synthetic_memory_bank(
            memory_count=memory_count,
            forgetting_threshold=threshold,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
        )
        estimated_storage = estimate_memory_storage_size(memory_bank)
        stats = memory_bank.get_stats()
        index_size = int(stats.get("index_size", 0) or 0)
        estimated_faiss_bytes = index_size * args.embedding_dim * 4

        for top_k, candidate_k, token_budget, probe in product(
            args.top_k,
            args.candidate_k,
            args.token_budget,
            PROBES,
        ):
            if top_k > candidate_k:
                continue
            rows.append(
                run_budget_probe(
                    memory_bank=memory_bank,
                    probe=probe,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    token_budget=token_budget,
                    memory_count=memory_count,
                    forgetting_threshold=threshold,
                    current_time=args.current_time,
                    run_id=run_id,
                    repeat_index=repeat_index,
                    seed=args.seed,
                    embedding_dim=args.embedding_dim,
                    estimated_storage_bytes=estimated_storage,
                    estimated_faiss_index_bytes=estimated_faiss_bytes,
                    process=process,
                )
            )
    return rows


def build_synthetic_memory_bank(
    memory_count: int,
    forgetting_threshold: float,
    embedding_dim: int,
    seed: int = 42,
) -> MemoryBank:
    try:
        memory_bank = MemoryBank(
            forgetting_threshold=forgetting_threshold,
            embedding_model=HashEmbeddingModel(dim=embedding_dim),
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    seed_memories = (
        (
            "User",
            "The Python book you recommended was Automate the Boring Stuff.",
            "2026-06-20 10:00",
        ),
        (
            "User",
            "My stomach is uncomfortable, so I should avoid spicy food for now.",
            "2026-06-23 11:00",
        ),
        (
            "User",
            "I like basketball and swimming on weekends.",
            "2026-06-20 10:05",
        ),
        (
            "User",
            "Old spicy preference: I used to like spicy hotpot before my stomach issue.",
            "2026-05-01 10:00",
        ),
    )
    for role, content, timestamp in seed_memories:
        memory_bank.store_dialog(role, content, timestamp)

    for index in range(memory_count - len(seed_memories)):
        topic = (index + seed) % 20
        content = (
            f"Filler memory {index:05d}: user discussed neutral topic {topic}, "
            f"daily planning detail {(index + seed) % 7}, and local note "
            f"{(index + seed) % 13}."
        )
        day = 1 + ((index + seed) % 28)
        memory_bank.store_dialog("User", content, f"2026-04-{day:02d} 09:00")

    memory_bank.update_user_portrait(
        "The user is study-oriented, health-conscious, and enjoys active hobbies."
    )
    memory_bank.update_global_summary(
        "The user studies Python, has a temporary stomach restriction, and likes sports."
    )
    return memory_bank


def run_budget_probe(
    memory_bank: MemoryBank,
    probe: BudgetProbe,
    top_k: int,
    candidate_k: int,
    token_budget: int,
    memory_count: int,
    forgetting_threshold: float,
    current_time: str,
    run_id: str,
    repeat_index: int,
    seed: int,
    embedding_dim: int,
    estimated_storage_bytes: int,
    estimated_faiss_index_bytes: int,
    process: Any | None = None,
) -> dict[str, Any]:
    rss_before = read_rss(process)
    started = time.perf_counter_ns()
    retrieval = memory_bank.retrieve_with_metadata(
        query=probe.query,
        top_k=top_k,
        candidate_k=candidate_k,
        current_time=current_time,
        reinforce=False,
    )
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    rss_after = read_rss(process)
    retrieved_memories = list(retrieval["memories"])
    selected_memories, selected_token_proxy = select_memories_for_token_budget(
        retrieved_memories, token_budget
    )
    selected_ids = [str(item.get("memory_id", "")) for item in selected_memories]
    answer = template_answer(probe.query, selected_memories)
    stats = memory_bank.get_stats()
    metric_input = {
        "query": probe.query,
        "retrieved_memories": selected_memories,
        "retrieved_memory_ids": selected_ids,
        "template_answer": answer,
        "latency_ms": latency_ms,
        "index_size": int(stats.get("index_size", 0) or 0),
        "prompt_token_estimate": selected_token_proxy,
    }
    metric_spec = MemoryBankMetricSpec(
        query_id=probe.query_id,
        relevant_keywords=probe.relevant_keywords,
        answer_keywords=probe.answer_keywords,
        stale_keywords=probe.stale_keywords,
    )
    metrics = evaluate_reproduction_row(metric_input, metric_spec)
    stale_rate = float(metrics["stale_retrieval_rate"])
    retrieval_accuracy = float(metrics["memory_retrieval_accuracy"])
    response_accuracy = float(metrics["response_correctness"])
    rss_peak = _max_optional(rss_before, rss_after)
    rss_delta = (
        rss_after - rss_before
        if rss_before is not None and rss_after is not None
        else None
    )
    return {
        "run_id": run_id,
        "repeat_index": repeat_index,
        "query_id": probe.query_id,
        "token_budget": token_budget,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "memory_count": memory_count,
        "forgetting_threshold": forgetting_threshold,
        "seed": seed,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_dim": embedding_dim,
        "current_time": current_time,
        "retrieval_latency_ms": latency_ms,
        "selected_token_proxy": selected_token_proxy,
        "selected_memory_count": len(selected_memories),
        "retrieved_count_before_budget": len(retrieved_memories),
        "candidate_count_before_forgetting": int(
            retrieval.get("candidate_count_before_forgetting", 0) or 0
        ),
        "candidate_count_after_forgetting": int(
            retrieval.get("candidate_count_after_forgetting", 0) or 0
        ),
        "memory_retrieval_accuracy": retrieval_accuracy,
        "answer_accuracy": response_accuracy,
        "response_correctness": response_accuracy,
        "stale_retrieval_rate": stale_rate,
        "relevant_loss_rate": float(retrieval_accuracy < 1.0),
        "stale_retrieval_case_rate": float(stale_rate > 0.0),
        "token_budget_used_ratio": selected_token_proxy / token_budget,
        "estimated_memory_storage_bytes": estimated_storage_bytes,
        "estimated_faiss_index_bytes": estimated_faiss_index_bytes,
        "faiss_index_size": int(stats.get("index_size", 0) or 0),
        "retrieval_rss_before_bytes": rss_before,
        "retrieval_rss_peak_bytes": rss_peak,
        "retrieval_rss_delta_bytes": rss_delta,
        "token_metric": TOKEN_METRIC,
        "selected_memory_ids": selected_ids,
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
    }


def select_memories_for_token_budget(
    retrieved_memories: list[dict[str, Any]],
    token_budget: int,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    used = 0
    for memory in retrieved_memories:
        cost = estimate_token_proxy(str(memory.get("content", "")))
        if used + cost > token_budget:
            continue
        selected.append(memory)
        used += cost
    return selected, used


def select_memories_for_prompt_budget(
    retrieved_memories: list[dict[str, Any]],
    prompt_token_budget: int,
) -> tuple[list[dict[str, Any]], int]:
    """Compatibility wrapper for the previous public helper name."""

    return select_memories_for_token_budget(retrieved_memories, prompt_token_budget)


def estimate_token_proxy(text: str) -> int:
    """Count ASCII alphanumeric runs plus non-ASCII non-whitespace chars."""

    ascii_runs = 0
    non_ascii_chars = 0
    in_ascii_run = False
    for char in text:
        if ord(char) < 128 and char.isalnum():
            if not in_ascii_run:
                ascii_runs += 1
                in_ascii_run = True
        else:
            in_ascii_run = False
            if ord(char) >= 128 and not char.isspace():
                non_ascii_chars += 1
    return ascii_runs + non_ascii_chars


def estimate_token_count(text: str) -> int:
    """Compatibility alias; the value is a proxy, not tokenizer output."""

    return estimate_token_proxy(text)


def template_answer(query: str, memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "I do not have enough retrieved memory to answer confidently."
    return f"Based on the selected memories for '{query}', " + " ".join(
        str(item.get("content", "")) for item in memories
    )


def estimate_memory_storage_size(memory_bank: MemoryBank) -> int:
    """Estimate UTF-8 bytes for memory content and selected metadata fields."""

    total = 0
    for memory in memory_bank.get_all():
        for attribute in ("content", "memory_id", "memory_type", "tags", "timestamp"):
            total += len(str(getattr(memory, attribute, "")).encode("utf-8"))
    total += len(str(getattr(memory_bank, "user_portrait", "")).encode("utf-8"))
    total += len(str(getattr(memory_bank, "global_summary", "")).encode("utf-8"))
    return total


def aggregate_configurations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "token_budget",
        "top_k",
        "candidate_k",
        "memory_count",
        "forgetting_threshold",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, selected in sorted(grouped.items()):
        output.append({**dict(zip(keys, values)), **summarize_rows(selected)})
    return output


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [
        {
            "memory_retrieval_accuracy": row["memory_retrieval_accuracy"],
            "response_correctness": row["response_correctness"],
            "stale_retrieval_rate": row["stale_retrieval_rate"],
            "retrieval_latency_ms": row["retrieval_latency_ms"],
            "faiss_index_size": row["faiss_index_size"],
            "prompt_token_cost": row["selected_token_proxy"],
        }
        for row in rows
    ]
    existing = summarize_metric_rows(metric_rows)
    latencies = [float(row["retrieval_latency_ms"]) for row in rows]
    rss_values = [
        int(value)
        for value in (row.get("retrieval_rss_peak_bytes") for row in rows)
        if value is not None
    ]
    return {
        "run_count": len(rows),
        "query_count": len({row["query_id"] for row in rows}),
        "memory_retrieval_accuracy": existing["memory_retrieval_accuracy"],
        "answer_accuracy": existing["response_correctness"],
        "response_correctness": existing["response_correctness"],
        "stale_retrieval_rate": existing["stale_retrieval_rate"],
        "relevant_loss_rate": _mean(rows, "relevant_loss_rate"),
        "stale_retrieval_case_rate": _mean(rows, "stale_retrieval_case_rate"),
        "mean_retrieval_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p95_retrieval_latency_ms": percentile(latencies, 0.95),
        "mean_selected_token_proxy": _mean(rows, "selected_token_proxy"),
        "mean_token_budget_used_ratio": _mean(rows, "token_budget_used_ratio"),
        "mean_selected_memory_count": _mean(rows, "selected_memory_count"),
        "mean_estimated_memory_storage_bytes": _mean(
            rows, "estimated_memory_storage_bytes"
        ),
        "max_estimated_memory_storage_bytes": max(
            (int(row["estimated_memory_storage_bytes"]) for row in rows), default=0
        ),
        "mean_estimated_faiss_index_bytes": _mean(
            rows, "estimated_faiss_index_bytes"
        ),
        "max_faiss_index_size": max(
            (int(row["faiss_index_size"]) for row in rows), default=0
        ),
        "mean_retrieval_rss_peak_bytes": (
            statistics.fmean(rss_values) if rss_values else None
        ),
        "max_retrieval_rss_peak_bytes": max(rss_values) if rss_values else None,
        "rss_available": bool(rss_values),
        "token_metric": TOKEN_METRIC,
    }


def aggregate_by_dimension(
    rows: list[dict[str, Any]], dimensions: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row[dimension]].append(row)
        output[dimension] = [
            {dimension: value, **summarize_rows(selected)}
            for value, selected in sorted(grouped.items())
        ]
    return output


def build_budget_summary(
    rows: list[dict[str, Any]],
    configuration_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    git_commit: str | None,
    git_error: str | None,
    skipped_invalid_combinations: list[dict[str, int]],
    elapsed_ms: float,
    output_paths: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_metadata": {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "git_commit_error": git_error,
            "elapsed_ms": elapsed_ms,
            "local_only": True,
            "cloud_api_used": False,
            "llm_called": False,
        },
        "dataset_source": DATASET_SOURCE,
        "experiment_scope": (
            "Deterministic synthetic budget probes for quality-resource trends; "
            "not a replacement for results/fair_comparison."
        ),
        "grid": grid_from_args(args),
        "repeat": args.repeat,
        "seed": args.seed,
        "commit": git_commit,
        "methods": ["memorybank_dense_hash_local"],
        "metric_definitions": metric_definitions(),
        "token_proxy_warning": TOKEN_WARNING,
        "skipped_invalid_combinations": skipped_invalid_combinations,
        "row_count": len(rows),
        "configuration_count": len(configuration_rows),
        "query_count": len(PROBES),
        "aggregate_by_configuration": configuration_rows,
        "aggregate_by_dimension": aggregate_by_dimension(
            rows, ("token_budget", "top_k", "candidate_k", "memory_count")
        ),
        "limitations": limitations(),
        "output_paths": output_paths,
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
    }


def build_budget_resources(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    elapsed_ms: float,
    peak_tracemalloc_bytes: int,
    rss_before_bytes: int | None,
    rss_after_bytes: int | None,
    rss_error: str | None,
    artifact_paths: list[Path],
) -> dict[str, Any]:
    row_rss = [
        int(value)
        for value in (row.get("retrieval_rss_peak_bytes") for row in rows)
        if value is not None
    ]
    rss_peak = max(
        [value for value in (rss_before_bytes, rss_after_bytes, *row_rss) if value is not None],
        default=None,
    )
    rss_delta = (
        rss_after_bytes - rss_before_bytes
        if rss_before_bytes is not None and rss_after_bytes is not None
        else None
    )
    sizes = {repository_relative(path): _file_size(path) for path in artifact_paths}
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": args.embedding_dim,
        "elapsed_ms": elapsed_ms,
        "rss_available": rss_peak is not None,
        "rss_unavailable_reason": rss_error if rss_peak is None else None,
        "rss_measurement": (
            "psutil.Process(os.getpid()).memory_info().rss sampled before/after "
            "the sweep and immediately before/after each retrieval"
        ),
        "rss_before_bytes": rss_before_bytes,
        "rss_after_bytes": rss_after_bytes,
        "rss_peak_bytes": rss_peak,
        "rss_delta_bytes": rss_delta,
        "peak_tracemalloc_bytes": peak_tracemalloc_bytes,
        "max_estimated_memory_storage_bytes": max(
            (int(row["estimated_memory_storage_bytes"]) for row in rows), default=0
        ),
        "estimated_memory_storage_definition": (
            "UTF-8 bytes of memory content and selected metadata plus portrait/summary; "
            "an estimate, not persisted disk usage"
        ),
        "max_faiss_index_size": max(
            (int(row["faiss_index_size"]) for row in rows), default=0
        ),
        "max_estimated_faiss_index_bytes": max(
            (int(row["estimated_faiss_index_bytes"]) for row in rows), default=0
        ),
        "result_artifact_bytes": sum(sizes.values()),
        "artifact_file_bytes": sizes,
        "token_metric": TOKEN_METRIC,
        "token_proxy_definition": metric_definitions()["token_proxy"],
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
    }


def build_manifest(
    args: argparse.Namespace,
    run_id: str,
    git_commit: str | None,
    git_error: str | None,
    rows: list[dict[str, Any]],
    configuration_rows: list[dict[str, Any]],
    skipped_invalid_combinations: list[dict[str, int]],
    artifacts: list[Path],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_commit_error": git_error,
        "command": reproducible_command(args, run_id),
        "seed": args.seed,
        "grid": grid_from_args(args),
        "dataset_source": DATASET_SOURCE,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_dim": args.embedding_dim,
        "current_time": args.current_time,
        "forgetting_threshold": list(args.forgetting_threshold),
        "repeat": args.repeat,
        "reinforcement": False,
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
        "row_count": len(rows),
        "configuration_count": len(configuration_rows),
        "probe_count": len(PROBES),
        "skipped_invalid_combinations": skipped_invalid_combinations,
        "token_metric": TOKEN_METRIC,
        "output_files": [
            artifact_metadata(path, base_dir=args.results_root)
            for path in artifacts
        ],
        "limitations": limitations(),
    }


def generate_figures(
    configurations: list[dict[str, Any]], figures_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figures_dir / "token_budget_quality.png",
        figures_dir / "topk_candidatek_quality.png",
        figures_dir / "candidatek_latency.png",
        figures_dir / "memory_count_resources.png",
        figures_dir / "quality_resource_tradeoff.png",
    ]

    token_rows, token_anchor = select_anchor_rows(
        configurations,
        varying="token_budget",
        preferred={"top_k": 3, "candidate_k": 20, "memory_count": 1000},
    )
    x = [row["token_budget"] for row in token_rows]
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    axis.plot(x, [row["memory_retrieval_accuracy"] for row in token_rows], marker="o", label="Retrieval accuracy")
    axis.plot(x, [row["stale_retrieval_rate"] for row in token_rows], marker="s", label="Stale retrieval rate")
    axis.set(xlabel="Token budget (proxy units)", ylabel="Rate", title=f"Token-budget quality ({format_anchor(token_anchor)})")
    axis.grid(alpha=0.3); axis.legend(); fig.tight_layout(); fig.savefig(paths[0], dpi=170); plt.close(fig)

    pair_rows, pair_anchor = select_anchor_rows(
        configurations,
        varying=("top_k", "candidate_k"),
        preferred={"token_budget": 256, "memory_count": 1000},
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    for candidate_k in sorted({row["candidate_k"] for row in pair_rows}):
        series = sorted((row for row in pair_rows if row["candidate_k"] == candidate_k), key=lambda row: row["top_k"])
        axis.plot([row["top_k"] for row in series], [row["memory_retrieval_accuracy"] for row in series], marker="o", label=f"candidate_k={candidate_k}")
    axis.set(xlabel="Top-K", ylabel="Retrieval accuracy", title=f"Top-K / candidate-K quality ({format_anchor(pair_anchor)})")
    axis.grid(alpha=0.3); axis.legend(); fig.tight_layout(); fig.savefig(paths[1], dpi=170); plt.close(fig)

    candidate_rows, candidate_anchor = select_anchor_rows(
        configurations,
        varying="candidate_k",
        preferred={"top_k": 3, "token_budget": 256, "memory_count": 1000},
    )
    x = [row["candidate_k"] for row in candidate_rows]
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    axis.plot(x, [row["mean_retrieval_latency_ms"] for row in candidate_rows], marker="o", label="Mean latency")
    axis.plot(x, [row["p95_retrieval_latency_ms"] for row in candidate_rows], marker="s", label="P95 latency")
    axis.set(xlabel="Candidate-K", ylabel="Retrieval latency (ms)", title=f"Candidate-K latency ({format_anchor(candidate_anchor)})")
    axis.grid(alpha=0.3); axis.legend(); fig.tight_layout(); fig.savefig(paths[2], dpi=170); plt.close(fig)

    memory_rows, memory_anchor = select_anchor_rows(
        configurations,
        varying="memory_count",
        preferred={"top_k": 3, "candidate_k": 20, "token_budget": 256},
    )
    x = [row["memory_count"] for row in memory_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    axes[0].plot(x, [row["mean_retrieval_latency_ms"] for row in memory_rows], marker="o", label="Mean latency")
    axes[0].plot(x, [row["p95_retrieval_latency_ms"] for row in memory_rows], marker="s", label="P95 latency")
    axes[0].set(xlabel="Memory count", ylabel="Latency (ms)"); axes[0].grid(alpha=0.3); axes[0].legend()
    storage_lines = axes[1].plot(
        x,
        [row["mean_estimated_memory_storage_bytes"] for row in memory_rows],
        color="tab:blue",
        marker="o",
        label="Estimated memory storage",
    )
    resource_lines = list(storage_lines)
    resource_axis = axes[1]
    if all(row["mean_retrieval_rss_peak_bytes"] is not None for row in memory_rows):
        resource_axis = axes[1].twinx()
        resource_lines.extend(
            resource_axis.plot(
                x,
                [row["mean_retrieval_rss_peak_bytes"] for row in memory_rows],
                color="tab:orange",
                marker="s",
                label="Process RSS",
            )
        )
        resource_axis.set_ylabel("Process RSS (bytes)", color="tab:orange")
    axes[1].set(
        xlabel="Memory count",
        ylabel="Estimated storage (bytes)",
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend(resource_lines, [line.get_label() for line in resource_lines])
    fig.suptitle(f"Memory-count resources ({format_anchor(memory_anchor)})"); fig.tight_layout(); fig.savefig(paths[3], dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    scatter = axis.scatter(
        [row["mean_selected_token_proxy"] for row in configurations],
        [row["memory_retrieval_accuracy"] for row in configurations],
        c=[row["mean_retrieval_latency_ms"] for row in configurations],
        s=28,
        alpha=0.7,
    )
    axis.set(xlabel="Mean selected token proxy (proxy units)", ylabel="Retrieval accuracy", title="Quality-resource tradeoff (all legal configurations)")
    fig.colorbar(scatter, ax=axis, label="Mean retrieval latency (ms)")
    axis.grid(alpha=0.2); fig.tight_layout(); fig.savefig(paths[4], dpi=170); plt.close(fig)
    return paths


def select_anchor_rows(
    rows: list[dict[str, Any]],
    varying: str | tuple[str, ...],
    preferred: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    varying_keys = {varying} if isinstance(varying, str) else set(varying)
    fixed_keys = [key for key in ("token_budget", "top_k", "candidate_k", "memory_count") if key not in varying_keys]
    available = {key: sorted({row[key] for row in rows}) for key in fixed_keys}
    anchor = {
        key: preferred.get(key) if preferred.get(key) in available[key] else available[key][0]
        for key in fixed_keys
    }
    selected = [row for row in rows if all(row[key] == value for key, value in anchor.items())]
    sort_keys = (varying,) if isinstance(varying, str) else varying
    selected.sort(key=lambda row: tuple(row[key] for key in sort_keys))
    return selected, anchor


def format_anchor(anchor: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in anchor.items())


def metric_definitions() -> dict[str, str]:
    return {
        "memory_retrieval_accuracy": "Keyword coverage of relevant probe terms in budget-selected memories.",
        "answer_accuracy": "Deterministic template-answer keyword coverage proxy; not LLM answer quality.",
        "stale_retrieval_rate": "Fraction of selected memories matching probe-specific stale keywords.",
        "retrieval_latency_ms": "Wall-clock dense retrieval time measured with perf_counter_ns; excludes token-budget selection.",
        "estimated_memory_storage_bytes": "Estimated UTF-8 bytes of memory content and selected metadata; not persisted disk usage.",
        "result_artifact_bytes": "Sum of generated compact CSV, JSON, and PNG artifact file sizes.",
        "rss": "Resident set size of the whole Python process sampled with psutil; includes runtime, NumPy, FAISS, and loaded dependencies.",
        "token_proxy": "ASCII alphanumeric runs count as one proxy unit; each non-ASCII non-whitespace character counts as one proxy unit. This is not a real tokenizer.",
    }


def limitations() -> list[str]:
    return [
        "Only three deterministic synthetic probes are used.",
        "Hash embedding is not MiniLM or another semantic production encoder.",
        "The token metric is a deterministic proxy, not a real tokenizer.",
        "Template-answer keyword coverage is not LLM answer quality.",
        "RSS includes the full Python process and is affected by allocator and OS behavior.",
        "Latency is specific to the local machine and run conditions.",
        "This sweep does not compare the overall quality of the four StateBudgetMem methods.",
        "Formal method conclusions remain in results/fair_comparison.",
    ]


def grid_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "token_budget": list(args.token_budget),
        "top_k": list(args.top_k),
        "candidate_k": list(args.candidate_k),
        "memory_count": list(args.memory_count),
        "forgetting_threshold": list(args.forgetting_threshold),
        "repeat": args.repeat,
        "probe_count": len(PROBES),
    }


def reproducible_command(args: argparse.Namespace, run_id: str) -> str:
    values = [
        ".venv\\Scripts\\python.exe tools\\memorybank\\run_budget_sweep.py",
        f"--results-root {repository_relative(args.results_root)}",
        f"--run-id {run_id}",
        "--token-budget " + " ".join(map(str, args.token_budget)),
        "--top-k " + " ".join(map(str, args.top_k)),
        "--candidate-k " + " ".join(map(str, args.candidate_k)),
        "--memory-count " + " ".join(map(str, args.memory_count)),
        "--forgetting-threshold " + " ".join(map(str, args.forgetting_threshold)),
        f"--repeat {args.repeat}",
        f"--seed {args.seed}",
        f"--embedding-dim {args.embedding_dim}",
        f'--current-time "{args.current_time}"',
    ]
    return " ".join(values)


def get_git_commit() -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip(), None
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, str(exc)


def get_rss_process() -> tuple[Any | None, str | None]:
    try:
        import psutil

        return psutil.Process(os.getpid()), None
    except (ImportError, OSError) as exc:
        return None, str(exc)


def read_rss(process: Any | None) -> int | None:
    if process is None:
        return None
    try:
        return int(process.memory_info().rss)
    except (AttributeError, OSError):
        return None


def percentile(values: Iterable[float], quantile: float) -> float:
    selected = sorted(float(value) for value in values)
    if not selected:
        return 0.0
    if len(selected) == 1:
        return selected[0]
    position = (len(selected) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return selected[lower]
    fraction = position - lower
    return selected[lower] + (selected[upper] - selected[lower]) * fraction


def artifact_metadata(
    path: Path,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": portable_artifact_path(path, base_dir=base_dir),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def portable_artifact_path(path: Path, base_dir: Path | None = None) -> str:
    """Return a non-absolute path while retaining output subdirectories."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        if base_dir is not None:
            try:
                return resolved.relative_to(base_dir.resolve()).as_posix()
            except ValueError:
                pass
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row.get(key, 0.0) or 0.0) for row in rows) if rows else 0.0


def _max_optional(*values: int | None) -> int | None:
    selected = [value for value in values if value is not None]
    return max(selected) if selected else None


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        serialized = dict(row)
        for key, value in serialized.items():
            if isinstance(value, (list, dict)):
                serialized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        writer.writerow(serialized)
    _write_utf8_lf_text(path, buffer.getvalue())


def _write_json(path: Path, payload: Any) -> None:
    _write_utf8_lf_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _write_utf8_lf_text(path: Path, text: str) -> None:
    """Write deterministic UTF-8 bytes without platform newline translation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_bytes(normalized.encode("utf-8"))


def _write_json_stable_size(path: Path, payload: dict[str, Any]) -> None:
    """Include the resource JSON's own size once its digit width stabilizes."""

    for _ in range(5):
        _write_json(path, payload)
        size = _file_size(path)
        files = payload["artifact_file_bytes"]
        previous = files.get(repository_relative(path))
        files[repository_relative(path)] = size
        payload["result_artifact_bytes"] = sum(files.values())
        if previous == size:
            break
    _write_json(path, payload)


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main())
