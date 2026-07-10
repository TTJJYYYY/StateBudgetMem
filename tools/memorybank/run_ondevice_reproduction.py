#!/usr/bin/env python3
"""Run the local/on-device MemoryBank reproduction pipeline.

The runner uses the built-in paper-aligned sample until a formal
MemoryBank-style dataset is available. It records retrieval and prompt
construction outputs without calling a cloud LLM.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statebudgetmem.baselines.memorybank import (  # noqa: E402
    MemoryBank,
    build_paper_aligned_storage,
)
from statebudgetmem.baselines.memorybank.embeddings import (  # noqa: E402
    deterministic_hash_embedding,
)
from statebudgetmem.baselines.memorybank.metrics import (  # noqa: E402
    MemoryBankMetricSpec,
    evaluate_reproduction_row,
    summarize_metric_rows,
)


DEFAULT_QUERIES = [
    "What book did you recommend?",
    "What food should I avoid now?",
    "What do you remember about my hobbies?",
]

DEFAULT_METRIC_SPECS = {
    "q001": MemoryBankMetricSpec(
        query_id="q001",
        relevant_keywords=("book", "python"),
        answer_keywords=("book", "python"),
    ),
    "q002": MemoryBankMetricSpec(
        query_id="q002",
        relevant_keywords=("avoid", "spicy", "stomach"),
        answer_keywords=("avoid", "spicy"),
    ),
    "q003": MemoryBankMetricSpec(
        query_id="q003",
        relevant_keywords=("basketball", "swimming"),
        answer_keywords=("basketball", "swimming"),
    ),
}


class HashEmbeddingModel:
    """Deterministic local encoder used for offline smoke reproduction."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, text: str):
        return deterministic_hash_embedding(text, dim=self.dim)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local MemoryBank reproduction and write raw, summary, and resource logs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/memorybank/ondevice"),
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to run. Repeat the flag for multiple queries.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--forgetting-threshold", type=float, default=0.3)
    parser.add_argument("--retention-time-unit-hours", type=float, default=24.0)
    parser.add_argument(
        "--exclude-forgotten",
        action="store_true",
        help=(
            "Exclude candidates whose retention is below the "
            "forgetting threshold from retrieval and prompt."
        ),
    )
    parser.add_argument(
        "--current-time",
        default="2026-06-24 10:00",
        help="Reference time passed to MemoryBank retrieval.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id. Defaults to a UTC timestamp.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "memorybank_ondevice_%Y%m%dT%H%M%SZ"
    )
    queries = args.queries or DEFAULT_QUERIES

    raw_dir = args.results_root / "raw"
    summary_dir = args.results_root / "summaries"
    resources_dir = args.results_root / "resources"
    for directory in (raw_dir, summary_dir, resources_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{run_id}.jsonl"
    summary_path = summary_dir / f"{run_id}.json"
    resources_path = resources_dir / f"{run_id}.json"

    try:
        memory_bank = MemoryBank(
            forgetting_threshold=args.forgetting_threshold,
            decay_interval_hours=args.retention_time_unit_hours,
            embedding_model=HashEmbeddingModel(dim=args.embedding_dim),
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    tracemalloc.start()
    run_started = time.perf_counter()
    storage_report = build_paper_aligned_storage(memory_bank)
    raw_rows = [
        _run_query(
            memory_bank=memory_bank,
            query=query,
            query_index=index,
            top_k=args.top_k,
            current_time=args.current_time,
            run_id=run_id,
            exclude_forgotten=args.exclude_forgotten,
            embedding_dim=args.embedding_dim,
            retention_time_unit_hours=args.retention_time_unit_hours,
        )
        for index, query in enumerate(queries, start=1)
    ]
    elapsed_ms = (time.perf_counter() - run_started) * 1000.0
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    _write_jsonl(raw_path, raw_rows)
    summary = _build_summary(
        run_id=run_id,
        args=args,
        raw_rows=raw_rows,
        raw_path=raw_path,
        summary_path=summary_path,
        resources_path=resources_path,
        storage_report=storage_report,
        elapsed_ms=elapsed_ms,
    )
    _write_json(summary_path, summary)

    resources = _build_resources(
        run_id=run_id,
        args=args,
        memory_bank=memory_bank,
        raw_path=raw_path,
        summary_path=summary_path,
        resources_path=resources_path,
        elapsed_ms=elapsed_ms,
        peak_memory_bytes=peak_memory_bytes,
    )
    _write_json(resources_path, resources)
    resources["output_file_bytes"]["resources_json"] = _file_size(resources_path)
    _write_json(resources_path, resources)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_query(
    memory_bank: MemoryBank,
    query: str,
    query_index: int,
    top_k: int,
    current_time: str,
    run_id: str,
    exclude_forgotten: bool,
    embedding_dim: int,
    retention_time_unit_hours: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt_context = memory_bank.build_augmented_prompt(
        query=query,
        current_time=current_time,
        top_k=top_k,
        exclude_forgotten=exclude_forgotten,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    prompt = str(prompt_context["prompt_template"])
    stats = memory_bank.get_stats()

    row = {
        "run_id": run_id,
        "query_id": f"q{query_index:03d}",
        "query": query,
        "current_time": current_time,
        "top_k": top_k,
        "retrieved_memory_ids": prompt_context["retrieved_memory_ids"],
        "retrieved_count": prompt_context["retrieved_count"],
        "retrieved_memories": prompt_context["retrieved_memories"],
        "prompt_sections": prompt_context["prompt_sections"],
        "prompt_template": prompt,
        "latency_ms": latency_ms,
        "prompt_token_estimate": prompt_context.get(
            "prompt_token_estimate",
            estimate_token_count(prompt),
        ),
        "forgotten_memory_ids": prompt_context.get("forgotten_memory_ids", []),
        "excluded_forgotten_memory_ids": prompt_context.get(
            "excluded_forgotten_memory_ids",
            [],
        ),
        "excluded_forgotten_count": prompt_context.get(
            "excluded_forgotten_count",
            0,
        ),
        "candidate_count_before_forgetting": prompt_context.get(
            "candidate_count_before_forgetting",
            0,
        ),
        "candidate_count_after_forgetting": prompt_context.get(
            "candidate_count_after_forgetting",
            0,
        ),
        "exclude_forgotten": prompt_context.get(
            "exclude_forgotten",
            exclude_forgotten,
        ),
        "forgetting_threshold": prompt_context.get("forgetting_threshold"),
        "retention_time_unit_hours": prompt_context.get(
            "retention_time_unit_hours",
            retention_time_unit_hours,
        ),
        "strength_before_after": prompt_context.get("strength_before_after", []),
        "last_accessed_before_after": prompt_context.get(
            "last_accessed_before_after",
            [],
        ),
        "access_count_before_after": prompt_context.get(
            "access_count_before_after",
            [],
        ),
        "index_size": int(stats.get("index_size", 0) or 0),
        "memory_count": int(stats.get("total_memories", 0) or 0),
        "embedding_backend": "hash",
        "embedding_model": "deterministic_hash_embedding",
        "embedding_dim": embedding_dim,
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
    }
    row["template_answer"] = _template_answer(row)
    metric_spec = DEFAULT_METRIC_SPECS.get(
        row["query_id"],
        MemoryBankMetricSpec(query_id=row["query_id"]),
    )
    row["metric_spec"] = {
        "relevant_keywords": list(metric_spec.relevant_keywords),
        "answer_keywords": list(metric_spec.answer_keywords),
        "stale_keywords": list(metric_spec.stale_keywords),
    }
    row["paper_metrics"] = evaluate_reproduction_row(row, metric_spec)
    return row


def estimate_token_count(text: str) -> int:
    """Cheap deterministic token proxy for local budget experiments."""
    ascii_words = 0
    non_ascii_chars = 0
    current_word = False
    for char in text:
        if ord(char) < 128 and char.isalnum():
            if not current_word:
                ascii_words += 1
                current_word = True
        else:
            current_word = False
            if not char.isspace() and ord(char) >= 128:
                non_ascii_chars += 1
    return ascii_words + non_ascii_chars


def _template_answer(row: dict[str, Any]) -> str:
    """Local deterministic answer used before a local LLM is connected."""

    memories = [
        str(item.get("content", "")).strip()
        for item in row.get("retrieved_memories", [])
        if str(item.get("content", "")).strip()
    ]
    if not memories:
        return "I do not have enough retrieved memory to answer confidently."
    return f"Based on the retrieved memories for '{row['query']}', " + " ".join(
        memories
    )


def _build_summary(
    run_id: str,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    raw_path: Path,
    summary_path: Path,
    resources_path: Path,
    storage_report: dict[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in raw_rows]
    token_counts = [int(row["prompt_token_estimate"]) for row in raw_rows]
    retrieved_counts = [int(row["retrieved_count"]) for row in raw_rows]
    forgotten_counts = [
        len(row.get("forgotten_memory_ids", [])) for row in raw_rows
    ]
    excluded_counts = [
        len(row.get("excluded_forgotten_memory_ids", [])) for row in raw_rows
    ]
    paper_metric_rows = [dict(row.get("paper_metrics", {})) for row in raw_rows]
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "memorybank_ondevice_reproduction",
        "dataset_source": "built_in_paper_storage_spec",
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
        "embedding_backend": "hash",
        "embedding_model": "deterministic_hash_embedding",
        "embedding_dim": args.embedding_dim,
        "forgetting_threshold": args.forgetting_threshold,
        "retention_time_unit_hours": args.retention_time_unit_hours,
        "exclude_forgotten": getattr(args, "exclude_forgotten", False),
        "top_k": args.top_k,
        "query_count": len(raw_rows),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "mean_prompt_token_estimate": (
            statistics.fmean(token_counts) if token_counts else 0.0
        ),
        "mean_retrieved_count": statistics.fmean(retrieved_counts)
        if retrieved_counts
        else 0.0,
        "total_forgotten_candidates": sum(forgotten_counts),
        "total_excluded_forgotten": sum(excluded_counts),
        "mean_forgotten_candidates_per_query": (
            statistics.fmean(forgotten_counts) if forgotten_counts else 0.0
        ),
        "mean_excluded_forgotten_per_query": (
            statistics.fmean(excluded_counts) if excluded_counts else 0.0
        ),
        "paper_metrics": summarize_metric_rows(paper_metric_rows),
        "elapsed_ms": elapsed_ms,
        "raw_jsonl_path": str(raw_path),
        "summary_json_path": str(summary_path),
        "resources_json_path": str(resources_path),
        "storage_report": storage_report,
    }


def _build_resources(
    run_id: str,
    args: argparse.Namespace,
    memory_bank: MemoryBank,
    raw_path: Path,
    summary_path: Path,
    resources_path: Path,
    elapsed_ms: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    stats = memory_bank.get_stats()
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "embedding_backend": "hash",
        "embedding_model": "deterministic_hash_embedding",
        "embedding_dim": args.embedding_dim,
        "forgetting_threshold": args.forgetting_threshold,
        "retention_time_unit_hours": args.retention_time_unit_hours,
        "exclude_forgotten": getattr(args, "exclude_forgotten", False),
        "top_k": args.top_k,
        "elapsed_ms": elapsed_ms,
        "peak_tracemalloc_bytes": peak_memory_bytes,
        "memory_stats": stats,
        "index_size": int(stats.get("index_size", 0) or 0),
        "total_memories": int(stats.get("total_memories", 0) or 0),
        "output_file_bytes": {
            "raw_jsonl": _file_size(raw_path),
            "summary_json": _file_size(summary_path),
            "resources_json": _file_size(resources_path),
        },
        "storage_size_bytes": _memory_storage_size(memory_bank),
    }


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _memory_storage_size(memory_bank: MemoryBank) -> int:
    total = 0
    for memory in memory_bank.get_all():
        total += len(str(getattr(memory, "content", "")).encode("utf-8"))
        total += len(str(getattr(memory, "memory_id", "")).encode("utf-8"))
        total += len(str(getattr(memory, "memory_type", "")).encode("utf-8"))
        total += len(str(getattr(memory, "tags", "")).encode("utf-8"))
    total += len(str(getattr(memory_bank, "user_portrait", "")).encode("utf-8"))
    total += len(str(getattr(memory_bank, "global_summary", "")).encode("utf-8"))
    return total


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
