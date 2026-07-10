#!/usr/bin/env python3
"""Run MemoryBank on-device budget sweep experiments.

This script can run before the formal MemoryBank-style dataset is available. It
constructs a deterministic local memory bank with controlled relevant, stale,
and filler memories, then sweeps retrieval and device-resource budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

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


DEFAULT_TOP_K = (1, 3, 5)
DEFAULT_PROMPT_TOKEN_BUDGETS = (128, 256, 512, 1024)
DEFAULT_MEMORY_COUNTS = (100, 500, 1000, 5000)
DEFAULT_FORGETTING_THRESHOLDS = (0.1, 0.3, 0.5)


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
    """Deterministic local encoder for budget-sweep smoke reproduction."""

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
    parser = argparse.ArgumentParser(
        description="Sweep MemoryBank top-k, prompt-token, storage, and forgetting budgets.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/memorybank/budget_sweep"),
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K))
    parser.add_argument(
        "--prompt-token-budget",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROMPT_TOKEN_BUDGETS),
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
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--current-time", default="2026-07-10 10:00")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke sweep: top_k=1,3; token=128,512; memory=100; threshold=0.3.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.top_k = [1, 3]
        args.prompt_token_budget = [128, 512]
        args.memory_count = [100]
        args.forgetting_threshold = [0.3]

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "memorybank_budget_sweep_%Y%m%dT%H%M%SZ"
    )
    raw_dir = args.results_root / "raw"
    summary_dir = args.results_root / "summaries"
    resources_dir = args.results_root / "resources"
    for directory in (raw_dir, summary_dir, resources_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{run_id}.jsonl"
    summary_path = summary_dir / f"{run_id}.json"
    resources_path = resources_dir / f"{run_id}.json"

    tracemalloc.start()
    started = time.perf_counter()
    rows = run_budget_sweep(args, run_id=run_id)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    _write_jsonl(raw_path, rows)
    summary = build_budget_summary(
        rows,
        args=args,
        run_id=run_id,
        raw_path=raw_path,
        summary_path=summary_path,
        resources_path=resources_path,
        elapsed_ms=elapsed_ms,
    )
    _write_json(summary_path, summary)
    resources = build_budget_resources(
        rows,
        args=args,
        run_id=run_id,
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


def run_budget_sweep(args: argparse.Namespace, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for memory_count, threshold in product(
        args.memory_count,
        args.forgetting_threshold,
    ):
        memory_bank = build_synthetic_memory_bank(
            memory_count=memory_count,
            forgetting_threshold=threshold,
            embedding_dim=args.embedding_dim,
        )
        forgetting_preview = memory_bank.forgetting_log(current_time=args.current_time)
        forgotten_memory_ids = set(forgetting_preview["forgotten_memory_ids"])

        for top_k, prompt_budget, probe in product(
            args.top_k,
            args.prompt_token_budget,
            PROBES,
        ):
            row = run_budget_probe(
                memory_bank=memory_bank,
                probe=probe,
                top_k=top_k,
                prompt_token_budget=prompt_budget,
                memory_count=memory_count,
                forgetting_threshold=threshold,
                forgotten_memory_ids=forgotten_memory_ids,
                current_time=args.current_time,
                run_id=run_id,
            )
            rows.append(row)
    return rows


def build_synthetic_memory_bank(
    memory_count: int,
    forgetting_threshold: float,
    embedding_dim: int,
) -> MemoryBank:
    try:
        memory_bank = MemoryBank(
            forgetting_threshold=forgetting_threshold,
            embedding_model=HashEmbeddingModel(dim=embedding_dim),
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    seed_memories = [
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
    ]
    for role, content, timestamp in seed_memories:
        memory_bank.store_dialog(role, content, timestamp)

    filler_needed = max(0, memory_count - len(seed_memories))
    for index in range(filler_needed):
        topic = index % 20
        content = (
            f"Filler memory {index:05d}: user discussed neutral topic {topic}, "
            f"daily planning detail {index % 7}, and local note {index % 13}."
        )
        day = 1 + (index % 28)
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
    prompt_token_budget: int,
    memory_count: int,
    forgetting_threshold: float,
    forgotten_memory_ids: set[str],
    current_time: str,
    run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt_context = memory_bank.build_augmented_prompt(
        query=probe.query,
        current_time=current_time,
        top_k=top_k,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    selected_memories, selected_token_cost = select_memories_for_prompt_budget(
        prompt_context["retrieved_memories"],
        prompt_token_budget,
    )
    answer = template_answer(probe.query, selected_memories)
    stats = memory_bank.get_stats()
    storage_size = estimate_memory_storage_size(memory_bank)
    row = {
        "run_id": run_id,
        "query_id": probe.query_id,
        "query": probe.query,
        "top_k": top_k,
        "prompt_token_budget": prompt_token_budget,
        "memory_count_budget": memory_count,
        "forgetting_threshold": forgetting_threshold,
        "retrieved_count_before_budget": prompt_context["retrieved_count"],
        "retrieved_memory_ids_before_budget": prompt_context["retrieved_memory_ids"],
        "selected_count_after_budget": len(selected_memories),
        "selected_memory_ids_after_budget": [
            str(item.get("memory_id", "")) for item in selected_memories
        ],
        "retrieved_memories": selected_memories,
        "dropped_memory_ids_by_prompt_budget": [
            str(item.get("memory_id", ""))
            for item in prompt_context["retrieved_memories"]
            if str(item.get("memory_id", ""))
            not in {str(row.get("memory_id", "")) for row in selected_memories}
        ],
        "forgotten_retrieved_memory_ids": [
            str(item.get("memory_id", ""))
            for item in selected_memories
            if str(item.get("memory_id", "")) in forgotten_memory_ids
        ],
        "template_answer": answer,
        "retrieval_latency_ms": latency_ms,
        "latency_ms": latency_ms,
        "prompt_token_cost": selected_token_cost,
        "prompt_token_estimate": selected_token_cost,
        "faiss_index_size": int(stats.get("index_size", 0) or 0),
        "index_size": int(stats.get("index_size", 0) or 0),
        "storage_size_bytes": storage_size,
        "memory_stats": stats,
        "local_only": True,
        "cloud_api_used": False,
    }
    metric_spec = MemoryBankMetricSpec(
        query_id=probe.query_id,
        relevant_keywords=probe.relevant_keywords,
        answer_keywords=probe.answer_keywords,
        stale_keywords=probe.stale_keywords,
    )
    row["paper_metrics"] = evaluate_reproduction_row(row, metric_spec)
    row["budget_pressure"] = {
        "relevant_memory_lost": row["paper_metrics"]["memory_retrieval_accuracy"] < 1.0,
        "stale_memory_retrieved": row["paper_metrics"]["stale_retrieval_rate"] > 0.0,
        "prompt_budget_used_ratio": (
            selected_token_cost / prompt_token_budget if prompt_token_budget else 0.0
        ),
    }
    return row


def select_memories_for_prompt_budget(
    retrieved_memories: list[dict[str, Any]],
    prompt_token_budget: int,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    used_tokens = 0
    for memory in retrieved_memories:
        token_cost = estimate_token_count(str(memory.get("content", "")))
        if selected and used_tokens + token_cost > prompt_token_budget:
            continue
        if not selected and token_cost > prompt_token_budget:
            continue
        selected.append(memory)
        used_tokens += token_cost
    return selected, used_tokens


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


def template_answer(query: str, memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "I do not have enough retrieved memory to answer confidently."
    return f"Based on the selected memories for '{query}', " + " ".join(
        str(item.get("content", "")) for item in memories
    )


def estimate_memory_storage_size(memory_bank: MemoryBank) -> int:
    total = 0
    for memory in memory_bank.get_all():
        total += len(str(getattr(memory, "content", "")).encode("utf-8"))
        total += len(str(getattr(memory, "memory_id", "")).encode("utf-8"))
        total += len(str(getattr(memory, "memory_type", "")).encode("utf-8"))
        total += len(str(getattr(memory, "tags", "")).encode("utf-8"))
    total += len(str(getattr(memory_bank, "user_portrait", "")).encode("utf-8"))
    total += len(str(getattr(memory_bank, "global_summary", "")).encode("utf-8"))
    return total


def build_budget_summary(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    raw_path: Path,
    summary_path: Path,
    resources_path: Path,
    elapsed_ms: float,
) -> dict[str, Any]:
    metric_rows = [dict(row.get("paper_metrics", {})) for row in rows]
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "memorybank_budget_sweep",
        "dataset_source": "synthetic_memorybank_budget_probe",
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
        "sweep_grid": {
            "top_k": list(args.top_k),
            "prompt_token_budget": list(args.prompt_token_budget),
            "memory_count": list(args.memory_count),
            "forgetting_threshold": list(args.forgetting_threshold),
        },
        "run_count": len(rows),
        "query_count": len(PROBES),
        "paper_metrics": summarize_metric_rows(metric_rows),
        "budget_pressure": summarize_budget_pressure(rows),
        "by_budget": summarize_by_budget(rows),
        "elapsed_ms": elapsed_ms,
        "raw_jsonl_path": str(raw_path),
        "summary_json_path": str(summary_path),
        "resources_json_path": str(resources_path),
    }


def summarize_budget_pressure(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "relevant_loss_rate": 0.0,
            "stale_retrieval_case_rate": 0.0,
            "mean_prompt_budget_used_ratio": 0.0,
        }
    return {
        "relevant_loss_rate": sum(
            bool(row["budget_pressure"]["relevant_memory_lost"]) for row in rows
        )
        / len(rows),
        "stale_retrieval_case_rate": sum(
            bool(row["budget_pressure"]["stale_memory_retrieved"]) for row in rows
        )
        / len(rows),
        "mean_prompt_budget_used_ratio": statistics.fmean(
            float(row["budget_pressure"]["prompt_budget_used_ratio"]) for row in rows
        ),
    }


def summarize_by_budget(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "top_k": grouped_metric_summary(rows, "top_k"),
        "prompt_token_budget": grouped_metric_summary(rows, "prompt_token_budget"),
        "memory_count_budget": grouped_metric_summary(rows, "memory_count_budget"),
        "forgetting_threshold": grouped_metric_summary(rows, "forgetting_threshold"),
    }


def grouped_metric_summary(
    rows: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    output = []
    for value, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        metric_summary = summarize_metric_rows(
            [dict(row.get("paper_metrics", {})) for row in group_rows]
        )
        output.append(
            {
                key: value,
                "run_count": len(group_rows),
                **metric_summary,
                **summarize_budget_pressure(group_rows),
                "mean_storage_size_bytes": statistics.fmean(
                    float(row.get("storage_size_bytes", 0.0) or 0.0)
                    for row in group_rows
                ),
            }
        )
    return output


def build_budget_resources(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    raw_path: Path,
    summary_path: Path,
    resources_path: Path,
    elapsed_ms: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    max_index_size = max((int(row.get("faiss_index_size", 0) or 0) for row in rows), default=0)
    max_storage_size = max(
        (int(row.get("storage_size_bytes", 0) or 0) for row in rows),
        default=0,
    )
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local_only": True,
        "cloud_api_used": False,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "embedding_model": "deterministic_hash_embedding",
        "embedding_dim": args.embedding_dim,
        "elapsed_ms": elapsed_ms,
        "peak_tracemalloc_bytes": peak_memory_bytes,
        "max_faiss_index_size": max_index_size,
        "max_storage_size_bytes": max_storage_size,
        "output_file_bytes": {
            "raw_jsonl": _file_size(raw_path),
            "summary_json": _file_size(summary_path),
            "resources_json": _file_size(resources_path),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main())
