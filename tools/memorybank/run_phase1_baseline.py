#!/usr/bin/env python3
"""Phase 1: On-Device MemoryBank Baseline — Formal Runner.

Complete pipeline:
    dataset → MemoryBank storage → probing → retrieval → metrics → resources

Supports two modes:
    --smoke    Run on built-in smoke sample (no dataset needed)
    (default)  Run on data/memorybank_reproduction/ (requires Huang's dataset)
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statebudgetmem.baselines.memorybank import MemoryBank  # noqa: E402
from statebudgetmem.baselines.memorybank.embeddings import (  # noqa: E402
    deterministic_hash_embedding,
)
from statebudgetmem.baselines.memorybank.metrics import (  # noqa: E402
    MemoryBankMetricSpec,
    evaluate_reproduction_row,
    summarize_metric_rows,
)
from statebudgetmem.baselines.memorybank.paper_storage import (  # noqa: E402
    PaperStorageSpec,
    build_paper_aligned_storage,
    default_paper_storage_spec,
    default_retrieval_probe,
)

# ── Smoke probes (with gold labels for testing) ──────────────────────────

SMOKE_PROBES = [
    {
        "query_id": "q_smoke_book",
        "user_id": "smoke_user",
        "question": "What book did you recommend and what food should I avoid?",
        "reference_answer": "Automate the Boring Stuff with Python. Avoid spicy food.",
        "gold_memory_ids": [],
        "expected_keywords": ["Automate", "spicy"],
        "question_type": "memory_recall",
    },
    {
        "query_id": "q_smoke_hobby",
        "user_id": "smoke_user",
        "question": "What does the user like to do on weekends?",
        "reference_answer": "Basketball and swimming.",
        "gold_memory_ids": [],
        "expected_keywords": ["basketball", "swimming"],
        "question_type": "user_portrait",
    },
    {
        "query_id": "q_smoke_name",
        "user_id": "smoke_user",
        "question": "What is the user's name?",
        "reference_answer": "Lin.",
        "gold_memory_ids": [],
        "expected_keywords": ["Lin"],
        "question_type": "memory_recall",
    },
    {
        "query_id": "q_smoke_negative",
        "user_id": "smoke_user",
        "question": "Did we write heap sort together?",
        "reference_answer": "No, heap sort was never mentioned.",
        "gold_memory_ids": [],
        "expected_keywords": [],
        "question_type": "negative_memory",
    },
    {
        "query_id": "q_smoke_temporal",
        "user_id": "smoke_user",
        "question": "What happened between June 20 and June 23?",
        "reference_answer": (
            "Lin discussed study, hobbies, Python book, and stomach issues."
        ),
        "gold_memory_ids": [],
        "expected_keywords": ["stomach", "book", "exam"],
        "question_type": "temporal_memory",
    },
]

# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: On-Device MemoryBank Baseline Runner",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/memorybank_reproduction",
        help="Path to reproduction dataset directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on built-in smoke sample (bypasses dataset dir).",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "sentence-transformer"],
        default="hash",
        help="Hash = deterministic CI; sentence-transformer = real embedding.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Model name for sentence-transformer backend.",
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument(
        "--exclude-forgotten",
        action="store_true",
        help="Exclude forgotten memories from prompt.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="results/memorybank/phase1",
        help="Output root directory.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run the first user.",
    )
    parser.add_argument(
        "--forgetting-threshold",
        type=float,
        default=0.3,
    )
    parser.add_argument("--retention-time-unit-hours", type=float, default=24.0)
    parser.add_argument(
        "--current-time",
        default="2026-06-24 10:00",
        help="Reference time for retrieval.",
    )
    return parser.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "phase1_%Y%m%dT%H%M%SZ"
    )

    raw_dir = Path(args.output_dir) / "raw"
    summary_dir = Path(args.output_dir) / "summaries"
    resources_dir = Path(args.output_dir) / "resources"
    for d in (raw_dir, summary_dir, resources_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{run_id}.jsonl"
    summary_path = summary_dir / f"{run_id}.json"
    csv_path = summary_dir / f"{run_id}.csv"
    resources_path = resources_dir / f"{run_id}.json"

    tracemalloc.start()
    started = time.perf_counter()

    # 1. Build MemoryBank
    memory_bank = _build_memory_bank(args)
    dataset_source = (
        "built_in_smoke_sample" if args.smoke else str(args.dataset_dir)
    )

    # 2. Ingest data
    if args.smoke:
        build_paper_aligned_storage(memory_bank, default_paper_storage_spec())
        probes = SMOKE_PROBES
    else:
        from statebudgetmem.baselines.memorybank.datasets import (
            load_reproduction_dataset,
        )
        users, probes_data = load_reproduction_dataset(args.dataset_dir)
        if args.quick:
            users = users[:1]
            probes_data = [p for p in probes_data if p.user_id == users[0].user_id]
        _ingest_users(memory_bank, users)
        probes = [
            {
                "query_id": p.query_id,
                "user_id": p.user_id,
                "question": p.question,
                "reference_answer": p.reference_answer,
                "gold_memory_ids": p.gold_memory_ids,
                "expected_keywords": p.expected_keywords,
                "question_type": p.question_type,
            }
            for p in probes_data
        ]

    # 3. Run probes
    rows = _run_probes(
        memory_bank,
        probes,
        args,
        run_id=run_id,
        dataset_source=dataset_source,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 4. Output
    _write_jsonl(raw_path, rows)
    summary = _build_summary(rows, run_id, args, raw_path, summary_path,
                             csv_path, resources_path, elapsed_ms, dataset_source)
    _write_json(summary_path, summary)
    _write_csv(csv_path, rows)
    resources = _build_resources(args, run_id, rows, raw_path, summary_path,
                                 resources_path, elapsed_ms, peak_memory,
                                 memory_bank, dataset_source)
    _write_json(resources_path, resources)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# ── Ingestion ────────────────────────────────────────────────────────────


def _build_memory_bank(args: argparse.Namespace) -> MemoryBank:
    """Build a MemoryBank with the selected embedding backend."""
    if args.embedding_backend == "hash":
        class _HashEmbed:
            def __init__(self, dim: int):
                self.dim = dim

            def encode(self, text: str):
                return deterministic_hash_embedding(text, dim=self.dim)

        args.actual_embedding_dim = args.embedding_dim
        return MemoryBank(
            embedding_dim=args.embedding_dim,
            forgetting_threshold=args.forgetting_threshold,
            decay_interval_hours=args.retention_time_unit_hours,
            embedding_model=_HashEmbed(args.embedding_dim),
        )
    else:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(args.embedding_model)
        except Exception as e:
            raise SystemExit(
                f"Failed to load sentence-transformer model "
                f"'{args.embedding_model}'. Install optional dependencies with "
                f"pip install -e \".[memorybank]\". For offline runs, use a "
                f"cached model or pass a local model directory. Original error: {e}"
            ) from e
        actual_dim = int(model.get_sentence_embedding_dimension())
        args.actual_embedding_dim = actual_dim
        return MemoryBank(
            embedding_dim=actual_dim,
            forgetting_threshold=args.forgetting_threshold,
            decay_interval_hours=args.retention_time_unit_hours,
            embedding_model=model,
        )


def _embedding_metadata(args: argparse.Namespace) -> tuple[str, str]:
    if args.embedding_backend == "hash":
        return "hash", "deterministic_hash_embedding"
    return "sentence-transformer", args.embedding_model


def _embedding_dim(args: argparse.Namespace) -> int:
    return int(getattr(args, "actual_embedding_dim", args.embedding_dim))


def _ingest_users(memory_bank: MemoryBank, users: list) -> None:
    """Ingest reproduction dataset users into MemoryBank.

    Preserves the original ``memory_id`` values from the dataset so that
    gold labels (probing_questions.jsonl) can match retrieved memories.
    """
    from statebudgetmem.core.online import MemoryPiece, MemoryType as MT

    for user in users:
        for day in user.days:
            for turn in day.get("dialogues", []):
                content = f"{turn.get('role', 'User')}: {turn.get('content', '')}"
                ts_str = str(turn.get("timestamp", ""))
                ts = memory_bank._parse_time(ts_str)
                memory = MemoryPiece(
                    content=content,
                    timestamp=ts,
                    memory_type=MT.DIALOG,
                    last_accessed=ts,
                    memory_id=str(turn.get("memory_id", "")),
                    tags=memory_bank._auto_tag(content),
                )
                memory_bank._insert_memory(memory)
            if day.get("daily_event_summary"):
                memory_bank.store_summary(
                    str(day["daily_event_summary"]),
                    str(day.get("date", "")),
                )
        if getattr(user, "global_summary", ""):
            memory_bank.update_global_summary(str(user.global_summary))
        if getattr(user, "user_portrait", ""):
            memory_bank.update_user_portrait(str(user.user_portrait))


# ── Probing ──────────────────────────────────────────────────────────────


def _run_probes(
    memory_bank: MemoryBank,
    probes: list[dict],
    args: argparse.Namespace,
    run_id: str,
    dataset_source: str,
) -> list[dict]:
    rows: list[dict] = []
    embedding_backend, embedding_model = _embedding_metadata(args)
    embedding_dim = _embedding_dim(args)
    for probe in probes:
        started = time.perf_counter()
        prompt_ctx = memory_bank.build_augmented_prompt(
            query=probe["question"],
            current_time=args.current_time,
            top_k=args.top_k,
            exclude_forgotten=args.exclude_forgotten,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        retrieved = prompt_ctx["retrieved_memories"]
        retrieved_ids = [str(r.get("memory_id", "")) for r in retrieved]
        template = _template_answer(probe["question"], retrieved)
        stats = memory_bank.get_stats()
        index_size = int(stats.get("index_size", 0) or 0)

        row = {
            "run_id": run_id,
            "dataset_source": dataset_source,
            "user_id": probe.get("user_id", ""),
            "query_id": probe["query_id"],
            "question": probe["question"],
            "query": probe["question"],
            "question_type": probe.get("question_type", ""),
            "reference_answer": probe.get("reference_answer", ""),
            "retrieved_memory_ids": retrieved_ids,
            "gold_memory_ids": probe.get("gold_memory_ids", []),
            "expected_keywords": probe.get("expected_keywords", []),
            "retrieved_memories": retrieved,
            "template_answer": template,
            "retrieval_latency_ms": latency_ms,
            "latency_ms": latency_ms,
            "prompt_sections": prompt_ctx.get("prompt_sections", {}),
            "prompt_template": prompt_ctx.get("prompt_template", ""),
            "prompt_token_estimate": prompt_ctx.get("prompt_token_estimate", 0),
            "retrieved_count": prompt_ctx.get("retrieved_count", len(retrieved)),
            "forgotten_memory_ids": prompt_ctx.get("forgotten_memory_ids", []),
            "excluded_forgotten_memory_ids": prompt_ctx.get(
                "excluded_forgotten_memory_ids",
                [],
            ),
            "excluded_forgotten_count": prompt_ctx.get(
                "excluded_forgotten_count",
                0,
            ),
            "candidate_count_before_forgetting": prompt_ctx.get(
                "candidate_count_before_forgetting",
                0,
            ),
            "candidate_count_after_forgetting": prompt_ctx.get(
                "candidate_count_after_forgetting",
                0,
            ),
            "exclude_forgotten": prompt_ctx.get(
                "exclude_forgotten",
                args.exclude_forgotten,
            ),
            "forgetting_threshold": prompt_ctx.get(
                "forgetting_threshold",
                args.forgetting_threshold,
            ),
            "strength_before_after": prompt_ctx.get("strength_before_after", []),
            "last_accessed_before_after": prompt_ctx.get(
                "last_accessed_before_after",
                [],
            ),
            "access_count_before_after": prompt_ctx.get(
                "access_count_before_after",
                [],
            ),
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "index_size": index_size,
            "retention_time_unit_hours": prompt_ctx.get(
                "retention_time_unit_hours",
                args.retention_time_unit_hours,
            ),
            "local_only": True,
            "cloud_api_used": False,
            "llm_called": False,
        }

        spec = MemoryBankMetricSpec(
            query_id=probe["query_id"],
            relevant_keywords=tuple(
                probe.get("expected_keywords", [])
            ),
            answer_keywords=tuple(
                probe.get("expected_keywords", [])
            ),
        )
        row["paper_metrics"] = evaluate_reproduction_row(row, spec)
        rows.append(row)
    return rows


def _template_answer(query: str, memories: list[dict]) -> str:
    if not memories:
        return "I do not have enough retrieved memory to answer."
    parts = [str(m.get("content", "")) for m in memories[:3]]
    return f"Based on memories for '{query}': " + " ".join(parts)


# ── Output ───────────────────────────────────────────────────────────────


def _build_summary(
    rows: list[dict],
    run_id: str,
    args: argparse.Namespace,
    raw_path: Path,
    summary_path: Path,
    csv_path: Path,
    resources_path: Path,
    elapsed_ms: float,
    dataset_source: str,
) -> dict:
    embedding_backend, embedding_model = _embedding_metadata(args)
    embedding_dim = _embedding_dim(args)
    metric_rows = [dict(r.get("paper_metrics", {})) for r in rows]
    forgotten_counts = [len(r.get("forgotten_memory_ids", [])) for r in rows]
    excluded_counts = [
        len(r.get("excluded_forgotten_memory_ids", [])) for r in rows
    ]
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        qt = r.get("question_type", "unknown")
        by_type.setdefault(qt, []).append(dict(r.get("paper_metrics", {})))

    by_question_type = {
        qt: summarize_metric_rows(group) for qt, group in by_type.items()
    }

    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "memorybank_phase1_baseline",
        "dataset_source": dataset_source,
        "smoke_mode": args.smoke,
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "forgetting_threshold": args.forgetting_threshold,
        "retention_time_unit_hours": args.retention_time_unit_hours,
        "exclude_forgotten": args.exclude_forgotten,
        "top_k": args.top_k,
        "probe_count": len(rows),
        "total_forgotten_candidates": sum(forgotten_counts),
        "total_excluded_forgotten": sum(excluded_counts),
        "mean_forgotten_candidates_per_query": (
            statistics.fmean(forgotten_counts) if forgotten_counts else 0.0
        ),
        "mean_excluded_forgotten_per_query": (
            statistics.fmean(excluded_counts) if excluded_counts else 0.0
        ),
        "paper_metrics": summarize_metric_rows(metric_rows),
        "by_question_type": by_question_type,
        "elapsed_ms": elapsed_ms,
        "raw_jsonl_path": str(raw_path),
        "summary_json_path": str(summary_path),
        "summary_csv_path": str(csv_path),
        "resources_json_path": str(resources_path),
    }


def _build_resources(
    args: argparse.Namespace,
    run_id: str,
    rows: list[dict],
    raw_path: Path,
    summary_path: Path,
    resources_path: Path,
    elapsed_ms: float,
    peak_memory: int,
    memory_bank: Any,
    dataset_source: str,
) -> dict:
    stats = memory_bank.get_stats()
    embedding_backend, embedding_model = _embedding_metadata(args)
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "local_only": True,
        "cloud_api_used": False,
        "llm_called": False,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "embedding_dim": _embedding_dim(args),
        "forgetting_threshold": args.forgetting_threshold,
        "retention_time_unit_hours": args.retention_time_unit_hours,
        "exclude_forgotten": args.exclude_forgotten,
        "top_k": args.top_k,
        "elapsed_ms": elapsed_ms,
        "peak_tracemalloc_bytes": peak_memory,
        "faiss_index_size": int(stats.get("index_size", 0) or 0),
        "storage_size_bytes": _estimate_storage(memory_bank),
        "output_file_bytes": {
            "raw_jsonl": _file_size(raw_path),
            "summary_json": _file_size(summary_path),
            "resources_json": _file_size(resources_path),
        },
    }


def _estimate_storage(memory_bank: Any) -> int:
    total = 0
    for m in memory_bank.get_all():
        total += len(str(getattr(m, "content", "")).encode("utf-8"))
    return total


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = [
        "query_id", "question", "question_type",
        "memory_retrieval_accuracy", "response_correctness",
        "contextual_coherence", "gold_precision", "gold_recall", "gold_f1",
        "retrieval_latency_ms",
    ]
    lines = [",".join(keys)]
    for r in rows:
        metrics = r.get("paper_metrics", {})
        vals = []
        for k in keys:
            if k in r:
                vals.append(str(r[k]).replace(",", ";"))
            elif k in metrics:
                vals.append(str(metrics[k]))
            else:
                vals.append("")
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main())
