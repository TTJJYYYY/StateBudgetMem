#!/usr/bin/env python3
"""Run a local MemoryBank reinforcement/forgetting logging demo."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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


class HashEmbeddingModel:
    """Small deterministic local encoder for the reproduction demo.

    It avoids downloading sentence-transformers while still exercising the
    MemoryBank FAISS path with numeric vectors.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, text: str):
        return deterministic_hash_embedding(text, dim=self.dim)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MemoryBank reinforcement and forgetting logs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/memorybank/forgetting_demo"),
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--forgetting-threshold", type=float, default=0.3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--retention-time-unit-hours", type=float, default=24.0)
    parser.add_argument(
        "--recall-time",
        default="2026-06-21 10:00",
        help="Reference time for retrieval/reinforcement.",
    )
    parser.add_argument(
        "--forgetting-time",
        default="2026-07-10 10:00",
        help="Reference time for forgetting update.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        reinforcement_bank = _build_seeded_memory_bank(args)
        default_bank = _build_seeded_memory_bank(args)
        filtered_bank = _build_seeded_memory_bank(args)
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    queries = [
        "What Python book did you recommend?",
        "What food should I avoid now?",
    ]

    reinforcement_rows: list[dict[str, Any]] = []
    for query in queries:
        retrieved = reinforcement_bank.retrieve(
            query,
            top_k=args.top_k,
            current_time=args.recall_time,
        )
        for item in retrieved:
            reinforcement_rows.append(_reinforcement_row(item))

    forgetting_report = reinforcement_bank.update_forgetting_with_log(
        current_time=args.forgetting_time,
    )
    forgetting_rows = forgetting_report["events"]

    comparison_query = queries[1]
    default_context = default_bank.build_augmented_prompt(
        comparison_query,
        current_time=args.forgetting_time,
        top_k=args.top_k,
        exclude_forgotten=False,
    )
    before_filtered_state = _memory_state(filtered_bank)
    filtered_context = filtered_bank.build_augmented_prompt(
        comparison_query,
        current_time=args.forgetting_time,
        top_k=args.top_k,
        exclude_forgotten=True,
    )
    excluded_ids = filtered_context["excluded_forgotten_memory_ids"]
    excluded_memories_unchanged = bool(excluded_ids) and all(
        before_filtered_state.get(memory_id) == _memory_state(filtered_bank).get(memory_id)
        for memory_id in excluded_ids
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    reinforcement_path = args.results_dir / "memorybank_reinforcement_log.jsonl"
    forgetting_path = args.results_dir / "memorybank_forgetting_log.jsonl"
    summary_path = args.results_dir / "memorybank_forgetting_summary.json"
    _write_jsonl(reinforcement_path, reinforcement_rows)
    _write_jsonl(forgetting_path, forgetting_rows)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local_only": True,
        "cloud_api_used": False,
        "top_k": args.top_k,
        "forgetting_threshold": args.forgetting_threshold,
        "retention_time_unit_hours": args.retention_time_unit_hours,
        "recall_time": args.recall_time,
        "forgetting_time": args.forgetting_time,
        "reinforcement_log_path": str(reinforcement_path),
        "forgetting_log_path": str(forgetting_path),
        "forgotten_memory_ids": forgetting_report["forgotten_memory_ids"],
        "reinforcement_event_count": len(reinforcement_rows),
        "forgetting_event_count": len(forgetting_rows),
        "default_prompt_retrieved_memory_ids": default_context[
            "retrieved_memory_ids"
        ],
        "filtered_prompt_retrieved_memory_ids": filtered_context[
            "retrieved_memory_ids"
        ],
        "default_forgotten_memory_ids": default_context["forgotten_memory_ids"],
        "filtered_forgotten_memory_ids": filtered_context["forgotten_memory_ids"],
        "excluded_forgotten_memory_ids": excluded_ids,
        "excluded_forgotten_count": filtered_context["excluded_forgotten_count"],
        "default_retention_time_unit_hours": default_context[
            "retention_time_unit_hours"
        ],
        "filtered_retention_time_unit_hours": filtered_context[
            "retention_time_unit_hours"
        ],
        "excluded_memories_unchanged": excluded_memories_unchanged,
        "memory_stats": reinforcement_bank.get_stats(),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_seeded_memory_bank(args: argparse.Namespace) -> MemoryBank:
    bank = MemoryBank(
        forgetting_threshold=args.forgetting_threshold,
        decay_interval_hours=args.retention_time_unit_hours,
        embedding_model=HashEmbeddingModel(dim=args.embedding_dim),
    )
    _seed_memories(bank)
    return bank


def _seed_memories(memory_bank: MemoryBank) -> None:
    memory_bank.store_dialog(
        "User",
        "I am preparing for a machine learning exam.",
        "2026-06-20 09:00",
    )
    memory_bank.store_dialog(
        "AI",
        "I recommended Automate the Boring Stuff with Python.",
        "2026-06-20 10:00",
    )
    memory_bank.store_dialog(
        "User",
        "My stomach is uncomfortable, so I should avoid spicy food for now.",
        "2026-06-20 11:00",
    )


def _memory_state(memory_bank: MemoryBank) -> dict[str, dict[str, Any]]:
    return {
        memory.memory_id: {
            "strength": memory.strength,
            "last_accessed": memory.last_accessed,
            "access_count": memory.access_count,
        }
        for memory in memory_bank.get_all()
    }


def _reinforcement_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": item["memory_id"],
        "query": item.get("query", ""),
        "before_strength": item.get("before_strength"),
        "after_strength": item.get("after_strength"),
        "before_last_accessed": item.get("before_last_accessed"),
        "after_last_accessed": item.get("after_last_accessed"),
        "retrieval_rank": item.get("retrieval_rank"),
        "retrieval_score": item.get("retrieval_score"),
        "semantic_score": item.get("semantic_score"),
        "composite_score": item.get("composite_score"),
        "timestamp": item.get("recall_timestamp"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
