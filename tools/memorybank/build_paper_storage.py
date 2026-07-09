#!/usr/bin/env python3
"""Build the MemoryBank paper's three-layer storage locally.

This is a small reproduction utility for the MemoryBank paper storage stage:
raw dialog memories, event summaries, and user portrait. It uses fixed local
sample text by default and does not call a cloud API.
"""

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

from statebudgetmem.baselines.memorybank import (  # noqa: E402
    MemoryBank,
    RetrievalProbe,
    build_paper_aligned_storage,
    run_paper_retrieval_probe,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MemoryBank paper-aligned three-layer storage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/memorybank/paper_storage/latest_summary.json"),
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("results/memorybank/paper_storage/memorybank_snapshot"),
        help="Path prefix for MemoryBank .json/.faiss snapshot files.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Build storage and report counts without writing MemoryBank snapshot files.",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Only build the three storage layers; do not run a retrieval probe.",
    )
    parser.add_argument(
        "--query",
        default="What book did you recommend and what food should I avoid now?",
        help="Retrieval probe query.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--current-time",
        default="2026-06-24 10:00",
        help="Reference time passed to MemoryBank.retrieve.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        memory_bank = MemoryBank()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    report = build_paper_aligned_storage(memory_bank)
    retrieval_report = None
    if not args.skip_retrieval:
        retrieval_report = run_paper_retrieval_probe(
            memory_bank,
            RetrievalProbe(
                query=args.query,
                top_k=args.top_k,
                current_time=args.current_time,
            ),
        )

    saved_snapshot: str | None = None
    if not args.skip_save:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        memory_bank.save(str(args.snapshot))
        saved_snapshot = str(args.snapshot)

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "MemoryBank paper three-layer storage smoke reproduction",
        "local_only": True,
        "cloud_api_used": False,
        "snapshot_prefix": saved_snapshot,
        **report,
        "retrieval_probe": retrieval_report,
    }
    _write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
