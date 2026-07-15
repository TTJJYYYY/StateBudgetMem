#!/usr/bin/env python3
"""Unified method comparison demo.

费哲瀚 — Phase M1: Demo (成员 D)

Reads unified experiment raw.jsonl and produces a side-by-side comparison of
all registered methods on the same queries.  Only reads results files — does
NOT import adapter internals.

Usage:
    # Full comparison (terminal table)
    python tools/phase1/compare_methods.py results/interface_smoke

    # JSON output for programmatic use
    python tools/phase1/compare_methods.py results/interface_smoke --json

    # Compact text summary
    python tools/phase1/compare_methods.py results/interface_smoke --summary
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    fmt = "table"

    results_root = None
    for a in args:
        if a in ("--json", "--summary"):
            fmt = a.lstrip("-")
        elif not a.startswith("-"):
            results_root = Path(a)

    if results_root is None:
        results_root = Path("results/interface_smoke")

    if not results_root.exists():
        print(f"Results not found: {results_root}")
        print("Run first:")
        print("  python -m statebudgetmem.unified_runner --config configs/memorybank_interface_smoke.yaml")
        return 1

    rows: list[dict[str, Any]] = []
    for raw_file in sorted(results_root.rglob("raw.jsonl")):
        with raw_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))

    if not rows:
        print("No data. Run the unified runner first.")
        return 1

    # Group: query_id → method_name → {recall, stale, latency, token, retrieved_ids}
    by_query: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        qid = row["query_id"]
        mname = row["method"]
        retrieved = row.get("retrieved_memory_ids", [])
        by_query[qid][mname] = {
            "retrieved": retrieved,
            "recall": row.get("recall_at_k", 0),
            "stale": row.get("stale_retrieval_rate", 0),
            "latency": row.get("retrieval_latency_ms", 0),
            "token": row.get("total_token_cost", 0),
        }

    method_names = sorted({row["method"] for row in rows})

    if fmt == "json":
        print(json.dumps({
            "methods": method_names,
            "queries": list(by_query.keys()),
            "by_query": {qid: {
                mn: data for mn, data in methods.items()
            } for qid, methods in by_query.items()},
        }, ensure_ascii=False, indent=2))
        return 0

    if fmt == "summary":
        _print_summary(rows, method_names)
        return 0

    # Default: rich table
    _print_table(by_query, method_names)
    return 0


def _print_table(by_query: dict, method_names: list[str]) -> None:
    col_w = 14 if len(method_names) <= 3 else 10
    header = f"{'query_id':<20s}"
    for mn in method_names:
        header += f" {mn:<{col_w}s}"
    print(header)
    print("-" * (20 + len(method_names) * (col_w + 1)))

    for qid, methods in sorted(by_query.items()):
        short_qid = qid[:18] if len(qid) > 18 else qid
        line = f"{short_qid:<20s}"
        for mn in method_names:
            data = methods.get(mn, {})
            recall = data.get("recall", 0)
            stale = data.get("stale", 0)
            line += f" R={recall:.2f} S={stale:.2f}"[:col_w].ljust(col_w + 1)
        print(line)


def _print_summary(rows: list[dict], method_names: list[str]) -> None:
    agg: dict[str, dict[str, list]] = {mn: {"recall": [], "stale": [], "latency": [], "token": []} for mn in method_names}
    for row in rows:
        mn = row["method"]
        agg[mn]["recall"].append(row.get("recall_at_k", 0))
        agg[mn]["stale"].append(row.get("stale_retrieval_rate", 0))
        agg[mn]["latency"].append(row.get("retrieval_latency_ms", 0))
        agg[mn]["token"].append(row.get("total_token_cost", 0))

    print(f"{'method':<22s} {'Recall@K':>10s} {'Stale':>8s} {'Lat(ms)':>9s} {'Tok':>6s}")
    print("-" * 60)
    for mn in method_names:
        d = agg[mn]
        r = sum(d["recall"]) / len(d["recall"]) if d["recall"] else 0
        s = sum(d["stale"]) / len(d["stale"]) if d["stale"] else 0
        l = sum(d["latency"]) / len(d["latency"]) if d["latency"] else 0
        t = sum(d["token"]) / len(d["token"]) if d["token"] else 0
        print(f"{mn:<22s} {r:>10.3f} {s:>8.3f} {l:>9.2f} {t:>6.1f}")
    print(f"\nqueries: {len(rows)}, methods: {len(method_names)}")


if __name__ == "__main__":
    raise SystemExit(main())
