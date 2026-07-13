#!/usr/bin/env python3
"""Resource measurement summary for unified experiment runs.

费哲瀚 — Phase M1: Resource Measurement (成员 D)

Reads unified experiment output and prints a compact resource comparison table.

Usage:
    python tools/phase1/resource_summary.py <results_dir>
    python tools/phase1/resource_summary.py results/interface_smoke

Output:
    method_name        ingest_ms  retrieve_ms  token_cost  storage_bytes
    --------------------------------------------------------------------
    tfidf_topk          2.1        1.5          18          0
    memorybank_core     1.8        3.2          22          15400
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if not args:
        results_root = Path("results/interface_smoke")
    else:
        results_root = Path(args[0])

    if not results_root.exists():
        print(f"Results directory not found: {results_root}")
        return 1

    rows: list[dict[str, Any]] = []
    for raw_file in sorted(results_root.rglob("raw.jsonl")):
        with raw_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))

    if not rows:
        print("No raw.jsonl found. Run the unified runner first:")
        print("  python -m statebudgetmem.unified_runner --config configs/memorybank_interface_smoke.yaml")
        return 1

    # Aggregate by method
    methods: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        name = row.get("method", "unknown")
        methods.setdefault(name, {"ingest": [], "retrieve": [], "token": []})
        methods[name]["ingest"].append(float(row.get("ingest_latency_ms", 0) or 0))
        methods[name]["retrieve"].append(float(row.get("retrieval_latency_ms", 0) or 0))
        methods[name]["token"].append(float(row.get("total_token_cost", 0) or 0))

    # Print table
    print(f"{'method_name':<22s} {'ingest_ms':>10s} {'retrieve_ms':>11s} {'token_avg':>10s} {'queries':>8s}")
    print("-" * 65)
    for name, data in sorted(methods.items()):
        ingest_avg = sum(data["ingest"]) / len(data["ingest"]) if data["ingest"] else 0
        ret_avg = sum(data["retrieve"]) / len(data["retrieve"]) if data["retrieve"] else 0
        tok_avg = sum(data["token"]) / len(data["token"]) if data["token"] else 0
        n = len(data["retrieve"])
        print(f"{name:<22s} {ingest_avg:>10.2f} {ret_avg:>11.2f} {tok_avg:>10.1f} {n:>8d}")

    # Environment
    env_files = sorted(results_root.rglob("environment.json"))
    if env_files:
        env = json.loads(env_files[0].read_text(encoding="utf-8"))
        print(f"\nenvironment: {env.get('platform','?')}, Python {env.get('python_version','?')}")
        print(f"embedding: {env.get('embedding_backend','?')} / {env.get('embedding_model','?')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
