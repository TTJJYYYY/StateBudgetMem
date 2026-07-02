#!/usr/bin/env python3
"""Run Flat / Current / History / Dual view retrieval experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from statebudgetmem.views import ViewsExperimentConfig, run_views_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 StateBudgetMem Views 对照实验")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/controlled/temporal_challenge_v1.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/views"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["flat", "current", "dual"],
        choices=["flat", "current", "history", "dual"],
    )
    parser.add_argument("--token-budget", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_views_experiment(
        ViewsExperimentConfig(
            dataset_path=args.dataset,
            top_k=args.top_k,
            random_seed=args.seed,
            results_dir=args.results_dir,
            methods=tuple(args.methods),
            token_budget=args.token_budget,
        )
    )

    print("Views 实验完成")
    print(f"  run_id        : {result['run_id']}")
    print(f"  raw           : {result['raw_path']}")
    print(f"  summary JSON  : {result['summary_json_path']}")
    print(f"  summary CSV   : {result['summary_csv_path']}")
    print("  routing       : oracle_query_type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
