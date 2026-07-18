from __future__ import annotations

import argparse
import json
from pathlib import Path

from statebudgetmem.evaluation.fair_comparison_analysis import (
    analyze_fair_comparison,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate formal fair-comparison per-query results "
            "by query type and generate diagnostic artifacts."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/fair_comparison/per_query_results.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fair_comparison_by_type"),
    )
    parser.add_argument(
        "--expected-queries",
        type=int,
        default=96,
        help=(
            "Expected records per method. Use 0 to disable the "
            "fixed-count check."
        ),
    )
    args = parser.parse_args(argv)

    metadata = analyze_fair_comparison(
        args.input,
        args.output_dir,
        expected_queries_per_method=(
            args.expected_queries
            if args.expected_queries > 0
            else None
        ),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
