from __future__ import annotations

import argparse
from pathlib import Path

from statebudgetmem.views import ViewsExperimentConfig, run_views_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Current/History/Dual view comparison."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/controlled/temporal_challenge_v1.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("results/views"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["flat", "current", "dual"],
        help="Any of: flat current dual",
    )
    parser.add_argument("--token-budget", type=int, default=None)

    args = parser.parse_args()

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

    print(f"run_id: {result['run_id']}")
    print(f"raw: {result['raw_path']}")
    print(f"summary_json: {result['summary_json_path']}")
    print(f"summary_csv: {result['summary_csv_path']}")


if __name__ == "__main__":
    main()
