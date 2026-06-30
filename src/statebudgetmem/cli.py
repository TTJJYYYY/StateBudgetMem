from __future__ import annotations

import argparse
from pathlib import Path

from statebudgetmem.data import read_flat_yaml
from statebudgetmem.experiments import BaselineConfig, run_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="statebudgetmem")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run the deterministic Task001 baseline.")
    run_parser.add_argument("--config", required=True, help="Path to baseline YAML config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        config_path = Path(args.config)
        raw_config = read_flat_yaml(config_path)
        config = BaselineConfig(
            method=str(raw_config["method"]),
            dataset_path=Path(str(raw_config["dataset_path"])),
            top_k=int(raw_config["top_k"]),
            random_seed=int(raw_config["random_seed"]),
            results_dir=Path(str(raw_config["results_dir"])),
            config_path=config_path,
        )
        result = run_baseline(config)
        print(f"run_id: {result['run_id']}")
        print(f"raw: {result['raw_path']}")
        print(f"summary_json: {result['summary_json_path']}")
        print(f"summary_csv: {result['summary_csv_path']}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
