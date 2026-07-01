from __future__ import annotations

import csv
import json
from pathlib import Path

from statebudgetmem.cli import main


def test_cli_run_writes_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.yaml"
    results_dir = tmp_path / "results"
    config_path.write_text(
        "\n".join(
            [
                "method: tfidf_topk",
                "dataset_path: data/controlled/baseline_scenarios.jsonl",
                "top_k: 3",
                "random_seed: 42",
                f"results_dir: {results_dir}",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["run", "--config", str(config_path)]) == 0

    raw_files = list((results_dir / "raw").glob("*.jsonl"))
    json_files = list((results_dir / "summaries").glob("*.json"))
    csv_files = list((results_dir / "summaries").glob("*.csv"))
    assert len(raw_files) == 1
    assert len(json_files) == 1
    assert len(csv_files) == 1

    raw_rows = [
        json.loads(line)
        for line in raw_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(json_files[0].read_text(encoding="utf-8"))
    with csv_files[0].open("r", encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))

    assert raw_rows
    assert summary["query_count"] == len(raw_rows)
    assert int(csv_row["query_count"]) == summary["query_count"]
    assert float(csv_row["mean_recall_at_k"]) == summary["mean_recall_at_k"]
