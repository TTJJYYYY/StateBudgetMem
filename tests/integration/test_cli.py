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


def test_cli_analyze_staleness_tfidf_writes_outputs(tmp_path: Path, capsys) -> None:
    results_dir = tmp_path / "staleness"

    assert main(
        [
            "analyze-staleness",
            "--backend",
            "tfidf",
            "--top-k",
            "3",
            "--results-dir",
            str(results_dir),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["backend"] == "tfidf"
    assert "mean_stale_retrieval_rate" in output
    assert output["queries_with_stale_retrieval"] > 0

    raw_path = Path(output["raw_path"])
    summary_json_path = Path(output["summary_json_path"])
    summary_csv_path = Path(output["summary_csv_path"])
    assert raw_path.exists()
    assert summary_json_path.exists()
    assert summary_csv_path.exists()

    raw_rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert any(row["stale_retrieval_rate"] > 0 for row in raw_rows)
    assert summary["mean_stale_retrieval_rate"] == output["mean_stale_retrieval_rate"]
