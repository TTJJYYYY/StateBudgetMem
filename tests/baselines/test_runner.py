from __future__ import annotations

import json
from pathlib import Path

from statebudgetmem.baselines.runner import (
    BaselineComparisonConfig,
    run_baseline_comparison,
)


def test_baseline_comparison_runner_writes_machine_readable_outputs(tmp_path: Path) -> None:
    config = BaselineComparisonConfig(
        dataset_path=Path("data/controlled/baseline_scenarios.jsonl"),
        results_dir=tmp_path,
        top_k=3,
        methods=("tfidf_topk",),
        random_seed=42,
    )

    result = run_baseline_comparison(config)

    raw_path = Path(result["raw_path"])
    summary_json_path = Path(result["summary_json_path"])
    summary_csv_path = Path(result["summary_csv_path"])

    assert raw_path.exists()
    assert summary_json_path.exists()
    assert summary_csv_path.exists()

    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    summary_rows = json.loads(summary_json_path.read_text(encoding="utf-8"))

    assert raw_rows
    assert summary_rows[0]["method"] == "tfidf_topk"
    assert "recall_at_k" in raw_rows[0]
    assert "valid_recall_at_k" in raw_rows[0]
    assert "stale_retrieval_rate" in raw_rows[0]
    assert "omr" in raw_rows[0]
    assert "cor" in raw_rows[0]
