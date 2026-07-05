from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS_DEMO = ROOT / "tools" / "demo"
if str(TOOLS_DEMO) not in sys.path:
    sys.path.insert(0, str(TOOLS_DEMO))

from run_defense_demo import DefenseDemoConfig, run_defense_demo


def test_defense_demo_writes_unified_summary_when_experiments_are_skipped(
    tmp_path: Path,
) -> None:
    summary = run_defense_demo(
        DefenseDemoConfig(
            baseline_dataset=ROOT / "data/controlled/baseline_scenarios.jsonl",
            views_dataset=ROOT / "data/controlled/temporal_challenge_v1.jsonl",
            results_dir=tmp_path,
            top_k=3,
            random_seed=42,
            skip_baseline=True,
            skip_views=True,
        )
    )

    latest_path = tmp_path / "latest_summary.json"
    saved = json.loads(latest_path.read_text(encoding="utf-8"))

    assert Path(summary["summary_json_path"]).exists()
    assert saved["sections"]["tfidf_baseline"] is None
    assert saved["sections"]["views_experiment"] is None
    assert saved["sections"]["versioning_example"]["validation_is_valid"] is True
    assert "RESTORE" in saved["sections"]["versioning_example"]["operation_counts"]
    assert len(saved["sections"]["routing_examples"]["examples"]) == 4
