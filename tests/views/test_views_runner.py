from __future__ import annotations

import json
from pathlib import Path

from statebudgetmem.views.runner import ViewsExperimentConfig, run_views_experiment


def _write_runner_dataset(path: Path) -> None:
    scenario = {
        "scenario_id": "runner_routing",
        "description": "Minimal scenario for views runner routing.",
        "memories": [
            {
                "memory_id": "m_current_home",
                "subject": "user",
                "attribute": "home_city",
                "value": "Hangzhou",
                "text": "用户现在住在杭州。",
                "event_time": "2026-06-01",
                "valid_from": "2026-06-01",
                "valid_to": None,
                "status": "CURRENT",
                "memory_type": "profile",
                "importance": 0.8,
                "confidence": 0.9,
                "token_cost": 8,
                "supersedes": [],
                "temporarily_invalidates": [],
                "metadata": {},
            }
        ],
        "queries": [
            {
                "query_id": "q_current_text_oracle_general",
                "text": "我现在住在哪里？",
                "query_type": "GENERAL",
                "reference_time": "2026-07-01",
                "gold_relevant_memory_ids": ["m_current_home"],
                "gold_valid_memory_ids": ["m_current_home"],
                "gold_stale_memory_ids": [],
            }
        ],
    }
    path.write_text(json.dumps(scenario, ensure_ascii=False) + "\n", encoding="utf-8")


def test_views_runner_rule_routing_overrides_oracle_query_type(tmp_path: Path) -> None:
    dataset_path = tmp_path / "scenarios.jsonl"
    _write_runner_dataset(dataset_path)

    result = run_views_experiment(
        ViewsExperimentConfig(
            dataset_path=dataset_path,
            top_k=3,
            random_seed=42,
            results_dir=tmp_path / "results",
            methods=("dual",),
            routing="rule",
        )
    )

    raw_rows = [
        json.loads(line)
        for line in Path(result["raw_path"]).read_text(encoding="utf-8").splitlines()
    ]

    assert raw_rows[0]["routing_mode"] == "rule"
    assert raw_rows[0]["oracle_query_type"] == "GENERAL"
    assert raw_rows[0]["predicted_query_type"] == "CURRENT"
    assert raw_rows[0]["query_type"] == "CURRENT"
    assert raw_rows[0]["retrieved_memory_ids"] == ["m_current_home"]
    assert result["summary"][0]["routing_mode"] == "rule"


def test_views_runner_oracle_routing_preserves_dataset_query_type(tmp_path: Path) -> None:
    dataset_path = tmp_path / "scenarios.jsonl"
    _write_runner_dataset(dataset_path)

    result = run_views_experiment(
        ViewsExperimentConfig(
            dataset_path=dataset_path,
            top_k=3,
            random_seed=42,
            results_dir=tmp_path / "results",
            methods=("dual",),
            routing="oracle",
        )
    )

    raw_rows = [
        json.loads(line)
        for line in Path(result["raw_path"]).read_text(encoding="utf-8").splitlines()
    ]

    assert raw_rows[0]["routing_mode"] == "oracle"
    assert raw_rows[0]["query_type"] == "GENERAL"
    assert raw_rows[0]["retrieved_memory_ids"] == []
