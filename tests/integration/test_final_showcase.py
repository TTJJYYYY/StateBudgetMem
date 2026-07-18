"""Validate the final showcase output and its conclusion boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS_DEMO = ROOT / "tools" / "demo"
if str(TOOLS_DEMO) not in sys.path:
    sys.path.insert(0, str(TOOLS_DEMO))

from build_final_showcase import build_final_showcase


FORMAL_METHODS = (
    "tfidf_topk",
    "memorybank_core",
    "memorybank_versioning",
    "memorybank_dual_views",
    "statebudgetmem_rule",
    "statebudgetmem_oracle",
)


@pytest.fixture()
def showcase_dir(tmp_path: Path) -> Path:
    fair_dir = tmp_path / "fair_comparison"
    resource_dir = tmp_path / "resource"
    output_dir = tmp_path / "final_showcase"

    fair_dir.mkdir(parents=True)
    method_rows = []
    for index, method in enumerate(FORMAL_METHODS):
        method_rows.append(
            {
                "method": method,
                "mean_recall_at_k": 0.40 + index * 0.02,
                "mean_valid_recall_at_k": 0.50 + index * 0.02,
                "mean_stale_retrieval_rate": max(0.0, 0.20 - index * 0.02),
                "mean_total_token_cost": 24.0 + index,
                "mean_retrieval_latency_ms": 8.0 + index,
                "mean_eligible_count": 6.0,
                "mean_candidate_count": 6.0,
                "query_count": 96,
                "top_k": 3,
                "candidate_k": 20,
                "token_budget": 256,
            }
        )

    write_json(
        fair_dir / "summary.json",
        {
            "dataset_path": "data/controlled/temporal_challenge_v1.jsonl",
            "methods": method_rows,
        },
    )
    write_json(fair_dir / "method_summary.json", method_rows)
    write_json(
        fair_dir / "run_config_resolved.json",
        {
            "dataset_path": "data/controlled/temporal_challenge_v1.jsonl",
            "config_path": "configs/fair_comparison/*.yaml",
            "embedding_backend": "sentence-transformer",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 3,
            "candidate_k": 20,
            "token_budget": 256,
            "random_seed": 42,
            "query_state_policy": "independent",
            "run_id": "test_run",
        },
    )
    write_json(fair_dir / "environment.json", {"python_version": "3.11"})
    write_json(
        resource_dir / "memorybank_resource_metrics.json",
        {
            "by_memory_count": [
                {
                    "memory_count": 100,
                    "storage_total_bytes": 1234,
                    "faiss_index_file_bytes": 567,
                    "index_loaded_rss_bytes": 890,
                }
            ]
        },
    )
    write_json(
        resource_dir / "metrics.json",
        {
            "by_memory_count_and_top_k": [
                {
                    "memory_count": 100,
                    "top_k": 3,
                    "mean_prompt_token_estimate": 24.0,
                    "mean_retrieval_latency_ms": 18.0,
                    "valid_recall_at_k": 1.0,
                }
            ]
        },
    )

    build_final_showcase(
        output_dir=output_dir,
        fair_results_dir=fair_dir,
        resource_dir=resource_dir,
    )
    return output_dir


def test_index_html_exists(showcase_dir: Path) -> None:
    index = showcase_dir / "index.html"
    assert index.exists(), f"index.html not found at {index}"
    assert len(index.read_text(encoding="utf-8")) > 1000


def test_three_layer_structure(showcase_dir: Path) -> None:
    html = (showcase_dir / "index.html").read_text(encoding="utf-8")

    assert "Case Entry" in html
    assert "MemoryExplorer" in html
    assert "Dashboard" in html


def test_case_entry_labeled_demo_only(showcase_dir: Path) -> None:
    html = (showcase_dir / "index.html").read_text(encoding="utf-8")

    assert "Demo-only" in html
    assert "formal performance metric" in html


def test_memory_explorer_labeled_analysis_tool(showcase_dir: Path) -> None:
    html = (showcase_dir / "index.html").read_text(encoding="utf-8")

    assert "Showcase and analysis tool" in html


def test_formal_results_source_path_present(showcase_dir: Path) -> None:
    html = (showcase_dir / "index.html").read_text(encoding="utf-8")

    assert "fair_comparison" in html


def test_showcase_data_json_valid(showcase_dir: Path) -> None:
    data = read_json(showcase_dir / "showcase_data.json")

    assert "case_entry" in data
    assert "memory_explorer" in data
    assert data["memory_explorer"]["resource_panel"]["cloud_api_calls"] == 0


def test_dashboard_data_json_valid(showcase_dir: Path) -> None:
    data = read_json(showcase_dir / "experiment_dashboard_data.json")

    assert "methods" in data
    assert [row["method"] for row in data["methods"]] == list(FORMAL_METHODS)
    assert data["metadata"]["missing_methods"] == []


def test_tfidf_is_explicit_baseline(showcase_dir: Path) -> None:
    data = read_json(showcase_dir / "experiment_dashboard_data.json")
    groups = {row["method"]: row["group"] for row in data["methods"]}

    assert groups["tfidf_topk"] == "baseline"


def test_dashboard_can_aggregate_per_query_without_method_summary(
    tmp_path: Path,
) -> None:
    fair_dir = tmp_path / "fair_comparison"
    resource_dir = tmp_path / "resource"
    output_dir = tmp_path / "showcase"
    fair_dir.mkdir(parents=True)

    with (fair_dir / "per_query_results.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for method in FORMAL_METHODS:
            for index in range(2):
                handle.write(
                    json.dumps(
                        {
                            "method": method,
                            "status": "success",
                            "query_id": f"Q{index}",
                            "recall_at_k": 0.5,
                            "valid_recall_at_k": 0.75,
                            "stale_retrieval_rate": 0.0,
                            "total_token_cost": 12,
                            "retrieval_latency_ms": 3.0,
                            "top_k": 3,
                            "candidate_k": 20,
                            "token_budget": 256,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    build_final_showcase(
        output_dir=output_dir,
        fair_results_dir=fair_dir,
        resource_dir=resource_dir,
    )
    data = read_json(output_dir / "experiment_dashboard_data.json")

    assert [row["method"] for row in data["methods"]] == list(FORMAL_METHODS)
    assert all(row["query_count"] == 2 for row in data["methods"])
    assert data["metadata"]["missing_methods"] == []


def test_no_demo_mixed_into_formal(showcase_dir: Path) -> None:
    data = read_json(showcase_dir / "experiment_dashboard_data.json")
    methods = [row["method"] for row in data.get("methods", [])]

    assert all("demo" not in method.lower() for method in methods)


def test_readme_exists(showcase_dir: Path) -> None:
    assert (showcase_dir / "README.md").exists()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
