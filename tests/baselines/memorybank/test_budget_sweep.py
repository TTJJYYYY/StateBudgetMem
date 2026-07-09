from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "memorybank" / "run_budget_sweep.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_budget_sweep", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prompt_budget_selection_drops_memories_when_budget_is_tight() -> None:
    runner = _load_runner()
    memories = [
        {"memory_id": "m1", "content": "short relevant Python book memory"},
        {
            "memory_id": "m2",
            "content": "very long filler memory " + "token " * 100,
        },
    ]

    selected, token_cost = runner.select_memories_for_prompt_budget(
        memories,
        prompt_token_budget=10,
    )

    assert [item["memory_id"] for item in selected] == ["m1"]
    assert token_cost <= 10


def test_budget_summary_groups_metrics_and_pressure(tmp_path: Path) -> None:
    runner = _load_runner()
    rows = [
        {
            "top_k": 1,
            "prompt_token_budget": 128,
            "memory_count_budget": 100,
            "forgetting_threshold": 0.3,
            "storage_size_bytes": 1000,
            "paper_metrics": {
                "memory_retrieval_accuracy": 0.5,
                "response_correctness": 0.5,
                "contextual_coherence": 1.0,
                "stale_retrieval_rate": 0.0,
                "retrieval_latency_ms": 2.0,
                "faiss_index_size": 100.0,
                "prompt_token_cost": 20.0,
            },
            "budget_pressure": {
                "relevant_memory_lost": True,
                "stale_memory_retrieved": False,
                "prompt_budget_used_ratio": 0.5,
            },
        },
        {
            "top_k": 3,
            "prompt_token_budget": 512,
            "memory_count_budget": 100,
            "forgetting_threshold": 0.3,
            "storage_size_bytes": 1000,
            "paper_metrics": {
                "memory_retrieval_accuracy": 1.0,
                "response_correctness": 1.0,
                "contextual_coherence": 1.0,
                "stale_retrieval_rate": 0.5,
                "retrieval_latency_ms": 3.0,
                "faiss_index_size": 100.0,
                "prompt_token_cost": 40.0,
            },
            "budget_pressure": {
                "relevant_memory_lost": False,
                "stale_memory_retrieved": True,
                "prompt_budget_used_ratio": 0.25,
            },
        },
    ]
    args = argparse.Namespace(
        top_k=[1, 3],
        prompt_token_budget=[128, 512],
        memory_count=[100],
        forgetting_threshold=[0.3],
    )

    summary = runner.build_budget_summary(
        rows,
        args=args,
        run_id="test_budget",
        raw_path=tmp_path / "raw.jsonl",
        summary_path=tmp_path / "summary.json",
        resources_path=tmp_path / "resources.json",
        elapsed_ms=5.0,
    )

    assert summary["run_count"] == 2
    assert summary["paper_metrics"]["memory_retrieval_accuracy"] == 0.75
    assert summary["budget_pressure"]["relevant_loss_rate"] == 0.5
    assert summary["budget_pressure"]["stale_retrieval_case_rate"] == 0.5
    assert len(summary["by_budget"]["top_k"]) == 2
    assert summary["by_budget"]["memory_count_budget"][0]["mean_storage_size_bytes"] == 1000


def test_budget_resources_record_device_costs(tmp_path: Path) -> None:
    runner = _load_runner()
    raw_path = tmp_path / "raw.jsonl"
    summary_path = tmp_path / "summary.json"
    resources_path = tmp_path / "resources.json"
    raw_path.write_text("{}\n", encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(embedding_dim=32)

    resources = runner.build_budget_resources(
        [
            {"faiss_index_size": 100, "storage_size_bytes": 2048},
            {"faiss_index_size": 500, "storage_size_bytes": 4096},
        ],
        args=args,
        run_id="test_budget",
        raw_path=raw_path,
        summary_path=summary_path,
        resources_path=resources_path,
        elapsed_ms=10.0,
        peak_memory_bytes=8192,
    )

    assert resources["max_faiss_index_size"] == 500
    assert resources["max_storage_size_bytes"] == 4096
    assert resources["peak_tracemalloc_bytes"] == 8192
    assert resources["output_file_bytes"]["raw_jsonl"] > 0
