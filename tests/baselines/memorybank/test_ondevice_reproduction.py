from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "memorybank" / "run_ondevice_reproduction.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_ondevice_reproduction", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeMemoryBank:
    def __init__(self) -> None:
        self.stats = {"index_size": 2, "total_memories": 4}

    def build_augmented_prompt(self, query: str, current_time: str, top_k: int):
        return {
            "retrieved_memory_ids": ["m1"],
            "retrieved_count": 1,
            "retrieved_memories": [
                {
                    "memory_id": "m1",
                    "memory_type": "dialog",
                    "content": "User likes swimming.",
                    "composite_score": 0.9,
                }
            ],
            "prompt_sections": {
                "relevant_memories": "[dialog] User likes swimming.",
                "global_user_portrait": "Active student.",
                "global_event_summary": "User discussed hobbies.",
                "current_user_query": query,
            },
            "prompt_template": (
                "【相关历史记忆】\n[dialog] User likes swimming.\n"
                "【全局用户画像】\nActive student.\n"
                "【全局事件摘要】\nUser discussed hobbies.\n"
                f"【用户当前问题】\n{query}"
            ),
        }

    def get_stats(self):
        return self.stats

    def get_all(self):
        return [
            SimpleNamespace(
                content="User likes swimming.",
                memory_id="m1",
                memory_type="dialog",
                tags=["hobby"],
            )
        ]


def test_ondevice_runner_raw_summary_and_resources_are_machine_readable(tmp_path):
    runner = _load_runner()
    args = argparse.Namespace(
        top_k=3,
        embedding_dim=32,
        forgetting_threshold=0.3,
    )

    row = runner._run_query(
        memory_bank=_FakeMemoryBank(),
        query="What are my hobbies?",
        query_index=1,
        top_k=3,
        current_time="2026-06-24 10:00",
        run_id="test_run",
    )

    assert row["query_id"] == "q001"
    assert row["retrieved_memory_ids"] == ["m1"]
    assert row["prompt_sections"]["global_user_portrait"] == "Active student."
    assert row["prompt_sections"]["global_event_summary"] == "User discussed hobbies."
    assert row["prompt_token_estimate"] > 0
    assert "paper_metrics" in row
    assert "memory_retrieval_accuracy" in row["paper_metrics"]
    assert row["template_answer"]
    assert row["local_only"] is True
    assert row["cloud_api_used"] is False

    raw_path = tmp_path / "raw.jsonl"
    summary_path = tmp_path / "summary.json"
    resources_path = tmp_path / "resources.json"
    raw_path.write_text("{}\n", encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")

    summary = runner._build_summary(
        run_id="test_run",
        args=args,
        raw_rows=[row],
        raw_path=raw_path,
        summary_path=summary_path,
        resources_path=resources_path,
        storage_report={"paper_layers": {}},
        elapsed_ms=12.5,
    )
    assert summary["raw_jsonl_path"] == str(raw_path)
    assert summary["resources_json_path"] == str(resources_path)
    assert summary["llm_called"] is False
    assert summary["mean_retrieved_count"] == 1.0
    assert "paper_metrics" in summary
    assert "mean_prompt_token_cost" in summary["paper_metrics"]

    resources = runner._build_resources(
        run_id="test_run",
        args=args,
        memory_bank=_FakeMemoryBank(),
        raw_path=raw_path,
        summary_path=summary_path,
        resources_path=resources_path,
        elapsed_ms=12.5,
        peak_memory_bytes=2048,
    )
    assert resources["embedding_model"] == "deterministic_hash_embedding"
    assert resources["index_size"] == 2
    assert resources["storage_size_bytes"] > 0
    assert resources["output_file_bytes"]["raw_jsonl"] > 0
