from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "memorybank" / "run_phase1_baseline.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_phase1_baseline", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "top_k": 3,
        "current_time": "2026-06-24 10:00",
        "exclude_forgotten": True,
        "forgetting_threshold": 0.3,
        "retention_time_unit_hours": 24.0,
        "embedding_backend": "hash",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": 32,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _FakeMemoryBank:
    def __init__(self) -> None:
        self.build_calls = 0

    def retrieve(self, *_args, **_kwargs):
        raise AssertionError("phase1 runner must not call retrieve directly")

    def get_stats(self):
        return {"index_size": 3}

    def build_augmented_prompt(
        self,
        query: str,
        current_time: str,
        top_k: int,
        exclude_forgotten: bool = False,
    ):
        self.build_calls += 1
        return {
            "retrieved_memory_ids": ["m1"],
            "retrieved_count": 1,
            "retrieved_memories": [
                {
                    "memory_id": "m1",
                    "content": f"memory for {query}",
                    "memory_type": "dialog",
                    "semantic_score": 0.9,
                    "composite_score": 0.8,
                    "retrieval_score": 0.8,
                    "score": 0.8,
                    "retrieval_rank": 1,
                    "time_decay": 1.0,
                    "strength_factor": 1.3,
                    "retention": 0.9,
                    "is_forgotten": False,
                    "forgetting_threshold": 0.3,
                    "timestamp": 1.0,
                    "age_hours": 0.0,
                    "status": "active",
                    "tags": [],
                    "before_strength": 1.0,
                    "after_strength": 2.0,
                    "before_last_accessed": 1.0,
                    "after_last_accessed": 2.0,
                    "before_access_count": 0,
                    "after_access_count": 1,
                    "query": query,
                    "recall_timestamp": 2.0,
                }
            ],
            "prompt_sections": {"relevant_memories": f"[dialog] memory for {query}"},
            "prompt_template": f"prompt uses memory for {query}",
            "prompt_token_estimate": 11,
            "forgotten_memory_ids": ["old"],
            "excluded_forgotten_memory_ids": ["old"] if exclude_forgotten else [],
            "excluded_forgotten_count": 1 if exclude_forgotten else 0,
            "candidate_count_before_forgetting": 2,
            "candidate_count_after_forgetting": 1,
            "exclude_forgotten": exclude_forgotten,
            "forgetting_threshold": 0.3,
            "retention_time_unit_hours": 24.0,
            "strength_before_after": [
                {"memory_id": "m1", "before": 1.0, "after": 2.0}
            ],
            "last_accessed_before_after": [
                {"memory_id": "m1", "before": 1.0, "after": 2.0}
            ],
            "access_count_before_after": [
                {"memory_id": "m1", "before": 0, "after": 1}
            ],
        }


def test_phase1_probe_uses_one_prompt_retrieval_path() -> None:
    runner = _load_runner()
    fake_bank = _FakeMemoryBank()
    probes = [
        {
            "query_id": "q1",
            "user_id": "u1",
            "question": "What should I remember?",
            "reference_answer": "memory",
            "gold_memory_ids": ["m1"],
            "expected_keywords": ["memory"],
            "question_type": "memory_recall",
        }
    ]

    rows = runner._run_probes(
        fake_bank,
        probes,
        _args(),
        run_id="phase1_test",
        dataset_source="built_in_smoke_sample",
    )

    assert fake_bank.build_calls == len(probes)
    assert rows[0]["run_id"] == "phase1_test"
    assert rows[0]["dataset_source"] == "built_in_smoke_sample"
    assert rows[0]["retrieved_memory_ids"] == ["m1"]
    assert rows[0]["retrieved_memories"][0]["content"] in rows[0]["prompt_template"]
    assert rows[0]["template_answer"].endswith(rows[0]["retrieved_memories"][0]["content"])


def test_three_tools_share_deterministic_hash_embedding() -> None:
    from statebudgetmem.baselines.memorybank.embeddings import (
        deterministic_hash_embedding,
    )

    ondevice = importlib.util.spec_from_file_location(
        "run_ondevice_reproduction",
        ROOT / "tools" / "memorybank" / "run_ondevice_reproduction.py",
    )
    assert ondevice and ondevice.loader
    ondevice_module = importlib.util.module_from_spec(ondevice)
    ondevice.loader.exec_module(ondevice_module)

    forgetting = importlib.util.spec_from_file_location(
        "run_memorybank_forgetting_demo",
        ROOT / "tools" / "memorybank" / "run_memorybank_forgetting_demo.py",
    )
    assert forgetting and forgetting.loader
    forgetting_module = importlib.util.module_from_spec(forgetting)
    forgetting.loader.exec_module(forgetting_module)

    runner = _load_runner()
    args = _args(embedding_backend="hash", embedding_dim=32)
    bank = runner._build_memory_bank(args)
    text = "Stable hash embedding text"
    expected = deterministic_hash_embedding(text, dim=32)

    assert expected.shape == (32,)
    assert (deterministic_hash_embedding(text, dim=32) == expected).all()
    assert (ondevice_module.HashEmbeddingModel(dim=32).encode(text) == expected).all()
    assert (forgetting_module.HashEmbeddingModel(dim=32).encode(text) == expected).all()
    assert (bank._embedding_model.encode(text) == expected).all()


def test_embedding_metadata_hash_and_sentence_transformer() -> None:
    runner = _load_runner()

    assert runner._embedding_metadata(
        _args(embedding_backend="hash", embedding_model="all-MiniLM-L6-v2")
    ) == ("hash", "deterministic_hash_embedding")
    assert runner._embedding_metadata(
        _args(embedding_backend="sentence-transformer", embedding_model="local-model")
    ) == ("sentence-transformer", "local-model")


def test_sentence_transformer_failure_message(monkeypatch) -> None:
    runner = _load_runner()

    class _FailingSentenceTransformer:
        def __init__(self, model_name):
            raise RuntimeError(f"cannot load {model_name}")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        type(
            "_Module",
            (),
            {"SentenceTransformer": _FailingSentenceTransformer},
        )(),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner._build_memory_bank(
            _args(
                embedding_backend="sentence-transformer",
                embedding_model="local-model",
            )
        )

    message = str(exc_info.value)
    assert "local-model" in message
    assert 'pip install -e ".[memorybank]"' in message
    assert "local model directory" in message or "cached model" in message
    assert "cannot load local-model" in message


def test_sentence_transformer_uses_model_dimension(monkeypatch) -> None:
    runner = _load_runner()

    class _FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def get_sentence_embedding_dimension(self):
            return 768

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        type(
            "_Module",
            (),
            {"SentenceTransformer": _FakeSentenceTransformer},
        )(),
    )

    args = _args(
        embedding_backend="sentence-transformer",
        embedding_model="local-model",
        embedding_dim=32,
    )
    bank = runner._build_memory_bank(args)

    assert bank.embedding_dim == 768
    assert args.actual_embedding_dim == 768
    assert runner._embedding_dim(args) == 768


def test_phase1_raw_row_keeps_forgetting_fields() -> None:
    runner = _load_runner()
    rows = runner._run_probes(
        _FakeMemoryBank(),
        [
            {
                "query_id": "q1",
                "question": "question",
                "expected_keywords": [],
            }
        ],
        _args(exclude_forgotten=True),
        run_id="run_with_fields",
        dataset_source="built_in_smoke_sample",
    )

    row = rows[0]
    assert row["question"] == "question"
    assert row["query"] == "question"
    assert row["index_size"] == 3
    assert row["embedding_dim"] == 32
    assert row["retention_time_unit_hours"] == 24.0
    assert row["forgotten_memory_ids"] == ["old"]
    assert row["excluded_forgotten_memory_ids"] == ["old"]
    assert row["exclude_forgotten"] is True
    assert row["prompt_sections"]
    assert row["prompt_template"]
    assert row["strength_before_after"]


def test_phase1_smoke_exclude_forgetting_fields(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/memorybank/run_phase1_baseline.py",
            "--smoke",
            "--embedding-backend",
            "hash",
            "--embedding-dim",
            "32",
            "--exclude-forgotten",
            "--run-id",
            "test_exclude",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

    raw_path = tmp_path / "raw" / "test_exclude.jsonl"
    summary_path = tmp_path / "summaries" / "test_exclude.json"
    resources_path = tmp_path / "resources" / "test_exclude.json"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resources = json.loads(resources_path.read_text(encoding="utf-8"))

    assert all(row["run_id"] == "test_exclude" for row in rows)
    assert all(row["exclude_forgotten"] is True for row in rows)
    assert all(row["embedding_model"] == "deterministic_hash_embedding" for row in rows)
    assert all(row["question"] == row["query"] for row in rows)
    assert all(row["index_size"] > 0 for row in rows)
    assert all(row["paper_metrics"]["faiss_index_size"] == row["index_size"] for row in rows)
    assert all(row["embedding_dim"] == 32 for row in rows)
    assert all(row["retention_time_unit_hours"] == 24.0 for row in rows)
    assert all(row["dataset_source"] == "built_in_smoke_sample" for row in rows)
    assert all("forgotten_memory_ids" in row for row in rows)
    assert all("excluded_forgotten_memory_ids" in row for row in rows)
    assert summary["exclude_forgotten"] is True
    assert summary["dataset_source"] == "built_in_smoke_sample"
    assert summary["embedding_model"] == "deterministic_hash_embedding"
    assert summary["embedding_dim"] == 32
    assert summary["retention_time_unit_hours"] == 24.0
    assert summary["paper_metrics"]["mean_faiss_index_size"] > 0
    assert "total_excluded_forgotten" in summary
    assert resources["exclude_forgotten"] is True
    assert resources["dataset_source"] == "built_in_smoke_sample"
    assert resources["embedding_backend"] == "hash"
    assert resources["embedding_model"] == "deterministic_hash_embedding"
    assert resources["embedding_dim"] == 32
    assert resources["retention_time_unit_hours"] == 24.0
    assert resources["cloud_api_used"] is False
    assert resources["llm_called"] is False
