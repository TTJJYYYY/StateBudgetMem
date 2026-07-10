from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from statebudgetmem.baselines.memorybank.datasets import ReproductionUser


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
        return {"index_size": 3, "total_memories": 3}

    def get_all(self):
        return []

    def build_augmented_prompt(
        self,
        query: str,
        current_time: str,
        top_k: int,
        exclude_forgotten: bool = False,
        filters: dict | None = None,
    ):
        self.build_calls += 1
        return {
            "retrieved_memory_ids": ["m1"],
            "provided_context_ids": [],
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


class _TypeErrorMemoryBank(_FakeMemoryBank):
    def __init__(self) -> None:
        super().__init__()
        self.seen_filters = []

    def build_augmented_prompt(
        self,
        query: str,
        current_time: str,
        top_k: int,
        exclude_forgotten: bool = False,
        filters: dict | None = None,
    ):
        self.build_calls += 1
        self.seen_filters.append(filters)
        raise TypeError("intentional prompt failure")


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
    assert rows[0]["bank_isolation_enabled"] is True
    assert rows[0]["user_memory_count"] == 3
    assert rows[0]["user_index_size"] == 3
    assert rows[0]["retrieved_memories"][0]["content"] in rows[0]["prompt_template"]
    assert rows[0]["template_answer"].endswith(rows[0]["retrieved_memories"][0]["content"])


def test_phase1_prompt_typeerror_is_not_retried_without_future_filter() -> None:
    runner = _load_runner()
    fake_bank = _TypeErrorMemoryBank()
    probes = [
        {
            "query_id": "q_typeerror",
            "user_id": "u1",
            "question": "What should fail?",
            "gold_memory_ids": [],
            "expected_keywords": [],
            "question_type": "memory_recall",
        }
    ]

    with pytest.raises(TypeError, match="intentional prompt failure"):
        runner._run_probes(
            {"u1": fake_bank},
            probes,
            _args(current_time="2026-06-24 10:00"),
            run_id="phase1_typeerror",
            dataset_source="unit",
        )

    assert fake_bank.build_calls == 1
    assert fake_bank.seen_filters == [
        {"max_timestamp": runner.MemoryBank._parse_time("2026-06-24 10:00")}
    ]


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
            raise AssertionError("new sentence-transformers path should not be used")

        def get_embedding_dimension(self):
            return 512

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

    assert bank.embedding_dim == 512
    assert args.actual_embedding_dim == 512
    assert runner._embedding_dim(args) == 512


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
    assert row["user_memory_count"] == 3
    assert row["user_index_size"] == 3
    assert row["bank_isolation_enabled"] is True
    assert row["future_memory_count_excluded"] == 0
    assert row["query_time_source"] == "cli_override"
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
    assert all(row["bank_isolation_enabled"] is True for row in rows)
    assert all("gold_retrieval_ids" in row for row in rows)
    assert all("gold_context_ids" in row for row in rows)
    assert all("provided_context_ids" in row for row in rows)
    assert all(row["dataset_source"] == "built_in_smoke_sample" for row in rows)
    assert all("forgotten_memory_ids" in row for row in rows)
    assert all("excluded_forgotten_memory_ids" in row for row in rows)
    assert summary["exclude_forgotten"] is True
    assert summary["dataset_source"] == "built_in_smoke_sample"
    assert summary["embedding_model"] == "deterministic_hash_embedding"
    assert summary["embedding_dim"] == 32
    assert summary["retention_time_unit_hours"] == 24.0
    assert summary["user_count"] == 1
    assert summary["bank_isolation_enabled"] is True
    assert summary["paper_metrics"]["mean_faiss_index_size"] > 0
    assert "retrieval_gold_recall" in summary["paper_metrics"]
    assert "context_coverage" in summary["paper_metrics"]
    assert "total_excluded_forgotten" in summary
    assert resources["exclude_forgotten"] is True
    assert resources["dataset_source"] == "built_in_smoke_sample"
    assert resources["embedding_backend"] == "hash"
    assert resources["embedding_model"] == "deterministic_hash_embedding"
    assert resources["embedding_dim"] == 32
    assert resources["retention_time_unit_hours"] == 24.0
    assert resources["user_count"] == 1
    assert resources["bank_isolation_enabled"] is True
    assert resources["cloud_api_used"] is False
    assert resources["llm_called"] is False


def test_phase1_isolates_banks_by_user_and_reinforcement() -> None:
    runner = _load_runner()
    args = _args(current_time=None, top_k=3, exclude_forgotten=False)
    users = [
        ReproductionUser(
            user_id="user_001",
            days=[
                {
                    "date": "2026-06-01",
                    "dialogues": [
                        {
                            "role": "user",
                            "content": "I keep a sapphire notebook for exams.",
                            "timestamp": "2026-06-01 09:00",
                            "memory_id": "user_001_dialog_sapphire",
                        }
                    ],
                }
            ],
            global_summary="User 001 studies with a sapphire notebook.",
            user_portrait="User 001 likes quiet study plans.",
            global_memory_ids={
                "event_summary_id": "user_001_global_event_summary",
                "portrait_id": "user_001_global_user_portrait",
            },
        ),
        ReproductionUser(
            user_id="user_002",
            days=[
                {
                    "date": "2026-06-01",
                    "dialogues": [
                        {
                            "role": "user",
                            "content": "I keep an amber notebook for recipes.",
                            "timestamp": "2026-06-01 09:00",
                            "memory_id": "user_002_dialog_amber",
                        }
                    ],
                }
            ],
            global_summary="User 002 cooks with an amber notebook.",
            user_portrait="User 002 likes recipe experiments.",
            global_memory_ids={
                "event_summary_id": "user_002_global_event_summary",
                "portrait_id": "user_002_global_user_portrait",
            },
        ),
    ]
    banks = {}
    for user in users:
        bank = runner._build_memory_bank(args)
        runner._ingest_user(bank, user)
        banks[user.user_id] = bank

    before_strength = banks["user_002"].get("user_002_dialog_amber").strength
    before_access_count = banks["user_002"].get("user_002_dialog_amber").access_count
    rows = runner._run_probes(
        banks,
        [
            {
                "query_id": "q_user_001",
                "user_id": "user_001",
                "question": "Which notebook do I use for exams?",
                "gold_memory_ids": ["user_001_dialog_sapphire"],
                "expected_keywords": ["sapphire"],
                "question_type": "memory_recall",
            }
        ],
        args,
        run_id="phase1_isolation_test",
        dataset_source="unit",
    )

    assert not any(mid.startswith("user_002") for mid in rows[0]["retrieved_memory_ids"])
    assert banks["user_001"].global_summary_id == "user_001_global_event_summary"
    assert banks["user_002"].global_summary_id == "user_002_global_event_summary"
    assert banks["user_001"].user_portrait_id == "user_001_global_user_portrait"
    assert banks["user_002"].user_portrait_id == "user_002_global_user_portrait"
    assert banks["user_002"].get("user_002_dialog_amber").strength == before_strength
    assert banks["user_002"].get("user_002_dialog_amber").access_count == before_access_count


def test_phase1_query_timestamp_excludes_future_memory() -> None:
    runner = _load_runner()
    args = _args(current_time=None, top_k=5, exclude_forgotten=False)
    user = ReproductionUser(
        user_id="user_future",
        days=[
            {
                "date": "2026-06-01",
                "dialogues": [
                    {
                        "role": "user",
                        "content": "Past memory says I owned a blue mug.",
                        "timestamp": "2026-06-01 09:00",
                        "memory_id": "past_mug",
                    },
                    {
                        "role": "user",
                        "content": "Future memory says I bought a red telescope.",
                        "timestamp": "2026-06-03 09:00",
                        "memory_id": "future_telescope",
                    },
                ],
            }
        ],
    )
    bank = runner._build_memory_bank(args)
    runner._ingest_user(bank, user)

    rows = runner._run_probes(
        {"user_future": bank},
        [
            {
                "query_id": "q_future",
                "user_id": "user_future",
                "question": "What telescope did I buy?",
                "query_timestamp": "2026-06-02 09:00",
                "gold_memory_ids": ["future_telescope"],
                "expected_keywords": ["telescope"],
                "question_type": "memory_recall",
            }
        ],
        args,
        run_id="phase1_future_test",
        dataset_source="unit",
    )

    row = rows[0]
    assert row["query_time_source"] == "probe"
    assert row["query_timestamp"] == "2026-06-02 09:00"
    assert row["future_memory_ids_excluded"] == ["future_telescope"]
    assert row["future_memory_count_excluded"] == 1
    assert "future_telescope" not in row["retrieved_memory_ids"]
    assert row["unknown_gold_ids"] == ["future_telescope"]


def test_phase1_splits_retrieval_context_and_unknown_gold_ids() -> None:
    runner = _load_runner()
    bank = SimpleNamespace(
        memories_by_id={"dialog_gold": object()},
        global_summary_id="summary_gold",
        user_portrait_id="portrait_gold",
    )

    split = runner._split_gold_ids(
        bank,
        ["dialog_gold", "portrait_gold", "summary_gold", "missing_gold"],
        visible_retrieval_ids={"dialog_gold"},
    )
    metrics = runner._split_gold_metrics(
        retrieved_ids=["dialog_gold"],
        provided_context_ids=["portrait_gold", "summary_gold"],
        gold_retrieval_ids=split["gold_retrieval_ids"],
        gold_context_ids=split["gold_context_ids"],
    )
    negative_metrics = runner._split_gold_metrics(
        retrieved_ids=["irrelevant"],
        provided_context_ids=[],
        gold_retrieval_ids=[],
        gold_context_ids=[],
    )

    assert split["gold_retrieval_ids"] == ["dialog_gold"]
    assert split["gold_context_ids"] == ["portrait_gold", "summary_gold"]
    assert split["unknown_gold_ids"] == ["missing_gold"]
    assert metrics["retrieval_gold_recall"] == 1.0
    assert metrics["context_coverage"] == 1.0
    assert metrics["overall_context_coverage"] == 1.0
    assert negative_metrics["retrieval_gold_recall"] == 0.0
    assert negative_metrics["context_coverage"] == 0.0
    assert negative_metrics["overall_context_coverage"] == 0.0
