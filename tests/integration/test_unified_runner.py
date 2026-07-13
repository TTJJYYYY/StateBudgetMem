from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from statebudgetmem.core.experiment import ExperimentConfig
from statebudgetmem.core.registry import MethodRegistry
from statebudgetmem.schemas.results import MethodResult
from statebudgetmem.unified_runner import load_experiment_config, main, run_unified_experiment


class GoldGuardMethod:
    name = "gold_guard"

    def reset(self) -> None:
        self.ingested = []

    def ingest(self, memories) -> None:
        self.ingested = list(memories)

    def retrieve(self, query, *, top_k, token_budget=None, mutate=False):
        assert query.gold_relevant_memory_ids == []
        assert query.gold_valid_memory_ids == []
        assert query.gold_stale_memory_ids == []
        return MethodResult(
            method_name=self.name,
            query_id=query.query_id,
            retrieved_memories=[],
            total_token_cost=0,
            latency_ms=0.0,
            metadata={"top_k": top_k, "token_budget": token_budget, "mutate": mutate},
        )


def test_unified_runner_hides_gold_and_writes_machine_readable_artifacts(tmp_path: Path) -> None:
    registry = MethodRegistry()
    contexts = []
    registry.register(
        "gold_guard", lambda context: contexts.append(context) or GoldGuardMethod()
    )
    result = run_unified_experiment(
        ExperimentConfig(
            dataset_path=Path("data/controlled/interface_smoke_v1.jsonl"),
            results_dir=tmp_path,
            methods=("gold_guard",),
            top_k=2,
            token_budget=32,
            random_seed=42,
        ),
        registry=registry,
    )
    raw_path = Path(result["raw_path"])
    summary_path = Path(result["summary_json_path"])
    csv_path = Path(result["summary_csv_path"])
    environment_path = Path(result["environment_path"])
    assert all(path.exists() for path in (raw_path, summary_path, csv_path, environment_path))
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    assert len(raw) == 3
    assert all(row["status"] == "success" and row["random_seed"] == 42 for row in raw)
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    assert len(environment["dataset_sha256"]) == 64
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["method"] == "gold_guard"
    assert contexts[0].experiment.embedding_backend == "method_default"
    assert contexts[0].work_dir.name == "gold_guard"


def test_unified_runner_runs_existing_tfidf_adapter(tmp_path: Path) -> None:
    result = run_unified_experiment(
        ExperimentConfig(
            dataset_path=Path("data/controlled/interface_smoke_v1.jsonl"),
            results_dir=tmp_path,
            methods=("tfidf_topk",),
            top_k=2,
            token_budget=20,
            random_seed=42,
        )
    )
    raw = [
        json.loads(line)
        for line in Path(result["raw_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert raw
    assert all(row["total_token_cost"] <= 20 for row in raw)
    assert {row["query_type"] for row in raw} == {"CURRENT", "HISTORICAL", "CHANGE"}


def test_unified_runner_runs_memorybank_core_adapter(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("faiss")
    result = run_unified_experiment(
        ExperimentConfig(
            dataset_path=Path("data/controlled/interface_smoke_v1.jsonl"),
            results_dir=tmp_path,
            methods=("memorybank_core",),
            top_k=2,
            candidate_k=3,
            token_budget=32,
            embedding_backend="hash",
            embedding_model="deterministic_hash_embedding",
            reinforcement_enabled=False,
            query_state_policy="independent",
        )
    )
    raw = [
        json.loads(line)
        for line in Path(result["raw_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw) == 3
    assert all(row["method"] == "memorybank_core" for row in raw)
    assert all(row["total_token_cost"] <= 32 for row in raw)
    assert all(row["status"] == "success" for row in raw)


def test_unified_runner_loads_frozen_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_path: data/controlled/interface_smoke_v1.jsonl",
                f"results_dir: {tmp_path / 'results'}",
                "methods: tfidf_topk",
                "top_k: 2",
                "token_budget: 20",
                "random_seed: 42",
                "repeat: 1",
                "candidate_k: 4",
                "forgetting_enabled: true",
                "forgetting_threshold: 0.3",
                "exclude_forgotten: false",
                "reinforcement_enabled: false",
                "query_state_policy: sequential",
                "embedding_backend: method_default",
                "embedding_model: method_default",
                "token_counter_name: memory_record_token_cost",
            ]
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    assert config.methods == ("tfidf_topk",)
    assert config.config_path == config_path
    assert config.query_state_policy == "sequential"
    assert main(["--config", str(config_path)]) == 0


def test_runner_applies_independent_and_sequential_query_state(tmp_path: Path) -> None:
    counts: dict[str, list[GoldGuardMethod]] = {"independent": [], "sequential": []}

    class CountingMethod(GoldGuardMethod):
        name = "counting"

        def __init__(self) -> None:
            self.reset_count = 0
            self.ingest_count = 0

        def reset(self) -> None:
            self.reset_count += 1
            super().reset()

        def ingest(self, memories) -> None:
            self.ingest_count += 1
            super().ingest(memories)

    for policy in ("independent", "sequential"):
        registry = MethodRegistry()

        def factory(_context, *, policy=policy):
            method = CountingMethod()
            counts[policy].append(method)
            return method

        registry.register("counting", factory)
        run_unified_experiment(
            ExperimentConfig(
                dataset_path=Path("data/controlled/interface_smoke_v1.jsonl"),
                results_dir=tmp_path / policy,
                methods=("counting",),
                query_state_policy=policy,
            ),
            registry=registry,
        )

    assert (counts["independent"][0].reset_count, counts["independent"][0].ingest_count) == (3, 3)
    assert (counts["sequential"][0].reset_count, counts["sequential"][0].ingest_count) == (1, 1)
