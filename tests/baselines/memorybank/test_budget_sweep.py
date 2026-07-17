from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


def _quality_signature(rows: list[dict]) -> list[tuple]:
    return sorted(
        (
            row["repeat_index"],
            row["query_id"],
            row["token_budget"],
            row["top_k"],
            row["candidate_k"],
            row["memory_count"],
            row["memory_retrieval_accuracy"],
            row["answer_accuracy"],
            row["stale_retrieval_rate"],
        )
        for row in rows
    )


def test_cli_aliases_and_defaults() -> None:
    runner = _load_runner()

    current = runner.parse_args(
        [
            "--token-budget",
            "64",
            "128",
            "--candidate-k",
            "5",
            "20",
            "--repeat",
            "3",
            "--seed",
            "7",
        ]
    )
    legacy = runner.parse_args(["--prompt-token-budget", "256"])

    assert current.token_budget == [64, 128]
    assert current.candidate_k == [5, 20]
    assert current.repeat == 3
    assert current.seed == 7
    assert legacy.token_budget == [256]
    assert current.results_root == Path("results/budget_sweep")


def test_quick_grid_matches_contract() -> None:
    runner = _load_runner()
    args = runner.parse_args(["--quick"])

    runner.apply_quick_grid(args)

    assert args.token_budget == [64, 128]
    assert args.top_k == [1, 3]
    assert args.candidate_k == [5, 20]
    assert args.memory_count == [100]
    assert args.forgetting_threshold == [0.3]
    assert args.repeat == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--token-budget", "0"], "token_budget"),
        (["--top-k", "0"], "top_k"),
        (["--candidate-k", "0"], "candidate_k"),
        (["--memory-count", "3"], "memory_count"),
        (["--repeat", "0"], "repeat"),
        (["--embedding-dim", "0"], "embedding_dim"),
    ],
)
def test_argument_validation_rejects_invalid_values(
    arguments: list[str], message: str
) -> None:
    runner = _load_runner()
    args = runner.parse_args(arguments)

    with pytest.raises(ValueError, match=message):
        runner.validate_args(args)


def test_prompt_budget_selection_and_token_proxy() -> None:
    runner = _load_runner()
    memories = [
        {"memory_id": "m1", "content": "short relevant Python book memory"},
        {"memory_id": "m2", "content": "very long " + "token " * 100},
    ]

    selected, token_cost = runner.select_memories_for_prompt_budget(
        memories, prompt_token_budget=10
    )

    assert [item["memory_id"] for item in selected] == ["m1"]
    assert token_cost <= 10
    assert runner.estimate_token_proxy("abc-123 中文") == 4
    assert runner.TOKEN_METRIC == "deterministic_proxy_not_real_tokenizer"


def test_candidate_k_is_passed_and_reinforcement_is_disabled() -> None:
    runner = _load_runner()
    calls: list[dict] = []

    class FakeBank:
        def retrieve_with_metadata(self, **kwargs):
            calls.append(kwargs)
            return {
                "memories": [
                    {
                        "memory_id": "m1",
                        "content": "Python book recommendation",
                    }
                ],
                "candidate_count_before_forgetting": 7,
                "candidate_count_after_forgetting": 7,
            }

        def get_stats(self):
            return {"index_size": 100, "total_memories": 100}

    row = runner.run_budget_probe(
        memory_bank=FakeBank(),
        probe=runner.PROBES[0],
        top_k=3,
        candidate_k=20,
        token_budget=64,
        memory_count=100,
        forgetting_threshold=0.3,
        current_time="2026-07-10 10:00",
        run_id="test",
        repeat_index=0,
        seed=42,
        embedding_dim=32,
        estimated_storage_bytes=1234,
        estimated_faiss_index_bytes=12800,
    )

    assert calls == [
        {
            "query": runner.PROBES[0].query,
            "top_k": 3,
            "candidate_k": 20,
            "current_time": "2026-07-10 10:00",
            "reinforce": False,
        }
    ]
    assert row["candidate_k"] == 20
    assert row["selected_memory_count"] == 1
    assert row["local_only"] is True
    assert row["cloud_api_used"] is False
    assert row["llm_called"] is False


def test_invalid_top_k_candidate_k_is_skipped_and_recorded() -> None:
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--token-budget",
            "64",
            "--top-k",
            "3",
            "5",
            "--candidate-k",
            "2",
            "5",
            "--memory-count",
            "4",
        ]
    )

    skipped = runner.invalid_combinations(args)
    rows = runner.run_budget_sweep(args, run_id="invalid-grid")

    assert skipped == [
        {"top_k": 3, "candidate_k": 2},
        {"top_k": 5, "candidate_k": 2},
    ]
    assert len(rows) == len(runner.PROBES) * 2
    assert all(row["top_k"] <= row["candidate_k"] for row in rows)


def test_repeat_aggregation_and_percentile() -> None:
    runner = _load_runner()
    rows = []
    for repeat in range(2):
        for query_id, latency in (("q1", 1.0), ("q2", 3.0)):
            rows.append(
                {
                    "repeat_index": repeat,
                    "query_id": query_id,
                    "token_budget": 64,
                    "top_k": 1,
                    "candidate_k": 5,
                    "memory_count": 100,
                    "forgetting_threshold": 0.3,
                    "memory_retrieval_accuracy": 1.0,
                    "answer_accuracy": 1.0,
                    "response_correctness": 1.0,
                    "stale_retrieval_rate": 0.0,
                    "relevant_loss_rate": 0.0,
                    "stale_retrieval_case_rate": 0.0,
                    "retrieval_latency_ms": latency,
                    "selected_token_proxy": 10,
                    "token_budget_used_ratio": 10 / 64,
                    "selected_memory_count": 1,
                    "estimated_memory_storage_bytes": 1000,
                    "estimated_faiss_index_bytes": 12800,
                    "faiss_index_size": 100,
                    "retrieval_rss_peak_bytes": None,
                }
            )

    summary = runner.aggregate_configurations(rows)

    assert len(summary) == 1
    assert summary[0]["run_count"] == 4
    assert summary[0]["query_count"] == 2
    assert summary[0]["mean_retrieval_latency_ms"] == 2.0
    assert summary[0]["p95_retrieval_latency_ms"] == pytest.approx(3.0)
    assert summary[0]["rss_available"] is False


def test_quality_is_deterministic_and_grid_order_independent() -> None:
    runner = _load_runner()
    first = runner.parse_args(
        [
            "--token-budget",
            "64",
            "128",
            "--top-k",
            "1",
            "3",
            "--candidate-k",
            "5",
            "--memory-count",
            "20",
            "--seed",
            "42",
        ]
    )
    reversed_grid = runner.parse_args(
        [
            "--token-budget",
            "128",
            "64",
            "--top-k",
            "3",
            "1",
            "--candidate-k",
            "5",
            "--memory-count",
            "20",
            "--seed",
            "42",
        ]
    )

    rows_a = runner.run_budget_sweep(first, run_id="same")
    rows_b = runner.run_budget_sweep(reversed_grid, run_id="same")

    assert _quality_signature(rows_a) == _quality_signature(rows_b)


def test_quick_main_writes_compact_verified_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    output = tmp_path / "budget-sweep"
    monkeypatch.setattr(runner, "get_git_commit", lambda: ("a" * 40, None))

    exit_code = runner.main(
        [
            "--quick",
            "--results-root",
            str(output),
            "--run-id",
            "quick-test",
            "--embedding-dim",
            "8",
        ]
    )

    assert exit_code == 0
    expected = [
        output / "budget_sweep_rows.csv",
        output / "budget_sweep_rows.json",
        output / "budget_sweep_summary.csv",
        output / "budget_sweep_summary.json",
        output / "resource_metrics.json",
        output / "manifest.json",
        output / "figures" / "token_budget_quality.png",
        output / "figures" / "topk_candidatek_quality.png",
        output / "figures" / "candidatek_latency.png",
        output / "figures" / "memory_count_resources.png",
        output / "figures" / "quality_resource_tradeoff.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)

    rows = json.loads((output / "budget_sweep_rows.json").read_text(encoding="utf-8"))
    with (output / "budget_sweep_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    summary = json.loads((output / "budget_sweep_summary.json").read_text(encoding="utf-8"))
    resources = json.loads((output / "resource_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == len(csv_rows) == 24
    assert summary["configuration_count"] == 8
    assert summary["grid"]["token_budget"] == [64, 128]
    assert summary["grid"]["candidate_k"] == [5, 20]
    assert summary["token_proxy_warning"] == runner.TOKEN_WARNING
    assert resources["token_metric"] == runner.TOKEN_METRIC
    assert resources["result_artifact_bytes"] > 0
    assert manifest["git_commit"] == "a" * 40
    assert manifest["row_count"] == 24
    assert all(row["local_only"] is True for row in rows)
    assert all(row["cloud_api_used"] is False for row in rows)
    assert all(row["llm_called"] is False for row in rows)
    assert all(row["token_metric"] == runner.TOKEN_METRIC for row in rows)
    assert "retrieved_memories" not in rows[0]
    assert "content" not in json.dumps(rows, ensure_ascii=False)
    assert str(tmp_path) not in json.dumps(summary, ensure_ascii=False)
    assert str(tmp_path) not in json.dumps(manifest, ensure_ascii=False)

    for artifact in manifest["output_files"]:
        path = output / Path(artifact["path"]).name
        if "figures/" in artifact["path"]:
            path = output / "figures" / Path(artifact["path"]).name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert artifact["sha256"] == digest
        assert artifact["size_bytes"] == path.stat().st_size


def test_json_writer_rejects_nan(tmp_path: Path) -> None:
    runner = _load_runner()

    with pytest.raises(ValueError):
        runner._write_json(tmp_path / "bad.json", {"value": float("nan")})
