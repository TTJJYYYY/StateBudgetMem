from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "memorybank" / "run_ondevice_memorybank_baseline.py"
SPEC = importlib.util.spec_from_file_location("ondevice_memorybank_baseline", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_runner_smoke_writes_required_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    assert runner.main(["--output-dir", str(output), "--smoke", "--embedding-dim", "32"]) == 0
    required = {
        "config.json", "environment.json", "predictions.csv", "metrics.json",
        "memorybank_resource_metrics.json", "memorybank_retrieval_log.jsonl",
        "memorybank_reinforcement_log.jsonl", "memorybank_forgetting_log.jsonl",
        "memorybank_run_summary.json", "summary.md",
    }
    assert required <= {p.name for p in output.iterdir()}
    summary = json.loads((output / "memorybank_run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert summary["local_only"] is True
    assert summary["network_calls"] == 0
    assert len(list((output / "figures").glob("*.png"))) == 7


def test_logs_reinforcement_forgetting_and_persistence(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    runner.main(["--output-dir", str(output), "--smoke", "--embedding-dim", "32"])
    retrieval = json.loads((output / "memorybank_retrieval_log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(("memory_id", "query", "rank", "retrieval_score", "before_strength",
                "after_strength", "before_last_accessed", "after_last_accessed",
                "retention", "is_forgotten", "forgotten_memory_ids", "timestamp")) <= set(retrieval)
    assert retrieval["after_strength"] == retrieval["before_strength"] + 1
    assert retrieval["after_last_accessed"] >= retrieval["before_last_accessed"]
    forgetting = json.loads((output / "memorybank_forgetting_log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "retention" in forgetting and "forgotten_memory_ids" in forgetting
    storage = output / "storage" / "n100_r0"
    assert (storage / "memory_data.jsonl").exists()
    assert (storage / "metadata.json").exists()
    assert (storage / "embeddings.npy").exists()
    assert (storage / "memorybank.faiss").exists()


def test_retention_is_deterministic_without_waiting() -> None:
    encoder = runner.HashEncoder(32)
    args = runner.parse_args(["--output-dir", "unused", "--smoke", "--embedding-dim", "32"])
    bank, _, _, _ = runner.build_bank(args, encoder, 100, 42, __import__("psutil").Process())
    event = bank.forgetting_log("2026-07-11 12:00")["events"][0]
    expected = __import__("math").exp(-event["elapsed_time_units"] / event["strength"])
    assert event["retention"] == expected
