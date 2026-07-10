"""Tests for Phase 1 evaluator (费哲瀚 — C1/C2/C3).

All tests use built-in smoke data; no external dataset or cloud API required.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from statebudgetmem.baselines.memorybank.metrics import (
    gold_retrieval_f1,
    gold_retrieval_precision,
    gold_retrieval_recall,
)

ROOT = Path(__file__).resolve().parents[3]


class TestGoldMetrics:
    def test_precision_all_hits(self):
        assert gold_retrieval_precision(["m1", "m2", "m3"], ["m1", "m2"]) == 2 / 3

    def test_precision_no_hits(self):
        assert gold_retrieval_precision(["m1"], ["m2"]) == 0.0

    def test_precision_empty_retrieved(self):
        assert gold_retrieval_precision([], ["m1"]) == 0.0

    def test_precision_empty_gold(self):
        assert gold_retrieval_precision(["m1"], []) == 0.0

    def test_recall_all_found(self):
        assert gold_retrieval_recall(["m1", "m2"], ["m1"]) == 1.0

    def test_recall_partial(self):
        assert gold_retrieval_recall(["m1"], ["m1", "m2"]) == 0.5

    def test_recall_empty_gold(self):
        assert gold_retrieval_recall(["m1"], []) == 1.0

    def test_f1_perfect(self):
        assert gold_retrieval_f1(1.0, 1.0) == 1.0

    def test_f1_zero(self):
        assert gold_retrieval_f1(0.0, 0.0) == 0.0


class TestSmokeRunner:
    def test_smoke_produces_raw_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r = subprocess.run(
                [sys.executable, "tools/memorybank/run_phase1_baseline.py",
                 "--smoke", "--top-k", "3", "--output-dir", tmpdir],
                capture_output=True, text=True, cwd=ROOT,
            )
            assert r.returncode == 0, r.stderr
            raw = list(Path(tmpdir, "raw").glob("*.jsonl"))
            assert len(raw) == 1
            lines = [
                json.loads(l)
                for l in raw[0].read_text(encoding="utf-8").strip().split("\n")
                if l
            ]
            assert len(lines) == 5
            for row in lines:
                assert "paper_metrics" in row
                assert row["local_only"] is True

    def test_smoke_produces_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [sys.executable, "tools/memorybank/run_phase1_baseline.py",
                 "--smoke", "--output-dir", tmpdir],
                capture_output=True, cwd=ROOT,
            )
            sf = list(Path(tmpdir, "summaries").glob("*.json"))
            assert len(sf) == 1
            s = json.loads(sf[0].read_text(encoding="utf-8"))
            assert s["probe_count"] == 5

    def test_smoke_produces_resources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [sys.executable, "tools/memorybank/run_phase1_baseline.py",
                 "--smoke", "--output-dir", tmpdir],
                capture_output=True, cwd=ROOT,
            )
            rf = list(Path(tmpdir, "resources").glob("*.json"))
            assert len(rf) == 1
            r = json.loads(rf[0].read_text(encoding="utf-8"))
            assert r["cloud_api_used"] is False
            assert r["llm_called"] is False

    def test_smoke_produces_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [sys.executable, "tools/memorybank/run_phase1_baseline.py",
                 "--smoke", "--output-dir", tmpdir],
                capture_output=True, cwd=ROOT,
            )
            cf = list(Path(tmpdir, "summaries").glob("*.csv"))
            assert len(cf) == 1
            lines = cf[0].read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 6

    def test_hash_embedding_no_cloud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [sys.executable, "tools/memorybank/run_phase1_baseline.py",
                 "--smoke", "--embedding-backend", "hash", "--output-dir", tmpdir],
                capture_output=True, cwd=ROOT,
            )
            rf = list(Path(tmpdir, "resources").glob("*.json"))
            r = json.loads(rf[0].read_text(encoding="utf-8"))
            assert r["embedding_backend"] == "hash"
            assert r["cloud_api_used"] is False
