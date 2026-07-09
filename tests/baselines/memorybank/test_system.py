import os
import subprocess
import sys
import textwrap
from pathlib import Path

from statebudgetmem.baselines.memorybank import TFIDFMemoryBank
from statebudgetmem.interfaces import MemoryStatus, UpdateOperation

ROOT = Path(__file__).resolve().parents[3]


def test_tfidf_memorybank_supports_shared_online_contract_without_heavy_models() -> None:
    memory = TFIDFMemoryBank()
    ids = memory.add([("用户", "我喜欢游泳", "2026-06-20 10:00")])

    assert len(ids) == 1
    stored = memory.get(ids[0])
    assert stored is not None
    assert stored.content == "用户: 我喜欢游泳"

    memory.update(ids[0], UpdateOperation.SUPERSEDE)
    assert memory.get(ids[0]).status == MemoryStatus.SUPERSEDED


def test_memorybank_package_imports_without_optional_dependencies() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked_roots = {"numpy", "faiss", "sentence_transformers"}

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked_roots:
                    raise ImportError(f"blocked optional dependency: {fullname}")
                return None

        sys.meta_path.insert(0, BlockOptional())

        import statebudgetmem.baselines.memorybank as memorybank

        assert memorybank.TFIDFMemoryBank.__name__ == "TFIDFMemoryBank"
        try:
            memorybank.MemoryBank()
        except ImportError as exc:
            message = str(exc)
            assert "pip install -e '.[memorybank]'" in message
            assert "numpy" in message
            assert "faiss-cpu" in message
        else:
            raise AssertionError("MemoryBank should require optional dependencies")
        """
    )
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_memorybank_similarity_update_uses_merge_operation() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    memory = object.__new__(MemoryBank)
    memory.memories = [object()]
    memory.retrieve = lambda query, top_k=3: [
        {"semantic_score": 0.8, "memory_id": "m1"}
    ]

    candidate = MemoryPiece(
        content="user likes swimming on weekends",
        timestamp=0.0,
        memory_type=MemoryType.FACT,
        memory_id="m2",
    )

    operation, target_id = MemoryBank._classify_operation(memory, candidate)

    assert operation is UpdateOperation.MERGE
    assert target_id == "m1"


def test_memorybank_merge_operation_updates_target_memory() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    memory = object.__new__(MemoryBank)
    target = MemoryPiece(
        content="user likes swimming",
        timestamp=0.0,
        memory_type=MemoryType.FACT,
        memory_id="m1",
    )
    candidate = MemoryPiece(
        content="user swims twice per week",
        timestamp=1.0,
        memory_type=MemoryType.FACT,
        memory_id="m2",
    )
    memory.memories_by_id = {"m1": target}

    updated_id = MemoryBank._execute_operation(
        memory,
        UpdateOperation.MERGE,
        candidate,
        "m1",
    )

    assert updated_id == "m1"
    assert "user swims twice per week" in target.content
    assert target.version == 2
    assert target.strength == 1.5


class _FakeVector:
    def reshape(self, *_args):
        return self


class _FakeIndex:
    ntotal = 1

    def search(self, _query, _top_k):
        return [[0.9]], [[0]]


def test_memorybank_retrieval_records_reinforcement_fields() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    memory = object.__new__(MemoryBank)
    memory.index = _FakeIndex()
    memory.faiss_id_to_mid = {0: "m1"}
    memory.forgetting_threshold = 0.3
    memory.access_count = 0
    piece = MemoryPiece(
        content="AI recommended a Python book",
        timestamp=MemoryBank._parse_time("2026-06-20 10:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="m1",
        last_accessed=MemoryBank._parse_time("2026-06-20 10:00"),
    )
    memory.memories_by_id = {"m1": piece}
    memory._ensure_index = lambda: None
    memory._get_embedding = lambda _text: _FakeVector()

    assert piece.strength == 1.0
    before_last_accessed = piece.last_accessed
    retrieved = MemoryBank.retrieve(
        memory,
        "What book did you recommend?",
        top_k=1,
        current_time="2026-06-21 10:00",
    )

    assert len(retrieved) == 1
    row = retrieved[0]
    assert row["before_strength"] == 1.0
    assert row["after_strength"] == 2.0
    assert row["strength"] == 2.0
    assert row["before_last_accessed"] == before_last_accessed
    assert row["after_last_accessed"] == MemoryBank._parse_time("2026-06-21 10:00")
    assert row["retrieval_rank"] == 1
    assert row["retrieval_score"] == row["composite_score"]
    assert piece.strength == 2.0
    assert piece.last_accessed == row["after_last_accessed"]


def test_memorybank_forgetting_log_records_retention_and_forgotten_ids() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    memory = object.__new__(MemoryBank)
    memory.forgetting_threshold = 0.5
    memory.forget_count = 0
    old = MemoryPiece(
        content="old memory",
        timestamp=MemoryBank._parse_time("2026-06-01 10:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="old",
        last_accessed=MemoryBank._parse_time("2026-06-01 10:00"),
    )
    recent = MemoryPiece(
        content="recent memory",
        timestamp=MemoryBank._parse_time("2026-06-10 10:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="recent",
        last_accessed=MemoryBank._parse_time("2026-06-10 09:59"),
    )
    recent.strength = 1000.0
    memory.memories = [old, recent]
    memory.memories_by_id = {"old": old, "recent": recent}

    preview = MemoryBank.forgetting_log(memory, current_time="2026-06-10 10:00")
    assert "old" in preview["forgotten_memory_ids"]
    old_event = next(row for row in preview["events"] if row["memory_id"] == "old")
    assert old_event["retention"] < 0.5
    assert old_event["is_forgotten"] is True

    report = MemoryBank.update_forgetting_with_log(
        memory,
        current_time="2026-06-10 10:00",
    )
    assert report["forgotten_memory_ids"] == ["old"]
    assert report["threshold"] == 0.5
    assert old.strength == 0.5
    assert memory.forget_count == 1

    legacy = MemoryBank.update_forgetting(memory, current_time="2026-06-10 10:00")
    assert all("retention" in row for row in legacy)
