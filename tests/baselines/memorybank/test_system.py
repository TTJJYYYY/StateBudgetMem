import math
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

        for module_name in list(sys.modules):
            if module_name.split(".", 1)[0] in blocked_roots:
                del sys.modules[module_name]

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


def test_memorybank_constructor_validates_core_parameters() -> None:
    import pytest

    from statebudgetmem.baselines.memorybank.system import MemoryBank

    with pytest.raises(ValueError, match="embedding_dim.*0"):
        MemoryBank(embedding_dim=0)
    with pytest.raises(ValueError, match="forgetting_threshold.*1.5"):
        MemoryBank(forgetting_threshold=1.5)
    with pytest.raises(ValueError, match="decay_interval_hours.*0"):
        MemoryBank(decay_interval_hours=0)


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


class _FakeIndexMany:
    def __init__(self, distances):
        self._distances = distances
        self.ntotal = len(distances)
        self.search_calls = []

    def search(self, _query, top_k):
        self.search_calls.append(top_k)
        distances = [score for score, _idx in self._distances[:top_k]]
        indices = [idx for _score, idx in self._distances[:top_k]]
        return [distances], [indices]


def _memorybank_with_pieces(pieces, distances=None):
    from statebudgetmem.baselines.memorybank.system import MemoryBank

    memory = object.__new__(MemoryBank)
    if distances is None:
        distances = [(0.9 - index * 0.1, index) for index in range(len(pieces))]
    memory.index = _FakeIndexMany(distances)
    memory.faiss_id_to_mid = {
        index: piece.memory_id for index, piece in enumerate(pieces)
    }
    memory.forgetting_threshold = 0.5
    memory.decay_interval_sec = 24.0 * 3600.0
    memory.access_count = 0
    memory.memories = pieces
    memory.memories_by_id = {piece.memory_id: piece for piece in pieces}
    memory._ensure_index = lambda: None
    memory._get_embedding = lambda _text: _FakeVector()
    return memory


def test_memorybank_retrieval_records_reinforcement_fields() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    memory = object.__new__(MemoryBank)
    memory.index = _FakeIndex()
    memory.faiss_id_to_mid = {0: "m1"}
    memory.forgetting_threshold = 0.3
    memory.decay_interval_sec = 24.0 * 3600.0
    memory.access_count = 0
    piece = MemoryPiece(
        content="AI recommended a Python book",
        timestamp=MemoryBank._parse_time("2026-06-20 10:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="m1",
        last_accessed=MemoryBank._parse_time("2026-06-20 10:00"),
    )
    piece.strength = 100.0
    memory.memories_by_id = {"m1": piece}
    memory._ensure_index = lambda: None
    memory._get_embedding = lambda _text: _FakeVector()

    assert piece.strength == 100.0
    before_last_accessed = piece.last_accessed
    retrieved = MemoryBank.retrieve(
        memory,
        "What book did you recommend?",
        top_k=1,
        current_time="2026-06-21 10:00",
    )

    assert len(retrieved) == 1
    row = retrieved[0]
    assert row["before_strength"] == 100.0
    assert row["after_strength"] == 101.0
    assert row["strength"] == 101.0
    assert row["before_last_accessed"] == before_last_accessed
    assert row["after_last_accessed"] == MemoryBank._parse_time("2026-06-21 10:00")
    assert row["retrieval_rank"] == 1
    assert row["retrieval_score"] == row["composite_score"]
    assert piece.strength == 101.0
    assert piece.last_accessed == row["after_last_accessed"]
    assert row["before_access_count"] == 0
    assert row["after_access_count"] == 1
    assert row["score"] == row["composite_score"]
    assert row["retention"] >= 0
    assert row["is_forgotten"] is False


def test_memorybank_default_mode_records_but_does_not_exclude_forgotten() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    current_time = MemoryBank._parse_time("2026-06-10 10:00")
    old = MemoryPiece(
        content="old spicy food note",
        timestamp=MemoryBank._parse_time("2026-06-01 10:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="old",
        last_accessed=MemoryBank._parse_time("2026-06-01 10:00"),
    )
    recent = MemoryPiece(
        content="recent spicy food note",
        timestamp=MemoryBank._parse_time("2026-06-10 09:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="recent",
        last_accessed=MemoryBank._parse_time("2026-06-10 09:00"),
    )
    recent.strength = 10.0
    bank = _memorybank_with_pieces([old, recent], distances=[(0.99, 0), (0.8, 1)])

    result = MemoryBank.retrieve_with_metadata(
        bank,
        "spicy food",
        top_k=2,
        current_time=current_time,
        exclude_forgotten=False,
    )

    assert "old" in result["forgotten_memory_ids"]
    assert result["excluded_forgotten_memory_ids"] == []
    assert "old" in [row["memory_id"] for row in result["memories"]]
    assert old.strength == 2.0
    assert recent.strength == 11.0


def test_memorybank_exclude_forgotten_filters_without_reinforcement() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    current_time = MemoryBank._parse_time("2026-06-10 10:00")
    old = MemoryPiece(
        content="old spicy food note",
        timestamp=MemoryBank._parse_time("2026-06-01 10:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="old",
        last_accessed=MemoryBank._parse_time("2026-06-01 10:00"),
    )
    recent = MemoryPiece(
        content="recent spicy food note",
        timestamp=MemoryBank._parse_time("2026-06-10 09:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="recent",
        last_accessed=MemoryBank._parse_time("2026-06-10 09:00"),
    )
    recent.strength = 10.0
    before = (old.strength, old.last_accessed, old.access_count)
    bank = _memorybank_with_pieces([old, recent], distances=[(0.99, 0), (0.8, 1)])

    result = MemoryBank.retrieve_with_metadata(
        bank,
        "spicy food",
        top_k=2,
        current_time=current_time,
        exclude_forgotten=True,
    )

    assert "old" in result["forgotten_memory_ids"]
    assert "old" in result["excluded_forgotten_memory_ids"]
    assert "old" not in [row["memory_id"] for row in result["memories"]]
    assert (old.strength, old.last_accessed, old.access_count) == before
    assert recent.strength == 11.0


def test_memorybank_does_not_reinforce_candidates_outside_final_top_k() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    pieces = [
        MemoryPiece(
            content=f"candidate {idx}",
            timestamp=MemoryBank._parse_time("2026-06-10 09:00"),
            memory_type=MemoryType.DIALOG,
            memory_id=f"m{idx}",
            last_accessed=MemoryBank._parse_time("2026-06-10 09:00"),
        )
        for idx in range(3)
    ]
    bank = _memorybank_with_pieces(
        pieces,
        distances=[(0.9, 0), (0.8, 1), (0.7, 2)],
    )
    before = {
        piece.memory_id: (piece.strength, piece.last_accessed, piece.access_count)
        for piece in pieces
    }

    result = MemoryBank.retrieve_with_metadata(
        bank,
        "candidate",
        top_k=1,
        current_time="2026-06-10 10:00",
    )

    selected_id = result["memories"][0]["memory_id"]
    for piece in pieces:
        old_strength, old_last_accessed, old_access_count = before[piece.memory_id]
        if piece.memory_id == selected_id:
            assert piece.strength == old_strength + 1
            assert piece.access_count == old_access_count + 1
        else:
            assert (piece.strength, piece.last_accessed, piece.access_count) == (
                old_strength,
                old_last_accessed,
                old_access_count,
            )
    assert bank.access_count == 1


def test_memorybank_filters_before_reinforcement() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    blocked = MemoryPiece(
        content="blocked diet note",
        timestamp=MemoryBank._parse_time("2026-06-10 09:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="blocked",
        last_accessed=MemoryBank._parse_time("2026-06-10 09:00"),
        tags=["blocked"],
    )
    allowed = MemoryPiece(
        content="allowed diet note",
        timestamp=MemoryBank._parse_time("2026-06-10 09:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="allowed",
        last_accessed=MemoryBank._parse_time("2026-06-10 09:00"),
        tags=["allowed"],
    )
    bank = _memorybank_with_pieces(
        [blocked, allowed],
        distances=[(0.99, 0), (0.8, 1)],
    )

    result = MemoryBank.retrieve_with_metadata(
        bank,
        "diet",
        top_k=2,
        filters={"tags": ["allowed"]},
        current_time="2026-06-10 10:00",
    )

    assert [row["memory_id"] for row in result["memories"]] == ["allowed"]
    assert blocked.strength == 1.0
    assert blocked.access_count == 0
    assert allowed.strength == 2.0


def test_memorybank_top_k_non_positive_returns_full_empty_metadata() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    piece = MemoryPiece(
        content="candidate",
        timestamp=MemoryBank._parse_time("2026-06-10 09:00"),
        memory_type=MemoryType.DIALOG,
        memory_id="m1",
        last_accessed=MemoryBank._parse_time("2026-06-10 09:00"),
    )
    bank = _memorybank_with_pieces([piece], distances=[(0.9, 0)])

    result = MemoryBank.retrieve_with_metadata(
        bank,
        "candidate",
        top_k=0,
        current_time="2026-06-10 10:00",
        exclude_forgotten=True,
    )

    assert result == {
        "memories": [],
        "forgotten_memory_ids": [],
        "excluded_forgotten_memory_ids": [],
        "candidate_count_before_forgetting": 0,
        "candidate_count_after_forgetting": 0,
        "exclude_forgotten": True,
        "forgetting_threshold": 0.5,
        "retention_time_unit_hours": 24.0,
    }
    assert bank.index.search_calls == []
    assert piece.strength == 1.0
    assert piece.access_count == 0


def test_memorybank_expands_search_after_forgotten_candidates() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    current_time = MemoryBank._parse_time("2026-06-10 10:00")
    pieces = []
    for index in range(4):
        last_accessed = (
            MemoryBank._parse_time("2026-06-01 10:00")
            if index < 3
            else MemoryBank._parse_time("2026-06-10 09:00")
        )
        piece = MemoryPiece(
            content=f"candidate {index}",
            timestamp=last_accessed,
            memory_type=MemoryType.DIALOG,
            memory_id=f"m{index}",
            last_accessed=last_accessed,
        )
        if index == 3:
            piece.strength = 10.0
        pieces.append(piece)
    bank = _memorybank_with_pieces(
        pieces,
        distances=[(0.99, 0), (0.98, 1), (0.97, 2), (0.4, 3)],
    )
    before = {
        piece.memory_id: (piece.strength, piece.last_accessed, piece.access_count)
        for piece in pieces
    }

    result = MemoryBank.retrieve_with_metadata(
        bank,
        "candidate",
        top_k=1,
        current_time=current_time,
        exclude_forgotten=True,
    )

    assert bank.index.search_calls == [3, 4]
    assert [row["memory_id"] for row in result["memories"]] == ["m3"]
    assert result["excluded_forgotten_memory_ids"] == ["m0", "m1", "m2"]
    for piece in pieces[:3]:
        assert (piece.strength, piece.last_accessed, piece.access_count) == before[
            piece.memory_id
        ]
    assert pieces[3].strength == before["m3"][0] + 1
    assert pieces[3].access_count == before["m3"][2] + 1


def test_memorybank_forgetting_log_records_retention_and_forgotten_ids() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    memory = object.__new__(MemoryBank)
    memory.forgetting_threshold = 0.5
    memory.decay_interval_sec = 24.0 * 3600.0
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


def test_memorybank_retention_uses_default_daily_time_unit() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank
    from statebudgetmem.interfaces import MemoryPiece, MemoryType

    memory = object.__new__(MemoryBank)
    memory.forgetting_threshold = 0.3
    memory.decay_interval_sec = 24.0 * 3600.0
    memory.memories = [
        MemoryPiece(
            content="one day old",
            timestamp=MemoryBank._parse_time("2026-06-09 10:00"),
            memory_type=MemoryType.DIALOG,
            memory_id="m1",
            last_accessed=MemoryBank._parse_time("2026-06-09 10:00"),
        )
    ]
    memory.memories_by_id = {"m1": memory.memories[0]}

    report = MemoryBank.forgetting_log(memory, current_time="2026-06-10 10:00")
    event = report["events"][0]

    assert memory.decay_interval_sec / 3600.0 == 24.0
    assert event["elapsed_hours"] == 24.0
    assert event["elapsed_time_units"] == 1.0
    assert event["retention_time_unit_hours"] == 24.0
    assert math.isclose(event["retention"], math.exp(-1), rel_tol=1e-12)


def test_memorybank_build_augmented_prompt_matches_paper_sections() -> None:
    from statebudgetmem.baselines.memorybank.system import MemoryBank

    memory = object.__new__(MemoryBank)
    memory.user_portrait = "The user is health-conscious and likes concise advice."
    memory.global_summary = "Recent events: the user asked for meal planning help."
    memory.retrieve = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("build_augmented_prompt should use retrieve_with_metadata")
    )
    memory.retrieve_with_metadata = lambda **_kwargs: {
        "memories": [
            {
                "memory_id": "m_diet",
                "memory_type": "dialog",
                "content": "User should avoid spicy food this week.",
                "semantic_score": 0.95,
                "composite_score": 0.91,
                "retrieval_score": 0.91,
                "score": 0.91,
                "retrieval_rank": 1,
                "time_decay": 1.0,
                "strength_factor": 1.3,
                "retention": 0.8,
                "is_forgotten": False,
                "forgetting_threshold": 0.3,
                "timestamp": 1.0,
                "age_hours": 0.0,
                "status": "active",
                "tags": ["diet"],
                "before_strength": 1.0,
                "after_strength": 2.0,
                "before_last_accessed": 123.0,
                "after_last_accessed": 456.0,
                "before_access_count": 0,
                "after_access_count": 1,
                "query": "What should I eat tonight?",
                "recall_timestamp": 456.0,
            }
        ],
        "forgotten_memory_ids": ["excluded_old"],
        "excluded_forgotten_memory_ids": ["excluded_old"],
        "candidate_count_before_forgetting": 2,
        "candidate_count_after_forgetting": 1,
        "exclude_forgotten": True,
        "forgetting_threshold": 0.3,
        "retention_time_unit_hours": 24.0,
    }

    result = MemoryBank.build_augmented_prompt(
        memory,
        query="What should I eat tonight?",
        current_time="2026-06-21 18:00",
        top_k=3,
        exclude_forgotten=True,
    )

    prompt = result["prompt_template"]
    assert result["retrieved_count"] == 1
    assert result["retrieved_memory_ids"] == ["m_diet"]
    assert result["global_user_portrait"] == memory.user_portrait
    assert result["global_event_summary"] == memory.global_summary
    assert result["current_user_query"] == "What should I eat tonight?"
    assert result["prompt_sections"]["relevant_memories"].startswith("[dialog]")
    assert result["prompt_token_estimate"] > 0
    assert result["forgotten_memory_ids"] == ["excluded_old"]
    assert result["excluded_forgotten_memory_ids"] == ["excluded_old"]
    assert result["excluded_forgotten_count"] == 1
    assert result["candidate_count_before_forgetting"] == 2
    assert result["candidate_count_after_forgetting"] == 1
    assert result["retention_time_unit_hours"] == 24.0
    assert result["strength_before_after"] == [
        {"memory_id": "m_diet", "before": 1.0, "after": 2.0}
    ]
    assert result["last_accessed_before_after"] == [
        {"memory_id": "m_diet", "before": 123.0, "after": 456.0}
    ]
    assert result["access_count_before_after"] == [
        {"memory_id": "m_diet", "before": 0, "after": 1}
    ]
    assert result["retrieved_memory_ids"] == [
        row["memory_id"] for row in result["retrieved_memories"]
    ]
    assert "User should avoid spicy food this week." in prompt
    assert "excluded_old" not in prompt

    memory_pos = prompt.index("【相关历史记忆】")
    portrait_pos = prompt.index("【全局用户画像】")
    summary_pos = prompt.index("【全局事件摘要】")
    query_pos = prompt.index("【用户当前问题】")
    assert memory_pos < portrait_pos < summary_pos < query_pos
