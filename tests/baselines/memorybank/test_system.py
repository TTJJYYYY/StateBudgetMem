from statebudgetmem.baselines.memorybank import TFIDFMemoryBank
from statebudgetmem.interfaces import MemoryStatus, UpdateOperation


def test_tfidf_memorybank_supports_shared_online_contract_without_heavy_models() -> None:
    memory = TFIDFMemoryBank()
    ids = memory.add([("用户", "我喜欢游泳", "2026-06-20 10:00")])

    assert len(ids) == 1
    stored = memory.get(ids[0])
    assert stored is not None
    assert stored.content == "用户: 我喜欢游泳"

    memory.update(ids[0], UpdateOperation.SUPERSEDE)
    assert memory.get(ids[0]).status == MemoryStatus.SUPERSEDED


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
