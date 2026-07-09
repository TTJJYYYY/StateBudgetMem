from __future__ import annotations

from dataclasses import dataclass

from statebudgetmem.baselines.memorybank import (
    DailySummary,
    DialogTurn,
    PaperStorageSpec,
    RetrievalProbe,
    build_paper_aligned_storage,
    default_paper_storage_spec,
    run_paper_retrieval_probe,
)


@dataclass
class FakeStoredMemory:
    memory_id: str
    memory_type: str


class FakePaperMemoryBank:
    def __init__(self) -> None:
        self.memories: list[FakeStoredMemory] = []
        self.global_summary = ""
        self.user_portrait = ""

    def store_dialog(self, role: str, content: str, timestamp: str) -> FakeStoredMemory:
        memory = FakeStoredMemory(f"d{len(self.memories)}", "dialog")
        self.memories.append(memory)
        return memory

    def store_summary(self, summary: str, timestamp: str) -> FakeStoredMemory:
        memory = FakeStoredMemory(f"s{len(self.memories)}", "summary")
        self.memories.append(memory)
        return memory

    def update_global_summary(self, summary: str) -> None:
        self.global_summary = summary

    def update_user_portrait(self, portrait: str) -> None:
        self.user_portrait = portrait

    def get_all(self, filters=None):
        return list(self.memories)

    def get_stats(self):
        return {
            "total_memories": len(self.memories),
            "index_size": len(self.memories),
            "global_summary_set": bool(self.global_summary),
            "user_portrait_set": bool(self.user_portrait),
        }

    def retrieve(self, query, top_k=5, filters=None, current_time=None):
        return [
            {
                "memory_id": memory.memory_id,
                "memory_type": memory.memory_type,
                "semantic_score": 0.8 - index * 0.1,
                "composite_score": 0.7 - index * 0.1,
                "time_decay": 0.95,
                "strength": 2 + index,
            }
            for index, memory in enumerate(self.memories[:top_k])
        ]


def test_build_paper_aligned_storage_populates_three_layers() -> None:
    memory_bank = FakePaperMemoryBank()
    spec = PaperStorageSpec(
        dialogs=[
            DialogTurn("User", "I like swimming.", "2026-06-20 10:00"),
            DialogTurn("AI", "I will remember that.", "2026-06-20 10:01"),
        ],
        daily_summaries=[
            DailySummary("The user likes swimming.", "2026-06-20 23:00")
        ],
        global_summary="The user has an active lifestyle.",
        user_portrait="Active and health-conscious.",
    )

    report = build_paper_aligned_storage(memory_bank, spec)

    assert report["paper_layers"]["raw_dialog"]["count"] == 2
    assert report["paper_layers"]["event_summary"]["count"] == 1
    assert report["paper_layers"]["event_summary"]["global_summary_set"] is True
    assert report["paper_layers"]["user_portrait"]["set"] is True
    assert report["memory_type_counts"] == {"dialog": 2, "summary": 1}
    assert memory_bank.global_summary == "The user has an active lifestyle."
    assert memory_bank.user_portrait == "Active and health-conscious."


def test_default_paper_storage_spec_is_local_and_nonempty() -> None:
    spec = default_paper_storage_spec()

    assert spec.dialogs
    assert spec.daily_summaries
    assert spec.global_summary
    assert spec.user_portrait


def test_run_paper_retrieval_probe_records_paper_fields() -> None:
    memory_bank = FakePaperMemoryBank()
    build_paper_aligned_storage(
        memory_bank,
        PaperStorageSpec(
            dialogs=[DialogTurn("User", "I like swimming.", "2026-06-20 10:00")],
            daily_summaries=[
                DailySummary("The user likes swimming.", "2026-06-20 23:00")
            ],
        ),
    )

    report = run_paper_retrieval_probe(
        memory_bank,
        RetrievalProbe(
            query="What activity do I like?",
            top_k=2,
            current_time="2026-06-21 10:00",
        ),
    )

    assert report["retrieval_path"] == [
        "query",
        "local_embedding",
        "faiss_index",
        "top_k_memories",
        "composite_score",
    ]
    assert report["retrieved_memory_ids"] == ["d0", "s1"]
    assert report["index_size"] == 2
    assert report["latency_ms"] >= 0.0
    assert report["retrieved"][0]["semantic_score"] == 0.8
    assert report["retrieved"][0]["composite_score"] == 0.7
    assert report["retrieved"][0]["time_decay"] == 0.95
    assert report["retrieved"][0]["strength"] == 2.0
