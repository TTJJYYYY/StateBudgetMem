from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class DialogTurn:
    role: str
    content: str
    timestamp: str


@dataclass(frozen=True)
class DailySummary:
    summary: str
    timestamp: str


@dataclass(frozen=True)
class PaperStorageSpec:
    """Three-layer MemoryBank storage payload aligned with the paper.

    The MemoryBank paper stores chronological conversations, hierarchical event
    summaries, and an evolving user portrait. This spec keeps those layers
    explicit so the on-device reproduction can use fixed local summaries before
    a local summarizer is added.
    """

    dialogs: list[DialogTurn]
    daily_summaries: list[DailySummary] = field(default_factory=list)
    global_summary: str = ""
    user_portrait: str = ""


@dataclass(frozen=True)
class RetrievalProbe:
    query: str
    top_k: int = 3
    current_time: str | None = None


class PaperStorageBackend(Protocol):
    def store_dialog(self, role: str, content: str, timestamp: str):
        ...

    def store_summary(self, summary: str, timestamp: str):
        ...

    def update_global_summary(self, summary: str) -> None:
        ...

    def update_user_portrait(self, portrait: str) -> None:
        ...

    def get_all(self, filters: dict | None = None) -> list[Any]:
        ...

    def get_stats(self) -> dict[str, Any]:
        ...


class PaperRetrievalBackend(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        current_time: str | float | int | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def get_stats(self) -> dict[str, Any]:
        ...


def build_paper_aligned_storage(
    memory_bank: PaperStorageBackend,
    spec: PaperStorageSpec | None = None,
) -> dict[str, Any]:
    """Populate a MemoryBank instance with the paper's three storage layers.

    This function deliberately does not call any cloud LLM. Daily summaries,
    global summaries, and portraits are supplied as fixed local text so the
    first on-device reproduction remains deterministic.
    """

    payload = spec or default_paper_storage_spec()
    dialog_ids = [
        memory_bank.store_dialog(turn.role, turn.content, turn.timestamp).memory_id
        for turn in payload.dialogs
    ]
    summary_ids = [
        memory_bank.store_summary(item.summary, item.timestamp).memory_id
        for item in payload.daily_summaries
    ]

    if payload.global_summary:
        memory_bank.update_global_summary(payload.global_summary)
    if payload.user_portrait:
        memory_bank.update_user_portrait(payload.user_portrait)

    memories = memory_bank.get_all()
    return {
        "paper_layers": {
            "raw_dialog": {
                "count": len(dialog_ids),
                "memory_ids": dialog_ids,
            },
            "event_summary": {
                "count": len(summary_ids),
                "memory_ids": summary_ids,
                "global_summary_set": bool(payload.global_summary),
            },
            "user_portrait": {
                "set": bool(payload.user_portrait),
            },
        },
        "memory_type_counts": _memory_type_counts(memories),
        "memory_stats": memory_bank.get_stats(),
    }


def run_paper_retrieval_probe(
    memory_bank: PaperRetrievalBackend,
    probe: RetrievalProbe | None = None,
) -> dict[str, Any]:
    """Run and record the paper-aligned MemoryBank retrieval path.

    The recorded path is query -> local embedding -> FAISS index -> top-k
    memories -> composite score. ``MemoryBank.retrieve`` owns the embedding,
    FAISS search, time decay, and spacing-effect update; this wrapper records
    the reproducibility fields needed by the on-device reproduction.
    """

    payload = probe or default_retrieval_probe()
    started = time.perf_counter()
    retrieved = memory_bank.retrieve(
        payload.query,
        top_k=payload.top_k,
        current_time=payload.current_time,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    stats = memory_bank.get_stats()

    return {
        "query": payload.query,
        "top_k": payload.top_k,
        "current_time": payload.current_time,
        "retrieval_path": [
            "query",
            "local_embedding",
            "faiss_index",
            "top_k_memories",
            "composite_score",
        ],
        "retrieved_memory_ids": [str(item.get("memory_id", "")) for item in retrieved],
        "latency_ms": latency_ms,
        "index_size": int(stats.get("index_size", 0) or 0),
        "retrieved": [_normalize_retrieved_item(item) for item in retrieved],
    }


def default_paper_storage_spec() -> PaperStorageSpec:
    """Small local sample for smoke testing the MemoryBank paper storage flow."""

    return PaperStorageSpec(
        dialogs=[
            DialogTurn(
                role="User",
                content="My name is Lin and I am preparing for a machine learning exam.",
                timestamp="2026-06-20 10:00",
            ),
            DialogTurn(
                role="AI",
                content="I can help you plan a focused review schedule.",
                timestamp="2026-06-20 10:01",
            ),
            DialogTurn(
                role="User",
                content="I like basketball and swimming on weekends.",
                timestamp="2026-06-20 10:05",
            ),
            DialogTurn(
                role="User",
                content="The book you recommended was Automate the Boring Stuff with Python.",
                timestamp="2026-06-21 15:00",
            ),
            DialogTurn(
                role="User",
                content="My stomach has been uncomfortable, so I should avoid spicy food for now.",
                timestamp="2026-06-23 11:00",
            ),
        ],
        daily_summaries=[
            DailySummary(
                summary=(
                    "Lin is preparing for a machine learning exam and wants a "
                    "focused review schedule."
                ),
                timestamp="2026-06-20 23:00",
            ),
            DailySummary(
                summary=(
                    "Lin remembered a Python book recommendation and later reported "
                    "a temporary stomach restriction against spicy food."
                ),
                timestamp="2026-06-23 23:00",
            ),
        ],
        global_summary=(
            "The user is a student preparing for exams, likes basketball and "
            "swimming, studies Python, and currently needs mild food due to stomach discomfort."
        ),
        user_portrait=(
            "Lin is study-oriented, health-conscious, and enjoys active weekend hobbies."
        ),
    )


def default_retrieval_probe() -> RetrievalProbe:
    return RetrievalProbe(
        query="What book did you recommend and what food should I avoid now?",
        top_k=3,
        current_time="2026-06-24 10:00",
    )


def _memory_type_counts(memories: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for memory in memories:
        memory_type = getattr(memory, "memory_type", "")
        value = getattr(memory_type, "value", str(memory_type))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _normalize_retrieved_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": str(item.get("memory_id", "")),
        "memory_type": str(item.get("memory_type", "")),
        "semantic_score": float(item.get("semantic_score", 0.0) or 0.0),
        "composite_score": float(item.get("composite_score", 0.0) or 0.0),
        "time_decay": float(item.get("time_decay", 0.0) or 0.0),
        "strength": float(item.get("strength", 0.0) or 0.0),
    }
