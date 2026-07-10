from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class MemoryBankMetricSpec:
    """Deterministic labels for MemoryBank reproduction metrics.

    The original paper uses human annotation for retrieval accuracy, response
    correctness, and coherence. Before the formal dataset is ready, this spec
    provides local keyword labels for the built-in reproduction sample.
    """

    query_id: str
    relevant_keywords: tuple[str, ...] = ()
    answer_keywords: tuple[str, ...] = ()
    stale_keywords: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def memory_retrieval_accuracy(
    retrieved_memories: Iterable[dict[str, Any]],
    relevant_keywords: Iterable[str],
) -> float:
    """Approximate paper retrieval accuracy using labeled keywords."""

    keywords = _normalize_keywords(relevant_keywords)
    if not keywords:
        return 0.0
    retrieved_text = _joined_memory_text(retrieved_memories)
    return _keyword_coverage(retrieved_text, keywords)


def response_correctness(answer: str, expected_keywords: Iterable[str]) -> float:
    """Approximate paper response correctness with expected keyword coverage."""

    keywords = _normalize_keywords(expected_keywords)
    if not keywords:
        return 0.0
    return _keyword_coverage(answer, keywords)


def contextual_coherence(
    query: str,
    answer: str,
    retrieved_memories: Iterable[dict[str, Any]],
) -> float:
    """Heuristic coherence score for retrieval-only local reproduction.

    A fully faithful paper reproduction should use human or judge labels. This
    deterministic proxy rewards answers that are non-empty, overlap the query,
    and are grounded in retrieved memory text.
    """

    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0

    score = 0.2
    query_tokens = _tokens(query)
    if query_tokens and answer_tokens & query_tokens:
        score += 0.4

    memory_tokens = _tokens(_joined_memory_text(retrieved_memories))
    if memory_tokens and answer_tokens & memory_tokens:
        score += 0.4

    return min(score, 1.0)


def stale_retrieval_rate_by_keywords(
    retrieved_memories: Iterable[dict[str, Any]],
    stale_keywords: Iterable[str],
) -> float:
    """Keyword-labeled stale retrieval rate for built-in reproduction data."""

    keywords = _normalize_keywords(stale_keywords)
    memories = list(retrieved_memories)
    if not memories or not keywords:
        return 0.0
    stale_count = 0
    for memory in memories:
        content = str(memory.get("content", "")).lower()
        if any(keyword in content for keyword in keywords):
            stale_count += 1
    return stale_count / len(memories)


def evaluate_reproduction_row(
    row: dict[str, Any],
    spec: MemoryBankMetricSpec,
) -> dict[str, float]:
    """Compute paper and on-device metrics for one reproduction row."""

    retrieved = list(row.get("retrieved_memories", []))
    answer = str(row.get("template_answer") or row.get("prompt_template", ""))
    retrieved_ids = row.get("retrieved_memory_ids") or [
        str(item.get("memory_id", "")) for item in retrieved
    ]
    gold_ids = list(row.get("gold_memory_ids", []) or [])

    gold_p = gold_retrieval_precision(retrieved_ids, gold_ids)
    gold_r = gold_retrieval_recall(retrieved_ids, gold_ids)

    return {
        "memory_retrieval_accuracy": memory_retrieval_accuracy(
            retrieved,
            spec.relevant_keywords,
        ),
        "response_correctness": response_correctness(
            answer,
            spec.answer_keywords,
        ),
        "contextual_coherence": contextual_coherence(
            str(row.get("query", "")),
            answer,
            retrieved,
        ),
        "stale_retrieval_rate": stale_retrieval_rate_by_keywords(
            retrieved,
            spec.stale_keywords,
        ),
        "gold_precision": gold_p,
        "gold_recall": gold_r,
        "gold_f1": gold_retrieval_f1(gold_p, gold_r),
        "retrieval_latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
        "faiss_index_size": float(row.get("index_size", 0.0) or 0.0),
        "prompt_token_cost": float(row.get("prompt_token_estimate", 0.0) or 0.0),
    }


def summarize_metric_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    selected = list(rows)
    if not selected:
        return {
            "memory_retrieval_accuracy": 0.0,
            "response_correctness": 0.0,
            "contextual_coherence": 0.0,
            "stale_retrieval_rate": 0.0,
            "gold_precision": 0.0,
            "gold_recall": 0.0,
            "gold_f1": 0.0,
            "mean_retrieval_latency_ms": 0.0,
            "mean_faiss_index_size": 0.0,
            "mean_prompt_token_cost": 0.0,
        }
    return {
        "memory_retrieval_accuracy": _mean(
            selected,
            "memory_retrieval_accuracy",
        ),
        "response_correctness": _mean(selected, "response_correctness"),
        "contextual_coherence": _mean(selected, "contextual_coherence"),
        "stale_retrieval_rate": _mean(selected, "stale_retrieval_rate"),
        "gold_precision": _mean(selected, "gold_precision"),
        "gold_recall": _mean(selected, "gold_recall"),
        "gold_f1": _mean(selected, "gold_f1"),
        "mean_retrieval_latency_ms": _mean(selected, "retrieval_latency_ms"),
        "mean_faiss_index_size": _mean(selected, "faiss_index_size"),
        "mean_prompt_token_cost": _mean(selected, "prompt_token_cost"),
    }


# ── Gold-label metrics (Phase 1 formal evaluation) ──────────────────────


def gold_retrieval_precision(
    retrieved_ids: list[str],
    gold_ids: list[str],
) -> float:
    """Precision: fraction of retrieved memories that are gold."""
    if not retrieved_ids or not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    hits = sum(1 for mid in retrieved_ids if mid in gold_set)
    return hits / len(retrieved_ids)


def gold_retrieval_recall(
    retrieved_ids: list[str],
    gold_ids: list[str],
) -> float:
    """Recall: fraction of gold memories that were retrieved."""
    if not gold_ids:
        return 1.0
    gold_set = set(gold_ids)
    hits = sum(1 for mid in retrieved_ids if mid in gold_set)
    return hits / len(gold_ids)


def gold_retrieval_f1(precision: float, recall: float) -> float:
    """F1 score for retrieval."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _normalize_keywords(keywords: Iterable[str]) -> list[str]:
    return [keyword.lower() for keyword in keywords if str(keyword).strip()]


def _joined_memory_text(retrieved_memories: Iterable[dict[str, Any]]) -> str:
    return " ".join(str(item.get("content", "")) for item in retrieved_memories)


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    lowered = text.lower()
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return hits / len(keywords)


def _tokens(text: str) -> set[str]:
    return {token for token in text.lower().replace("_", " ").split() if len(token) > 2}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)


__all__ = [
    "MemoryBankMetricSpec",
    "contextual_coherence",
    "evaluate_reproduction_row",
    "gold_retrieval_f1",
    "gold_retrieval_precision",
    "gold_retrieval_recall",
    "memory_retrieval_accuracy",
    "response_correctness",
    "stale_retrieval_rate_by_keywords",
    "summarize_metric_rows",
]
