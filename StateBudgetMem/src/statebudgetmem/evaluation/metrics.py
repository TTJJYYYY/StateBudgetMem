from __future__ import annotations

from statebudgetmem.schemas import QueryRecord, RetrievedMemory


def recall_at_k(retrieved_ids: list[str], gold_relevant_ids: list[str]) -> float:
    if not gold_relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & set(gold_relevant_ids)) / len(set(gold_relevant_ids))


def valid_recall_at_k(retrieved_ids: list[str], gold_valid_ids: list[str]) -> float:
    if not gold_valid_ids:
        return 0.0
    return len(set(retrieved_ids) & set(gold_valid_ids)) / len(set(gold_valid_ids))


def stale_retrieval_rate(retrieved_ids: list[str], gold_stale_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return len(set(retrieved_ids) & set(gold_stale_ids)) / len(retrieved_ids)


def average_retrieved_token_cost(retrieved: list[RetrievedMemory]) -> float:
    if not retrieved:
        return 0.0
    return sum(item.memory.token_cost for item in retrieved) / len(retrieved)


def evaluate_retrieval(
    query: QueryRecord,
    retrieved: list[RetrievedMemory],
    retrieval_latency_ms: float,
) -> dict[str, float]:
    retrieved_ids = [item.memory.memory_id for item in retrieved]
    return {
        "recall_at_k": recall_at_k(retrieved_ids, query.gold_relevant_memory_ids),
        "valid_recall_at_k": valid_recall_at_k(retrieved_ids, query.gold_valid_memory_ids),
        "stale_retrieval_rate": stale_retrieval_rate(
            retrieved_ids,
            query.gold_stale_memory_ids,
        ),
        "average_token_cost": average_retrieved_token_cost(retrieved),
        "retrieval_latency_ms": retrieval_latency_ms,
    }
