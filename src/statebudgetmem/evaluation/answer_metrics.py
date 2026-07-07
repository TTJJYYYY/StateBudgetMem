from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from statebudgetmem.schemas import QueryType


@dataclass(frozen=True)
class AnswerEvaluationRecord:
    """One judged answer used by answer-level evaluation.

    ``is_correct`` should come from a deterministic label, keyword scorer, or
    judge output owned by the caller. ``stale_used`` can be provided directly
    when a judge labels stale usage; otherwise it is inferred from overlap
    between ``used_memory_ids`` and ``gold_stale_memory_ids``.
    """

    query_id: str
    query_type: QueryType | str
    is_correct: bool
    used_memory_ids: tuple[str, ...] = ()
    gold_stale_memory_ids: tuple[str, ...] = ()
    stale_used: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_query_type(self) -> QueryType:
        return QueryType(self.query_type)

    def used_stale_memory(self) -> bool:
        if self.stale_used is not None:
            return self.stale_used
        return bool(set(self.used_memory_ids) & set(self.gold_stale_memory_ids))


def answer_accuracy(records: Iterable[AnswerEvaluationRecord]) -> float:
    selected = list(records)
    if not selected:
        return 0.0
    return sum(record.is_correct for record in selected) / len(selected)


def stale_usage_rate(records: Iterable[AnswerEvaluationRecord]) -> float:
    selected = list(records)
    if not selected:
        return 0.0
    return sum(record.used_stale_memory() for record in selected) / len(selected)


def query_type_accuracy(
    records: Iterable[AnswerEvaluationRecord],
    query_type: QueryType | str,
) -> float:
    target = QueryType(query_type)
    selected = [
        record for record in records if record.normalized_query_type() is target
    ]
    return answer_accuracy(selected)


def current_state_accuracy(records: Iterable[AnswerEvaluationRecord]) -> float:
    return query_type_accuracy(records, QueryType.CURRENT)


def historical_accuracy(records: Iterable[AnswerEvaluationRecord]) -> float:
    return query_type_accuracy(records, QueryType.HISTORICAL)


def change_accuracy(records: Iterable[AnswerEvaluationRecord]) -> float:
    return query_type_accuracy(records, QueryType.CHANGE)


def answer_accuracy_by_query_type(
    records: Iterable[AnswerEvaluationRecord],
) -> dict[str, float]:
    selected = list(records)
    return {
        query_type.value.lower(): query_type_accuracy(selected, query_type)
        for query_type in QueryType
    }


def evaluate_answer_layer(
    records: Iterable[AnswerEvaluationRecord],
) -> dict[str, float | int]:
    selected = list(records)
    by_type = answer_accuracy_by_query_type(selected)
    return {
        "answer_count": len(selected),
        "answer_accuracy": answer_accuracy(selected),
        "stale_usage_rate": stale_usage_rate(selected),
        "current_state_accuracy": by_type["current"],
        "historical_accuracy": by_type["historical"],
        "change_accuracy": by_type["change"],
        "general_accuracy": by_type["general"],
    }


__all__ = [
    "AnswerEvaluationRecord",
    "answer_accuracy",
    "answer_accuracy_by_query_type",
    "change_accuracy",
    "current_state_accuracy",
    "evaluate_answer_layer",
    "historical_accuracy",
    "query_type_accuracy",
    "stale_usage_rate",
]
