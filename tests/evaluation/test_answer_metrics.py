from __future__ import annotations

from statebudgetmem.evaluation import (
    AnswerEvaluationRecord,
    answer_accuracy,
    answer_accuracy_by_query_type,
    change_accuracy,
    current_state_accuracy,
    evaluate_answer_layer,
    historical_accuracy,
    query_type_accuracy,
    stale_usage_rate,
)
from statebudgetmem.schemas import QueryType


def test_answer_accuracy_counts_correct_answers() -> None:
    records = [
        AnswerEvaluationRecord("q1", QueryType.CURRENT, True),
        AnswerEvaluationRecord("q2", QueryType.CURRENT, False),
        AnswerEvaluationRecord("q3", QueryType.HISTORICAL, True),
    ]

    assert answer_accuracy(records) == 2 / 3


def test_empty_answer_metrics_return_zero() -> None:
    assert answer_accuracy([]) == 0.0
    assert stale_usage_rate([]) == 0.0
    assert current_state_accuracy([]) == 0.0
    assert historical_accuracy([]) == 0.0
    assert change_accuracy([]) == 0.0


def test_stale_usage_rate_can_be_inferred_from_used_memory_ids() -> None:
    records = [
        AnswerEvaluationRecord(
            "q1",
            QueryType.CURRENT,
            False,
            used_memory_ids=("m_old", "m_current"),
            gold_stale_memory_ids=("m_old",),
        ),
        AnswerEvaluationRecord(
            "q2",
            QueryType.CURRENT,
            True,
            used_memory_ids=("m_current",),
            gold_stale_memory_ids=("m_old",),
        ),
    ]

    assert stale_usage_rate(records) == 0.5


def test_stale_usage_rate_prefers_explicit_stale_used_label() -> None:
    records = [
        AnswerEvaluationRecord(
            "q1",
            QueryType.CURRENT,
            False,
            used_memory_ids=("m_current",),
            gold_stale_memory_ids=("m_old",),
            stale_used=True,
        ),
        AnswerEvaluationRecord(
            "q2",
            QueryType.CURRENT,
            True,
            used_memory_ids=("m_old",),
            gold_stale_memory_ids=("m_old",),
            stale_used=False,
        ),
    ]

    assert stale_usage_rate(records) == 0.5


def test_query_type_accuracy_splits_current_historical_and_change() -> None:
    records = [
        AnswerEvaluationRecord("q_current_ok", "current", True),
        AnswerEvaluationRecord("q_current_bad", "CURRENT", False),
        AnswerEvaluationRecord("q_history_ok", QueryType.HISTORICAL, True),
        AnswerEvaluationRecord("q_change_bad", QueryType.CHANGE, False),
        AnswerEvaluationRecord("q_change_ok", QueryType.CHANGE, True),
    ]

    assert current_state_accuracy(records) == 0.5
    assert historical_accuracy(records) == 1.0
    assert change_accuracy(records) == 0.5
    assert query_type_accuracy(records, "historical") == 1.0


def test_answer_accuracy_by_query_type_includes_all_query_types() -> None:
    records = [
        AnswerEvaluationRecord("q_current", QueryType.CURRENT, True),
        AnswerEvaluationRecord("q_general", QueryType.GENERAL, False),
    ]

    assert answer_accuracy_by_query_type(records) == {
        "current": 1.0,
        "historical": 0.0,
        "change": 0.0,
        "general": 0.0,
    }


def test_evaluate_answer_layer_returns_summary_dict() -> None:
    records = [
        AnswerEvaluationRecord(
            "q_current",
            QueryType.CURRENT,
            True,
            used_memory_ids=("m_current",),
            gold_stale_memory_ids=("m_old",),
        ),
        AnswerEvaluationRecord(
            "q_history",
            QueryType.HISTORICAL,
            False,
            used_memory_ids=("m_old",),
            gold_stale_memory_ids=("m_old",),
        ),
        AnswerEvaluationRecord("q_change", QueryType.CHANGE, True),
    ]

    assert evaluate_answer_layer(records) == {
        "answer_count": 3,
        "answer_accuracy": 2 / 3,
        "stale_usage_rate": 1 / 3,
        "current_state_accuracy": 1.0,
        "historical_accuracy": 0.0,
        "change_accuracy": 1.0,
        "general_accuracy": 0.0,
    }
