from __future__ import annotations

from statebudgetmem.baselines.memorybank.metrics import (
    MemoryBankMetricSpec,
    contextual_coherence,
    evaluate_reproduction_row,
    memory_retrieval_accuracy,
    response_correctness,
    stale_retrieval_rate_by_keywords,
    summarize_metric_rows,
)


def test_memorybank_paper_metric_proxies_are_deterministic() -> None:
    retrieved = [
        {
            "memory_id": "m1",
            "content": "The recommended Python book was Automate the Boring Stuff.",
        },
        {
            "memory_id": "m2",
            "content": "The user used to like spicy hotpot.",
        },
    ]

    assert memory_retrieval_accuracy(retrieved, ("python", "book")) == 1.0
    assert response_correctness("I recommended a Python book.", ("python", "book")) == 1.0
    assert stale_retrieval_rate_by_keywords(retrieved, ("hotpot",)) == 0.5
    assert contextual_coherence(
        "What Python book did you recommend?",
        "The Python book was retrieved from memory.",
        retrieved,
    ) == 1.0


def test_evaluate_reproduction_row_and_summary_include_paper_and_device_metrics() -> None:
    row = {
        "query_id": "q001",
        "query": "What Python book did you recommend?",
        "retrieved_memories": [
            {
                "memory_id": "m1",
                "content": "The recommended Python book was Automate the Boring Stuff.",
            }
        ],
        "template_answer": "The Python book was Automate the Boring Stuff.",
        "latency_ms": 2.5,
        "index_size": 7,
        "prompt_token_estimate": 42,
    }
    metrics = evaluate_reproduction_row(
        row,
        MemoryBankMetricSpec(
            query_id="q001",
            relevant_keywords=("python", "book"),
            answer_keywords=("python", "book"),
        ),
    )

    assert metrics["memory_retrieval_accuracy"] == 1.0
    assert metrics["response_correctness"] == 1.0
    assert metrics["retrieval_latency_ms"] == 2.5
    assert metrics["faiss_index_size"] == 7.0
    assert metrics["prompt_token_cost"] == 42.0

    summary = summarize_metric_rows([metrics])
    assert summary["memory_retrieval_accuracy"] == 1.0
    assert summary["mean_retrieval_latency_ms"] == 2.5
    assert summary["mean_faiss_index_size"] == 7.0
    assert summary["mean_prompt_token_cost"] == 42.0


def test_gold_context_summary_averages_only_applicable_rows() -> None:
    summary = summarize_metric_rows(
        [
            {
                "context_coverage": 1.0,
                "has_context_gold": True,
                "overall_context_coverage": 1.0,
                "has_any_gold": True,
            },
            {
                "context_coverage": 0.0,
                "has_context_gold": False,
                "overall_context_coverage": 0.0,
                "has_any_gold": False,
            },
        ]
    )

    assert summary["context_coverage"] == 1.0
    assert summary["context_gold_applicable_count"] == 1.0
    assert summary["overall_context_coverage"] == 1.0
    assert summary["overall_gold_applicable_count"] == 1.0
