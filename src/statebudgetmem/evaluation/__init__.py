"""Method-independent retrieval metrics.

Method-specific evaluation code belongs beside the method implementation. For
example, MemoryBank answer comparison and staleness analysis live under
``statebudgetmem.baselines.memorybank``.
"""

from statebudgetmem.evaluation.metrics import (
    average_retrieved_token_cost,
    evaluate_retrieval,
    recall_at_k,
    stale_retrieval_rate,
    valid_recall_at_k,
)

__all__ = [
    "average_retrieved_token_cost",
    "evaluate_retrieval",
    "recall_at_k",
    "stale_retrieval_rate",
    "valid_recall_at_k",
]
