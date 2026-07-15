"""MemoryBank baseline package.

Everything specific to the original MemoryBank comparison lives here:
backend implementations, lightweight agent wrappers, demo datasets, answer
comparison evaluation, stale-memory analysis, and the Gradio demo.

The package intentionally does not import the Gradio demo at module import time,
so the core project remains usable without the optional ``demo`` dependencies.
"""

from statebudgetmem.baselines.memorybank.agents import BaselineAgent, MemoryAugmentedAgent
from statebudgetmem.baselines.memorybank.datasets import (
    DEMO_HISTORY,
    DEMO_QUESTIONS,
    History,
    Probe,
    load_json_dataset,
    load_memora_data,
)
from statebudgetmem.baselines.memorybank.evaluator import (
    DeepSeekLLM,
    EvaluationResult,
    MemoryEvaluator,
    MockLLM,
    OpenAICompatibleLLM,
    print_summary,
    run_memora_batch,
    summarize_results,
)
from statebudgetmem.baselines.memorybank.metrics import (
    MemoryBankMetricSpec,
    contextual_coherence,
    evaluate_reproduction_row,
    memory_retrieval_accuracy,
    response_correctness,
    stale_retrieval_rate_by_keywords,
    summarize_metric_rows,
)
from statebudgetmem.baselines.memorybank.paper_storage import (
    DailySummary,
    DialogTurn,
    PaperStorageSpec,
    RetrievalProbe,
    build_paper_aligned_storage,
    default_paper_storage_spec,
    default_retrieval_probe,
    run_paper_retrieval_probe,
)
from statebudgetmem.baselines.memorybank.staleness import (
    ObsoleteDetector,
    calculate_omr,
    calculate_outdated_memory_rate,
    label_demo_memory,
    label_memora_with_evidence,
)
from statebudgetmem.baselines.memorybank.system import MemoryBank, TFIDFMemoryBank


def __getattr__(name: str) -> object:
    """Keep the dense adapter lazy so base imports do not require NumPy/FAISS."""
    if name == "MemoryBankMethod":
        from statebudgetmem.baselines.memorybank.adapter import MemoryBankMethod

        return MemoryBankMethod
    if name in {"StateBudgetMemDenseMethod", "StateBudgetMemMode"}:
        from statebudgetmem.baselines.memorybank.statebudgetmem_adapter import (
            StateBudgetMemDenseMethod,
            StateBudgetMemMode,
        )

        return {
            "StateBudgetMemDenseMethod": StateBudgetMemDenseMethod,
            "StateBudgetMemMode": StateBudgetMemMode,
        }[name]
    raise AttributeError(name)

__all__ = [
    "MemoryBank",
    "TFIDFMemoryBank",
    "MemoryBankMethod",
    "StateBudgetMemDenseMethod",
    "StateBudgetMemMode",
    "BaselineAgent",
    "MemoryAugmentedAgent",
    "History",
    "Probe",
    "DEMO_HISTORY",
    "DEMO_QUESTIONS",
    "load_json_dataset",
    "load_memora_data",
    "DeepSeekLLM",
    "EvaluationResult",
    "MemoryEvaluator",
    "MockLLM",
    "OpenAICompatibleLLM",
    "summarize_results",
    "print_summary",
    "run_memora_batch",
    "MemoryBankMetricSpec",
    "contextual_coherence",
    "evaluate_reproduction_row",
    "memory_retrieval_accuracy",
    "response_correctness",
    "stale_retrieval_rate_by_keywords",
    "summarize_metric_rows",
    "DailySummary",
    "DialogTurn",
    "PaperStorageSpec",
    "RetrievalProbe",
    "build_paper_aligned_storage",
    "default_paper_storage_spec",
    "default_retrieval_probe",
    "run_paper_retrieval_probe",
    "ObsoleteDetector",
    "calculate_omr",
    "calculate_outdated_memory_rate",
    "label_demo_memory",
    "label_memora_with_evidence",
]
