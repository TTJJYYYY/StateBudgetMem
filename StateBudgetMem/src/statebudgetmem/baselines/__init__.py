"""Reference baselines used by StateBudgetMem experiments.

Baselines are grouped by method so that each contributor can work inside one
self-contained subpackage:

- :mod:`statebudgetmem.baselines.tfidf`
- :mod:`statebudgetmem.baselines.memorybank`
"""

from statebudgetmem.baselines.tfidf import (
    BaselineConfig,
    TfidfCosineRetriever,
    TfidfMemoryMethod,
    run_baseline,
)

__all__ = [
    "TfidfMemoryMethod",
    "TfidfCosineRetriever",
    "BaselineConfig",
    "run_baseline",
    "MemoryBank",
    "TFIDFMemoryBank",
    "BaselineAgent",
    "MemoryAugmentedAgent",
]


def __getattr__(name: str):
    if name in {
        "MemoryBank",
        "TFIDFMemoryBank",
        "BaselineAgent",
        "MemoryAugmentedAgent",
    }:
        from statebudgetmem.baselines import memorybank

        return getattr(memorybank, name)
    raise AttributeError(name)
