"""Public contracts shared by all StateBudgetMem modules.

There are two intentionally related interface layers:

1. ``MemoryPiece`` / ``MemorySystem`` for an online memory backend.
2. ``MemoryMethod`` / ``MethodResult`` for reproducible controlled experiments.

All modules should import these objects from ``statebudgetmem.interfaces`` or the
corresponding schema modules instead of defining private copies.
"""

from statebudgetmem.core.method import MemoryMethod
from statebudgetmem.core.online import (
    MemoryPiece,
    MemoryStatus,
    MemorySystem,
    MemoryType,
    QueryRouter,
    UpdateOperation,
    VersionManager,
    ViewManager,
    ViewType,
    filter_memories,
    messages_to_memory_pieces,
)
from statebudgetmem.schemas import (
    MemoryAnnotation,
    MemoryRecord,
    QueryRecord,
    QueryType,
    RetrievedMemory,
    Scenario,
)
from statebudgetmem.schemas.results import MethodResult

__all__ = [
    "MemoryMethod",
    "MethodResult",
    "MemoryPiece",
    "MemorySystem",
    "MemoryType",
    "MemoryStatus",
    "UpdateOperation",
    "VersionManager",
    "ViewManager",
    "QueryRouter",
    "ViewType",
    "MemoryAnnotation",
    "MemoryRecord",
    "QueryRecord",
    "QueryType",
    "RetrievedMemory",
    "Scenario",
    "filter_memories",
    "messages_to_memory_pieces",
]
