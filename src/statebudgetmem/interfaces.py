"""Public contracts shared by all StateBudgetMem modules.

There are two intentionally related interface layers:

1. ``MemoryPiece`` / ``MemorySystem`` for an online memory backend.
2. ``MemoryRecord`` / ``MemoryMethod`` for reproducible controlled experiments.

Version-management contracts are exported from the real versioning module so
that the public facade and ``VersioningEngine`` use the same operation enum and
API. All modules should import public objects from ``statebudgetmem.interfaces``
instead of defining private copies.
"""

from statebudgetmem.core.method import MemoryMethod
from statebudgetmem.core.online import (
    MemoryPiece,
    MemoryStatus,
    MemorySystem,
    MemoryType,
    QueryRouter,
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
from statebudgetmem.versioning.contracts import UpdateOperation, VersionManager

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
