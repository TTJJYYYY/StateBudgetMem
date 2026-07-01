"""Shared contracts used across StateBudgetMem modules.

Prefer importing from ``statebudgetmem.interfaces``. This module keeps the
existing convenience exports while pointing version-management names at the
canonical versioning implementation.
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
from statebudgetmem.schemas import QueryType
from statebudgetmem.versioning.contracts import UpdateOperation, VersionManager

__all__ = [
    "MemoryMethod",
    "QueryType",
    "MemoryPiece",
    "MemoryStatus",
    "MemorySystem",
    "MemoryType",
    "QueryRouter",
    "UpdateOperation",
    "VersionManager",
    "ViewManager",
    "ViewType",
    "filter_memories",
    "messages_to_memory_pieces",
]
