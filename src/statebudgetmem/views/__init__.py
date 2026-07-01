"""
StateBudgetMem / Views Module

Manages two memory views:
- Current View: only currently active memories (status CURRENT/ACTIVE)
- History View: full version chains and historical states

Bridges the routing module (which determines which view to use) with the
versioning module (which resolves current vs historical state).
"""

from statebudgetmem.views.manager import MemoryViewManager

__all__ = ["MemoryViewManager"]
