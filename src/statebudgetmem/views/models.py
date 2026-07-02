from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from statebudgetmem.schemas import QueryType


class ViewName(str, Enum):
    """Canonical view labels used in experiment outputs."""

    FLAT = "flat"
    CURRENT = "current"
    HISTORY = "history"
    DUAL = "dual"


class ViewPolicy(BaseModel):
    """Configuration for view-based retrieval.

    ``CURRENT`` and the current side of ``CHANGE`` are resolved at the query's
    ``reference_time`` by default. ``HISTORICAL`` is resolved as a point-in-time
    snapshot at the same reference time, while ``CHANGE`` may access the full
    version history.
    """

    model_config = ConfigDict(extra="forbid")

    current_as_of_latest: bool = False
    history_for_change_queries: bool = True
    expand_change_lineage: bool = True

    # Kept only for backwards-compatible configuration loading. Dual-view
    # ranking now uses one shared TF-IDF candidate space, so score offsets are
    # intentionally ignored.
    current_score_boost: float = Field(default=0.0, ge=0.0)
    history_score_penalty_for_current: float = Field(default=0.0, ge=0.0)


class CandidateMemory(BaseModel):
    """Memory plus the view that made it available to retrieval."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    source_view: ViewName
    metadata: dict[str, Any] = Field(default_factory=dict)


class ViewDecision(BaseModel):
    """Explain which view was selected for a query."""

    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    selected_views: list[ViewName]
    reason: str
