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

    current_as_of_latest=True 表示 Current View 始终代表“当前最新状态”。
    这符合本项目的实验设定：Current View 只保留当前有效状态，
    因此无法回答完整历史问题。
    """

    model_config = ConfigDict(extra="forbid")

    current_as_of_latest: bool = True
    history_for_change_queries: bool = True
    expand_change_lineage: bool = True
    current_score_boost: float = Field(default=0.02, ge=0.0)
    history_score_penalty_for_current: float = Field(default=0.15, ge=0.0)


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
