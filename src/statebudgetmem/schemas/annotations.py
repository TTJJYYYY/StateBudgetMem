from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    gold_status: str | None = None
    gold_operation: str | None = None
    gold_target_memory_ids: list[str] = Field(default_factory=list)
    gold_supersedes: list[str] = Field(default_factory=list)
    gold_temporarily_invalidates: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
