from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryStatus(str, Enum):
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    INVALIDATED = "INVALIDATED"
    UNKNOWN = "UNKNOWN"


class QueryType(str, Enum):
    """Canonical query type shared by schemas, routing, and evaluation.

    Enum values remain uppercase to preserve the controlled-dataset contract,
    while ``_missing_`` accepts lowercase/config spellings such as ``"current"``.
    """

    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    CHANGE = "CHANGE"
    GENERAL = "GENERAL"

    @classmethod
    def _missing_(cls, value: object) -> "QueryType | None":
        if isinstance(value, str):
            normalized = value.strip().upper()
            for member in cls:
                if member.value == normalized or member.name == normalized:
                    return member
        return None


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    subject: str
    attribute: str
    value: str
    text: str
    event_time: date
    valid_from: date | None = None
    valid_to: date | None = None
    status: MemoryStatus
    memory_type: str
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    token_cost: int = Field(ge=0)
    dimensions: dict[str, str] = Field(default_factory=dict)
    supersedes: list[str] = Field(default_factory=list)
    temporarily_invalidates: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_dimensions_are_strings(cls, dimensions: Any) -> Any:
        if dimensions is None:
            return dimensions
        if not isinstance(dimensions, dict):
            return dimensions
        for key, value in dimensions.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("dimensions keys and values must be strings")
        return dimensions

    @model_validator(mode="after")
    def validate_valid_range(self) -> "MemoryRecord":
        if self.valid_from is not None and self.valid_to is not None:
            if self.valid_to < self.valid_from:
                raise ValueError("valid_to must be on or after valid_from")
        return self


class QueryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    text: str
    query_type: QueryType
    reference_time: date
    gold_relevant_memory_ids: list[str] = Field(default_factory=list)
    gold_valid_memory_ids: list[str] = Field(default_factory=list)
    gold_stale_memory_ids: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    description: str
    memories: list[MemoryRecord]
    queries: list[QueryRecord]

    @field_validator("memories")
    @classmethod
    def validate_unique_memory_ids(cls, memories: list[MemoryRecord]) -> list[MemoryRecord]:
        ids = [memory.memory_id for memory in memories]
        duplicates = sorted({memory_id for memory_id in ids if ids.count(memory_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate memory_id values: {duplicates}")
        return memories

    @field_validator("queries")
    @classmethod
    def validate_unique_query_ids(cls, queries: list[QueryRecord]) -> list[QueryRecord]:
        ids = [query.query_id for query in queries]
        duplicates = sorted({query_id for query_id in ids if ids.count(query_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate query_id values: {duplicates}")
        return queries

    @model_validator(mode="after")
    def validate_query_gold_ids(self) -> "Scenario":
        memory_ids = {memory.memory_id for memory in self.memories}
        for query in self.queries:
            gold_ids = (
                query.gold_relevant_memory_ids
                + query.gold_valid_memory_ids
                + query.gold_stale_memory_ids
            )
            unknown = sorted({memory_id for memory_id in gold_ids if memory_id not in memory_ids})
            if unknown:
                raise ValueError(f"query {query.query_id} references unknown memory ids: {unknown}")
        return self


class RetrievedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: MemoryRecord
    score: float
    rank: int
