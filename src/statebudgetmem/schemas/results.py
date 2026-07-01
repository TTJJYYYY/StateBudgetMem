from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from statebudgetmem.schemas.records import QueryType


class RetrievedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    score: float
    rank: int = Field(ge=1)
    token_cost: int = Field(ge=0)
    source_view: str = "flat"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method_name: str
    query_id: str
    retrieved_memories: list[RetrievedMemory]
    predicted_query_type: QueryType | None = None
    total_token_cost: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_retrieval_contract(self) -> "MethodResult":
        ranks = [item.rank for item in self.retrieved_memories]
        expected_ranks = list(range(1, len(self.retrieved_memories) + 1))
        if ranks != expected_ranks:
            raise ValueError("retrieved_memories ranks must start at 1 and be consecutive")

        expected_token_cost = sum(item.token_cost for item in self.retrieved_memories)
        if self.total_token_cost != expected_token_cost:
            raise ValueError("total_token_cost must equal the sum of retrieved memory token_cost")
        return self
