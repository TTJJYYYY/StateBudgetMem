from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from statebudgetmem.versioning.operations import (
    ComputedStatus,
    UpdateOperation,
    VersionRelation,
)


class MatchType(str, Enum):
    EXACT_SLOT = "EXACT_SLOT"
    BROADER_SCOPE = "BROADER_SCOPE"
    NARROWER_SCOPE = "NARROWER_SCOPE"
    COMPATIBLE_SCOPE = "COMPATIBLE_SCOPE"


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class StateDimension(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: str

    @field_validator("name", "value", mode="before")
    @classmethod
    def strip_non_empty_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("state dimension fields must not be empty")
            return stripped
        return value


class StateKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str
    attribute: str
    dimensions: tuple[StateDimension, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def convert_legacy_condition(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        converted = dict(data)
        dimensions = converted.get("dimensions", ())
        if isinstance(dimensions, dict):
            dimension_items: list[Any] = [
                {"name": name, "value": value} for name, value in dimensions.items()
            ]
        elif dimensions is None:
            dimension_items = []
        else:
            dimension_items = list(dimensions)

        if "condition" in converted:
            condition = converted.pop("condition")
            if condition is not None:
                dimension_items.append({"name": "condition", "value": condition})

        converted["dimensions"] = dimension_items
        return converted

    @field_validator("subject", "attribute", mode="before")
    @classmethod
    def strip_non_empty_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("state key fields must not be empty")
            return stripped
        return value

    @model_validator(mode="after")
    def normalize_dimensions(self) -> "StateKey":
        names = [dimension.name for dimension in self.dimensions]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(f"duplicate state dimension names: {duplicate_names}")
        sorted_dimensions = tuple(
            sorted(self.dimensions, key=lambda dimension: (dimension.name, dimension.value))
        )
        object.__setattr__(self, "dimensions", sorted_dimensions)
        return self

    @classmethod
    def from_condition(cls, *, subject: str, attribute: str, condition: str) -> "StateKey":
        return cls(subject=subject, attribute=attribute, condition=condition)

    def dimension_map(self) -> dict[str, str]:
        return {dimension.name: dimension.value for dimension in self.dimensions}

    def __str__(self) -> str:
        if not self.dimensions:
            return f"{self.subject}:{self.attribute}"
        if len(self.dimensions) == 1 and self.dimensions[0].name == "condition":
            return f"{self.subject}:{self.attribute}[{self.dimensions[0].value}]"
        dimensions = ",".join(
            f"{dimension.name}={dimension.value}" for dimension in self.dimensions
        )
        return f"{self.subject}:{self.attribute}[{dimensions}]"


class StateObservation(BaseModel):
    """Observable, atomic state fact accepted by the versioning core."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    state_key: StateKey
    value: str
    text: str
    event_time: date
    valid_from: date | None = None
    valid_to: date | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_id", "value", "text", mode="before")
    @classmethod
    def strip_non_empty_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("state observation fields must not be empty")
            return stripped
        return value

    @model_validator(mode="after")
    def validate_valid_range(self) -> "StateObservation":
        if self.valid_from is not None and self.valid_to is not None:
            if self.valid_to < self.valid_from:
                raise ValueError("valid_to must be on or after valid_from")
        return self

    @property
    def effective_from(self) -> date:
        return self.valid_from or self.event_time


class MatchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_memory_id: str
    state_key: StateKey
    candidate_status: ComputedStatus
    candidate_event_time: date
    match_type: MatchType
    score: float = Field(ge=0.0, le=1.0)
    dimension_distance: int = Field(default=0, ge=0)
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_memory_id", mode="before")
    @classmethod
    def strip_non_empty_memory_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("candidate_memory_id must not be empty")
            return stripped
        return value


class UpdateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_memory_id: str
    operation: UpdateOperation
    target_memory_ids: list[str] = Field(default_factory=list)
    state_key: StateKey
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: list[str] = Field(default_factory=list)
    requires_review: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("new_memory_id", "reason", mode="before")
    @classmethod
    def strip_required_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("decision fields must not be empty")
            return stripped
        return value

    @field_validator("target_memory_ids")
    @classmethod
    def validate_unique_targets(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("target_memory_ids must be unique")
        return values


class VersionNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    state_key: StateKey
    computed_status: ComputedStatus
    valid_from: date | None = None
    valid_to: date | None = None
    invalidated_by: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_valid_range(self) -> "VersionNode":
        if self.valid_from is not None and self.valid_to is not None:
            if self.valid_to < self.valid_from:
                raise ValueError("valid_to must be on or after valid_from")
        if len(self.invalidated_by) != len(set(self.invalidated_by)):
            raise ValueError("invalidated_by must contain unique memory IDs")
        return self


class VersionEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predecessor_id: str
    successor_id: str
    relation: VersionRelation
    effective_from: date
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_distinct_endpoints(self) -> "VersionEdge":
        if self.predecessor_id == self.successor_id:
            raise ValueError("predecessor_id and successor_id must be different")
        return self


class UpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: UpdateDecision
    created_node_ids: list[str] = Field(default_factory=list)
    updated_node_ids: list[str] = Field(default_factory=list)
    created_edges: list[VersionEdge] = Field(default_factory=list)
    skipped: bool = False


class BatchUpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[UpdateResult] = Field(default_factory=list)

    @property
    def processed_memory_ids(self) -> tuple[str, ...]:
        return tuple(result.decision.new_memory_id for result in self.results)


class ResolvedState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    state_key: StateKey
    value: str
    effective_from: date
    effective_to: date | None = None
    computed_status: ComputedStatus
    is_conflict: bool = False
    reasons: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: ValidationSeverity
    code: str
    message: str
    memory_ids: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is ValidationSeverity.WARNING)
