from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentConfig(BaseModel):
    """Frozen v0.2 configuration shared by comparable retrieval methods."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: Path
    results_dir: Path = Path("results/unified")
    methods: tuple[str, ...] = ("tfidf_topk",)
    top_k: int = Field(default=3, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    token_budget: int | None = Field(default=None, ge=0)
    random_seed: int = 42
    repeat: int = Field(default=1, ge=1)
    embedding_backend: str = "method_default"
    embedding_model: str = "method_default"
    forgetting_enabled: bool = True
    forgetting_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    exclude_forgotten: bool = False
    reinforcement_enabled: bool = False
    query_state_policy: Literal["independent", "sequential"] = "independent"
    token_counter_name: str = "memory_record_token_cost"
    config_path: Path | None = None

    @field_validator("methods")
    @classmethod
    def validate_methods(cls, methods: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(method.strip() for method in methods if method.strip())
        if not normalized:
            raise ValueError("methods must contain at least one method name")
        if len(set(normalized)) != len(normalized):
            raise ValueError("methods must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_candidate_k(self) -> "ExperimentConfig":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class MethodBuildContext(BaseModel):
    """Construction inputs shared by every method in one experiment run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    experiment: ExperimentConfig
    work_dir: Path


class RunMetadata(BaseModel):
    """Environment and provenance saved once for every unified run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime
    dataset_path: str
    dataset_sha256: str
    config_path: str | None = None
    git_commit: str | None = None
    dirty_worktree: bool | None = None
    python_version: str
    platform: str
    hardware: dict[str, Any] = Field(default_factory=dict)
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    embedding_backend: str
    embedding_model: str
    token_counter_name: str
    random_seed: int


class ResourceMetrics(BaseModel):
    """Method-independent resource fields; detailed profiling follows later."""

    model_config = ConfigDict(extra="forbid")

    ingest_latency_ms: float = Field(default=0.0, ge=0.0)
    retrieval_latency_ms: float = Field(default=0.0, ge=0.0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    storage_bytes: int | None = Field(default=None, ge=0)
    total_token_cost: int = Field(default=0, ge=0)
    repeat_index: int = Field(default=0, ge=0)
