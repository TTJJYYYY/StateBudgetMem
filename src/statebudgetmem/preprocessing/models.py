from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from statebudgetmem.preprocessing.normalizer import estimate_token_cost
from statebudgetmem.schemas import MemoryRecord, MemoryStatus


class OperationHint(str, Enum):
    """预处理给 versioning 的更新提示。

    注意：这只是 hint，不是最终版本更新结果。

    TODO: 和 versioning/views 小组合并时，需要最终确认：
    1. operation_hint 的枚举值是否够用；
    2. previous_value 是否足够表达旧状态；
    3. 这些字段是否需要移动到全项目公共 schema。
    """

    ADD = "ADD"
    MERGE = "MERGE"
    SUPERSEDE = "SUPERSEDE"
    TEMP_INVALIDATE = "TEMP_INVALIDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"
    UNKNOWN = "UNKNOWN"


class RawMemoryInput(BaseModel):
    """预处理模块的原始输入格式。

    不管原始数据来自聊天、笔记还是 benchmark，进入 preprocessing 前都先转成这个结构。
    """

    model_config = ConfigDict(extra="forbid")

    raw_id: str
    text: str
    observed_at: date
    subject: str = "user"
    source_type: str = "raw"
    speaker: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredMemory(BaseModel):
    """预处理后的结构化草稿。

    它还不是最终版本管理结果，只是从自然语言中抽出来的结构化事实。

    TODO: 这里是项目需要统一规定的结构化字段协议。
    目前先保留 attribute/value/previous_value/operation_hint/evidence_span。
    后续和 versioning/views 对接时，如果字段名要改，只应该在这里集中修改。
    """

    model_config = ConfigDict(extra="forbid")

    subject: str = "user"
    attribute: str
    value: str
    text: str
    event_time: date

    status: MemoryStatus = MemoryStatus.CURRENT
    memory_type: str = "profile"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    previous_value: str | None = None
    operation_hint: OperationHint = OperationHint.ADD
    evidence_span: str | None = None
    needs_review: bool = False
    source_raw_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_evidence(self) -> "StructuredMemory":
        if self.evidence_span is None:
            self.evidence_span = self.text
        return self

    def to_memory_record(self, memory_id: str) -> MemoryRecord:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "source_raw_id": self.source_raw_id,
                "previous_value": self.previous_value,
                "operation_hint": self.operation_hint.value,
                "evidence_span": self.evidence_span,
                "needs_review": self.needs_review,
            }
        )

        return MemoryRecord(
            memory_id=memory_id,
            subject=self.subject,
            attribute=self.attribute,
            value=self.value,
            text=self.text,
            event_time=self.event_time,
            valid_from=self.event_time if self.status == MemoryStatus.CURRENT else None,
            valid_to=None,
            status=self.status,
            memory_type=self.memory_type,
            importance=self.importance,
            confidence=self.confidence,
            token_cost=estimate_token_cost(self.text),
            metadata=metadata,
        )


class PreprocessConfig(BaseModel):
    """预处理配置。"""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = "S_PREPROCESSED"
    description: str = "Scenario generated from raw user text."

    parser_type: str = "hybrid"  # rule / api / hybrid
    api_model: str = "gpt-4o-mini"
    fallback_to_rule: bool = True

    keep_note_fallback: bool = True
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
