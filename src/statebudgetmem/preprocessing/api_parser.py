from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from statebudgetmem.preprocessing.models import OperationHint, RawMemoryInput, StructuredMemory
from statebudgetmem.preprocessing.normalizer import canonical_attribute, clean_value, infer_memory_type
from statebudgetmem.schemas import MemoryStatus


class ApiMemoryItem(BaseModel):
    attribute: str = Field(description="标准化属性名，例如 home_location, preference, allergy")
    value: str = Field(description="当前值")
    previous_value: str | None = Field(default=None, description="旧值，没有则为 null")
    operation_hint: OperationHint = Field(default=OperationHint.ADD)
    evidence_span: str = Field(description="原文中支持该记忆的片段")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    needs_review: bool = False


class ApiParseResult(BaseModel):
    memories: list[ApiMemoryItem]


class ApiParser:
    """外部 API 版信息预处理器。

    不在代码里写死 API key，而是从 OPENAI_API_KEY 环境变量读取。
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def parse(self, raw: RawMemoryInput) -> list[StructuredMemory]:
        result = self._call_openai(raw)
        memories: list[StructuredMemory] = []

        for item in result.memories:
            attribute = canonical_attribute(item.attribute)
            memory_type = infer_memory_type(attribute)

            memories.append(
                StructuredMemory(
                    subject=raw.subject,
                    attribute=attribute,
                    value=clean_value(item.value),
                    text=item.evidence_span,
                    event_time=raw.observed_at,
                    status=MemoryStatus.CURRENT,
                    memory_type=memory_type,
                    importance=_default_importance(memory_type, attribute),
                    confidence=item.confidence,
                    previous_value=clean_value(item.previous_value) if item.previous_value else None,
                    operation_hint=item.operation_hint,
                    evidence_span=item.evidence_span,
                    needs_review=item.needs_review or item.confidence < 0.45,
                    source_raw_id=raw.raw_id,
                    metadata={
                        "source_type": raw.source_type,
                        "speaker": raw.speaker,
                        "parser": "api",
                        **raw.metadata,
                    },
                )
            )

        return memories

    def _call_openai(self, raw: RawMemoryInput) -> ApiParseResult:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖。请先安装：pip install openai") from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("未检测到 OPENAI_API_KEY。请先设置环境变量，不要把 key 写进代码。")

        client = OpenAI()

        response = client.responses.create(
            model=self.model,
            input=_build_prompt(raw),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "memory_preprocessing_result",
                    "strict": True,
                    "schema": _json_schema(),
                }
            },
        )

        content = getattr(response, "output_text", None)
        if not content:
            content = _extract_text_from_response(response)

        try:
            return ApiParseResult.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RuntimeError(f"API 返回结果无法解析为结构化记忆：{content}") from exc


def _build_prompt(raw: RawMemoryInput) -> str:
    return f"""
你是 StateBudgetMem 项目的信息预处理模块。
任务：把用户自然语言记忆转换成结构化记忆。

请抽取用户当前状态、偏好、习惯、健康信息、位置、身份信息等。

特别注意：
1. “以前 A，现在 B” 应抽取 B，并把 A 放入 previous_value，operation_hint 为 SUPERSEDE。
2. “暂时/这周/这几天” 这类表达，operation_hint 可为 TEMP_INVALIDATE。
3. “别记/忘掉/删除” 这类表达，operation_hint 为 DELETE。
4. 无明显事实的信息不要强行抽取，可返回空列表。
5. attribute 尽量使用稳定字段名，例如 home_location, preference, allergy, breakfast, company。

原始输入：
raw_id: {raw.raw_id}
observed_at: {raw.observed_at}
text: {raw.text}
"""


def _json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "attribute": {"type": "string"},
                        "value": {"type": "string"},
                        "previous_value": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        },
                        "operation_hint": {
                            "type": "string",
                            "enum": [
                                "ADD",
                                "MERGE",
                                "SUPERSEDE",
                                "TEMP_INVALIDATE",
                                "DELETE",
                                "NOOP",
                                "UNKNOWN",
                            ],
                        },
                        "evidence_span": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "needs_review": {"type": "boolean"},
                    },
                    "required": [
                        "attribute",
                        "value",
                        "previous_value",
                        "operation_hint",
                        "evidence_span",
                        "confidence",
                        "needs_review",
                    ],
                },
            }
        },
        "required": ["memories"],
    }


def _extract_text_from_response(response: Any) -> str:
    """兼容不同 SDK 返回对象，尽量从 response.output 中取文本。"""
    try:
        output = response.output
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    return text
    except Exception:
        pass

    raise RuntimeError("无法从 API response 中提取文本结果。")


def _default_importance(memory_type: str, attribute: str) -> float:
    if attribute == "allergy":
        return 0.9
    if memory_type == "health":
        return 0.85
    if memory_type == "profile":
        return 0.7
    if memory_type == "preference":
        return 0.65
    if memory_type == "habit":
        return 0.6
    return 0.4
