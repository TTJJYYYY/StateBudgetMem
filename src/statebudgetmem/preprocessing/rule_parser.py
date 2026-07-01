from __future__ import annotations

import re

from statebudgetmem.preprocessing.models import OperationHint, RawMemoryInput, StructuredMemory
from statebudgetmem.preprocessing.normalizer import (
    canonical_attribute,
    clean_value,
    infer_memory_type,
    normalize_text,
    split_clauses,
)
from statebudgetmem.schemas import MemoryStatus


class RuleBasedParser:
    """规则版信息抽取器。

    用于：
    1. 无 API key 时仍能跑；
    2. 做离线 baseline；
    3. API 失败时作为 fallback。
    """

    def __init__(self, keep_note_fallback: bool = True) -> None:
        self.keep_note_fallback = keep_note_fallback

    def parse(self, raw: RawMemoryInput) -> list[StructuredMemory]:
        text = normalize_text(raw.text)
        memories: list[StructuredMemory] = []

        state_change = self._parse_state_change(raw, text)
        if state_change is not None:
            memories.append(state_change)

        for clause in split_clauses(text):
            memory = self._parse_clause(raw, clause)
            if memory is not None and not _duplicated(memory, memories):
                memories.append(memory)

        if not memories and self.keep_note_fallback:
            memories.append(
                self._make_memory(
                    raw=raw,
                    text=text,
                    attribute="note",
                    value=text,
                    confidence=0.35,
                    operation_hint=OperationHint.ADD,
                    needs_review=True,
                )
            )

        return memories

    def _parse_state_change(self, raw: RawMemoryInput, text: str) -> StructuredMemory | None:
        location = re.search(
            r"(?:以前|之前|原来).{0,8}(?:住在|住|在)(?P<old>.+?)(?:,|，|但|但是|不过|现在|目前|后来).{0,8}(?:搬到|搬去|住在|住|到)(?P<new>.+)",
            text,
        )
        if location:
            return self._make_memory(
                raw=raw,
                text=text,
                attribute="home_location",
                value=clean_value(location.group("new")),
                previous_value=clean_value(location.group("old")),
                confidence=0.86,
                operation_hint=OperationHint.SUPERSEDE,
            )

        preference = re.search(
            r"(?:以前|之前|原来).{0,8}(?:喜欢|爱|喝|吃)(?P<old>.+?)(?:,|，|但|但是|不过|现在|目前|后来).{0,10}(?:改成|改为|换成|改喝|改吃|喜欢|喝|吃)(?P<new>.+)",
            text,
        )
        if preference:
            return self._make_memory(
                raw=raw,
                text=text,
                attribute="preference",
                value=f"like:{clean_value(preference.group('new'))}",
                previous_value=f"like:{clean_value(preference.group('old'))}",
                confidence=0.78,
                operation_hint=OperationHint.SUPERSEDE,
            )

        return None

    def _parse_clause(self, raw: RawMemoryInput, text: str) -> StructuredMemory | None:
        text = normalize_text(text)
        if not text:
            return None

        if any(marker in text for marker in ["别记", "不要记", "不用记", "忘掉", "删除"]):
            return self._make_memory(
                raw=raw,
                text=text,
                attribute="control",
                value=text,
                confidence=0.7,
                operation_hint=OperationHint.DELETE,
                needs_review=True,
            )

        allergy = re.search(r"(?:我|用户)?对(?P<value>.+?)过敏", text)
        if allergy:
            return self._make_memory(
                raw=raw,
                text=text,
                attribute="allergy",
                value=clean_value(allergy.group("value")),
                confidence=0.88,
            )

        home = re.search(r"(?:我|用户)?(?:现在|目前|最近)?(?:住在|住|搬到|搬去)(?P<value>.+)", text)
        if home:
            return self._make_memory(
                raw=raw,
                text=text,
                attribute="home_location",
                value=clean_value(home.group("value")),
                confidence=0.75,
            )

        meal = re.search(r"(?P<meal>早餐|早饭|午餐|午饭|晚餐|晚饭)(?:通常|一般|喜欢|吃|是)?(?P<value>.+)", text)
        if meal:
            return self._make_memory(
                raw=raw,
                text=text,
                attribute=canonical_attribute(meal.group("meal")),
                value=clean_value(meal.group("value")),
                confidence=0.68,
            )

        if any(marker in text for marker in ["不喜欢", "不吃", "不喝", "讨厌", "戒了", "避免"]):
            value = re.sub(r".*?(不喜欢|不吃|不喝|讨厌|戒了|避免)", "", text)
            return self._make_memory(
                raw=raw,
                text=text,
                attribute="preference",
                value=f"avoid:{clean_value(value)}",
                confidence=0.7,
            )

        like = re.search(r"(?:我|用户)?(?:很|挺|比较|特别)?(?:喜欢|爱|偏好)(?P<value>.+)", text)
        if like:
            return self._make_memory(
                raw=raw,
                text=text,
                attribute="preference",
                value=f"like:{clean_value(like.group('value'))}",
                confidence=0.7,
            )

        generic = re.search(r"(?:我的|用户的)?(?P<attr>[\u4e00-\u9fffA-Za-z0-9_]{1,12})(?:是|为|=)(?P<value>.+)", text)
        if generic:
            attribute = canonical_attribute(generic.group("attr"))
            return self._make_memory(
                raw=raw,
                text=text,
                attribute=attribute,
                value=clean_value(generic.group("value")),
                confidence=0.58,
            )

        if self.keep_note_fallback and len(text) >= 4:
            return self._make_memory(
                raw=raw,
                text=text,
                attribute="note",
                value=text,
                confidence=0.35,
                operation_hint=OperationHint.ADD,
                needs_review=True,
            )

        return None

    def _make_memory(
        self,
        raw: RawMemoryInput,
        text: str,
        attribute: str,
        value: str,
        confidence: float,
        *,
        previous_value: str | None = None,
        operation_hint: OperationHint | None = None,
        needs_review: bool = False,
    ) -> StructuredMemory:
        attribute = canonical_attribute(attribute)
        memory_type = infer_memory_type(attribute)

        if operation_hint is None:
            operation_hint = _guess_operation_hint(text, previous_value)

        return StructuredMemory(
            subject=raw.subject,
            attribute=attribute,
            value=clean_value(value),
            text=text,
            event_time=raw.observed_at,
            status=MemoryStatus.CURRENT,
            memory_type=memory_type,
            importance=_default_importance(memory_type, attribute),
            confidence=confidence,
            previous_value=previous_value,
            operation_hint=operation_hint,
            evidence_span=text,
            needs_review=needs_review or confidence < 0.45,
            source_raw_id=raw.raw_id,
            metadata={
                "source_type": raw.source_type,
                "speaker": raw.speaker,
                "parser": "rule",
                **raw.metadata,
            },
        )


def _guess_operation_hint(text: str, previous_value: str | None) -> OperationHint:
    if previous_value:
        return OperationHint.SUPERSEDE
    if any(marker in text for marker in ["暂时", "这周", "这几天", "临时"]):
        return OperationHint.TEMP_INVALIDATE
    if any(marker in text for marker in ["改成", "改为", "换成", "现在", "目前", "不再", "戒了"]):
        return OperationHint.SUPERSEDE
    if any(marker in text for marker in ["也", "另外", "同时", "而且"]):
        return OperationHint.MERGE
    return OperationHint.ADD


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


def _duplicated(memory: StructuredMemory, memories: list[StructuredMemory]) -> bool:
    return any(item.attribute == memory.attribute and item.value == memory.value for item in memories)
