from __future__ import annotations

import re

from statebudgetmem.preprocessing.compat import MemoryType, UpdateOperation
from statebudgetmem.preprocessing.models import ParsedMemory, RawMessage, parse_timestamp
from statebudgetmem.preprocessing.normalizer import (
    canonical_attribute,
    clean_value,
    normalize_text,
    split_clauses,
)


class RuleBasedParser:
    """规则版预处理器。

    作用：
    1. 无 API key 时仍能运行；
    2. 作为离线 baseline；
    3. hybrid 模式下作为 API 失败后的 fallback。
    """

    def __init__(self, keep_note_fallback: bool = True) -> None:
        self.keep_note_fallback = keep_note_fallback

    def parse(self, raw: RawMessage) -> list[ParsedMemory]:
        text = normalize_text(raw.content)
        timestamp = parse_timestamp(raw.timestamp)
        memories: list[ParsedMemory] = []

        state_change = self._parse_state_change(raw, text, timestamp)
        if state_change is not None:
            memories.append(state_change)

        for clause in split_clauses(text):
            memory = self._parse_clause(raw, clause, timestamp)
            if memory is not None and not _duplicated(memory, memories):
                memories.append(memory)

        if not memories and self.keep_note_fallback:
            memories.append(
                ParsedMemory(
                    content=text,
                    timestamp=timestamp,
                    memory_type=MemoryType.DIALOG,
                    operation=UpdateOperation.ADD,
                    attribute="note",
                    value=text,
                    evidence_span=text,
                    tags=["note"],
                    confidence=0.35,
                    source=raw.source,
                    needs_review=True,
                )
            )

        return memories

    def _parse_state_change(
        self,
        raw: RawMessage,
        text: str,
        timestamp: float,
    ) -> ParsedMemory | None:
        location = re.search(
            r"(?:以前|之前|原来).{0,8}(?:住在|住|在)(?P<old>.+?)(?:,|，|但|但是|不过|现在|目前|后来).{0,8}(?:搬到|搬去|住在|住|到)(?P<new>.+)",
            text,
        )
        if location:
            new_value = clean_value(location.group("new"))
            old_value = clean_value(location.group("old"))
            return ParsedMemory(
                content=f"用户当前居住地是{new_value}",
                timestamp=timestamp,
                memory_type=MemoryType.FACT,
                operation=UpdateOperation.SUPERSEDE,
                attribute="home_location",
                value=new_value,
                previous_value=old_value,
                evidence_span=text,
                tags=["profile", "location", "home_location"],
                confidence=0.86,
                source=raw.source,
            )

        preference = re.search(
            r"(?:以前|之前|原来).{0,8}(?:喜欢|爱|喝|吃)(?P<old>.+?)(?:,|，|但|但是|不过|现在|目前|后来).{0,10}(?:改成|改为|换成|改喝|改吃|喜欢|喝|吃)(?P<new>.+)",
            text,
        )
        if preference:
            new_value = clean_value(preference.group("new"))
            old_value = clean_value(preference.group("old"))
            return ParsedMemory(
                content=f"用户当前偏好是{new_value}",
                timestamp=timestamp,
                memory_type=MemoryType.PREFERENCE,
                operation=UpdateOperation.SUPERSEDE,
                attribute="preference",
                value=new_value,
                previous_value=old_value,
                evidence_span=text,
                tags=["preference"],
                confidence=0.78,
                source=raw.source,
            )

        return None

    def _parse_clause(
        self,
        raw: RawMessage,
        text: str,
        timestamp: float,
    ) -> ParsedMemory | None:
        text = normalize_text(text)
        if not text:
            return None

        if any(marker in text for marker in ["别记", "不要记", "不用记", "忘掉", "删除"]):
            return ParsedMemory(
                content=text,
                timestamp=timestamp,
                memory_type=MemoryType.FACT,
                operation=UpdateOperation.DELETE,
                attribute="control",
                value=text,
                evidence_span=text,
                tags=["control", "delete"],
                confidence=0.7,
                source=raw.source,
                needs_review=True,
            )

        allergy = re.search(r"(?:我|用户)?对(?P<value>.+?)过敏", text)
        if allergy:
            value = clean_value(allergy.group("value"))
            return ParsedMemory(
                content=f"用户对{value}过敏",
                timestamp=timestamp,
                memory_type=MemoryType.FACT,
                operation=UpdateOperation.ADD,
                attribute="allergy",
                value=value,
                evidence_span=text,
                tags=["health", "allergy"],
                confidence=0.88,
                source=raw.source,
            )

        home = re.search(r"(?:我|用户)?(?:现在|目前|最近)?(?:住在|住|搬到|搬去)(?P<value>.+)", text)
        if home:
            value = clean_value(home.group("value"))
            return ParsedMemory(
                content=f"用户当前居住地是{value}",
                timestamp=timestamp,
                memory_type=MemoryType.FACT,
                operation=_guess_operation(text),
                attribute="home_location",
                value=value,
                evidence_span=text,
                tags=["profile", "location", "home_location"],
                confidence=0.75,
                source=raw.source,
            )

        meal = re.search(
            r"(?P<meal>早餐|早饭|午餐|午饭|晚餐|晚饭)(?:通常|一般|喜欢|吃|是)?(?P<value>.+)",
            text,
        )
        if meal:
            attribute = canonical_attribute(meal.group("meal"))
            value = clean_value(meal.group("value"))
            return ParsedMemory(
                content=f"用户的{meal.group('meal')}习惯是{value}",
                timestamp=timestamp,
                memory_type=MemoryType.FACT,
                operation=UpdateOperation.ADD,
                attribute=attribute,
                value=value,
                evidence_span=text,
                tags=["habit", attribute],
                confidence=0.68,
                source=raw.source,
            )

        if any(marker in text for marker in ["不喜欢", "不吃", "不喝", "讨厌", "戒了", "避免"]):
            value = re.sub(r".*?(不喜欢|不吃|不喝|讨厌|戒了|避免)", "", text)
            value = clean_value(value)
            return ParsedMemory(
                content=f"用户避免或不喜欢{value}",
                timestamp=timestamp,
                memory_type=MemoryType.PREFERENCE,
                operation=_guess_operation(text),
                attribute="preference",
                value=f"avoid:{value}",
                evidence_span=text,
                tags=["preference", "negative_preference"],
                confidence=0.7,
                source=raw.source,
            )

        like = re.search(r"(?:我|用户)?(?:很|挺|比较|特别)?(?:喜欢|爱|偏好)(?P<value>.+)", text)
        if like:
            value = clean_value(like.group("value"))
            return ParsedMemory(
                content=f"用户喜欢{value}",
                timestamp=timestamp,
                memory_type=MemoryType.PREFERENCE,
                operation=UpdateOperation.ADD,
                attribute="preference",
                value=f"like:{value}",
                evidence_span=text,
                tags=["preference"],
                confidence=0.7,
                source=raw.source,
            )

        generic = re.search(
            r"(?:我的|用户的)?(?P<attr>[\u4e00-\u9fffA-Za-z0-9_]{1,12})(?:是|为|=)(?P<value>.+)",
            text,
        )
        if generic:
            attribute = canonical_attribute(generic.group("attr"))
            value = clean_value(generic.group("value"))
            return ParsedMemory(
                content=f"用户的{attribute}是{value}",
                timestamp=timestamp,
                memory_type=MemoryType.FACT,
                operation=UpdateOperation.ADD,
                attribute=attribute,
                value=value,
                evidence_span=text,
                tags=["fact", attribute],
                confidence=0.58,
                source=raw.source,
            )

        if self.keep_note_fallback and len(text) >= 4:
            return ParsedMemory(
                content=text,
                timestamp=timestamp,
                memory_type=MemoryType.DIALOG,
                operation=UpdateOperation.ADD,
                attribute="note",
                value=text,
                evidence_span=text,
                tags=["note"],
                confidence=0.35,
                source=raw.source,
                needs_review=True,
            )

        return None


def _guess_operation(text: str) -> UpdateOperation:
    if any(marker in text for marker in ["暂时", "这周", "这几天", "临时"]):
        return UpdateOperation.TEMP_INVALIDATE

    if any(marker in text for marker in ["改成", "改为", "换成", "现在", "目前", "不再", "戒了"]):
        return UpdateOperation.SUPERSEDE

    if any(marker in text for marker in ["也", "另外", "同时", "而且"]):
        return UpdateOperation.MERGE

    return UpdateOperation.ADD


def _duplicated(memory: ParsedMemory, memories: list[ParsedMemory]) -> bool:
    return any(
        item.attribute == memory.attribute and item.value == memory.value
        for item in memories
    )
