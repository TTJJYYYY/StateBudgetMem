from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

from statebudgetmem.interfaces import MemoryPiece
from statebudgetmem.preprocessing.api_parser import ApiParser
from statebudgetmem.preprocessing.models import (
    ParsedMemory,
    PreprocessConfig,
    RawMessage,
    messages_to_raw_messages,
)
from statebudgetmem.preprocessing.rule_parser import RuleBasedParser
from statebudgetmem.schemas import MemoryRecord


class MemoryPreprocessor:
    """预处理主入口。

    输入：
    - RawMessage
    - 或主接口使用的 messages: [(role, content, timestamp), ...]

    输出：
    - ParsedMemory: 带 operation / previous_value / evidence_span
    - MemoryPiece: 对齐当前 main 分支统一接口
    - MemoryRecord: 对齐 versioning / views / evaluation 实验层接口
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.rule_parser = RuleBasedParser(keep_note_fallback=self.config.keep_note_fallback)
        self.api_parser = ApiParser(model=self.config.api_model)

    def parse_raw_messages(self, raw_messages: Iterable[RawMessage]) -> list[ParsedMemory]:
        results: list[ParsedMemory] = []

        for raw in raw_messages:
            parsed_items = self._parse_one(raw)
            for item in parsed_items:
                if item.confidence >= self.config.min_confidence:
                    results.append(item)

        return results

    def parse_messages(
        self,
        messages: Iterable[Tuple[str, str, str | float | int]],
    ) -> list[ParsedMemory]:
        raw_messages = messages_to_raw_messages(messages)
        return self.parse_raw_messages(raw_messages)

    def to_memory_pieces(self, parsed_memories: Iterable[ParsedMemory]) -> list[MemoryPiece]:
        return [item.to_memory_piece() for item in parsed_memories]

    def preprocess_raw_messages(self, raw_messages: Iterable[RawMessage]) -> list[MemoryPiece]:
        return self.to_memory_pieces(self.parse_raw_messages(raw_messages))

    def preprocess_messages(
        self,
        messages: Iterable[Tuple[str, str, str | float | int]],
    ) -> list[MemoryPiece]:
        return self.to_memory_pieces(self.parse_messages(messages))

    def parse_raw_messages_to_records(
        self,
        raw_messages: Iterable[RawMessage],
        *,
        subject: str = "user",
    ) -> list[MemoryRecord]:
        """Parse RawMessage objects and return experiment-layer MemoryRecord objects.

        This method is the recommended bridge from preprocessing to
        versioning/views/evaluation. It keeps ParsedMemory as the intermediate
        result so structured fields are not lost.
        """

        from statebudgetmem.preprocessing.record_adapter import parsed_memories_to_records

        parsed = self.parse_raw_messages(raw_messages)
        return parsed_memories_to_records(parsed, subject=subject)

    def parse_messages_to_records(
        self,
        messages: Iterable[Tuple[str, str, str | float | int]],
        *,
        subject: str = "user",
    ) -> list[MemoryRecord]:
        """Parse message tuples and return experiment-layer MemoryRecord objects."""

        raw_messages = messages_to_raw_messages(messages)
        return self.parse_raw_messages_to_records(raw_messages, subject=subject)

    def _parse_one(self, raw: RawMessage) -> list[ParsedMemory]:
        if self.config.parser_type == "rule":
            return self.rule_parser.parse(raw)

        if self.config.parser_type == "api":
            return self.api_parser.parse(raw)

        if self.config.parser_type == "hybrid":
            try:
                api_result = self.api_parser.parse(raw)
                if api_result:
                    return api_result
            except Exception:
                if not self.config.fallback_to_rule:
                    raise

            return self.rule_parser.parse(raw)

        raise ValueError(f"unknown parser_type: {self.config.parser_type}")
