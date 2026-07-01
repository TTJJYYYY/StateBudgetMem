from __future__ import annotations

from collections.abc import Iterable

from statebudgetmem.preprocessing.api_parser import ApiParser
from statebudgetmem.preprocessing.models import PreprocessConfig, RawMemoryInput, StructuredMemory
from statebudgetmem.preprocessing.rule_parser import RuleBasedParser
from statebudgetmem.schemas import MemoryRecord, Scenario


class MemoryPreprocessor:
    """预处理主入口：RawMemoryInput -> Scenario/MemoryRecord。

    支持三种模式：
    - rule：只用规则解析；
    - api：只用外部 API；
    - hybrid：优先 API，失败后回退规则解析。
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.rule_parser = RuleBasedParser(keep_note_fallback=self.config.keep_note_fallback)
        self.api_parser = ApiParser(model=self.config.api_model)

    def preprocess_memories(self, raw_inputs: Iterable[RawMemoryInput]) -> list[MemoryRecord]:
        memories: list[MemoryRecord] = []
        counter = 1

        for raw in raw_inputs:
            for item in self._parse_one(raw):
                if item.confidence < self.config.min_confidence:
                    continue

                memory_id = f"{self.config.scenario_id}_M{counter:04d}"
                memories.append(item.to_memory_record(memory_id))
                counter += 1

        return memories

    def preprocess_scenario(self, raw_inputs: Iterable[RawMemoryInput]) -> Scenario:
        return Scenario(
            scenario_id=self.config.scenario_id,
            description=self.config.description,
            memories=self.preprocess_memories(raw_inputs),
            queries=[],
        )

    def _parse_one(self, raw: RawMemoryInput) -> list[StructuredMemory]:
        if self.config.parser_type == "rule":
            return self.rule_parser.parse(raw)

        if self.config.parser_type == "api":
            return self.api_parser.parse(raw)

        if self.config.parser_type == "hybrid":
            try:
                api_result = self.api_parser.parse(raw)
                if api_result:
                    return api_result
            except Exception as exc:
                if not self.config.fallback_to_rule:
                    raise exc

            return self.rule_parser.parse(raw)

        raise ValueError(f"unknown parser_type: {self.config.parser_type}")
