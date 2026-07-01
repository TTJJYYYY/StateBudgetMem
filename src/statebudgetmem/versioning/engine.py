from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from datetime import date

from statebudgetmem.schemas import MemoryRecord
from statebudgetmem.versioning.adapters import MemoryAdapter, MemoryRecordAdapter
from statebudgetmem.versioning.classifier import (
    OperationClassifier,
    RuleBasedOperationClassifier,
)
from statebudgetmem.versioning.exceptions import (
    DuplicateObservationError,
    VersioningInvariantError,
)
from statebudgetmem.versioning.graph import VersionGraph
from statebudgetmem.versioning.matcher import StateMatcher, StructuredStateMatcher
from statebudgetmem.versioning.models import (
    BatchUpdateResult,
    StateKey,
    StateObservation,
    UpdateDecision,
    UpdateResult,
)
from statebudgetmem.versioning.operations import UpdateOperation
from statebudgetmem.versioning.resolver import VersionResolver
from statebudgetmem.versioning.updater import VersionUpdater
from statebudgetmem.versioning.validator import VersionGraphValidator


class VersioningEngine:
    """End-to-end deterministic state-version management service."""

    def __init__(
        self,
        *,
        adapter: MemoryAdapter | None = None,
        matcher: StateMatcher | None = None,
        classifier: OperationClassifier | None = None,
        updater: VersionUpdater | None = None,
        validator: VersionGraphValidator | None = None,
    ) -> None:
        self.adapter = adapter or MemoryRecordAdapter()
        self.matcher = matcher or StructuredStateMatcher()
        self.classifier = classifier or RuleBasedOperationClassifier()
        self.updater = updater or VersionUpdater()
        self.validator = validator or VersionGraphValidator()
        self.graph = VersionGraph()
        self._observations: dict[str, StateObservation] = {}

    @property
    def observations_by_id(self) -> dict[str, StateObservation]:
        return {
            memory_id: observation.model_copy(deep=True)
            for memory_id, observation in self._observations.items()
        }

    def reset(self) -> None:
        self.graph = VersionGraph()
        self._observations = {}

    def ingest(self, memory: MemoryRecord) -> BatchUpdateResult:
        results = [
            self.ingest_observation(observation)
            for observation in self.adapter.to_observations(memory)
        ]
        return BatchUpdateResult(results=results)

    def ingest_many(
        self,
        memories: Iterable[MemoryRecord],
        *,
        sort_by_event_time: bool = True,
    ) -> BatchUpdateResult:
        observations: list[StateObservation] = []
        for memory in memories:
            observations.extend(self.adapter.to_observations(memory))
        if sort_by_event_time:
            observations.sort(key=lambda item: (item.event_time, item.memory_id))
        return BatchUpdateResult(
            results=[self.ingest_observation(item) for item in observations]
        )

    def ingest_observation(self, observation: StateObservation) -> UpdateResult:
        existing = self._observations.get(observation.memory_id)
        if existing is not None:
            if existing == observation:
                decision = UpdateDecision(
                    new_memory_id=observation.memory_id,
                    operation=UpdateOperation.NOOP,
                    target_memory_ids=[observation.memory_id]
                    if observation.memory_id in self.graph
                    else [],
                    state_key=observation.state_key,
                    confidence=1.0,
                    reason="identical observation with the same memory_id was already ingested",
                    evidence=[observation.memory_id],
                    metadata={"idempotent_replay": True},
                )
                return UpdateResult(decision=decision, skipped=True)
            raise DuplicateObservationError(
                f"memory_id {observation.memory_id} was reused for different content"
            )

        candidates = self.matcher.match(
            observation,
            self.graph.nodes,
            self._observations,
        )
        decision = self.classifier.classify(
            observation,
            candidates,
            self.graph,
            self._observations,
        )

        candidate_graph = self.graph.clone()
        candidate_observations = {
            memory_id: item.model_copy(deep=True)
            for memory_id, item in self._observations.items()
        }
        candidate_observations[observation.memory_id] = observation.model_copy(deep=True)
        result = self.updater.apply(
            candidate_graph,
            observation,
            decision,
            candidate_observations,
        )
        report = self.validator.validate(candidate_graph, candidate_observations)
        if not report.is_valid:
            messages = "; ".join(issue.message for issue in report.errors)
            raise VersioningInvariantError(messages)

        self.graph = candidate_graph
        self._observations = candidate_observations
        return result

    def resolve_at(
        self,
        state_key: StateKey,
        reference_time: date | str,
    ):
        return self.resolver().resolve_at(state_key, reference_time)

    def resolve_current(
        self,
        state_key: StateKey,
        *,
        reference_time: date | str | None = None,
    ):
        return self.resolver().resolve_current(
            state_key,
            reference_time=reference_time,
        )

    def current_view(self, *, reference_time: date | str | None = None):
        return self.resolver().current_view(reference_time=reference_time)

    def history(self, state_key: StateKey):
        return self.resolver().history(state_key)

    def lineage(self, memory_id: str):
        return self.resolver().lineage(memory_id)

    def resolver(self) -> VersionResolver:
        return VersionResolver(self.graph, self._observations)

    def validate(self):
        return self.validator.validate(self.graph, self._observations)

    def snapshot(self) -> dict[str, object]:
        return {
            "graph": self.graph.model_dump(),
            "observations": {
                memory_id: observation.model_dump(mode="json")
                for memory_id, observation in sorted(self._observations.items())
            },
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> "VersioningEngine":
        engine = cls()
        raw_graph = payload.get("graph")
        raw_observations = payload.get("observations")
        if not isinstance(raw_graph, dict) or not isinstance(raw_observations, dict):
            raise ValueError("snapshot requires graph and observations mappings")
        engine.graph = VersionGraph.model_validate(raw_graph)
        engine._observations = {
            str(memory_id): StateObservation.model_validate(observation)
            for memory_id, observation in raw_observations.items()
        }
        report = engine.validate()
        if not report.is_valid:
            messages = "; ".join(issue.message for issue in report.errors)
            raise VersioningInvariantError(messages)
        return engine
