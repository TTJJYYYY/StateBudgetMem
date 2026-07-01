from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
import unicodedata

from statebudgetmem.versioning.graph import VersionGraph
from statebudgetmem.versioning.models import (
    MatchCandidate,
    MatchType,
    StateObservation,
    UpdateDecision,
)
from statebudgetmem.versioning.operations import (
    ComputedStatus,
    UpdateOperation,
    VersionRelation,
)


@dataclass(frozen=True)
class RuleClassifierPolicy:
    """Observable metadata contract for deterministic operation classification."""

    intent_key: str = "versioning_intent"
    target_ids_key: str = "versioning_target_ids"
    temporary_key: str = "temporary"
    temporal_type_key: str = "temporal_type"
    restore_signal_key: str = "restore_signal"
    delete_request_key: str = "delete_request"
    merge_request_key: str = "merge_request"
    delete_scope_key: str = "versioning_delete_scope"


class OperationClassifier(Protocol):
    def classify(
        self,
        new_observation: StateObservation,
        candidates: Sequence[MatchCandidate],
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation],
    ) -> UpdateDecision:
        ...


class RuleBasedOperationClassifier:
    """Conservative classifier over already structured state observations.

    The classifier never reads public-schema gold fields. Optional upstream
    semantic hints live under explicitly named metadata keys in
    ``RuleClassifierPolicy``. Without hints, it uses value equality, bounded
    validity, exact-slot candidates, and existing graph relations.
    """

    def __init__(self, policy: RuleClassifierPolicy | None = None) -> None:
        self.policy = policy or RuleClassifierPolicy()

    def classify(
        self,
        new_observation: StateObservation,
        candidates: Sequence[MatchCandidate],
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation],
    ) -> UpdateDecision:
        explicit_intent = self._explicit_intent(new_observation.metadata)
        explicit_targets = self._explicit_targets(new_observation.metadata)

        if explicit_intent is UpdateOperation.DELETE or self._bool_hint(
            new_observation.metadata, self.policy.delete_request_key
        ):
            return self._classify_delete(
                new_observation,
                candidates,
                explicit_targets,
            )

        restore = self._find_restore_context(
            new_observation,
            candidates,
            graph,
            observations_by_id,
        )
        if (
            explicit_intent is UpdateOperation.RESTORE
            or self._bool_hint(new_observation.metadata, self.policy.restore_signal_key)
            or restore is not None
        ):
            return self._classify_restore(
                new_observation,
                candidates,
                graph,
                observations_by_id,
                explicit_targets,
                restore,
            )

        if explicit_intent is UpdateOperation.NOOP:
            return self._decision(
                new_observation,
                UpdateOperation.NOOP,
                explicit_targets,
                0.99,
                "explicit observable intent requested no state change",
                requires_review=False,
            )

        active_exact = self._active_candidates(
            candidates,
            graph,
            observations_by_id,
            new_observation,
            allowed_match_types={MatchType.EXACT_SLOT},
        )
        same_value = [
            candidate
            for candidate in active_exact
            if self._values_equal(
                observations_by_id[candidate.candidate_memory_id].value,
                new_observation.value,
            )
        ]

        if explicit_intent is UpdateOperation.MERGE or self._bool_hint(
            new_observation.metadata, self.policy.merge_request_key
        ):
            targets = explicit_targets or self._candidate_ids(active_exact[:1])
            if not targets:
                return self._decision(
                    new_observation,
                    UpdateOperation.ADD,
                    [],
                    0.80,
                    "merge was requested but no exact active state exists; add a new state",
                    requires_review=True,
                )
            return self._decision(
                new_observation,
                UpdateOperation.MERGE,
                targets,
                0.97 if explicit_intent is UpdateOperation.MERGE else 0.94,
                "observable merge intent targets the closest exact active state",
            )

        is_temporary = self._is_temporary(new_observation)
        if explicit_intent is UpdateOperation.TEMP_INVALIDATE or is_temporary:
            temporary_targets = explicit_targets or self._candidate_ids(
                self._active_candidates(
                    candidates,
                    graph,
                    observations_by_id,
                    new_observation,
                    allowed_match_types={
                        MatchType.EXACT_SLOT,
                        MatchType.BROADER_SCOPE,
                    },
                )
            )
            if temporary_targets:
                return self._decision(
                    new_observation,
                    UpdateOperation.TEMP_INVALIDATE,
                    temporary_targets,
                    0.98 if explicit_intent is UpdateOperation.TEMP_INVALIDATE else 0.90,
                    "bounded or explicitly temporary state overrides active exact/broader states",
                )
            return self._decision(
                new_observation,
                UpdateOperation.ADD,
                [],
                0.88,
                "temporary state has no active state to invalidate; add it as an independent state",
            )

        if explicit_intent is UpdateOperation.ADD:
            return self._decision(
                new_observation,
                UpdateOperation.ADD,
                [],
                0.99,
                "explicit observable intent requested a new independent state",
            )

        if same_value:
            return self._decision(
                new_observation,
                UpdateOperation.NOOP,
                [same_value[0].candidate_memory_id],
                0.99,
                "incoming value is equivalent to the closest active exact-slot value",
            )

        if explicit_intent is UpdateOperation.SUPERSEDE:
            targets = explicit_targets or self._candidate_ids(active_exact[:1])
            if not targets:
                return self._decision(
                    new_observation,
                    UpdateOperation.ADD,
                    [],
                    0.75,
                    "supersede was requested but no exact active target exists",
                    requires_review=True,
                )
            return self._decision(
                new_observation,
                UpdateOperation.SUPERSEDE,
                targets,
                0.98,
                "explicit observable intent permanently replaces the selected state",
            )

        if active_exact:
            targets = self._candidate_ids(active_exact[:1])
            return self._decision(
                new_observation,
                UpdateOperation.SUPERSEDE,
                targets,
                0.92,
                "incoming value differs from the closest active exact-slot state",
                requires_review=len(active_exact) > 1,
                metadata={"alternative_active_targets": self._candidate_ids(active_exact[1:])},
            )

        if not candidates:
            return self._decision(
                new_observation,
                UpdateOperation.ADD,
                [],
                0.99,
                "no earlier state shares the subject and attribute",
            )

        return self._decision(
            new_observation,
            UpdateOperation.ADD,
            [],
            0.80,
            "related states exist only in different scopes; preserve them and add a scoped state",
            requires_review=any(
                candidate.match_type is MatchType.NARROWER_SCOPE for candidate in candidates
            ),
            metadata={"related_candidate_ids": self._candidate_ids(candidates)},
        )

    def _classify_delete(
        self,
        new_observation: StateObservation,
        candidates: Sequence[MatchCandidate],
        explicit_targets: list[str],
    ) -> UpdateDecision:
        if explicit_targets:
            targets = explicit_targets
        else:
            scope = str(
                new_observation.metadata.get(self.policy.delete_scope_key, "exact")
            ).strip().casefold()
            if scope == "attribute":
                targets = self._candidate_ids(candidates)
            else:
                targets = self._candidate_ids(
                    candidate
                    for candidate in candidates
                    if candidate.match_type is MatchType.EXACT_SLOT
                )
        if not targets:
            return self._decision(
                new_observation,
                UpdateOperation.NOOP,
                [],
                0.95,
                "delete request found no matching version node",
            )
        return self._decision(
            new_observation,
            UpdateOperation.DELETE,
            targets,
            0.99,
            "explicit observable delete request targets matching state nodes",
        )

    def _classify_restore(
        self,
        new_observation: StateObservation,
        candidates: Sequence[MatchCandidate],
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation],
        explicit_targets: list[str],
        inferred_context: tuple[list[str], list[str]] | None,
    ) -> UpdateDecision:
        if inferred_context is not None:
            restore_source_ids, temporary_ids = inferred_context
        else:
            restore_source_ids, temporary_ids = self._restore_targets_from_candidates(
                new_observation,
                candidates,
                graph,
                observations_by_id,
            )
        if explicit_targets:
            known_sources = [
                item
                for item in explicit_targets
                if graph.get_node(item).computed_status
                in {ComputedStatus.TEMP_INVALIDATED, ComputedStatus.HISTORICAL}
            ]
            known_temporaries = [
                item
                for item in explicit_targets
                if graph.get_node(item).computed_status
                in {ComputedStatus.CURRENT, ComputedStatus.UNKNOWN}
            ]
            restore_source_ids = known_sources or restore_source_ids
            temporary_ids = known_temporaries or temporary_ids

        targets = [*temporary_ids, *restore_source_ids]
        targets = list(dict.fromkeys(targets))
        if not restore_source_ids:
            return self._decision(
                new_observation,
                UpdateOperation.ADD,
                [],
                0.65,
                "restore signal exists but no recoverable prior state was found",
                requires_review=True,
            )
        return self._decision(
            new_observation,
            UpdateOperation.RESTORE,
            targets,
            0.96 if inferred_context is not None else 0.88,
            "incoming state restores a previously temporarily invalidated value",
            requires_review=not temporary_ids,
            metadata={
                "restore_source_ids": restore_source_ids,
                "temporary_target_ids": temporary_ids,
            },
        )

    def _find_restore_context(
        self,
        new_observation: StateObservation,
        candidates: Sequence[MatchCandidate],
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation],
    ) -> tuple[list[str], list[str]] | None:
        candidate_ids = {candidate.candidate_memory_id for candidate in candidates}
        for edge in graph.edges:
            if edge.relation is not VersionRelation.TEMP_INVALIDATES:
                continue
            if edge.predecessor_id not in candidate_ids:
                continue
            if edge.successor_id not in candidate_ids:
                continue
            source_observation = observations_by_id.get(edge.predecessor_id)
            temporary_observation = observations_by_id.get(edge.successor_id)
            if source_observation is None or temporary_observation is None:
                continue
            if not self._values_equal(source_observation.value, new_observation.value):
                continue
            if self._values_equal(temporary_observation.value, new_observation.value):
                continue
            if not self._observation_active(temporary_observation, new_observation.event_time):
                continue
            return ([edge.predecessor_id], [edge.successor_id])
        return None

    def _restore_targets_from_candidates(
        self,
        new_observation: StateObservation,
        candidates: Sequence[MatchCandidate],
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation],
    ) -> tuple[list[str], list[str]]:
        source_ids: list[str] = []
        temporary_ids: list[str] = []
        for candidate in candidates:
            candidate_id = candidate.candidate_memory_id
            observation = observations_by_id[candidate_id]
            node = graph.get_node(candidate_id)
            if (
                node.computed_status is ComputedStatus.TEMP_INVALIDATED
                and self._values_equal(observation.value, new_observation.value)
            ):
                source_ids.append(candidate_id)
                temporary_ids.extend(node.invalidated_by)
        return (list(dict.fromkeys(source_ids)), list(dict.fromkeys(temporary_ids)))

    def _active_candidates(
        self,
        candidates: Sequence[MatchCandidate],
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation],
        incoming: StateObservation,
        *,
        allowed_match_types: set[MatchType],
    ) -> list[MatchCandidate]:
        return [
            candidate
            for candidate in candidates
            if candidate.match_type in allowed_match_types
            and self._candidate_effectively_active(
                candidate,
                graph,
                observations_by_id,
                incoming.event_time,
            )
        ]

    def _candidate_effectively_active(
        self,
        candidate: MatchCandidate,
        graph: VersionGraph,
        observations_by_id: Mapping[str, StateObservation],
        at: Any,
    ) -> bool:
        if candidate.candidate_status in {
            ComputedStatus.DELETED,
            ComputedStatus.HISTORICAL,
        }:
            return False
        observation = observations_by_id[candidate.candidate_memory_id]
        if not self._observation_active(observation, at):
            return False
        if candidate.candidate_status in {
            ComputedStatus.CURRENT,
            ComputedStatus.UNKNOWN,
        }:
            return True
        if candidate.candidate_status is ComputedStatus.TEMP_INVALIDATED:
            for edge in graph.outgoing_edges(
                candidate.candidate_memory_id,
                VersionRelation.TEMP_INVALIDATES,
            ):
                invalidator_observation = observations_by_id.get(edge.successor_id)
                if invalidator_observation is None:
                    continue
                if self._observation_active(invalidator_observation, at):
                    return False
            return True
        return False

    @staticmethod
    def _observation_active(observation: StateObservation, at: Any) -> bool:
        if at < observation.effective_from:
            return False
        return observation.valid_to is None or at < observation.valid_to

    def _explicit_intent(self, metadata: Mapping[str, Any]) -> UpdateOperation | None:
        raw = metadata.get(self.policy.intent_key)
        if raw is None:
            return None
        try:
            return UpdateOperation(str(raw).strip().upper())
        except ValueError:
            return None

    def _explicit_targets(self, metadata: Mapping[str, Any]) -> list[str]:
        raw = metadata.get(self.policy.target_ids_key)
        if raw is None:
            return []
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, Sequence):
            values = [str(item) for item in raw]
        else:
            return []
        return list(dict.fromkeys(item.strip() for item in values if item.strip()))

    def _is_temporary(self, observation: StateObservation) -> bool:
        if observation.valid_to is not None:
            return True
        if self._bool_hint(observation.metadata, self.policy.temporary_key):
            return True
        temporal_type = str(
            observation.metadata.get(self.policy.temporal_type_key, "")
        ).strip().upper()
        return temporal_type in {"TEMPORARY", "BOUNDED"}

    @staticmethod
    def _bool_hint(metadata: Mapping[str, Any], key: str) -> bool:
        value = metadata.get(key, False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().casefold() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _normalize_value(value: str) -> str:
        return unicodedata.normalize("NFKC", value).strip().casefold()

    def _values_equal(self, left: str, right: str) -> bool:
        return self._normalize_value(left) == self._normalize_value(right)

    @staticmethod
    def _candidate_ids(candidates: Sequence[MatchCandidate] | Any) -> list[str]:
        return list(
            dict.fromkeys(candidate.candidate_memory_id for candidate in candidates)
        )

    @staticmethod
    def _decision(
        observation: StateObservation,
        operation: UpdateOperation,
        targets: list[str],
        confidence: float,
        reason: str,
        *,
        requires_review: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> UpdateDecision:
        unique_targets = list(dict.fromkeys(targets))
        return UpdateDecision(
            new_memory_id=observation.memory_id,
            operation=operation,
            target_memory_ids=unique_targets,
            state_key=observation.state_key,
            confidence=confidence,
            reason=reason,
            evidence=[observation.memory_id, *unique_targets],
            requires_review=requires_review,
            metadata=metadata or {},
        )
