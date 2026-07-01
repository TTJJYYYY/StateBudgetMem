from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from statebudgetmem.versioning.exceptions import MissingObservationError
from statebudgetmem.versioning.models import (
    MatchCandidate,
    MatchType,
    StateObservation,
    VersionNode,
)
from statebudgetmem.versioning.operations import ComputedStatus

_MATCH_SCORES = {
    MatchType.EXACT_SLOT: 1.00,
    MatchType.BROADER_SCOPE: 0.85,
    MatchType.NARROWER_SCOPE: 0.80,
    MatchType.COMPATIBLE_SCOPE: 0.70,
}


class StateMatcher(Protocol):
    def match(
        self,
        new_observation: StateObservation,
        existing_nodes: Sequence[VersionNode],
        observations_by_id: Mapping[str, StateObservation],
        *,
        top_k: int | None = None,
    ) -> Sequence[MatchCandidate]:
        ...


class StructuredStateMatcher:
    """Find structured candidate states without choosing an update operation."""

    def match(
        self,
        new_observation: StateObservation,
        existing_nodes: Sequence[VersionNode],
        observations_by_id: Mapping[str, StateObservation],
        *,
        top_k: int | None = None,
    ) -> Sequence[MatchCandidate]:
        _validate_top_k(top_k)
        for node in existing_nodes:
            if node.memory_id not in observations_by_id:
                raise MissingObservationError(
                    f"missing observation for memory_id: {node.memory_id}"
                )

        candidates: list[MatchCandidate] = []
        incoming_dimensions = new_observation.state_key.dimension_map()
        for node in existing_nodes:
            if node.memory_id == new_observation.memory_id:
                continue
            if node.computed_status is ComputedStatus.DELETED:
                continue
            if node.state_key.subject != new_observation.state_key.subject:
                continue
            if node.state_key.attribute != new_observation.state_key.attribute:
                continue

            candidate_observation = observations_by_id[node.memory_id]
            if candidate_observation.event_time > new_observation.event_time:
                continue

            candidate_dimensions = node.state_key.dimension_map()
            match_type = _dimension_match_type(candidate_dimensions, incoming_dimensions)
            if match_type is None:
                continue
            dimension_distance = _dimension_distance(
                candidate_dimensions,
                incoming_dimensions,
            )

            candidates.append(
                MatchCandidate(
                    candidate_memory_id=node.memory_id,
                    state_key=node.state_key,
                    candidate_status=node.computed_status,
                    candidate_event_time=candidate_observation.event_time,
                    match_type=match_type,
                    score=_MATCH_SCORES[match_type],
                    dimension_distance=dimension_distance,
                    reasons=_reasons(
                        new_observation,
                        node,
                        match_type,
                        dimension_distance,
                    ),
                    metadata={},
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.dimension_distance,
                -candidate.candidate_event_time.toordinal(),
                candidate.candidate_memory_id,
            )
        )
        if top_k is not None:
            candidates = candidates[:top_k]
        return tuple(candidates)


def _validate_top_k(top_k: int | None) -> None:
    if top_k is None:
        return
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer or None")


def _dimension_match_type(
    candidate_dimensions: dict[str, str],
    incoming_dimensions: dict[str, str],
) -> MatchType | None:
    common_names = set(candidate_dimensions) & set(incoming_dimensions)
    for name in common_names:
        if candidate_dimensions[name] != incoming_dimensions[name]:
            return None

    candidate_items = set(candidate_dimensions.items())
    incoming_items = set(incoming_dimensions.items())
    if candidate_items == incoming_items:
        return MatchType.EXACT_SLOT
    if candidate_items < incoming_items:
        return MatchType.BROADER_SCOPE
    if incoming_items < candidate_items:
        return MatchType.NARROWER_SCOPE
    return MatchType.COMPATIBLE_SCOPE


def _dimension_distance(
    candidate_dimensions: dict[str, str],
    incoming_dimensions: dict[str, str],
) -> int:
    candidate_items = set(candidate_dimensions.items())
    incoming_items = set(incoming_dimensions.items())
    return len(candidate_items.symmetric_difference(incoming_items))


def _reasons(
    new_observation: StateObservation,
    node: VersionNode,
    match_type: MatchType,
    dimension_distance: int,
) -> list[str]:
    return [
        f"same subject: {new_observation.state_key.subject}",
        f"same attribute: {new_observation.state_key.attribute}",
        "candidate event_time is not later than incoming event_time",
        _dimension_reason(match_type),
        f"dimension distance: {dimension_distance}",
        f"candidate status: {node.computed_status.value}",
    ]


def _dimension_reason(match_type: MatchType) -> str:
    if match_type is MatchType.EXACT_SLOT:
        return "candidate dimensions exactly match incoming dimensions"
    if match_type is MatchType.BROADER_SCOPE:
        return "candidate dimensions are broader than incoming dimensions"
    if match_type is MatchType.NARROWER_SCOPE:
        return "candidate dimensions are narrower than incoming dimensions"
    return "candidate dimensions are compatible with incoming dimensions"
