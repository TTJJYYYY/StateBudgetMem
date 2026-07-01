from __future__ import annotations

from datetime import date

import pytest

from statebudgetmem.versioning import (
    ComputedStatus,
    MatchType,
    MissingObservationError,
    StateDimension,
    StateKey,
    StateObservation,
    StructuredStateMatcher,
    VersionNode,
)


def _obs(
    memory_id: str,
    *,
    value: str = "v",
    event_time: str = "2026-01-01",
    dimensions: dict[str, str] | None = None,
) -> StateObservation:
    return StateObservation(
        memory_id=memory_id,
        state_key=StateKey(
            subject="user",
            attribute="commute",
            dimensions=dimensions or {},
        ),
        value=value,
        text=value,
        event_time=event_time,
        confidence=1.0,
    )


def _node(
    observation: StateObservation,
    status: ComputedStatus = ComputedStatus.CURRENT,
) -> VersionNode:
    return VersionNode(
        memory_id=observation.memory_id,
        state_key=observation.state_key,
        computed_status=status,
        valid_from=observation.effective_from,
    )


def test_matcher_classifies_dimension_relations() -> None:
    matcher = StructuredStateMatcher()
    incoming = _obs(
        "new",
        event_time="2026-02-01",
        dimensions={"weather": "rainy", "city": "beijing"},
    )
    exact = _obs(
        "exact",
        dimensions={"city": "beijing", "weather": "rainy"},
    )
    broader = _obs("broader", dimensions={"weather": "rainy"})
    narrower = _obs(
        "narrower",
        dimensions={
            "weather": "rainy",
            "city": "beijing",
            "day_type": "weekday",
        },
    )
    compatible = _obs("compatible", dimensions={"day_type": "weekday"})
    observations = {
        item.memory_id: item for item in (exact, broader, narrower, compatible)
    }
    candidates = matcher.match(
        incoming,
        [_node(item) for item in observations.values()],
        observations,
    )
    by_id = {item.candidate_memory_id: item for item in candidates}
    assert by_id["exact"].match_type is MatchType.EXACT_SLOT
    assert by_id["broader"].match_type is MatchType.BROADER_SCOPE
    assert by_id["narrower"].match_type is MatchType.NARROWER_SCOPE
    assert by_id["compatible"].match_type is MatchType.COMPATIBLE_SCOPE
    assert by_id["exact"].dimension_distance == 0


def test_matcher_excludes_conflicting_dimension_values() -> None:
    incoming = _obs("new", event_time="2026-02-01", dimensions={"weather": "rainy"})
    old = _obs("old", dimensions={"weather": "sunny"})
    result = StructuredStateMatcher().match(incoming, [_node(old)], {"old": old})
    assert result == ()


def test_matcher_prefers_closer_scope_before_recency() -> None:
    incoming = _obs(
        "new",
        event_time="2026-03-01",
        dimensions={"weather": "rainy", "city": "beijing"},
    )
    broad_recent = _obs("broad", event_time="2026-02-28", dimensions={})
    close_old = _obs("close", event_time="2026-01-01", dimensions={"weather": "rainy"})
    result = StructuredStateMatcher().match(
        incoming,
        [_node(broad_recent), _node(close_old)],
        {"broad": broad_recent, "close": close_old},
    )
    assert [item.candidate_memory_id for item in result] == ["close", "broad"]
    assert result[0].dimension_distance == 1
    assert result[1].dimension_distance == 2


def test_matcher_filters_future_deleted_and_self() -> None:
    incoming = _obs("new", event_time="2026-02-01")
    future = _obs("future", event_time="2026-03-01")
    deleted = _obs("deleted", event_time="2026-01-01")
    result = StructuredStateMatcher().match(
        incoming,
        [_node(incoming), _node(future), _node(deleted, ComputedStatus.DELETED)],
        {"new": incoming, "future": future, "deleted": deleted},
    )
    assert result == ()


def test_matcher_requires_observation_for_every_graph_node() -> None:
    old = _obs("old")
    incoming = _obs("new", event_time="2026-02-01")
    with pytest.raises(MissingObservationError, match="old"):
        StructuredStateMatcher().match(incoming, [_node(old)], {})


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_matcher_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(ValueError):
        StructuredStateMatcher().match(_obs("new"), [], {}, top_k=top_k)  # type: ignore[arg-type]


def test_matcher_top_k_and_stable_id_tie_break() -> None:
    incoming = _obs("new", event_time="2026-02-01")
    b = _obs("b")
    a = _obs("a")
    result = StructuredStateMatcher().match(
        incoming,
        [_node(b), _node(a)],
        {"a": a, "b": b},
        top_k=1,
    )
    assert [item.candidate_memory_id for item in result] == ["a"]
