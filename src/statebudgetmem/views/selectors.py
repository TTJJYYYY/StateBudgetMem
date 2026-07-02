from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date

from statebudgetmem.schemas import MemoryRecord, QueryRecord, QueryType
from statebudgetmem.versioning.contracts import VersionManager
from statebudgetmem.versioning.models import StateDimension, StateKey


def state_key_from_memory(memory: MemoryRecord) -> StateKey:
    """Build the versioning state key for a public MemoryRecord."""

    return StateKey(
        subject=memory.subject,
        attribute=memory.attribute,
        dimensions=tuple(
            StateDimension(name=name, value=value)
            for name, value in memory.dimensions.items()
        ),
    )


def records_by_id(memories: Iterable[MemoryRecord]) -> dict[str, MemoryRecord]:
    return {memory.memory_id: memory for memory in memories}


def ordered_records_by_ids(
    memories_by_id: Mapping[str, MemoryRecord],
    memory_ids: Iterable[str],
) -> list[MemoryRecord]:
    """Return records in deterministic chronological order."""

    selected = [memories_by_id[memory_id] for memory_id in memory_ids if memory_id in memories_by_id]
    selected.sort(key=lambda item: (item.event_time, item.memory_id))
    return selected


def current_memory_ids(
    version_manager: VersionManager,
    *,
    reference_time: date | str | None = None,
) -> set[str]:
    view = version_manager.current_view(reference_time=reference_time)
    return {
        state.memory_id
        for resolved_states in view.values()
        for state in resolved_states
    }


def history_memory_ids(
    version_manager: VersionManager,
    *,
    state_keys: Iterable[StateKey] | None = None,
) -> set[str]:
    keys = tuple(state_keys) if state_keys is not None else tuple(version_manager.current_view().keys())
    result: set[str] = set()
    for key in keys:
        result.update(state.memory_id for state in version_manager.history(key))
    return result


def all_state_keys(memories: Iterable[MemoryRecord]) -> tuple[StateKey, ...]:
    keys = {state_key_from_memory(memory) for memory in memories}
    return tuple(sorted(keys, key=str))


def query_prefers_history(query: QueryRecord) -> bool:
    return query.query_type in {QueryType.HISTORICAL, QueryType.CHANGE}


def query_prefers_current(query: QueryRecord) -> bool:
    return query.query_type in {QueryType.CURRENT, QueryType.GENERAL}
