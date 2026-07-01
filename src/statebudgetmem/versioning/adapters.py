from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Protocol

from statebudgetmem.schemas import MemoryRecord
from statebudgetmem.versioning.models import StateDimension, StateKey, StateObservation


class MemoryAdapter(Protocol):
    def to_observations(
        self,
        memory: MemoryRecord,
    ) -> Sequence[StateObservation]:
        ...


class MemoryRecordAdapter:
    """Convert public MemoryRecord objects into versioning observations."""

    def to_observations(
        self,
        memory: MemoryRecord,
    ) -> Sequence[StateObservation]:
        state_key = StateKey(
            subject=memory.subject,
            attribute=memory.attribute,
            dimensions=tuple(
                StateDimension(name=name, value=value)
                for name, value in memory.dimensions.items()
            ),
        )
        observation = StateObservation(
            memory_id=memory.memory_id,
            state_key=state_key,
            value=memory.value,
            text=memory.text,
            event_time=memory.event_time,
            valid_from=memory.valid_from,
            valid_to=memory.valid_to,
            confidence=memory.confidence,
            metadata=deepcopy(memory.metadata),
        )
        return (observation,)
