from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from enum import Enum
from typing import Any, Protocol

from statebudgetmem.schemas import MemoryRecord
from statebudgetmem.versioning.models import (
    StateDimension,
    StateKey,
    StateObservation,
)
from statebudgetmem.versioning.operations import UpdateOperation


class MemoryAdapter(Protocol):
    def to_observations(
        self,
        memory: MemoryRecord,
    ) -> Sequence[StateObservation]: ...


def normalize_versioning_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize preprocessing metadata into the versioning contract.

    Canonical keys are consumed by ``RuleBasedOperationClassifier``. The
    preprocessing module historically emitted ``operation_hint``; this bridge
    accepts it and converts it to ``versioning_intent``.

    Malformed or contradictory hints are handled conservatively: the record is
    marked ``needs_review`` and its intent becomes ``NOOP`` so that bad metadata
    cannot silently mutate the version graph.
    """

    normalized: dict[str, Any] = deepcopy(metadata or {})
    errors: list[str] = []

    canonical_raw = normalized.get("versioning_intent")
    legacy_raw = normalized.get("operation_hint")

    canonical = _parse_operation(canonical_raw, "versioning_intent", errors)
    legacy = _parse_operation(legacy_raw, "operation_hint", errors)

    if canonical is not None and legacy is not None and canonical is not legacy:
        errors.append(
            "versioning_intent conflicts with legacy operation_hint "
            f"({canonical.value} != {legacy.value})"
        )

    selected = canonical or legacy
    normalized.pop("operation_hint", None)
    if selected is not None:
        normalized["versioning_intent"] = selected.value

    if "versioning_target_ids" in normalized:
        raw_targets = normalized["versioning_target_ids"]
        targets = _normalize_target_ids(raw_targets)
        if targets is None:
            errors.append("versioning_target_ids must be a string or a collection of IDs")
            normalized.pop("versioning_target_ids", None)
        else:
            normalized["versioning_target_ids"] = targets

    for key in (
        "temporary",
        "restore_signal",
        "delete_request",
        "merge_request",
    ):
        if key not in normalized:
            continue
        parsed = _normalize_bool(normalized[key])
        if parsed is None:
            errors.append(f"{key} must be boolean-like")
            normalized.pop(key, None)
        else:
            normalized[key] = parsed

    if errors:
        normalized["needs_review"] = True
        normalized["versioning_contract_error"] = "; ".join(errors)
        normalized["versioning_intent"] = UpdateOperation.NOOP.value

    return normalized


def _parse_operation(
    raw: object,
    key: str,
    errors: list[str],
) -> UpdateOperation | None:
    if raw is None:
        return None
    if isinstance(raw, Enum):
        raw = raw.value
    try:
        return UpdateOperation(raw)
    except (TypeError, ValueError):
        errors.append(f"{key} contains an unknown operation: {raw!r}")
        return None


def _normalize_target_ids(raw: object) -> list[str] | None:
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = list(raw)
    else:
        return None

    result: list[str] = []
    for item in values:
        if item is None:
            continue
        value = str(item).strip()
        if value and value not in result:
            result.append(value)
    return result


def _normalize_bool(raw: object) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in {0, 1}:
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().casefold()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _metadata_with_explicit_relations(memory: MemoryRecord) -> dict[str, Any]:
    """Bridge public relation fields into the versioning metadata contract."""

    metadata = normalize_versioning_metadata(memory.metadata)
    supersedes = _normalize_target_ids(memory.supersedes) or []
    temporarily_invalidates = _normalize_target_ids(memory.temporarily_invalidates) or []

    if not supersedes and not temporarily_invalidates:
        return metadata

    explicit_targets = _normalize_target_ids(metadata.get("versioning_target_ids")) or []
    all_targets = list(
        dict.fromkeys([*explicit_targets, *supersedes, *temporarily_invalidates])
    )
    metadata["versioning_target_ids"] = all_targets

    if supersedes and temporarily_invalidates:
        metadata["needs_review"] = True
        metadata["versioning_contract_error"] = (
            "MemoryRecord cannot declare both supersedes and "
            "temporarily_invalidates in one atomic observation"
        )
        metadata["versioning_intent"] = UpdateOperation.NOOP.value
        return metadata

    relation_intent = (
        UpdateOperation.SUPERSEDE
        if supersedes
        else UpdateOperation.TEMP_INVALIDATE
    )
    existing_raw = metadata.get("versioning_intent")
    existing = None
    if existing_raw is not None:
        try:
            existing = UpdateOperation(existing_raw)
        except (TypeError, ValueError):
            existing = None

    if existing is not None and existing is not relation_intent:
        metadata["needs_review"] = True
        metadata["versioning_contract_error"] = (
            "MemoryRecord relation fields conflict with versioning_intent "
            f"({relation_intent.value} != {existing.value})"
        )
        metadata["versioning_intent"] = UpdateOperation.NOOP.value
        return metadata

    metadata["versioning_intent"] = relation_intent.value
    if relation_intent is UpdateOperation.TEMP_INVALIDATE:
        metadata["temporary"] = True
    return metadata


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
        metadata = _metadata_with_explicit_relations(memory)
        observation = StateObservation(
            memory_id=memory.memory_id,
            state_key=state_key,
            value=memory.value,
            text=memory.text,
            event_time=memory.event_time,
            valid_from=memory.valid_from,
            valid_to=memory.valid_to,
            confidence=memory.confidence,
            metadata=metadata,
        )
        return (observation,)
