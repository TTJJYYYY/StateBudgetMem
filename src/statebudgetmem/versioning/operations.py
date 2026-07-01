from __future__ import annotations

from enum import Enum


class UpdateOperation(str, Enum):
    """Canonical state-version update operation.

    Enum values stay uppercase because the controlled datasets and the
    versioning graph already use uppercase operation names. ``_missing_``
    accepts lowercase/config spellings and the legacy online-interface name
    ``UPDATE``.

    ``UPDATE`` is treated as ``MERGE`` for backward compatibility: the old
    online contract described UPDATE as supplementing an existing memory,
    which is the closest versioning operation.
    """

    ADD = "ADD"
    MERGE = "MERGE"
    SUPERSEDE = "SUPERSEDE"
    TEMP_INVALIDATE = "TEMP_INVALIDATE"
    RESTORE = "RESTORE"
    DELETE = "DELETE"
    NOOP = "NOOP"

    @classmethod
    def _missing_(cls, value: object) -> "UpdateOperation | None":
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, str):
            return None

        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        if normalized == "UPDATE":
            normalized = "MERGE"

        for member in cls:
            if normalized in {member.name, member.value}:
                return member
        return None


class VersionRelation(str, Enum):
    """A directed version-graph edge from an old version to a new version.

    RESTORES records that the successor version restores from the predecessor
    version. It is a graph relation, distinct from UpdateOperation.RESTORE.
    """

    SUPERSEDES = "SUPERSEDES"
    TEMP_INVALIDATES = "TEMP_INVALIDATES"
    MERGES_INTO = "MERGES_INTO"
    RESTORES = "RESTORES"
    DELETES = "DELETES"


class ComputedStatus(str, Enum):
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    TEMP_INVALIDATED = "TEMP_INVALIDATED"
    DELETED = "DELETED"
    UNKNOWN = "UNKNOWN"
