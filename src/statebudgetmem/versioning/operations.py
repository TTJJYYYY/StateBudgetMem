from __future__ import annotations

from enum import Enum


class UpdateOperation(str, Enum):
    """A single update decision for an incoming memory.

    RESTORE means a new recovery signal restores a long-term state that was
    previously temporarily overridden.
    """

    ADD = "ADD"
    MERGE = "MERGE"
    SUPERSEDE = "SUPERSEDE"
    TEMP_INVALIDATE = "TEMP_INVALIDATE"
    RESTORE = "RESTORE"
    DELETE = "DELETE"
    NOOP = "NOOP"


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
