from __future__ import annotations


class VersioningError(RuntimeError):
    """Base exception for version-management failures."""


class DuplicateObservationError(VersioningError):
    """Raised when a memory ID is reused for different observable content."""


class MissingObservationError(VersioningError):
    """Raised when a graph node has no corresponding StateObservation."""


class DuplicateNodeError(VersioningError):
    """Raised when a graph node ID already exists."""


class InvalidDecisionError(VersioningError):
    """Raised when an UpdateDecision cannot be applied safely."""


class VersioningInvariantError(VersioningError):
    """Raised when a graph update would violate a versioning invariant."""
