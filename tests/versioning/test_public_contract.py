from __future__ import annotations

from statebudgetmem.interfaces import (
    UpdateOperation as PublicUpdateOperation,
    VersionManager as PublicVersionManager,
)
from statebudgetmem.versioning import (
    UpdateOperation,
    VersioningEngine,
)


def test_public_update_operation_is_the_versioning_enum() -> None:
    assert PublicUpdateOperation is UpdateOperation


def test_versioning_engine_implements_public_manager_protocol() -> None:
    assert isinstance(VersioningEngine(), PublicVersionManager)


def test_operation_parsing_is_case_insensitive_and_legacy_compatible() -> None:
    assert UpdateOperation("supersede") is UpdateOperation.SUPERSEDE
    assert UpdateOperation("temp-invalidate") is UpdateOperation.TEMP_INVALIDATE
    assert UpdateOperation("UPDATE") is UpdateOperation.MERGE
