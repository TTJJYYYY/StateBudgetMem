from pathlib import Path

from statebudgetmem.core.online import (
    UpdateOperation as OnlineUpdateOperation,
    VersionManager as OnlineVersionManager,
)
from statebudgetmem.interfaces import (
    MemoryMethod,
    MemoryPiece,
    MemoryRecord,
    MemorySystem,
    MethodResult,
    QueryRecord,
    QueryType,
    UpdateOperation,
    VersionManager,
    ViewType,
)
from statebudgetmem.versioning import (
    UpdateOperation as VersioningUpdateOperation,
    VersionManager as VersioningVersionManager,
)


ROOT = Path(__file__).resolve().parents[2]


def test_controlled_datasets_and_previous_results_are_preserved() -> None:
    baseline = ROOT / "data/controlled/baseline_scenarios.jsonl"
    temporal = ROOT / "data/controlled/temporal_challenge_v1.jsonl"
    assert len(baseline.read_text(encoding="utf-8").splitlines()) == 12
    assert len(temporal.read_text(encoding="utf-8").splitlines()) == 32
    assert (ROOT / "results/memorybank/obsolete_analysis.json").exists()
    assert len(list((ROOT / "results/raw").glob("*.jsonl"))) >= 2


def test_routing_tools_are_preserved() -> None:
    assert (ROOT / "tools/routing/debug_routing.py").exists()
    assert (ROOT / "tools/routing/run_real_routing.py").exists()


def test_public_interface_facade_contains_both_contract_layers() -> None:
    assert MemoryPiece is not None
    assert MemorySystem is not None
    assert MemoryRecord is not None
    assert QueryRecord is not None
    assert MemoryMethod is not None
    assert MethodResult is not None
    assert QueryType.GENERAL.name == "GENERAL"
    assert ViewType.NONE.value == "none"
    assert UpdateOperation.SUPERSEDE.value == "SUPERSEDE"
    assert UpdateOperation("supersede") is UpdateOperation.SUPERSEDE
    assert UpdateOperation("UPDATE") is UpdateOperation.MERGE
    assert OnlineUpdateOperation is UpdateOperation is VersioningUpdateOperation
    assert OnlineVersionManager is VersionManager is VersioningVersionManager
