from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_method_specific_code_is_grouped_by_baseline() -> None:
    memorybank = ROOT / "src/statebudgetmem/baselines/memorybank"
    tfidf = ROOT / "src/statebudgetmem/baselines/tfidf"

    assert {
        "system.py",
        "agents.py",
        "datasets.py",
        "evaluator.py",
        "staleness.py",
        "demo.py",
    } <= {path.name for path in memorybank.iterdir()}
    assert {"retriever.py", "adapter.py", "runner.py"} <= {
        path.name for path in tfidf.iterdir()
    }


def test_obsolete_flat_modules_are_not_reintroduced() -> None:
    package = ROOT / "src/statebudgetmem"
    obsolete = [
        package / "baselines/memorybank.py",
        package / "baselines/agents.py",
        package / "baselines/tfidf_adapter.py",
        package / "apps/memorybank_demo.py",
        package / "evaluation/memorybank.py",
        package / "evaluation/staleness.py",
        package / "experiments/baseline.py",
    ]
    assert not any(path.exists() for path in obsolete)
