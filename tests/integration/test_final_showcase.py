"""Validate the final showcase output.

费哲瀚 — 组员 C, Phase M2 Showcase
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_DIR = ROOT / "results" / "final_showcase"


def test_index_html_exists():
    index = SHOWCASE_DIR / "index.html"
    assert index.exists(), f"index.html not found at {index}"
    content = index.read_text(encoding="utf-8")
    assert len(content) > 1000, "index.html appears too small"


def test_three_layer_structure():
    """Verify the 3-layer boundary: Case Entry, MemoryExplorer, Dashboard."""
    index = SHOWCASE_DIR / "index.html"
    html = index.read_text(encoding="utf-8")

    assert "Case Entry" in html, "Missing Case Entry section"
    assert "MemoryExplorer" in html, "Missing MemoryExplorer section"
    assert "Fair Experiment" in html or "Dashboard" in html or "fair" in html.lower(), (
        "Missing Fair Experiment / Dashboard section"
    )


def test_case_entry_labeled_demo_only():
    index = SHOWCASE_DIR / "index.html"
    html = index.read_text(encoding="utf-8")

    assert "Demo-only" in html or "entry case for explanation" in html.lower(), (
        "Case Entry must be labeled as demo-only"
    )


def test_memory_explorer_labeled_analysis_tool():
    index = SHOWCASE_DIR / "index.html"
    html = index.read_text(encoding="utf-8")

    assert ("analysis tool" in html.lower()
            or "showcase and analysis" in html.lower()), (
        "MemoryExplorer must be labeled as analysis/showcase tool"
    )


def test_formal_results_source_path_present():
    """Formal results must reference data source path."""
    index = SHOWCASE_DIR / "index.html"
    html = index.read_text(encoding="utf-8")

    assert "fair_comparison" in html.lower(), (
        "Formal section must reference fair_comparison data source"
    )


def test_showcase_data_json_valid():
    data_path = SHOWCASE_DIR / "showcase_data.json"
    assert data_path.exists(), f"showcase_data.json not found at {data_path}"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert "case_entry" in data, "Missing case_entry"
    assert "memory_explorer" in data, "Missing memory_explorer"
    assert "metadata" in data, "Missing metadata"


def test_dashboard_data_json_valid():
    data_path = SHOWCASE_DIR / "experiment_dashboard_data.json"
    assert data_path.exists(), f"experiment_dashboard_data.json not found at {data_path}"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert "methods" in data, "Missing methods in dashboard"
    assert len(data["methods"]) >= 6, f"Expected >=6 methods, got {len(data['methods'])}"


def test_no_demo_mixed_into_formal():
    """Formal methods data must not contain demo/dialogue entries."""
    data_path = SHOWCASE_DIR / "experiment_dashboard_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    methods = [m["method"] for m in data.get("methods", [])]
    for m in methods:
        assert "demo" not in m.lower(), f"Demo method leaked into formal data: {m}"


def test_readme_exists():
    readme = SHOWCASE_DIR / "README.md"
    assert readme.exists(), f"README.md not found at {readme}"
