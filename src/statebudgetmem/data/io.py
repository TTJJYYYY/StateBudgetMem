from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from statebudgetmem.schemas import Scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"dataset not found: {scenario_path}")
    scenarios: list[Scenario] = []
    with scenario_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                scenarios.append(Scenario.model_validate(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"invalid scenario at {scenario_path}:{line_number}: {exc}") from exc
    return scenarios


def read_flat_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    data: dict[str, Any] = {}
    with config_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"invalid config line {line_number}: {line.rstrip()}")
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            value = raw_value.strip().strip("'\"")
            if value.isdigit():
                data[key] = int(value)
            else:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
    return data
