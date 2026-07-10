"""Build deterministic fixture summaries for the Phase 1 reproduction data.

The user JSON files are the source of truth. Before writing ``summary.json``
this tool runs the strict B3 probing-question and gold-label validation from
``statebudgetmem.baselines.memorybank.datasets``.
"""

from __future__ import annotations

import json
from pathlib import Path

from statebudgetmem.baselines.memorybank.datasets import (
    load_reproduction_dataset,
    reproduction_dataset_stats,
)

ROOT = Path("data/memorybank_reproduction")
USER_DIR = ROOT / "users"
OUTPUT_DIR = Path("results/memorybank/reproduction_storage")
OUTPUT_FILE = OUTPUT_DIR / "summary.json"


def build_summary_dataset() -> dict[str, object]:
    """Validate the dataset and write the B2 summary/portrait fixtures."""

    users, probes = load_reproduction_dataset(ROOT)
    stats = reproduction_dataset_stats(users, probes)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, object]] = []

    for user_file in sorted(USER_DIR.glob("*.json")):
        with user_file.open("r", encoding="utf-8") as fh:
            user = json.load(fh)

        user_id = str(user["user_id"])
        user_summary: dict[str, object] = {
            "user_id": user_id,
            "summary_mode": "fixture",
            "portrait_mode": "fixture",
            "source": "manual_fixture",
            "user_memory_ids": {
                "event_summary_id": f"{user_id}_global_event_summary",
                "portrait_id": f"{user_id}_global_user_portrait",
            },
            "days": [],
        }

        days: list[dict[str, str]] = []
        for day in user["days"]:
            days.append(
                {
                    "date": str(day["date"]),
                    "daily_event_summary": str(
                        day.get("daily_event_summary", "")
                    ),
                    "daily_personality": str(
                        day.get("daily_personality", "")
                    ),
                    "source": "manual_fixture",
                }
            )
        user_summary["days"] = days
        user_summary["global_event_summary"] = str(
            user.get("global_event_summary", "")
        )
        user_summary["global_user_portrait"] = str(
            user.get("global_user_portrait", "")
        )
        result.append(user_summary)

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(
        "Validated reproduction dataset: "
        f"{stats['user_count']} users, "
        f"{stats['user_day_count']} user-days, "
        f"{stats['probe_count']} probes, "
        f"{stats['memory_source_count']} addressable memory sources."
    )
    print(f"Saved fixture summary to {OUTPUT_FILE}")
    return stats


if __name__ == "__main__":
    build_summary_dataset()