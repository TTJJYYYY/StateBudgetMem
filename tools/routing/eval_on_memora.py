#!/usr/bin/env python3
"""
eval_on_memora.py — Evaluate query routing accuracy on real Memora data.

Reads Memora evaluation questions from a persona directory, classifies each
question using a RuleBasedRouter (offline) or LLMQueryRouter (online), and
reports accuracy broken down by task type (remembering / reasoning / recommending).

═══════════════════════════════════════════════════════════════════
Usage:
═══════════════════════════════════════════════════════════════════

    # Offline (rule-based, no API key needed):
    python tools/routing/eval_on_memora.py --memora-dir Memora --persona software_engineer

    # Online (LLM-based, using SiliconFlow / DeepSeek):
    export OPENAI_API_KEY="sk-xxx"
    export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
    python tools/routing/eval_on_memora.py \\
        --memora-dir Memora \\
        --persona software_engineer \\
        --mode llm \\
        --model "deepseek-ai/DeepSeek-V4-Flash"

    # All personas:
    python tools/routing/eval_on_memora.py --memora-dir Memora --all-personas

Output is written as JSON to results/routing_eval_<persona>_<mode>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── path setup ─────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR, _SRC_DIR / "statebudgetmem"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from statebudgetmem.routing import LLMQueryRouter, QueryRecord, RuleBasedRouter

# ── Memora task_type → expected QueryType mapping ───────────────────────
# This is a heuristic: the Memora dataset doesn't explicitly label temporal
# query types.  "remembering" questions ask about current or past facts;
# "reasoning" often involves changes/comparisons; "recommending" asks for
# current-state-based suggestions.  We report raw counts per type rather
# than claiming a single ground truth.
_TASK_TYPE_HINT: dict[str, list[str]] = {
    "remembering": ["CURRENT", "HISTORICAL"],
    "reasoning": ["CHANGE", "HISTORICAL"],
    "recommending": ["CURRENT"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate query routing on Memora evaluation questions",
    )
    p.add_argument(
        "--memora-dir",
        required=True,
        help="Path to the Memora dataset root (e.g. Memora/ or Memora/data)",
    )
    p.add_argument("--persona", default="software_engineer")
    p.add_argument("--period", default="monthly", choices=["weekly", "monthly"])
    p.add_argument("--all-personas", action="store_true")
    p.add_argument(
        "--mode", default="rule", choices=["rule", "llm"],
        help="Router mode (rule = offline, llm = online)",
    )
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    p.add_argument("--output", default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def find_memora_data(memora_dir: Path) -> Path:
    """Resolve the Memora data directory (handles both Memora/ and Memora/data/)."""
    if memora_dir.name == "data":
        return memora_dir
    data_dir = memora_dir / "data"
    if data_dir.is_dir():
        return data_dir
    raise FileNotFoundError(f"Could not find Memora data/ under {memora_dir}")


def load_eval_questions(data_dir: Path, persona: str, period: str) -> list[dict[str, Any]]:
    """
    Load all evaluation questions for one persona.

    Returns a flat list of dicts with keys: question, task_type, question_date.
    """
    eval_file = data_dir / period / persona / f"evaluation_questions_{persona}.json"
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    data = json.loads(eval_file.read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = []
    for task_type in ("remembering", "reasoning", "recommending"):
        for q in data.get("questions", {}).get(task_type, []):
            questions.append(
                {
                    "question": str(q.get("question", "")).strip(),
                    "task_type": task_type,
                    "question_date": str(q.get("question_date", "")),
                    "question_id": str(q.get("question_id", "")),
                }
            )
    return questions


def evaluate(
    questions: list[dict[str, Any]],
    router: Any,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Classify every question and return per-task-type statistics.

    Because Memora doesn't have ground-truth temporal labels we report:
    - distribution of predicted QueryTypes per task_type
    - individual results for inspection
    """
    by_type: dict[str, dict[str, int]] = {}  # task_type → {query_type: count}
    results: list[dict[str, str]] = []

    for q in questions:
        text = q["question"]
        qr = QueryRecord(text=text)
        try:
            qt = router.classify(qr)
        except Exception as exc:
            qt = router.fallback_type if hasattr(router, "fallback_type") else None
            if verbose:
                print(f"  ⚠ {text[:60]}… → ERROR: {exc}", file=sys.stderr)

        if qt is None:
            continue

        qt_str = qt.value if hasattr(qt, "value") else str(qt)
        task = q["task_type"]
        by_type.setdefault(task, {}).setdefault(qt_str, 0)
        by_type[task][qt_str] += 1

        results.append(
            {
                "question": text,
                "task_type": task,
                "question_date": q["question_date"],
                "predicted_query_type": qt_str,
            }
        )

        if verbose:
            print(f"  [{qt_str:11}] {task:13} | {text[:80]}")

    # ── summary ──
    total = len(results)
    summary = {
        "persona": "",
        "total_questions": total,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "rule" if isinstance(router, RuleBasedRouter) else "llm",
        "by_task_type": {},
    }
    for task, counts in sorted(by_type.items()):
        task_total = sum(counts.values())
        summary["by_task_type"][task] = {
            "total": task_total,
            "distribution": counts,
        }

    summary["all_results"] = results
    return summary


def main() -> int:
    args = parse_args()

    # ── resolve Memora data ──────────────────────────────────────
    memora_dir = Path(args.memora_dir).resolve()
    if not memora_dir.is_dir():
        print(f"❌ Memora directory not found: {memora_dir}")
        return 1
    data_dir = find_memora_data(memora_dir)

    # ── personas ─────────────────────────────────────────────────
    personas = [args.persona]
    if args.all_personas:
        period_dir = data_dir / args.period
        if not period_dir.is_dir():
            print(f"❌ Period directory not found: {period_dir}")
            return 1
        personas = sorted(
            d.name for d in period_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    # ── build router ─────────────────────────────────────────────
    if args.mode == "llm":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("❌ LLM mode requires --api-key or OPENAI_API_KEY")
            return 1
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "")
        router = LLMQueryRouter(
            api_key=api_key,
            base_url=base_url or None,
            model=args.model,
            log_raw_response=args.verbose,
        )
        print(f"🤖 LLM mode: model={args.model}, base_url={base_url or '(default)'}")
    else:
        router = RuleBasedRouter()
        print("📏 Rule-based mode (offline)")

    # ── evaluate ─────────────────────────────────────────────────
    results_dir = _PROJECT_ROOT / "results" / "routing"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []

    for persona in personas:
        print(f"\n═══ {persona} ═══")
        try:
            questions = load_eval_questions(data_dir, persona, args.period)
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping: {e}")
            continue

        print(f"  Questions: {len(questions)}")

        summary = evaluate(questions, router, verbose=args.verbose)
        summary["persona"] = persona
        all_summaries.append(summary)

        # ── print summary ──
        for task, info in summary["by_task_type"].items():
            dist = info["distribution"]
            dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
            print(f"  {task:15} ({info['total']:3d}): {dist_str}")

        # ── save individual ──
        out_path = str(results_dir / f"routing_eval_{persona}_{args.mode}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  💾 Saved: {out_path}")

    # ── cross-persona summary ──
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print(f"Cross-Persona Summary ({len(all_summaries)} personas, {args.period})")
        print(f"{'='*60}")

        global_counts: dict[str, dict[str, int]] = {}
        total_q = 0
        for s in all_summaries:
            for task, info in s["by_task_type"].items():
                global_counts.setdefault(task, {})
                for qt, cnt in info["distribution"].items():
                    global_counts[task][qt] = global_counts[task].get(qt, 0) + cnt
                total_q += info["total"]

        for task in sorted(global_counts):
            dist = global_counts[task]
            d_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
            task_total = sum(dist.values())
            print(f"  {task:15} ({task_total:3d}): {d_str}")

        print(f"  {'─'*50}")
        print(f"  {'TOTAL':15} ({total_q:3d} questions across {len(all_summaries)} personas)")

        combined_path = str(results_dir / f"routing_eval_ALL_{args.period}_{args.mode}.json")
        combined = {
            "period": args.period,
            "mode": args.mode,
            "persona_count": len(all_summaries),
            "total_questions": total_q,
            "global_distribution": global_counts,
            "per_persona": all_summaries,
        }
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 Combined: {combined_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
