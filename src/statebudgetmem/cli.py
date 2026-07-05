from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from statebudgetmem.data import read_flat_yaml
from statebudgetmem.baselines.tfidf import BaselineConfig, run_baseline

MEMORYBANK_OPTIONAL_HINT = (
    "MemoryBank command requires optional dependencies. "
    "Install them with: pip install -e '.[memorybank]'"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="statebudgetmem")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Run the deterministic controlled-data baseline."
    )
    run_parser.add_argument("--config", required=True, help="Path to baseline YAML config.")

    route_parser = subparsers.add_parser(
        "route", help="Classify one query and select a memory view."
    )
    route_parser.add_argument("query", help="User query text.")
    route_parser.add_argument("--reference-time", default=None)
    route_parser.add_argument("--mode", choices=["rule", "llm"], default="rule")
    route_parser.add_argument("--api-key", default=None)
    route_parser.add_argument("--base-url", default=None)
    route_parser.add_argument("--model", default="gpt-4o-mini")

    eval_parser = subparsers.add_parser(
        "evaluate-memorybank",
        help="Run the stateless-versus-MemoryBank answer comparison.",
    )
    source_group = eval_parser.add_mutually_exclusive_group()
    source_group.add_argument("--dataset", help="Generic JSON dataset path.")
    source_group.add_argument("--memora-dir", help="Path to Memora/data.")
    eval_parser.add_argument("--sample-index", type=int, default=0)
    eval_parser.add_argument("--persona", default="software_engineer")
    eval_parser.add_argument("--period", default="weekly")
    eval_parser.add_argument("--online", action="store_true")
    eval_parser.add_argument("--api-key", default=None)
    eval_parser.add_argument("--base-url", default=None)
    eval_parser.add_argument("--model", default="deepseek-chat")
    eval_parser.add_argument("--output", default="results/memorybank_evaluation.json")

    stale_parser = subparsers.add_parser(
        "analyze-staleness",
        help="Measure outdated-memory retrieval on the demo or Memora data.",
    )
    stale_parser.add_argument("--memora-dir", default=None)
    stale_parser.add_argument("--persona", default="software_engineer")
    stale_parser.add_argument("--period", default="weekly")
    stale_parser.add_argument("--sample", type=int, default=20)
    stale_parser.add_argument("--top-k", type=int, default=5)
    stale_parser.add_argument(
        "--dataset-path",
        default="data/controlled/baseline_scenarios.jsonl",
        help="Controlled JSONL dataset used by the offline TF-IDF backend.",
    )
    stale_parser.add_argument(
        "--results-dir",
        default="results/staleness",
        help="Directory for machine-readable staleness analysis outputs.",
    )
    stale_parser.add_argument(
        "--backend", choices=["tfidf", "memorybank"], default="tfidf"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_controlled_baseline(args)
    if args.command == "route":
        return _route_query(args)
    if args.command == "evaluate-memorybank":
        return _evaluate_memorybank(args)
    if args.command == "analyze-staleness":
        return _analyze_staleness(args)

    parser.print_help()
    return 0


def _run_controlled_baseline(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    raw_config = read_flat_yaml(config_path)
    config = BaselineConfig(
        method=str(raw_config["method"]),
        dataset_path=Path(str(raw_config["dataset_path"])),
        top_k=int(raw_config["top_k"]),
        random_seed=int(raw_config["random_seed"]),
        results_dir=Path(str(raw_config["results_dir"])),
        config_path=config_path,
    )
    result = run_baseline(config)
    print(f"run_id: {result['run_id']}")
    print(f"raw: {result['raw_path']}")
    print(f"summary_json: {result['summary_json_path']}")
    print(f"summary_csv: {result['summary_csv_path']}")
    return 0


def _route_query(args: argparse.Namespace) -> int:
    from statebudgetmem.routing import LLMQueryRouter, QueryRecord, RuleBasedRouter

    query = QueryRecord(text=args.query, reference_time=args.reference_time)
    if args.mode == "llm":
        router = LLMQueryRouter(
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
            model=args.model,
        )
    else:
        router = RuleBasedRouter()

    query_type = router.classify(query)
    view_type = router.route(query.text, query_type)
    print(json.dumps({"query_type": query_type.value, "view": view_type.value}, ensure_ascii=False))
    return 0


def _memorybank_dependency_error(exc: ImportError) -> SystemExit:
    message = str(exc)
    if "MemoryBank requires optional dependencies" in message:
        return SystemExit(message)
    if "FAISS MemoryBank requires optional dependencies" in message:
        return SystemExit(message)
    return SystemExit(f"{MEMORYBANK_OPTIONAL_HINT}. Original error: {message}")


def _evaluate_memorybank(args: argparse.Namespace) -> int:
    try:
        from statebudgetmem.baselines.memorybank import (
            MemoryEvaluator,
            MockLLM,
            OpenAICompatibleLLM,
            load_json_dataset,
            load_memora_data,
        )
    except ImportError as exc:
        raise _memorybank_dependency_error(exc) from exc

    if args.online:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("online evaluation requires --api-key or OPENAI_API_KEY")
        llm = OpenAICompatibleLLM(
            api_key=api_key,
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL") or None,
            model=args.model,
        )
    else:
        llm = MockLLM()

    history = probes = portrait = None
    if args.dataset:
        history, probes, portrait = load_json_dataset(args.dataset, args.sample_index)
    elif args.memora_dir:
        history, probes, portrait = load_memora_data(
            args.memora_dir,
            persona=args.persona,
            period=args.period,
        )

    evaluator = MemoryEvaluator(llm_caller=llm, judge_caller=llm if args.online else None)
    try:
        result = evaluator.run_evaluation(history, probes, portrait or "")
    except ImportError as exc:
        raise _memorybank_dependency_error(exc) from exc
    output = evaluator.export_results(result, args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"output: {output}")
    return 0


def _analyze_staleness(args: argparse.Namespace) -> int:
    if args.backend == "tfidf":
        return _analyze_tfidf_staleness(args)

    try:
        from statebudgetmem.baselines import MemoryBank, TFIDFMemoryBank
        from statebudgetmem.baselines.memorybank import (
            DEMO_HISTORY,
            DEMO_QUESTIONS,
            ObsoleteDetector,
            calculate_outdated_memory_rate,
            label_demo_memory,
            load_memora_data,
        )
    except ImportError as exc:
        raise _memorybank_dependency_error(exc) from exc

    if args.memora_dir:
        history, probes, _ = load_memora_data(
            args.memora_dir,
            persona=args.persona,
            period=args.period,
        )
        questions = probes[: args.sample]
        detector = ObsoleteDetector()
    else:
        history = DEMO_HISTORY
        questions = DEMO_QUESTIONS[: args.sample]
        detector = None

    try:
        memory_bank = MemoryBank() if args.backend == "memorybank" else TFIDFMemoryBank()
        memory_bank.add(history)
    except ImportError as exc:
        raise _memorybank_dependency_error(exc) from exc
    all_memories = [
        {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "timestamp": memory.timestamp,
        }
        for memory in memory_bank.get_all()
    ]
    if detector is None:
        ground_truth = {
            item["memory_id"]: label_demo_memory(item["content"])
            for item in all_memories
        }
    else:
        ground_truth = detector.detect_transitions(all_memories)

    totals = {"retrieved": 0, "obsolete": 0, "current": 0}
    rows: list[dict[str, Any]] = []
    for probe in questions:
        query = str(probe.get("question", ""))
        retrieved = memory_bank.retrieve(query, top_k=args.top_k)
        metrics = calculate_outdated_memory_rate(retrieved, ground_truth)
        rows.append({"query": query, **metrics})
        totals["retrieved"] += int(metrics["total"])
        totals["obsolete"] += int(metrics["obsolete"])
        totals["current"] += int(metrics["current"])

    summary = {
        "questions": len(rows),
        "retrieved": totals["retrieved"],
        "omr": totals["obsolete"] / totals["retrieved"] if totals["retrieved"] else 0.0,
        "cor": totals["current"] / totals["retrieved"] if totals["retrieved"] else 0.0,
        "details": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _analyze_tfidf_staleness(args: argparse.Namespace) -> int:
    config = BaselineConfig(
        method="tfidf_topk",
        dataset_path=Path(args.dataset_path),
        top_k=int(args.top_k),
        random_seed=42,
        results_dir=Path(args.results_dir),
        config_path=Path("README.md"),
    )
    result = run_baseline(config)
    raw_path = Path(result["raw_path"])
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stale_rows = [
        row for row in rows if float(row.get("stale_retrieval_rate", 0.0)) > 0.0
    ]
    summary = {
        "backend": "tfidf",
        "method": "tfidf_topk",
        "dataset_path": str(config.dataset_path),
        "top_k": config.top_k,
        "query_count": len(rows),
        "queries_with_stale_retrieval": len(stale_rows),
        "mean_stale_retrieval_rate": result["summary"]["mean_stale_retrieval_rate"],
        "max_stale_retrieval_rate": max(
            (float(row["stale_retrieval_rate"]) for row in rows),
            default=0.0,
        ),
        "raw_path": result["raw_path"],
        "summary_json_path": result["summary_json_path"],
        "summary_csv_path": result["summary_csv_path"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
