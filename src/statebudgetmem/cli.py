from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from statebudgetmem.data import read_flat_yaml
from statebudgetmem.baselines.tfidf import BaselineConfig, run_baseline


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
        "--backend", choices=["tfidf", "memorybank"], default="tfidf"
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the end-to-end memory pipeline (route -> view -> retrieve).",
    )
    pipeline_parser.add_argument("--query", "-q", required=True, help="User query text.")
    pipeline_parser.add_argument(
        "--dataset", default="data/controlled/baseline_scenarios.jsonl",
        help="Controlled JSONL dataset path.",
    )
    pipeline_parser.add_argument(
        "--mode", choices=["rule", "llm"], default="rule", help="Router mode.",
    )
    pipeline_parser.add_argument("--top-k", type=int, default=5)
    pipeline_parser.add_argument("--api-key", default=None)
    pipeline_parser.add_argument("--base-url", default=None)
    pipeline_parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    pipeline_parser.add_argument("--output", default=None)
    pipeline_parser.add_argument("--verbose", "-v", action="store_true")

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
    if args.command == "pipeline":
        return _run_pipeline(args)

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


def _run_pipeline(args: argparse.Namespace) -> int:
    """Run the end-to-end memory pipeline on controlled data."""
    from statebudgetmem.apps.pipeline import build_pipeline

    pipeline = build_pipeline(
        llm_api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        llm_base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        llm_model=args.model,
        use_llm_router=(args.mode == "llm"),
        top_k=args.top_k,
    )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    count = pipeline.ingest_controlled(str(dataset_path))
    if args.verbose:
        print(f"Ingested {count} memories from {dataset_path}")

    result = pipeline.ask(args.query)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved: {out_path}")

    return 0


def _evaluate_memorybank(args: argparse.Namespace) -> int:
    from statebudgetmem.baselines.memorybank import (
        MemoryEvaluator,
        MockLLM,
        OpenAICompatibleLLM,
        load_json_dataset,
        load_memora_data,
    )

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
    result = evaluator.run_evaluation(history, probes, portrait or "")
    output = evaluator.export_results(result, args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"output: {output}")
    return 0


def _analyze_staleness(args: argparse.Namespace) -> int:
    from statebudgetmem.baselines import MemoryBank, TFIDFMemoryBank
    from statebudgetmem.baselines.memorybank import (
        DEMO_HISTORY,
        DEMO_QUESTIONS,
        ObsoleteDetector,
        calculate_outdated_memory_rate,
        label_demo_memory,
        load_memora_data,
    )

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

    memory_bank = MemoryBank() if args.backend == "memorybank" else TFIDFMemoryBank()
    memory_bank.add(history)
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


if __name__ == "__main__":
    raise SystemExit(main())
