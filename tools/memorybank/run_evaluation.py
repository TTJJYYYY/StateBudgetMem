#!/usr/bin/env python3
"""Run the complete MemoryBank baseline evaluation.

This replaces the former root-level ``evaluation_v2.py`` entry point while
keeping its generic JSON loader, Memora adapter, LLM-as-Judge mode, per-persona
exports, and batch summary behavior.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from statebudgetmem.baselines.memorybank import (
    MemoryEvaluator,
    MockLLM,
    OpenAICompatibleLLM,
    load_json_dataset,
    load_memora_data,
    print_summary,
    run_memora_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MemoryBank baseline evaluation")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", help="Generic JSON dataset")
    source.add_argument("--memora-dir", help="Path to Memora/data")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--persona", default="software_engineer")
    parser.add_argument("--period", default="weekly")
    parser.add_argument("--all-personas", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output", default="results/memorybank/evaluation_results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.online:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("online mode requires --api-key or OPENAI_API_KEY")
        llm = OpenAICompatibleLLM(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
        )
        judge = llm
    else:
        llm = MockLLM()
        judge = None

    if args.memora_dir and args.all_personas:
        summaries = run_memora_batch(
            data_dir=args.memora_dir,
            llm_caller=llm,
            judge_caller=judge,
            period=args.period,
            output_dir=Path(args.output).parent,
        )
        for persona, summary in summaries.items():
            print(f"\n## {persona}")
            print(summary)
        return 0

    history = probes = None
    portrait = ""
    if args.dataset:
        history, probes, portrait = load_json_dataset(args.dataset, args.sample_index)
    elif args.memora_dir:
        history, probes, portrait = load_memora_data(
            args.memora_dir, persona=args.persona, period=args.period
        )

    evaluator = MemoryEvaluator(llm_caller=llm, judge_caller=judge)
    result = evaluator.run_evaluation(history, probes, portrait)
    evaluator.export_results(result, args.output)
    print_summary(result["summary"], result.get("memory_stats"))
    print(f"\n结果已导出: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
