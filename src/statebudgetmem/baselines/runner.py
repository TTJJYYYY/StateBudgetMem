from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from statebudgetmem.baselines.tfidf.adapter import TfidfMemoryMethod
from statebudgetmem.data import load_scenarios
from statebudgetmem.evaluation.metrics import (
    recall_at_k,
    stale_retrieval_rate,
    valid_recall_at_k,
)
from statebudgetmem.interfaces import MemoryPiece, MemoryStatus, MemoryType
from statebudgetmem.schemas import MemoryRecord, QueryRecord, Scenario
from statebudgetmem.schemas.results import MethodResult, RetrievedMemory


@dataclass(frozen=True)
class BaselineComparisonConfig:
    dataset_path: Path
    results_dir: Path = Path("results/baselines")
    top_k: int = 3
    methods: tuple[str, ...] = ("tfidf_topk", "tfidf_memorybank")
    random_seed: int = 42
    config_path: Path | None = None


class TfidfMemoryBankMethod:
    """Adapter for the lightweight TFIDFMemoryBank using controlled records.

    The original MemoryBank-style API stores dialogue turns and generates its
    own IDs. For controlled evaluation we preserve MemoryRecord.memory_id so
    Recall@K and stale-memory metrics remain comparable with tfidf_topk.
    """

    name = "tfidf_memorybank"

    def __init__(self) -> None:
        from statebudgetmem.baselines.memorybank import TFIDFMemoryBank

        self._bank = TFIDFMemoryBank()
        self._records_by_id: dict[str, MemoryRecord] = {}

    def reset(self) -> None:
        self.__init__()

    def ingest(self, memories: list[MemoryRecord]) -> None:
        self._records_by_id = {memory.memory_id: memory for memory in memories}
        self._bank.memories = [_record_to_piece(memory) for memory in memories]
        self._bank.memories_by_id = {
            memory.memory_id: piece
            for memory, piece in zip(memories, self._bank.memories)
        }
        self._bank._needs_rebuild = True

    def retrieve(
        self,
        query: QueryRecord,
        *,
        top_k: int,
        token_budget: int | None = None,
        mutate: bool = False,
    ) -> MethodResult:
        started = time.perf_counter()
        raw = self._bank.retrieve(query.text, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0

        retrieved: list[RetrievedMemory] = []
        used_tokens = 0
        for rank, item in enumerate(raw, start=1):
            memory_id = str(item.get("memory_id", ""))
            record = self._records_by_id.get(memory_id)
            token_cost = record.token_cost if record is not None else 0
            if token_budget is not None and used_tokens + token_cost > token_budget:
                continue
            used_tokens += token_cost
            retrieved.append(
                RetrievedMemory(
                    memory_id=memory_id,
                    score=float(item.get("composite_score", item.get("semantic_score", 0.0))),
                    rank=len(retrieved) + 1,
                    token_cost=token_cost,
                    source_view="flat",
                    metadata={
                        "semantic_score": item.get("semantic_score"),
                        "composite_score": item.get("composite_score"),
                        "status": item.get("status"),
                        "backend_rank": rank,
                    },
                )
            )

        return MethodResult(
            method_name=self.name,
            query_id=query.query_id,
            retrieved_memories=retrieved,
            predicted_query_type=None,
            total_token_cost=sum(item.token_cost for item in retrieved),
            latency_ms=latency_ms,
            metadata={
                "token_budget": token_budget,
                "mutate": mutate,
                "source_retriever": "TFIDFMemoryBank",
            },
        )


def run_baseline_comparison(config: BaselineComparisonConfig) -> dict[str, Any]:
    if config.top_k < 1:
        raise ValueError("top_k must be at least 1")

    scenarios = load_scenarios(config.dataset_path)
    methods = [_build_method(name) for name in config.methods]
    run_id = _run_id(config)
    raw_dir = config.results_dir / "raw"
    summary_dir = config.results_dir / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    raw_rows: list[dict[str, Any]] = []
    for method in methods:
        for scenario in scenarios:
            method.reset()
            method.ingest(scenario.memories)
            for query in scenario.queries:
                result = method.retrieve(query, top_k=config.top_k, mutate=False)
                raw_rows.append(_row_from_result(run_id, scenario, query, result, config))

    raw_path = raw_dir / f"{run_id}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary_rows = _summarize(raw_rows, config, run_id, raw_path, time.perf_counter() - started)
    summary_json_path = summary_dir / f"{run_id}.json"
    summary_csv_path = summary_dir / f"{run_id}.csv"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=False, indent=2, sort_keys=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "run_id": run_id,
        "raw_path": str(raw_path),
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "summary": summary_rows,
    }


def _build_method(name: str):
    if name == "tfidf_topk":
        return TfidfMemoryMethod()
    if name == "tfidf_memorybank":
        return TfidfMemoryBankMethod()
    raise ValueError(f"unsupported baseline method: {name}")


def _record_to_piece(memory: MemoryRecord) -> MemoryPiece:
    timestamp = datetime.combine(memory.event_time, datetime_time.min).timestamp()
    status = MemoryStatus.ACTIVE if memory.status.value == "CURRENT" else MemoryStatus.SUPERSEDED
    return MemoryPiece(
        content=memory.text,
        timestamp=timestamp,
        memory_type=MemoryType.FACT,
        memory_id=memory.memory_id,
        status=status,
        tags=[memory.attribute, memory.memory_type],
        confidence=memory.confidence,
        source=memory.metadata.get("source") if isinstance(memory.metadata, dict) else None,
    )


def _row_from_result(
    run_id: str,
    scenario: Scenario,
    query: QueryRecord,
    result: MethodResult,
    config: BaselineComparisonConfig,
) -> dict[str, Any]:
    retrieved_ids = [item.memory_id for item in result.retrieved_memories]
    retrieved_count = len(retrieved_ids)
    current_retrieved = len(set(retrieved_ids) & set(query.gold_valid_memory_ids))
    stale_rate = stale_retrieval_rate(retrieved_ids, query.gold_stale_memory_ids)
    return {
        "run_id": run_id,
        "method": result.method_name,
        "scenario_id": scenario.scenario_id,
        "query_id": query.query_id,
        "query_text": query.text,
        "query_type": query.query_type.value,
        "top_k": config.top_k,
        "random_seed": config.random_seed,
        "retrieved_memory_ids": retrieved_ids,
        "retrieved_scores": [round(item.score, 12) for item in result.retrieved_memories],
        "recall_at_k": recall_at_k(retrieved_ids, query.gold_relevant_memory_ids),
        "valid_recall_at_k": valid_recall_at_k(retrieved_ids, query.gold_valid_memory_ids),
        "stale_retrieval_rate": stale_rate,
        "omr": stale_rate,
        "cor": current_retrieved / retrieved_count if retrieved_count else 0.0,
        "total_token_cost": result.total_token_cost,
        "average_token_cost": result.total_token_cost / retrieved_count if retrieved_count else 0.0,
        "retrieval_latency_ms": result.latency_ms,
        "metadata": result.metadata,
    }


def _summarize(
    rows: list[dict[str, Any]],
    config: BaselineComparisonConfig,
    run_id: str,
    raw_path: Path,
    elapsed_seconds: float,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "run_id": run_id,
                "method": method,
                "query_count": len(method_rows),
                "mean_recall_at_k": _mean(method_rows, "recall_at_k"),
                "mean_valid_recall_at_k": _mean(method_rows, "valid_recall_at_k"),
                "mean_stale_retrieval_rate": _mean(method_rows, "stale_retrieval_rate"),
                "mean_omr": _mean(method_rows, "omr"),
                "mean_cor": _mean(method_rows, "cor"),
                "mean_token_cost": _mean(method_rows, "average_token_cost"),
                "mean_total_token_cost": _mean(method_rows, "total_token_cost"),
                "mean_retrieval_latency_ms": _mean(method_rows, "retrieval_latency_ms"),
                "top_k": config.top_k,
                "dataset_path": str(config.dataset_path),
                "random_seed": config.random_seed,
                "config_path": str(config.config_path) if config.config_path else "",
                "results_raw_path": str(raw_path),
                "model_name": "offline_tfidf_family",
                "run_time_seconds": elapsed_seconds,
                "hardware_info": platform.platform(),
            }
        )
    return summaries


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def _run_id(config: BaselineComparisonConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    methods = "_".join(config.methods)
    return f"baseline_comparison_{methods}_seed{config.random_seed}_k{config.top_k}_{timestamp}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m statebudgetmem.baselines.runner")
    parser.add_argument("--dataset", default="data/controlled/baseline_scenarios.jsonl")
    parser.add_argument("--results-dir", default="results/baselines")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = BaselineComparisonConfig(
        dataset_path=Path(args.dataset),
        results_dir=Path(args.results_dir),
        top_k=args.top_k,
        methods=tuple(args.methods) if args.methods else ("tfidf_topk", "tfidf_memorybank"),
        random_seed=args.random_seed,
    )
    result = run_baseline_comparison(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
