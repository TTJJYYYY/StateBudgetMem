#!/usr/bin/env python3
"""Run the local-only MemoryBank core baseline and write auditable artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

from statebudgetmem.baselines.memorybank import MemoryBank  # noqa: E402
from statebudgetmem.baselines.memorybank.embeddings import (  # noqa: E402
    deterministic_hash_embedding,
)
from statebudgetmem.core import MemoryStatus  # noqa: E402


@dataclass(frozen=True)
class Probe:
    query_id: str
    query: str
    relevant_key: str
    valid_key: str
    stale_key: str


PROBES = (
    Probe("book", "Which Python book was recommended?", "python_book", "python_book", ""),
    Probe("food", "What food should I avoid now?", "current_food", "current_food", "stale_food"),
    Probe("sport", "Which sports do I enjoy?", "current_sport", "current_sport", "stale_sport"),
    Probe("city", "Where do I currently live?", "current_city", "current_city", "stale_city"),
)


class HashEncoder:
    name = "deterministic_hash_embedding"
    network_used = False

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        return deterministic_hash_embedding(text, self.dim)


class SentenceTransformerEncoder:
    network_used = False

    def __init__(self, model_name: str, local_only: bool) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=local_only)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-counts", type=int, nargs="+", default=[100, 500, 1000, 2000])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding-backend",
        choices=("hash", "sentence-transformers"),
        default="hash",
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--forgetting-threshold", type=float, default=0.3)
    parser.add_argument("--retention-time-unit-hours", type=float, default=24.0)
    parser.add_argument("--current-time", default="2026-07-11 12:00")
    parser.add_argument("--enable-forgetting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-reinforcement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detailed-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.smoke:
        args.memory_counts = [100]
        args.top_k = [1, 3]
        args.repeat = 1

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((out / ".matplotlib").resolve()))
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(out)
    write_json(out / "config.json", config)

    encoder = build_encoder(args)
    environment = environment_info(args, encoder)
    write_json(out / "environment.json", environment)

    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    rows: list[dict[str, Any]] = []
    retrieval_logs: list[dict[str, Any]] = []
    reinforcement_logs: list[dict[str, Any]] = []
    forgetting_logs: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for memory_count in args.memory_counts:
        for repeat in range(args.repeat):
            seed = args.seed + repeat
            bank, labels, timings, build_peak = build_bank(args, encoder, memory_count, seed, process)
            persistence = persist_bank(bank, out / "storage" / f"n{memory_count}_r{repeat}")
            loaded = MemoryBank(
                forgetting_threshold=args.forgetting_threshold,
                embedding_model=encoder,
                decay_interval_hours=args.retention_time_unit_hours,
            )
            loaded.load(str((out / "storage" / f"n{memory_count}_r{repeat}" / "memorybank").resolve()))
            index_loaded_rss = process.memory_info().rss

            forgetting_started = time.perf_counter_ns()
            forgetting = loaded.forgetting_log(args.current_time) if args.enable_forgetting else {
                "forgotten_memory_ids": [], "events": []
            }
            forgetting_ms = (time.perf_counter_ns() - forgetting_started) / 1e6
            forgotten_ids = set(forgetting["forgotten_memory_ids"])
            for event in forgetting["events"]:
                forgetting_logs.append({
                    **event,
                    "query": None,
                    "rank": None,
                    "retrieval_score": None,
                    "before_strength": event["strength"],
                    "after_strength": event["strength"],
                    "before_last_accessed": event["last_accessed"],
                    "after_last_accessed": event["last_accessed"],
                    "forgotten_memory_ids": forgetting["forgotten_memory_ids"],
                    "timestamp": forgetting.get("current_time"),
                    "memory_count": memory_count,
                    "repeat": repeat,
                })

            for top_k in args.top_k:
                for probe in PROBES:
                    before = {m.memory_id: (m.strength, m.last_accessed) for m in loaded.get_all()}
                    rss_before = process.memory_info().rss
                    started = time.perf_counter_ns()
                    result = loaded.retrieve_with_metadata(
                        probe.query,
                        top_k=top_k,
                        current_time=args.current_time,
                    )
                    latency_ms = (time.perf_counter_ns() - started) / 1e6
                    peak_rss = max(rss_before, process.memory_info().rss)
                    selected = result["memories"]
                    if not args.enable_reinforcement:
                        for item in selected:
                            mem = loaded.get(item["memory_id"])
                            prior = before[item["memory_id"]]
                            if mem:
                                mem.strength, mem.last_accessed = prior
                                item["after_strength"], item["after_last_accessed"] = prior
                    ids = [item["memory_id"] for item in selected]
                    relevant_id = labels[probe.relevant_key]
                    valid_id = labels[probe.valid_key]
                    stale_id = labels.get(probe.stale_key, "")
                    prompt_chars = sum(len(item["content"]) for item in selected)
                    prompt_tokens = estimate_tokens("\n".join(item["content"] for item in selected))
                    row = {
                        "memory_count": memory_count,
                        "repeat": repeat,
                        "seed": seed,
                        "top_k": top_k,
                        "query_id": probe.query_id,
                        "query": probe.query,
                        "recall_at_k": float(relevant_id in ids),
                        "valid_recall_at_k": float(valid_id in ids),
                        "stale_retrieval_rate": (sum(mid == stale_id for mid in ids) / len(ids)) if ids and stale_id else 0.0,
                        "retrieval_latency_ms": latency_ms,
                        "prompt_characters": prompt_chars,
                        "prompt_token_estimate": prompt_tokens,
                        "retrieved_memory_ids": ids,
                        "forgotten_retrieved_ids": [mid for mid in ids if mid in forgotten_ids],
                        "retrieval_peak_rss_bytes": peak_rss,
                        "forgetting_update_ms": forgetting_ms,
                    }
                    rows.append(row)
                    for item in selected:
                        log = {
                            "memory_id": item["memory_id"], "query": probe.query,
                            "rank": item["retrieval_rank"], "retrieval_score": item["retrieval_score"],
                            "before_strength": item["before_strength"], "after_strength": item.get("after_strength"),
                            "before_last_accessed": item["before_last_accessed"],
                            "after_last_accessed": item.get("after_last_accessed"),
                            "retention": item["retention"], "is_forgotten": item["is_forgotten"],
                            "forgotten_memory_ids": result["forgotten_memory_ids"],
                            "timestamp": item["recall_timestamp"], "memory_count": memory_count,
                            "top_k": top_k, "repeat": repeat,
                        }
                        retrieval_logs.append(log)
                        reinforcement_logs.append({
                            **log,
                            "reinforcement_applied": args.enable_reinforcement,
                            "reinforcement_update_ms": latency_ms,
                        })

            storage_rows.append({
                "memory_count": memory_count, "repeat": repeat, **timings,
                "baseline_rss_bytes": baseline_rss,
                "index_loaded_rss_bytes": index_loaded_rss,
                "index_build_peak_rss_bytes": build_peak,
                **persistence,
            })

    write_csv(out / "predictions.csv", rows)
    write_csv(out / "resource_metrics.csv", storage_rows)
    write_jsonl(out / "memorybank_retrieval_log.jsonl", retrieval_logs if args.detailed_logs else [])
    write_jsonl(out / "memorybank_reinforcement_log.jsonl", reinforcement_logs if args.detailed_logs else [])
    write_jsonl(out / "memorybank_forgetting_log.jsonl", forgetting_logs if args.detailed_logs else [])
    metrics = summarize(rows, storage_rows)
    resources = summarize_resources(storage_rows, rows, out)
    write_json(out / "metrics.json", metrics)
    write_json(out / "memorybank_resource_metrics.json", resources)
    summary = {
        "status": "success", "local_only": args.local_only, "network_calls": 0,
        "cloud_api_calls": 0, "memory_system_baseline": True,
        "conversational_agent_baseline": False, "embedding_backend": args.embedding_backend,
        "embedding_model": encoder.name, "memory_counts": args.memory_counts,
        "top_k": args.top_k, "repeat": args.repeat, "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - run_started,
        "metrics": metrics, "resources": resources,
        "paper_mechanism": "R=exp(-t/S), S starts at 1, recalled memories use S+=1 and reset last_accessed",
        "project_extensions": "Threshold-based forgotten candidate flagging is evaluated; strength*=0.5 is not used by this runner.",
    }
    write_json(out / "memorybank_run_summary.json", summary)
    create_figures(rows, storage_rows, figures)
    write_summary_md(out / "summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if not args.memory_counts or any(n < len(PROBES) * 2 for n in args.memory_counts):
        raise ValueError("memory counts must leave room for controlled valid/stale memories")
    if any(k <= 0 for k in args.top_k) or args.repeat <= 0:
        raise ValueError("top-k and repeat must be positive")
    if args.local_only and args.embedding_backend == "sentence-transformers":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def build_encoder(args: argparse.Namespace):
    if args.embedding_backend == "hash":
        return HashEncoder(args.embedding_dim)
    return SentenceTransformerEncoder(args.embedding_model, local_only=args.local_only)


def build_bank(args, encoder, memory_count: int, seed: int, process):
    random.seed(seed)
    bank = MemoryBank(args.embedding_dim, args.forgetting_threshold, embedding_model=encoder,
                      decay_interval_hours=args.retention_time_unit_hours)
    controlled = [
        ("python_book", "The recommended Python book is Automate the Boring Stuff.", "2026-07-01 09:00", MemoryStatus.ACTIVE),
        ("current_food", "I currently avoid spicy food because of my stomach.", "2026-07-10 09:00", MemoryStatus.ACTIVE),
        ("stale_food", "I used to prefer very spicy hotpot.", "2026-01-01 09:00", MemoryStatus.SUPERSEDED),
        ("current_sport", "I currently enjoy swimming and basketball.", "2026-07-09 09:00", MemoryStatus.ACTIVE),
        ("stale_sport", "I previously trained for marathon running.", "2026-02-01 09:00", MemoryStatus.SUPERSEDED),
        ("current_city", "I currently live in Tokyo.", "2026-07-08 09:00", MemoryStatus.ACTIVE),
        ("stale_city", "I used to live in Osaka.", "2025-01-01 09:00", MemoryStatus.SUPERSEDED),
        ("extra", "I enjoy reading local technology news.", "2026-07-05 09:00", MemoryStatus.ACTIVE),
    ]
    labels: dict[str, str] = {}
    write_ns = 0
    embed_index_ns = 0
    peak = process.memory_info().rss
    for key, text, timestamp, status in controlled:
        started = time.perf_counter_ns()
        mem = bank.store_dialog("User", text, timestamp)
        elapsed = time.perf_counter_ns() - started
        embed_index_ns += elapsed
        labels[key] = mem.memory_id
        mem.status = status
        peak = max(peak, process.memory_info().rss)
    for index in range(memory_count - len(controlled)):
        text = f"Neutral local note {index:05d} about project topic {index % 31} and schedule day {index % 7}."
        started = time.perf_counter_ns()
        bank.store_dialog("User", text, f"2026-06-{1 + index % 28:02d} 08:00")
        embed_index_ns += time.perf_counter_ns() - started
        peak = max(peak, process.memory_info().rss)
    # The current implementation embeds and updates IndexFlatIP during each store.
    write_ns = embed_index_ns
    return bank, labels, {
        "memory_write_ms": write_ns / 1e6,
        "embedding_and_index_build_ms": embed_index_ns / 1e6,
        "embedding_ms": embed_index_ns / 1e6,
        "index_build_ms": 0.0,
    }, peak


def persist_bank(bank: MemoryBank, directory: Path) -> dict[str, int]:
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / "memorybank"
    bank.save(str(prefix))
    memory_path = directory / "memory_data.jsonl"
    metadata_path = directory / "metadata.json"
    embedding_path = directory / "embeddings.npy"
    index_path = directory / "memorybank.faiss"
    write_jsonl(memory_path, [{"memory_id": m.memory_id, "content": m.content} for m in bank.get_all()])
    write_json(metadata_path, {"memories": [m.to_dict() for m in bank.get_all()]})
    vectors = np.stack([np.asarray(m.embedding, dtype=np.float32) for m in bank.get_all()])
    np.save(embedding_path, vectors)
    sizes = {
        "memory_file_bytes": memory_path.stat().st_size,
        "metadata_file_bytes": metadata_path.stat().st_size,
        "embedding_file_bytes": embedding_path.stat().st_size,
        "faiss_index_file_bytes": index_path.stat().st_size,
    }
    sizes["storage_total_bytes"] = sum(sizes.values())
    return sizes


def estimate_tokens(text: str) -> int:
    ascii_words = len([token for token in text.split() if token])
    non_ascii = sum(1 for char in text if ord(char) >= 128 and not char.isspace())
    return ascii_words + non_ascii


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def summarize(rows, storage_rows):
    groups = []
    for n in sorted({r["memory_count"] for r in rows}):
        for k in sorted({r["top_k"] for r in rows}):
            subset = [r for r in rows if r["memory_count"] == n and r["top_k"] == k]
            latency = [r["retrieval_latency_ms"] for r in subset]
            groups.append({
                "memory_count": n, "top_k": k, "queries": len(subset),
                "recall_at_k": statistics.fmean(r["recall_at_k"] for r in subset),
                "valid_recall_at_k": statistics.fmean(r["valid_recall_at_k"] for r in subset),
                "stale_retrieval_rate": statistics.fmean(r["stale_retrieval_rate"] for r in subset),
                "mean_retrieval_latency_ms": statistics.fmean(latency),
                "p50_retrieval_latency_ms": percentile(latency, 50),
                "p95_retrieval_latency_ms": percentile(latency, 95),
                "mean_prompt_token_estimate": statistics.fmean(r["prompt_token_estimate"] for r in subset),
            })
    return {"metric_definitions": {
        "recall_at_k": "fraction of probes whose controlled relevant memory is in Top-K",
        "valid_recall_at_k": "fraction whose current valid memory is in Top-K",
        "stale_retrieval_rate": "stale controlled memories divided by retrieved memories",
        "prompt_token_estimate": "whitespace words plus non-ASCII characters; not a model tokenizer",
    }, "by_memory_count_and_top_k": groups}


def summarize_resources(storage_rows, rows, out):
    groups = []
    for n in sorted({r["memory_count"] for r in storage_rows}):
        subset = [r for r in storage_rows if r["memory_count"] == n]
        groups.append({"memory_count": n, **{
            key: statistics.fmean(float(r[key]) for r in subset)
            for key in ("memory_write_ms", "embedding_ms", "index_build_ms", "memory_file_bytes",
                        "metadata_file_bytes", "embedding_file_bytes", "faiss_index_file_bytes",
                        "storage_total_bytes", "baseline_rss_bytes", "index_loaded_rss_bytes",
                        "index_build_peak_rss_bytes")
        }})
    return {"by_memory_count": groups, "log_file_bytes": sum(
        p.stat().st_size for p in out.glob("*.jsonl") if p.is_file()
    )}


def environment_info(args, encoder):
    vm = psutil.virtual_memory()
    versions = {}
    for name in ("numpy", "faiss-cpu", "psutil", "matplotlib", "sentence-transformers"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {
        "os": platform.platform(), "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": psutil.cpu_count(), "ram_total_bytes": vm.total,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "not_configured"),
        "python": platform.python_version(), "dependencies": versions,
        "embedding_backend": args.embedding_backend, "embedding_model": encoder.name,
        "local_only": args.local_only, "network_calls": 0, "cloud_api_calls": 0,
        "cloud_database_used": False,
    }


def create_figures(rows, storage_rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resources = summarize_resources(storage_rows, rows, directory.parent)["by_memory_count"]
    metrics = summarize(rows, storage_rows)["by_memory_count_and_top_k"]
    ns = sorted({x["memory_count"] for x in metrics})
    def line(name, ys, xlabel, ylabel):
        plt.figure(figsize=(6.4, 4.0)); plt.plot(ns, ys, marker="o")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(directory / name, dpi=160); plt.close()
    best = [next(x for x in metrics if x["memory_count"] == n and x["top_k"] == max(y["top_k"] for y in metrics)) for n in ns]
    line("memory_count_mean_latency.png", [x["mean_retrieval_latency_ms"] for x in best], "Memory count", "Mean retrieval latency (ms)")
    line("memory_count_p95_latency.png", [x["p95_retrieval_latency_ms"] for x in best], "Memory count", "P95 retrieval latency (ms)")
    line("memory_count_index_size.png", [x["faiss_index_file_bytes"] for x in resources], "Memory count", "FAISS index bytes")
    line("memory_count_peak_memory.png", [x["index_build_peak_rss_bytes"] for x in resources], "Memory count", "Peak RSS bytes")
    line("memory_count_stale_rate.png", [x["stale_retrieval_rate"] for x in best], "Memory count", "Stale retrieval rate")
    plt.figure(figsize=(6.4, 4.0))
    for n in ns:
        sub = [x for x in metrics if x["memory_count"] == n]
        plt.plot([x["top_k"] for x in sub], [x["mean_prompt_token_estimate"] for x in sub], marker="o", label=str(n))
    plt.xlabel("Top-K"); plt.ylabel("Prompt token estimate"); plt.legend(title="Memories"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(directory / "topk_prompt_tokens.png", dpi=160); plt.close()
    times = np.arange(0, 11); plt.figure(figsize=(6.4, 4.0))
    for strength in (1, 2, 3): plt.plot(times, np.exp(-times / strength), label=f"S={strength}")
    plt.xlabel("Elapsed retention time units"); plt.ylabel("Retention R"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(directory / "strength_retention_example.png", dpi=160); plt.close()


def write_summary_md(path, summary):
    lines = ["# On-device MemoryBank Core Baseline", "", "- Status: success", "- Local-only: true",
             "- Cloud/API calls: 0", "- Scope: memory-system baseline; no local LLM answer generation", "",
             "The run uses local files, local embeddings and FAISS. Threshold-based forgotten candidates are",
             "reported as a project evaluation policy; this runner does not apply the non-paper `strength *= 0.5` extension."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path, rows):
    if not rows: return path.write_text("", encoding="utf-8")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows: writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


if __name__ == "__main__":
    raise SystemExit(main())
