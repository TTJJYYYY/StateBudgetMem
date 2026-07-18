#!/usr/bin/env python3
"""Build the final local-only showcase for StateBudgetMem.

The generated HTML is a presentation and analysis entry point. It reads formal
numbers from ``results/fair_comparison`` but keeps the dialogue and explorer
case clearly labeled as demo material.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_OUTPUT_DIR = ROOT / "results" / "final_showcase"
DEFAULT_FAIR_RESULTS_DIR = ROOT / "results" / "fair_comparison"
DEFAULT_RESOURCE_DIR = ROOT / "results" / "ondevice_memorybank" / "baseline_run"

from statebudgetmem.answering import (
    AnswerRequest,
    AnswerResult,
    LocalLLMAnswerer,
    LocalLLMUnavailable,
    TemplateAnswerer,
)


METHOD_GROUPS = {
    "tfidf_topk": "baseline",
    "memorybank_core": "baseline",
    "memorybank_versioning": "proposed variant",
    "memorybank_dual_views": "proposed variant",
    "statebudgetmem_rule": "proposed variant",
    "statebudgetmem_oracle": "oracle upper bound",
}

EXPECTED_FORMAL_METHODS = tuple(METHOD_GROUPS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the static StateBudgetMem final showcase.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fair-results-dir", type=Path, default=DEFAULT_FAIR_RESULTS_DIR)
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument(
        "--answerer",
        choices=("template", "local_llm"),
        default="template",
        help="Optional answerer for the demo-only Case Entry and MemoryExplorer.",
    )
    parser.add_argument("--local-llm-model", default="qwen2.5:3b")
    parser.add_argument(
        "--local-llm-endpoint",
        default="http://localhost:11434/api/generate",
    )
    parser.add_argument("--local-llm-timeout-s", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_final_showcase(
        output_dir=args.output_dir,
        fair_results_dir=args.fair_results_dir,
        resource_dir=args.resource_dir,
        answerer=args.answerer,
        local_llm_model=args.local_llm_model,
        local_llm_endpoint=args.local_llm_endpoint,
        local_llm_timeout_s=args.local_llm_timeout_s,
    )
    print("StateBudgetMem final showcase built:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    return 0


def build_final_showcase(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fair_results_dir: Path = DEFAULT_FAIR_RESULTS_DIR,
    resource_dir: Path = DEFAULT_RESOURCE_DIR,
    answerer: str = "template",
    local_llm_model: str = "qwen2.5:3b",
    local_llm_endpoint: str = "http://localhost:11434/api/generate",
    local_llm_timeout_s: float = 30.0,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard_data = build_experiment_dashboard_data(
        fair_results_dir=fair_results_dir,
        resource_dir=resource_dir,
    )
    showcase_data = build_showcase_data(
        dashboard_data=dashboard_data,
        answerer=answerer,
        local_llm_model=local_llm_model,
        local_llm_endpoint=local_llm_endpoint,
        local_llm_timeout_s=local_llm_timeout_s,
    )

    showcase_path = output_dir / "showcase_data.json"
    dashboard_path = output_dir / "experiment_dashboard_data.json"
    index_path = output_dir / "index.html"
    readme_path = output_dir / "README.md"

    write_json(showcase_path, showcase_data)
    write_json(dashboard_path, dashboard_data)
    index_path.write_text(render_html(showcase_data, dashboard_data), encoding="utf-8")
    readme_path.write_text(render_readme(showcase_data, dashboard_data), encoding="utf-8")

    return {
        "index": str(index_path),
        "showcase_data": str(showcase_path),
        "experiment_dashboard_data": str(dashboard_path),
        "readme": str(readme_path),
    }


def build_showcase_data(
    *,
    dashboard_data: dict[str, Any],
    answerer: str = "template",
    local_llm_model: str = "qwen2.5:3b",
    local_llm_endpoint: str = "http://localhost:11434/api/generate",
    local_llm_timeout_s: float = 30.0,
) -> dict[str, Any]:
    resource_panel = build_resource_panel(dashboard_data)
    data = {
        "metadata": {
            "title": "StateBudgetMem Final Showcase",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "local static showcase; no cloud API; optional local LLM for demo only",
            "formal_conclusion_source": "results/fair_comparison",
        },
        "case_entry": {
            "label": "Demo-only entrance",
            "claim_boundary": (
                "This dialogue is an entry case for explanation. It is not used as "
                "a formal performance metric."
            ),
            "conversation": [
                {
                    "time": "2026-03-01",
                    "speaker": "user",
                    "text": "我喜欢吃辣，尤其是川菜和火锅。",
                    "memory_id": "D1",
                },
                {
                    "time": "2026-05-10",
                    "speaker": "user",
                    "text": "最近胃不舒服，医生建议这段时间少吃辣。",
                    "memory_id": "D2",
                },
                {
                    "time": "2026-07-15",
                    "speaker": "user",
                    "text": "今晚适合吃什么？",
                    "memory_id": "Q_CURRENT",
                },
            ],
            "answers": [
                {
                    "method": "MemoryBank baseline",
                    "answer": (
                        "你之前喜欢辣味和火锅，可以考虑川菜或微辣火锅。"
                    ),
                    "cited_memory_ids": ["D1"],
                    "risk": "Uses an older preference as if it were current.",
                },
                {
                    "method": "StateBudgetMem",
                    "answer": (
                        "你长期偏好辣味，但当前有胃部不适；今晚更适合清淡、少油、少辣的晚餐。"
                    ),
                    "cited_memory_ids": ["D1", "D2"],
                    "risk": "Separates historical preference from current constraint.",
                },
            ],
        },
        "memory_explorer": {
            "label": "Showcase and analysis tool, not a formal experiment",
            "memories": [
                {
                    "memory_id": "D1",
                    "time": "2026-03-01",
                    "text": "用户喜欢吃辣，尤其是川菜和火锅。",
                    "status_by_current_query": "STALE",
                    "status_by_historical_query": "HISTORICAL",
                    "status_by_change_query": "CHANGE",
                    "valid_from": "2026-03-01",
                    "valid_to": "2026-05-10",
                    "token_cost": 12,
                    "operation": "ADD",
                },
                {
                    "memory_id": "D2",
                    "time": "2026-05-10",
                    "text": "用户最近胃不舒服，当前应少吃辣。",
                    "status_by_current_query": "CURRENT",
                    "status_by_historical_query": "HISTORICAL",
                    "status_by_change_query": "CHANGE",
                    "valid_from": "2026-05-10",
                    "valid_to": None,
                    "token_cost": 14,
                    "operation": "TEMP_INVALIDATE",
                    "temporarily_invalidates": ["D1"],
                },
                {
                    "memory_id": "D3",
                    "time": "2026-06-01",
                    "text": "用户工作日晚餐通常在家简单解决。",
                    "status_by_current_query": "CURRENT",
                    "status_by_historical_query": "HISTORICAL",
                    "status_by_change_query": "HISTORICAL",
                    "valid_from": "2026-06-01",
                    "valid_to": None,
                    "token_cost": 10,
                    "operation": "ADD",
                },
            ],
            "queries": [
                {
                    "query_id": "Q_CURRENT",
                    "query_type": "CURRENT",
                    "text": "今晚适合吃什么？",
                    "memorybank_retrieved": ["D1", "D2", "D3"],
                    "statebudgetmem_retrieved": ["D2", "D3"],
                    "answer_memory_ids": ["D2", "D3"],
                    "statebudgetmem_answer": "当前应少吃辣，可选清淡汤面、粥或低油家常菜。",
                },
                {
                    "query_id": "Q_HISTORICAL",
                    "query_type": "HISTORICAL",
                    "text": "五月之前我是不是喜欢吃辣？",
                    "memorybank_retrieved": ["D1", "D2"],
                    "statebudgetmem_retrieved": ["D1"],
                    "answer_memory_ids": ["D1"],
                    "statebudgetmem_answer": "是的，五月之前的历史偏好是喜欢辣味和火锅。",
                },
                {
                    "query_id": "Q_CHANGE",
                    "query_type": "CHANGE",
                    "text": "我的饮食偏好发生了什么变化？",
                    "memorybank_retrieved": ["D1", "D2", "D3"],
                    "statebudgetmem_retrieved": ["D1", "D2"],
                    "answer_memory_ids": ["D1", "D2"],
                    "statebudgetmem_answer": "从喜欢辣味，变化为因胃部不适而当前少吃辣。",
                },
            ],
            "resource_panel": resource_panel,
        },
    }
    apply_demo_answerer(
        data,
        answerer=answerer,
        local_llm_model=local_llm_model,
        local_llm_endpoint=local_llm_endpoint,
        local_llm_timeout_s=local_llm_timeout_s,
    )
    return data


def apply_demo_answerer(
    showcase_data: dict[str, Any],
    *,
    answerer: str,
    local_llm_model: str,
    local_llm_endpoint: str,
    local_llm_timeout_s: float,
) -> None:
    memories_by_id = {
        memory["memory_id"]: memory
        for memory in showcase_data["memory_explorer"]["memories"]
    }
    template = TemplateAnswerer()
    local_unavailable_reason: str | None = None
    latest_result: AnswerResult | None = None

    for query in showcase_data["memory_explorer"]["queries"]:
        retrieved = [
            memories_by_id[memory_id]
            for memory_id in query["statebudgetmem_retrieved"]
            if memory_id in memories_by_id
        ]
        prompt = build_demo_augmented_prompt(query["text"], retrieved)
        request = AnswerRequest(
            query=query["text"],
            retrieved_memories=retrieved,
            augmented_prompt=prompt,
            metadata={
                "source_name": "final_showcase_demo",
                "global_summary": "The user used to like spicy food, but currently has stomach discomfort and should eat lightly.",
                "reference_answer": query.get("statebudgetmem_answer", ""),
                "prefer_reference_answer": True,
            },
        )
        template_result = template.answer(request)
        query["template_answer"] = template_result.answer_text
        query["answerer_result"] = template_result.to_dict()
        query["answerer_type"] = template_result.answerer_type
        query["model_name"] = template_result.model_name

        if answerer == "local_llm" and local_unavailable_reason is None:
            try:
                local_result = LocalLLMAnswerer(
                    model_name=local_llm_model,
                    endpoint=local_llm_endpoint,
                    timeout_s=local_llm_timeout_s,
                    temperature=0.0,
                ).answer(request)
                query["local_llm_answer"] = local_result.answer_text
                query["statebudgetmem_answer"] = local_result.answer_text
                query["answerer_result"] = local_result.to_dict()
                query["answerer_type"] = local_result.answerer_type
                query["model_name"] = local_result.model_name
                latest_result = local_result
            except (LocalLLMUnavailable, ValueError) as exc:
                local_unavailable_reason = str(exc)
                query["local_llm_answer"] = None
                query["local_llm_unavailable_reason"] = local_unavailable_reason
                query["statebudgetmem_answer"] = template_result.answer_text
                query["answerer_type"] = "template_fallback"
                fallback_payload = template_result.to_dict()
                fallback_payload["answerer_type"] = "template_fallback"
                fallback_payload["metadata"]["fallback_reason"] = local_unavailable_reason
                fallback_payload["metadata"]["fallback_to"] = "template"
                query["answerer_result"] = fallback_payload
                latest_result = template_result
        else:
            query["statebudgetmem_answer"] = template_result.answer_text
            latest_result = template_result
            if local_unavailable_reason is not None:
                query["local_llm_answer"] = None
                query["local_llm_unavailable_reason"] = local_unavailable_reason
                query["answerer_type"] = "template_fallback"

    current_query = showcase_data["memory_explorer"]["queries"][0]
    statebudgetmem_answer = next(
        item
        for item in showcase_data["case_entry"]["answers"]
        if item["method"] == "StateBudgetMem"
    )
    statebudgetmem_answer["answer"] = current_query["statebudgetmem_answer"]
    statebudgetmem_answer["cited_memory_ids"] = current_query["answer_memory_ids"]
    statebudgetmem_answer["answerer_type"] = current_query["answerer_type"]
    statebudgetmem_answer["model_name"] = current_query["model_name"]

    panel = showcase_data["memory_explorer"]["resource_panel"]
    panel.update(
        {
            "answerer_requested": answerer,
            "answerer_type": current_query["answerer_type"],
            "model_name": current_query["model_name"],
            "generation_latency_ms": current_query["answerer_result"]["latency_ms"],
            "tokens_per_second": current_query["answerer_result"]["tokens_per_second"],
            "prompt_tokens": current_query["answerer_result"]["prompt_tokens"],
            "generated_tokens": current_query["answerer_result"]["generated_tokens"],
            "local_llm_endpoint": local_llm_endpoint if answerer == "local_llm" else None,
            "local_llm_unavailable_reason": local_unavailable_reason,
            "local_only": True,
            "cloud_api_calls": 0,
        }
    )
    showcase_data["metadata"]["answerer"] = {
        "requested": answerer,
        "actual": current_query["answerer_type"],
        "model_name": current_query["model_name"],
        "local_only": True,
        "cloud_api_calls": 0,
        "local_llm_unavailable_reason": local_unavailable_reason,
        "latest_result": latest_result.to_dict() if latest_result is not None else None,
    }


def build_demo_augmented_prompt(query: str, memories: list[dict[str, Any]]) -> str:
    memory_text = "\n".join(
        f"[{memory['memory_id']}] {memory['text']}" for memory in memories
    )
    return (
        "You are a local on-device demo answerer. Use only the provided memories. "
        "Separate current constraints from historical preferences.\n\n"
        f"Memories:\n{memory_text}\n\n"
        f"User query:\n{query}\n\n"
        "Answer briefly in Chinese."
    )


def build_experiment_dashboard_data(
    *,
    fair_results_dir: Path,
    resource_dir: Path,
) -> dict[str, Any]:
    summary = read_json(fair_results_dir / "summary.json", default={"methods": []})
    method_summary = read_json(
        fair_results_dir / "method_summary.json",
        default=summary.get("methods", []),
    )
    if not method_summary:
        method_summary = summarize_per_query_results(
            fair_results_dir / "per_query_results.jsonl"
        )
    config = read_json(fair_results_dir / "run_config_resolved.json", default={})
    environment = read_json(fair_results_dir / "environment.json", default={})
    resource_metrics = read_json(resource_dir / "memorybank_resource_metrics.json", default={})
    topk_metrics = read_json(resource_dir / "metrics.json", default={})

    methods = []
    for row in method_summary:
        methods.append(
            {
                "method": row["method"],
                "group": METHOD_GROUPS.get(row["method"], "other"),
                "recall_at_k": row.get("mean_recall_at_k"),
                "valid_recall_at_k": row.get("mean_valid_recall_at_k"),
                "stale_retrieval_rate": row.get("mean_stale_retrieval_rate"),
                "token_proxy": row.get("mean_prompt_token_proxy", row.get("mean_total_token_cost")),
                "retrieval_latency_ms": row.get("mean_retrieval_latency_ms"),
                "eligible_count": row.get("mean_eligible_count"),
                "candidate_count": row.get("mean_candidate_count"),
                "query_count": row.get("query_count"),
                "top_k": row.get("top_k"),
                "candidate_k": row.get("candidate_k"),
                "token_budget": row.get("token_budget"),
            }
        )

    method_names = {method["method"] for method in methods}
    missing_methods = [
        method for method in EXPECTED_FORMAL_METHODS if method not in method_names
    ]
    return {
        "metadata": {
            "title": "Fair Experiment Dashboard Data",
            "formal_source_dir": str(fair_results_dir),
            "resource_source_dir": str(resource_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expected_methods": list(EXPECTED_FORMAL_METHODS),
            "missing_methods": missing_methods,
            "conclusion_boundary": (
                "Formal conclusions come from the unified fair-comparison runner. "
                "Case Entry and MemoryExplorer are display and analysis tools."
            ),
        },
        "config": {
            "dataset_path": config.get("dataset_path", summary.get("dataset_path")),
            "config_path": config.get("config_path"),
            "embedding_backend": config.get("embedding_backend"),
            "embedding_model": config.get("embedding_model"),
            "top_k": config.get("top_k"),
            "candidate_k": config.get("candidate_k"),
            "token_budget": config.get("token_budget"),
            "random_seed": config.get("random_seed"),
            "query_state_policy": config.get("query_state_policy"),
            "run_id": config.get("run_id"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
        },
        "methods": methods,
        "resource_metrics": {
            "by_memory_count": resource_metrics.get("by_memory_count", []),
            "log_file_bytes": resource_metrics.get("log_file_bytes"),
            "topk_sweep": topk_metrics.get("by_memory_count_and_top_k", []),
            "metric_definitions": topk_metrics.get("metric_definitions", {}),
            "note": (
                "Resource and top-k sweep data are on-device MemoryBank resource "
                "measurements. They support the endpoint story but are separate "
                "from the fair comparison table unless rerun through the same runner."
            ),
        },
        "environment": environment,
        "method_groups": METHOD_GROUPS,
    }


def summarize_per_query_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("status") == "success":
                rows.append(payload)

    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for method in EXPECTED_FORMAL_METHODS:
        selected = by_method.get(method, [])
        if not selected:
            continue
        summaries.append(
            {
                "method": method,
                "mean_recall_at_k": mean_numeric(selected, "recall_at_k"),
                "mean_valid_recall_at_k": mean_numeric(
                    selected, "valid_recall_at_k"
                ),
                "mean_stale_retrieval_rate": mean_numeric(
                    selected, "stale_retrieval_rate"
                ),
                "mean_total_token_cost": mean_numeric(
                    selected, "total_token_cost"
                ),
                "mean_retrieval_latency_ms": mean_numeric(
                    selected, "retrieval_latency_ms"
                ),
                "mean_eligible_count": mean_numeric(
                    selected, "eligible_memory_count"
                ),
                "mean_candidate_count": mean_numeric(
                    selected, "candidate_count_after_scope"
                ),
                "query_count": len(selected),
                "top_k": single_value(selected, "top_k"),
                "candidate_k": single_value(selected, "candidate_k"),
                "token_budget": single_value(selected, "token_budget"),
            }
        )
    return summaries


def mean_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def single_value(rows: list[dict[str, Any]], field: str) -> Any:
    values = {row.get(field) for row in rows if row.get(field) is not None}
    if len(values) == 1:
        return next(iter(values))
    return None


def build_resource_panel(dashboard_data: dict[str, Any]) -> dict[str, Any]:
    resource_rows = dashboard_data.get("resource_metrics", {}).get("by_memory_count", [])
    latest = resource_rows[-1] if resource_rows else {}
    topk_rows = dashboard_data.get("resource_metrics", {}).get("topk_sweep", [])
    topk3 = [row for row in topk_rows if row.get("top_k") == 3]
    latency_row = topk3[-1] if topk3 else (topk_rows[-1] if topk_rows else {})
    config = dashboard_data.get("config", {})
    return {
        "local_only": True,
        "cloud_api_calls": 0,
        "embedding": config.get("embedding_model") or "local hash / MiniLM embedding",
        "vector_index": "local FAISS IndexFlatIP",
        "storage": "local JSONL / metadata / embedding files",
        "retrieval_latency_ms": latency_row.get("mean_retrieval_latency_ms")
        or latest.get("memory_write_ms"),
        "storage_bytes": latest.get("storage_total_bytes"),
        "token_budget": config.get("token_budget"),
        "token_proxy_metric": "memory_record_token_cost / prompt token proxy",
        "memory_count": latest.get("memory_count"),
    }


def render_html(showcase_data: dict[str, Any], dashboard_data: dict[str, Any]) -> str:
    payload = {
        "showcase": showcase_data,
        "dashboard": dashboard_data,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    escaped_json = data_json.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StateBudgetMem Final Showcase</title>
  <style>
    :root {{
      --ink: #17212b;
      --muted: #5b6673;
      --line: #d9e0e8;
      --bg: #f7f9fb;
      --panel: #ffffff;
      --blue: #2367c8;
      --green: #158057;
      --red: #c2413b;
      --amber: #b47109;
      --violet: #6a55b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }}
    header {{
      padding: 26px 36px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    p {{ margin: 0 0 10px; color: var(--muted); line-height: 1.55; }}
    nav {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    nav a {{
      color: var(--ink);
      text-decoration: none;
      border: 1px solid var(--line);
      padding: 8px 12px;
      border-radius: 6px;
      background: #f9fbfd;
      font-size: 14px;
    }}
    main {{ padding: 24px 36px 42px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 20px;
      box-shadow: 0 1px 2px rgba(20, 31, 43, 0.05);
    }}
    .notice {{
      border-left: 4px solid var(--amber);
      background: #fff8eb;
      padding: 10px 12px;
      color: #5f430d;
      margin-bottom: 14px;
    }}
    .grid {{ display: grid; gap: 16px; }}
    .two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
    }}
    .conversation {{
      display: grid;
      grid-template-columns: 112px 1fr;
      gap: 10px 14px;
      align-items: start;
      margin-top: 8px;
    }}
    .time {{ color: var(--muted); font-size: 13px; padding-top: 6px; }}
    .bubble {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 12px;
      color: var(--muted);
      margin: 2px 4px 2px 0;
      white-space: nowrap;
    }}
    .baseline {{ border-color: #cfd8e5; }}
    .proposed {{ border-color: #b8dcca; }}
    .oracle {{ border-color: #cfc7eb; }}
    .timeline {{
      display: grid;
      gap: 12px;
      margin: 12px 0;
    }}
    .memory {{
      display: grid;
      grid-template-columns: 92px 1fr 130px;
      gap: 12px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .memory.highlight {{ outline: 3px solid rgba(21, 128, 87, 0.22); }}
    .status-CURRENT {{ color: var(--green); border-color: rgba(21,128,87,.4); }}
    .status-HISTORICAL {{ color: var(--blue); border-color: rgba(35,103,200,.4); }}
    .status-CHANGE {{ color: var(--violet); border-color: rgba(106,85,184,.4); }}
    .status-STALE {{ color: var(--red); border-color: rgba(194,65,59,.4); }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 9px 12px;
      cursor: pointer;
      color: var(--ink);
    }}
    button.active {{
      border-color: var(--blue);
      background: #eaf2ff;
      color: #174e9c;
      font-weight: 600;
    }}
    .method-list {{ display: grid; gap: 8px; }}
    .retrieval-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid #edf1f5;
      padding: 7px 0;
      font-size: 14px;
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .metric-table th, .metric-table td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: middle;
    }}
    .metric-table th {{ color: var(--muted); font-weight: 600; }}
    .bar {{
      height: 10px;
      background: #e7edf4;
      border-radius: 999px;
      overflow: hidden;
      min-width: 90px;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: var(--blue);
      border-radius: 999px;
    }}
    .bar.valid > span {{ background: var(--green); }}
    .bar.stale > span {{ background: var(--red); }}
    .resource-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .resource {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }}
    .resource strong {{ display: block; font-size: 18px; margin-top: 5px; }}
    .small {{ font-size: 13px; color: var(--muted); }}
    @media (max-width: 900px) {{
      main, header {{ padding-left: 18px; padding-right: 18px; }}
      .two, .three, .resource-grid {{ grid-template-columns: 1fr; }}
      .memory {{ grid-template-columns: 1fr; }}
      .conversation {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>StateBudgetMem Final Showcase</h1>
    <p>三层展示结构：简短端到端入口、MemoryExplorer 分析过程、统一 runner 的公平实验 Dashboard。</p>
    <nav>
      <a href="#case-entry">Case Entry</a>
      <a href="#memory-explorer">MemoryExplorer</a>
      <a href="#experiment-dashboard">Experiment Dashboard</a>
      <a href="#on-device">On-device Panel</a>
    </nav>
  </header>
  <main>
    <section id="case-entry"></section>
    <section id="memory-explorer"></section>
    <section id="experiment-dashboard"></section>
    <section id="on-device"></section>
  </main>
  <script id="showcase-payload" type="application/json">{escaped_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('showcase-payload').textContent);
    const showcase = payload.showcase;
    const dashboard = payload.dashboard;
    const fmt = (value, digits = 3) => value === null || value === undefined ? 'n/a' : Number(value).toFixed(digits);
    const pct = (value) => value === null || value === undefined ? 'n/a' : (Number(value) * 100).toFixed(1) + '%';
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]));

    function renderCaseEntry() {{
      const caseEntry = showcase.case_entry;
      const convo = caseEntry.conversation.map(item => `
        <div class="time">${{esc(item.time)}}</div>
        <div class="bubble"><span class="pill">${{esc(item.memory_id)}}</span>${{esc(item.text)}}</div>
      `).join('');
      const answers = caseEntry.answers.map(item => `
        <div class="card">
          <h3>${{esc(item.method)}}</h3>
          <p>${{esc(item.answer)}}</p>
          <p class="small">引用 memories: ${{item.cited_memory_ids.map(id => `<span class="pill">${{esc(id)}}</span>`).join('')}} </p>
          <p class="small">${{esc(item.risk)}}</p>
        </div>
      `).join('');
      document.getElementById('case-entry').innerHTML = `
        <h2>1. Case Entry</h2>
        <div class="notice">${{esc(caseEntry.claim_boundary)}}</div>
        <div class="grid two">
          <div>
            <h3>端到端入口案例</h3>
            <div class="conversation">${{convo}}</div>
          </div>
          <div>
            <h3>模板回答差异</h3>
            <div class="grid">${{answers}}</div>
          </div>
        </div>
      `;
    }}

    function statusFor(memory, queryType) {{
      if (queryType === 'CURRENT') return memory.status_by_current_query;
      if (queryType === 'HISTORICAL') return memory.status_by_historical_query;
      return memory.status_by_change_query;
    }}

    function renderExplorer(selectedId) {{
      const explorer = showcase.memory_explorer;
      const query = explorer.queries.find(item => item.query_id === selectedId) || explorer.queries[0];
      const queryButtons = explorer.queries.map(item => `
        <button class="${{item.query_id === query.query_id ? 'active' : ''}}" onclick="renderExplorer('${{item.query_id}}')">
          ${{esc(item.query_type)}}
        </button>
      `).join('');
      const timeline = explorer.memories.map(memory => {{
        const status = statusFor(memory, query.query_type);
        const highlighted = query.answer_memory_ids.includes(memory.memory_id) ? 'highlight' : '';
        return `
          <div class="memory ${{highlighted}}">
            <div><strong>${{esc(memory.time)}}</strong><br><span class="pill">${{esc(memory.memory_id)}}</span></div>
            <div>${{esc(memory.text)}}<br><span class="small">operation=${{esc(memory.operation)}}; token=${{esc(memory.token_cost)}}</span></div>
            <div><span class="pill status-${{esc(status)}}">${{esc(status)}}</span></div>
          </div>
        `;
      }}).join('');
      const methodBlock = (title, ids) => `
        <div class="card">
          <h3>${{esc(title)}}</h3>
          <div class="method-list">
            ${{ids.map((id, index) => `<div class="retrieval-row"><span>#${{index + 1}} <strong>${{esc(id)}}</strong></span><span>${{query.answer_memory_ids.includes(id) ? 'used in answer' : 'retrieved only'}}</span></div>`).join('')}}
          </div>
        </div>
      `;
      document.getElementById('memory-explorer').innerHTML = `
        <h2>2. MemoryExplorer</h2>
        <div class="notice">${{esc(explorer.label)}}</div>
        <div style="margin-bottom: 12px;">${{queryButtons}}</div>
        <p><strong>${{esc(query.query_type)}} query:</strong> ${{esc(query.text)}}</p>
        <div class="timeline">${{timeline}}</div>
        <div class="grid two">
          ${{methodBlock('MemoryBank retrieved memories', query.memorybank_retrieved)}}
          ${{methodBlock('StateBudgetMem retrieved memories', query.statebudgetmem_retrieved)}}
        </div>
        <div class="card" style="margin-top: 14px;">
          <h3>Answer trace</h3>
          <p>${{esc(query.statebudgetmem_answer)}}</p>
          <p class="small">answerer=${{esc(query.answerer_type)}}; model=${{esc(query.model_name)}}. Highlighted memories are the records cited by the answer.</p>
        </div>
      `;
    }}

    function renderDashboard() {{
      const methods = dashboard.methods;
      const maxLatency = Math.max(...methods.map(m => Number(m.retrieval_latency_ms || 0)), 1);
      const rows = methods.map(m => `
        <tr>
          <td><strong>${{esc(m.method)}}</strong><br><span class="pill ${{m.group.includes('oracle') ? 'oracle' : m.group.includes('proposed') ? 'proposed' : 'baseline'}}">${{esc(m.group)}}</span></td>
          <td>${{pct(m.recall_at_k)}}<div class="bar"><span style="width:${{Number(m.recall_at_k || 0) * 100}}%"></span></div></td>
          <td>${{pct(m.valid_recall_at_k)}}<div class="bar valid"><span style="width:${{Number(m.valid_recall_at_k || 0) * 100}}%"></span></div></td>
          <td>${{pct(m.stale_retrieval_rate)}}<div class="bar stale"><span style="width:${{Number(m.stale_retrieval_rate || 0) * 100}}%"></span></div></td>
          <td>${{fmt(m.token_proxy, 2)}}</td>
          <td>${{fmt(m.retrieval_latency_ms, 2)}} ms<div class="bar"><span style="width:${{Number(m.retrieval_latency_ms || 0) / maxLatency * 100}}%"></span></div></td>
        </tr>
      `).join('');
      const cfg = dashboard.config;
      document.getElementById('experiment-dashboard').innerHTML = `
        <h2>3. Experiment Dashboard</h2>
        <div class="notice">${{esc(dashboard.metadata.conclusion_boundary)}}</div>
        <p>正式结果目录：<strong>${{esc(dashboard.metadata.formal_source_dir)}}</strong></p>
        <p class="small">dataset=${{esc(cfg.dataset_path)}}; top_k=${{esc(cfg.top_k)}}; candidate_k=${{esc(cfg.candidate_k)}}; token_budget=${{esc(cfg.token_budget)}}; seed=${{esc(cfg.random_seed)}}; embedding=${{esc(cfg.embedding_model)}}.</p>
        <table class="metric-table">
          <thead><tr><th>Method</th><th>Recall@K</th><th>Valid Recall@K</th><th>Stale Retrieval Rate</th><th>Token Proxy</th><th>Retrieval Latency</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    function renderOnDevice() {{
      const panel = showcase.memory_explorer.resource_panel;
      const resourceRows = dashboard.resource_metrics.by_memory_count || [];
      const sweepRows = dashboard.resource_metrics.topk_sweep || [];
      const resourceCards = [
        ['local-only', panel.local_only ? 'true' : 'false'],
        ['cloud API calls', panel.cloud_api_calls],
        ['embedding', panel.embedding],
        ['vector index', panel.vector_index],
        ['storage', panel.storage],
        ['latency', fmt(panel.retrieval_latency_ms, 2) + ' ms'],
        ['generation latency', fmt(panel.generation_latency_ms, 2) + ' ms'],
        ['tokens/s', fmt(panel.tokens_per_second, 2)],
        ['answer model', panel.model_name || 'n/a'],
        ['storage bytes', panel.storage_bytes ? Math.round(panel.storage_bytes).toLocaleString() : 'n/a'],
        ['token budget', panel.token_budget || 'n/a'],
      ].map(([label, value]) => `<div class="resource"><span class="small">${{esc(label)}}</span><strong>${{esc(value)}}</strong></div>`).join('');
      const resourceRowsHtml = resourceRows.map(row => `
        <tr><td>${{esc(row.memory_count)}}</td><td>${{Math.round(row.storage_total_bytes || 0).toLocaleString()}}</td><td>${{Math.round(row.faiss_index_file_bytes || 0).toLocaleString()}}</td><td>${{Math.round(row.index_loaded_rss_bytes || 0).toLocaleString()}}</td></tr>
      `).join('');
      const sweepHtml = sweepRows.slice(0, 12).map(row => `
        <tr><td>${{esc(row.memory_count)}}</td><td>${{esc(row.top_k)}}</td><td>${{fmt(row.mean_prompt_token_estimate, 2)}}</td><td>${{fmt(row.mean_retrieval_latency_ms, 2)}} ms</td><td>${{pct(row.valid_recall_at_k)}}</td></tr>
      `).join('');
      document.getElementById('on-device').innerHTML = `
        <h2>On-device Resource Panel</h2>
        <div class="resource-grid">${{resourceCards}}</div>
        <div class="grid two" style="margin-top: 16px;">
          <div>
            <h3>Storage and index growth</h3>
            <table class="metric-table"><thead><tr><th>Memories</th><th>Storage bytes</th><th>FAISS bytes</th><th>Loaded RSS bytes</th></tr></thead><tbody>${{resourceRowsHtml}}</tbody></table>
          </div>
          <div>
            <h3>Available top-k/resource sweep</h3>
            <table class="metric-table"><thead><tr><th>Memories</th><th>Top-K</th><th>Token proxy</th><th>Latency</th><th>Valid recall</th></tr></thead><tbody>${{sweepHtml}}</tbody></table>
            <p class="small">${{esc(dashboard.resource_metrics.note)}}</p>
          </div>
        </div>
      `;
    }}

    renderCaseEntry();
    renderExplorer(showcase.memory_explorer.queries[0].query_id);
    renderDashboard();
    renderOnDevice();
  </script>
</body>
</html>
"""


def render_readme(showcase_data: dict[str, Any], dashboard_data: dict[str, Any]) -> str:
    config = dashboard_data["config"]
    answerer = showcase_data.get("metadata", {}).get("answerer", {})
    return f"""# StateBudgetMem Final Showcase

Open `index.html` directly in a browser. The page is self-contained and does
not require a local server, a cloud API, or a live LLM by default.

## Layers

1. Case Entry: a short spicy-food dialogue used only as a presentation entry.
2. MemoryExplorer: an interactive visualization of temporal memory status,
   retrieval differences, answer citations, and on-device resource signals.
3. Experiment Dashboard: formal fair-comparison metrics loaded from
   `{dashboard_data['metadata']['formal_source_dir']}`.

## Optional Local LLM

- Requested answerer: `{answerer.get('requested')}`
- Actual answerer: `{answerer.get('actual')}`
- Model: `{answerer.get('model_name')}`
- Local only: `{answerer.get('local_only')}`
- Cloud API calls: `{answerer.get('cloud_api_calls')}`

To try Ollama locally, run:

```powershell
.venv\\Scripts\\python.exe tools\\demo\\build_final_showcase.py --answerer local_llm --local-llm-model <model_name>
```

If Ollama or the requested model is unavailable, the page is still generated
with the Template Answer and records the fallback reason in `showcase_data.json`.

## Formal Experiment Source

- Dataset: `{config.get('dataset_path')}`
- Config: `{config.get('config_path')}`
- Run ID: `{config.get('run_id')}`
- Top-K: `{config.get('top_k')}`
- Candidate-K: `{config.get('candidate_k')}`
- Token budget: `{config.get('token_budget')}`
- Seed: `{config.get('random_seed')}`
- Embedding: `{config.get('embedding_model')}`

## Boundary

The Case Entry and MemoryExplorer are display and analysis tools. Formal
performance conclusions come from the unified runner outputs in
`results/fair_comparison/`.
"""


def read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
