#!/usr/bin/env python3
"""Build the final local-only showcase for StateBudgetMem.

The generated HTML is a presentation and analysis entry point. It reads formal
numbers from ``results/fair_comparison_v2`` by default but keeps the dialogue
and explorer cases clearly labeled as demo material.
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
DEFAULT_FAIR_RESULTS_DIR = ROOT / "results" / "fair_comparison_v2"
DEFAULT_RESOURCE_DIR = ROOT / "results" / "ondevice_memorybank" / "baseline_run"
DEFAULT_CONTROLLED_DATASET = ROOT / "data" / "controlled" / "temporal_challenge_v1.jsonl"

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

EXTRA_SHOWCASE_SCENARIO_IDS = (
    "S103_work_role",
    "S106_running_injury",
    "S108_caffeine_medication",
    "S111_business_trip_return",
    "S120_reading_format_distractors",
    "S125_weather_commute",
)


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
    fixed_cases = build_fixed_showcase_cases(resource_panel)
    entry_case = fixed_cases[0]
    data = {
        "metadata": {
            "title": "StateBudgetMem Final Showcase",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "local static showcase; no cloud API; optional local LLM for demo only",
            "formal_conclusion_source": "results/fair_comparison_v2",
        },
        "case_entry": {
            "label": "Demo-only entrance",
            "claim_boundary": (
                "This dialogue is an entry case for explanation. It is not used as "
                "a formal performance metric."
            ),
            "conversation": entry_case["conversation"],
            "answers": [
                {
                    "method": "MemoryBank baseline",
                    "answer": entry_case["queries"][0]["memorybank_wrong_answer"],
                    "cited_memory_ids": entry_case["queries"][0]["memorybank_retrieved"][:1],
                    "risk": entry_case["failure_mode"],
                },
                {
                    "method": "StateBudgetMem",
                    "answer": entry_case["queries"][0]["statebudgetmem_answer"],
                    "cited_memory_ids": entry_case["queries"][0]["answer_memory_ids"],
                    "risk": entry_case["why_statebudgetmem"],
                },
            ],
        },
        "fixed_cases": fixed_cases,
        "memory_explorer": entry_case,
        "free_question_demo": {
            "enabled": True,
            "default_case_id": entry_case["case_id"],
            "top_k": 3,
            "answerer": "browser_template_answerer",
            "retrieval_boundary": (
                "Free question demo is illustrative only. It compares context "
                "construction strategies over fixed demo memories in the browser "
                "and is not used for formal metrics."
            ),
            "retrieval_boundary_zh": (
                "自由提问区仅用于展示检索策略差异；它只在浏览器中复用固定案例记忆，"
                "不计入正式实验指标。正式结论来自 results/fair_comparison_v2。"
            ),
            "statebudgetmem_demo_name": "StateBudgetMem-style scoped retrieval demo",
            "llm_policy": (
                "Template answers are the default. Optional DeepSeek answers require "
                "the local final_showcase demo server; API keys are never written into "
                "the generated files."
            ),
            "deepseek_server_endpoint": "/api/free-question-answer",
            "deepseek_default_model": "deepseek-chat",
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


def build_fixed_showcase_cases(resource_panel: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fixed no-free-input cases for defense-friendly explanation."""

    primary_cases = [
        {
            "case_id": "temporary_invalidation",
            "label": "临时失效：口腔手术后暂时禁辣",
            "label_en": "Temporary invalidation",
            "source_scenario": "S105_spicy_oral_surgery",
            "label_note": "Showcase and analysis tool, not a formal experiment",
            "thesis": "长期偏好没有消失，但当前问题必须优先使用临时健康限制。",
            "failure_mode": "MemoryBank 容易把“平时可以微辣”当成今天仍然有效。",
            "why_statebudgetmem": "StateBudgetMem 将长期偏好保留为历史/底层状态，并把临时禁辣标为当前有效约束。",
            "conversation": [
                {
                    "time": "2026-01-12",
                    "speaker": "user",
                    "text": "用户平时可以接受微辣口味。",
                    "memory_id": "S105_M1",
                },
                {
                    "time": "2026-06-20",
                    "speaker": "user",
                    "text": "口腔手术后，医生要求六月二十日至七月五日避免辣味食物。",
                    "memory_id": "S105_M2",
                },
                {
                    "time": "2026-06-29",
                    "speaker": "user",
                    "text": "今天点菜时辣度应该怎么选？",
                    "memory_id": "S105_Q_CURRENT",
                },
            ],
            "memories": [
                {
                    "memory_id": "S105_M1",
                    "time": "2026-01-12",
                    "text": "用户平时可以接受微辣口味。",
                    "status_by_current_query": "STALE",
                    "status_by_historical_query": "CURRENT",
                    "status_by_change_query": "HISTORICAL",
                    "valid_from": "2026-01-12",
                    "valid_to": None,
                    "token_cost": 8,
                    "operation": "ADD",
                },
                {
                    "memory_id": "S105_M2",
                    "time": "2026-06-20",
                    "text": "口腔手术后，医生要求用户六月二十日至七月五日避免辣味食物。",
                    "status_by_current_query": "CURRENT",
                    "status_by_historical_query": "STALE",
                    "status_by_change_query": "CURRENT",
                    "valid_from": "2026-06-20",
                    "valid_to": "2026-07-05",
                    "token_cost": 17,
                    "operation": "TEMP_INVALIDATE",
                    "temporarily_invalidates": ["S105_M1"],
                },
                {
                    "memory_id": "S105_M3",
                    "time": "2026-02-01",
                    "text": "用户一直很喜欢川菜的香味。",
                    "status_by_current_query": "HISTORICAL",
                    "status_by_historical_query": "HISTORICAL",
                    "status_by_change_query": "HISTORICAL",
                    "valid_from": "2026-02-01",
                    "valid_to": None,
                    "token_cost": 8,
                    "operation": "ADD",
                },
            ],
            "queries": [
                {
                    "query_id": "S105_Q_CURRENT",
                    "query_type": "CURRENT",
                    "text": "今天点菜时辣度应该怎么选？",
                    "memorybank_retrieved": ["S105_M1", "S105_M2", "S105_M3"],
                    "statebudgetmem_retrieved": ["S105_M2"],
                    "answer_memory_ids": ["S105_M2"],
                    "memorybank_wrong_answer": "可以选择微辣，因为你平时能接受微辣。",
                    "statebudgetmem_answer": "今天处在术后禁辣期，应选择不辣或清淡口味。",
                },
                {
                    "query_id": "S105_Q_HISTORICAL",
                    "query_type": "HISTORICAL",
                    "text": "五月中旬我能接受什么辣度？",
                    "memorybank_retrieved": ["S105_M2", "S105_M1", "S105_M3"],
                    "statebudgetmem_retrieved": ["S105_M1", "S105_M3"],
                    "answer_memory_ids": ["S105_M1"],
                    "memorybank_wrong_answer": "五月中旬也应避免辣味食物。",
                    "statebudgetmem_answer": "五月中旬还未进入术后禁辣期，历史状态是可以接受微辣。",
                },
                {
                    "query_id": "S105_Q_CHANGE",
                    "query_type": "CHANGE",
                    "text": "我的辣度状态为什么暂时发生了变化？",
                    "memorybank_retrieved": ["S105_M1", "S105_M2", "S105_M3"],
                    "statebudgetmem_retrieved": ["S105_M1", "S105_M2"],
                    "answer_memory_ids": ["S105_M1", "S105_M2"],
                    "memorybank_wrong_answer": "你一直喜欢微辣，没有明显变化。",
                    "statebudgetmem_answer": "原本可以微辣，但术后医生要求短期禁辣，所以当前状态被临时覆盖。",
                },
            ],
            "resource_panel": resource_panel,
        },
        {
            "case_id": "permanent_supersede",
            "label": "永久替代：通勤方式从自驾改为地铁",
            "label_en": "Permanent supersede",
            "source_scenario": "S101_commute_mode",
            "label_note": "Showcase and analysis tool, not a formal experiment",
            "thesis": "新状态永久替代旧状态，当前推荐不能再使用旧通勤方式。",
            "failure_mode": "MemoryBank 可能同时召回开车和地铁，并把旧自驾习惯混入当前答案。",
            "why_statebudgetmem": "StateBudgetMem 用 SUPERSEDE 关闭旧状态的当前有效期，但仍允许历史查询访问它。",
            "conversation": [
                {
                    "time": "2026-01-03",
                    "speaker": "user",
                    "text": "今年一月至三月，用户工作日通常自己开车去公司。",
                    "memory_id": "S101_M1",
                },
                {
                    "time": "2026-04-01",
                    "speaker": "user",
                    "text": "从四月开始，用户工作日改为乘地铁上下班。",
                    "memory_id": "S101_M2",
                },
                {
                    "time": "2026-06-29",
                    "speaker": "user",
                    "text": "我现在工作日上班主要怎么去？",
                    "memory_id": "S101_Q_CURRENT",
                },
            ],
            "memories": [
                {
                    "memory_id": "S101_M1",
                    "time": "2026-01-03",
                    "text": "今年一月至三月，用户工作日通常自己开车去公司。",
                    "status_by_current_query": "STALE",
                    "status_by_historical_query": "CURRENT",
                    "status_by_change_query": "HISTORICAL",
                    "valid_from": "2026-01-03",
                    "valid_to": "2026-03-31",
                    "token_cost": 13,
                    "operation": "ADD",
                },
                {
                    "memory_id": "S101_M2",
                    "time": "2026-04-01",
                    "text": "从四月开始，用户工作日改为乘地铁上下班。",
                    "status_by_current_query": "CURRENT",
                    "status_by_historical_query": "STALE",
                    "status_by_change_query": "CURRENT",
                    "valid_from": "2026-04-01",
                    "valid_to": None,
                    "token_cost": 12,
                    "operation": "SUPERSEDE",
                    "supersedes": ["S101_M1"],
                },
                {
                    "memory_id": "S101_M3",
                    "time": "2026-04-06",
                    "text": "周末采购较多时，用户仍会开车去超市。",
                    "status_by_current_query": "HISTORICAL",
                    "status_by_historical_query": "HISTORICAL",
                    "status_by_change_query": "HISTORICAL",
                    "valid_from": "2026-04-06",
                    "valid_to": None,
                    "token_cost": 10,
                    "operation": "ADD",
                },
            ],
            "queries": [
                {
                    "query_id": "S101_Q_CURRENT",
                    "query_type": "CURRENT",
                    "text": "我现在工作日上班主要怎么去？",
                    "memorybank_retrieved": ["S101_M2", "S101_M1", "S101_M3"],
                    "statebudgetmem_retrieved": ["S101_M2"],
                    "answer_memory_ids": ["S101_M2"],
                    "memorybank_wrong_answer": "你工作日通常自己开车去公司。",
                    "statebudgetmem_answer": "现在工作日主要乘地铁上下班。",
                },
                {
                    "query_id": "S101_Q_HISTORICAL",
                    "query_type": "HISTORICAL",
                    "text": "二月份我工作日通常怎么去公司？",
                    "memorybank_retrieved": ["S101_M2", "S101_M3", "S101_M1"],
                    "statebudgetmem_retrieved": ["S101_M1"],
                    "answer_memory_ids": ["S101_M1"],
                    "memorybank_wrong_answer": "二月份你已经主要坐地铁。",
                    "statebudgetmem_answer": "二月份的历史状态是工作日通常自己开车去公司。",
                },
                {
                    "query_id": "S101_Q_CHANGE",
                    "query_type": "CHANGE",
                    "text": "我的工作日通勤方式发生了什么变化？",
                    "memorybank_retrieved": ["S101_M1", "S101_M2", "S101_M3"],
                    "statebudgetmem_retrieved": ["S101_M1", "S101_M2"],
                    "answer_memory_ids": ["S101_M1", "S101_M2"],
                    "memorybank_wrong_answer": "你既开车也坐地铁，无法看出变化。",
                    "statebudgetmem_answer": "一月至三月主要自驾，四月起永久改为地铁通勤。",
                },
            ],
            "resource_panel": resource_panel,
        },
        {
            "case_id": "historical_change",
            "label": "历史/变化查询：手机平台从安卓换成 iPhone",
            "label_en": "Historical and change query",
            "source_scenario": "S102_phone_platform",
            "label_note": "Showcase and analysis tool, not a formal experiment",
            "thesis": "同一条旧记忆对当前问题是过期信息，对历史问题却是正确证据。",
            "failure_mode": "MemoryBank 只看语义相似，容易把五月后的 iPhone 状态用于三月份历史问题。",
            "why_statebudgetmem": "StateBudgetMem 先路由到历史视图，再按 reference_time 取当时有效状态。",
            "conversation": [
                {
                    "time": "2026-01-08",
                    "speaker": "user",
                    "text": "一月至四月，用户的主力手机是一部安卓手机。",
                    "memory_id": "S102_M1",
                },
                {
                    "time": "2026-05-01",
                    "speaker": "user",
                    "text": "五月起，用户把 iPhone 设为日常使用的主力手机。",
                    "memory_id": "S102_M2",
                },
                {
                    "time": "2026-03-15",
                    "speaker": "user",
                    "text": "三月份我的主力手机是什么系统？",
                    "memory_id": "S102_Q_HISTORICAL",
                },
            ],
            "memories": [
                {
                    "memory_id": "S102_M1",
                    "time": "2026-01-08",
                    "text": "一月至四月，用户的主力手机是一部安卓手机。",
                    "status_by_current_query": "STALE",
                    "status_by_historical_query": "CURRENT",
                    "status_by_change_query": "HISTORICAL",
                    "valid_from": "2026-01-08",
                    "valid_to": "2026-04-30",
                    "token_cost": 12,
                    "operation": "ADD",
                },
                {
                    "memory_id": "S102_M2",
                    "time": "2026-05-01",
                    "text": "五月起，用户把 iPhone 设为日常使用的主力手机。",
                    "status_by_current_query": "CURRENT",
                    "status_by_historical_query": "STALE",
                    "status_by_change_query": "CURRENT",
                    "valid_from": "2026-05-01",
                    "valid_to": None,
                    "token_cost": 16,
                    "operation": "SUPERSEDE",
                    "supersedes": ["S102_M1"],
                },
                {
                    "memory_id": "S102_M3",
                    "time": "2026-02-12",
                    "text": "用户家里的平板仍然使用安卓系统。",
                    "status_by_current_query": "HISTORICAL",
                    "status_by_historical_query": "HISTORICAL",
                    "status_by_change_query": "HISTORICAL",
                    "valid_from": "2026-02-12",
                    "valid_to": None,
                    "token_cost": 9,
                    "operation": "ADD",
                },
            ],
            "queries": [
                {
                    "query_id": "S102_Q_HISTORICAL",
                    "query_type": "HISTORICAL",
                    "text": "三月份我的主力手机是什么系统？",
                    "memorybank_retrieved": ["S102_M2", "S102_M1", "S102_M3"],
                    "statebudgetmem_retrieved": ["S102_M1"],
                    "answer_memory_ids": ["S102_M1"],
                    "memorybank_wrong_answer": "三月份你的主力手机是 iPhone。",
                    "statebudgetmem_answer": "三月份的历史状态是主力手机为安卓。",
                },
                {
                    "query_id": "S102_Q_CURRENT",
                    "query_type": "CURRENT",
                    "text": "我现在的主力手机用什么系统？",
                    "memorybank_retrieved": ["S102_M1", "S102_M2", "S102_M3"],
                    "statebudgetmem_retrieved": ["S102_M2"],
                    "answer_memory_ids": ["S102_M2"],
                    "memorybank_wrong_answer": "你现在的主力手机仍是安卓。",
                    "statebudgetmem_answer": "现在的主力手机是 iPhone / iOS。",
                },
                {
                    "query_id": "S102_Q_CHANGE",
                    "query_type": "CHANGE",
                    "text": "我的主力手机平台是怎么更换的？",
                    "memorybank_retrieved": ["S102_M1", "S102_M2", "S102_M3"],
                    "statebudgetmem_retrieved": ["S102_M1", "S102_M2"],
                    "answer_memory_ids": ["S102_M1", "S102_M2"],
                    "memorybank_wrong_answer": "你一直同时使用安卓和 iPhone。",
                    "statebudgetmem_answer": "一月至四月主力手机是安卓，五月起更换为 iPhone。",
                },
            ],
            "resource_panel": resource_panel,
        },
    ]
    return primary_cases + build_dataset_showcase_cases(
        resource_panel=resource_panel,
        scenario_ids=EXTRA_SHOWCASE_SCENARIO_IDS,
        dataset_path=DEFAULT_CONTROLLED_DATASET,
    )


def build_dataset_showcase_cases(
    *,
    resource_panel: dict[str, Any],
    scenario_ids: tuple[str, ...],
    dataset_path: Path,
) -> list[dict[str, Any]]:
    """Append representative formal-dataset scenarios to the demo selector."""

    scenarios = {
        scenario["scenario_id"]: scenario
        for scenario in iter_jsonl(dataset_path)
        if scenario.get("scenario_id") in scenario_ids
    }
    cases: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        scenario = scenarios.get(scenario_id)
        if not scenario:
            continue
        memories = scenario.get("memories", [])
        queries = scenario.get("queries", [])
        if not memories or not queries:
            continue
        case = {
            "case_id": f"dataset_{scenario_id}",
            "label": f"扩展场景：{scenario_id}",
            "label_en": "Formal dataset extension",
            "source_scenario": scenario_id,
            "label_note": (
                "Extended case from data/controlled/temporal_challenge_v1.jsonl; "
                "showcase only, not a new formal result."
            ),
            "thesis": str(scenario.get("description") or scenario_id),
            "failure_mode": (
                "MemoryBank flat retrieval may mix old, current, and distractor "
                "records when they are semantically close."
            ),
            "why_statebudgetmem": (
                "StateBudgetMem-style demo first applies query-type/view scoping, "
                "then retrieves from memories that are valid for that question view."
            ),
            "conversation": build_dataset_conversation(memories, queries),
            "memories": [
                build_dataset_memory(memory, queries)
                for memory in memories
            ],
            "queries": [
                build_dataset_query(query, memories)
                for query in queries
            ],
            "resource_panel": resource_panel,
        }
        cases.append(case)
    return cases


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_dataset_conversation(
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for memory in memories[:4]:
        timeline.append(
            {
                "time": memory.get("event_time") or memory.get("valid_from") or "",
                "speaker": "user",
                "text": memory.get("text", ""),
                "memory_id": memory.get("memory_id", ""),
            }
        )
    if queries:
        query = queries[0]
        timeline.append(
            {
                "time": query.get("reference_time", ""),
                "speaker": "user",
                "text": query.get("text", ""),
                "memory_id": query.get("query_id", ""),
            }
        )
    return timeline


def build_dataset_memory(
    memory: dict[str, Any],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "memory_id": memory["memory_id"],
        "time": memory.get("event_time") or memory.get("valid_from") or "",
        "text": memory.get("text", ""),
        "status_by_current_query": status_for_query_type(memory["memory_id"], queries, "CURRENT"),
        "status_by_historical_query": status_for_query_type(memory["memory_id"], queries, "HISTORICAL"),
        "status_by_change_query": status_for_query_type(memory["memory_id"], queries, "CHANGE"),
        "valid_from": memory.get("valid_from"),
        "valid_to": memory.get("valid_to"),
        "token_cost": memory.get("token_cost", estimate_text_token_proxy(memory.get("text", ""))),
        "operation": infer_memory_operation(memory),
        "supersedes": memory.get("supersedes", []),
        "temporarily_invalidates": memory.get("temporarily_invalidates", []),
    }


def build_dataset_query(
    query: dict[str, Any],
    memories: list[dict[str, Any]],
) -> dict[str, Any]:
    memory_ids = [memory["memory_id"] for memory in memories]
    valid_ids = [
        memory_id
        for memory_id in query.get("gold_valid_memory_ids", [])
        if memory_id in memory_ids
    ]
    stale_ids = [
        memory_id
        for memory_id in query.get("gold_stale_memory_ids", [])
        if memory_id in memory_ids
    ]
    relevant_ids = [
        memory_id
        for memory_id in query.get("gold_relevant_memory_ids", [])
        if memory_id in memory_ids
    ]
    distractor_ids = [
        memory_id
        for memory_id in memory_ids
        if memory_id not in stale_ids and memory_id not in valid_ids
    ]
    return {
        "query_id": query["query_id"],
        "query_type": query.get("query_type", "CURRENT"),
        "text": query.get("text", ""),
        "memorybank_retrieved": (stale_ids + valid_ids + distractor_ids)[:3],
        "statebudgetmem_retrieved": (valid_ids + [
            memory_id for memory_id in relevant_ids if memory_id not in valid_ids
        ])[:3],
        "answer_memory_ids": valid_ids[:3],
        "memorybank_wrong_answer": (
            "如果优先引用旧记忆或干扰记忆，回答可能与当前/历史视图不一致。"
        ),
        "statebudgetmem_answer": build_dataset_reference_answer(query, valid_ids, memories),
    }


def status_for_query_type(
    memory_id: str,
    queries: list[dict[str, Any]],
    query_type: str,
) -> str:
    selected = next(
        (query for query in queries if query.get("query_type") == query_type),
        None,
    )
    if selected is None:
        return "HISTORICAL"
    if memory_id in selected.get("gold_stale_memory_ids", []):
        return "STALE"
    if memory_id in selected.get("gold_valid_memory_ids", []):
        return "CURRENT"
    if memory_id in selected.get("gold_relevant_memory_ids", []):
        return "HISTORICAL"
    return "HISTORICAL"


def infer_memory_operation(memory: dict[str, Any]) -> str:
    if memory.get("temporarily_invalidates"):
        return "TEMP_INVALIDATE"
    if memory.get("supersedes"):
        return "SUPERSEDE"
    if memory.get("status") == "HISTORICAL":
        return "ADD/HISTORICAL"
    return "ADD"


def build_dataset_reference_answer(
    query: dict[str, Any],
    valid_ids: list[str],
    memories: list[dict[str, Any]],
) -> str:
    if not valid_ids:
        return "当前展示数据中没有可确认的有效记忆，应保守回答无法确定。"
    by_id = {memory["memory_id"]: memory for memory in memories}
    evidence = [
        by_id[memory_id]["text"]
        for memory_id in valid_ids
        if memory_id in by_id
    ]
    joined = "；".join(evidence[:2])
    return f"根据有效记忆 {', '.join(valid_ids[:3])}，可回答：{joined}"


def estimate_text_token_proxy(text: str) -> int:
    return max(1, len(str(text).split()))


def apply_demo_answerer(
    showcase_data: dict[str, Any],
    *,
    answerer: str,
    local_llm_model: str,
    local_llm_endpoint: str,
    local_llm_timeout_s: float,
) -> None:
    template = TemplateAnswerer()
    local_unavailable_reason: str | None = None
    latest_result: AnswerResult | None = None
    cases = showcase_data.get("fixed_cases") or [showcase_data["memory_explorer"]]

    for case in cases:
        memories_by_id = {
            memory["memory_id"]: memory
            for memory in case["memories"]
        }
        for query in case["queries"]:
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
                    "source_name": f"final_showcase_demo:{case.get('case_id', 'case')}",
                    "global_summary": case.get("thesis", ""),
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

    showcase_data["memory_explorer"] = cases[0]

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
                "Formal conclusions come from the v2 unified fair-comparison runner. "
                "Case Entry and MemoryExplorer are fixed-case display and analysis "
                "tools; the Free Question Demo is illustrative only and is not used "
                "for formal metrics."
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
    .case-selector {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 14px;
    }}
    .explain-band {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
    }}
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
    .free-input {{
      width: 100%;
      min-height: 74px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }}
    .free-controls {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
      margin: 12px 0;
    }}
    .free-column {{
      display: grid;
      gap: 10px;
      align-content: start;
    }}
    .context-list {{
      display: grid;
      gap: 8px;
      max-height: 260px;
      overflow: auto;
    }}
    .context-item {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px;
      font-size: 13px;
    }}
    .mono {{
      font-family: Consolas, "SFMono-Regular", Menlo, monospace;
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      main, header {{ padding-left: 18px; padding-right: 18px; }}
      .two, .three, .resource-grid, .explain-band {{ grid-template-columns: 1fr; }}
      .free-controls {{ grid-template-columns: 1fr; }}
      .memory {{ grid-template-columns: 1fr; }}
      .conversation {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>StateBudgetMem Final Showcase</h1>
    <p>固定案例入口、MemoryExplorer、自由提问三栏对比、v2 统一 runner 公平实验 Dashboard。Free question demo is illustrative only; formal conclusions come from results/fair_comparison_v2.</p>
    <nav>
      <a href="#case-entry">Case Entry</a>
      <a href="#memory-explorer">MemoryExplorer</a>
      <a href="#free-question-demo">Free Question Demo</a>
      <a href="#experiment-dashboard">Experiment Dashboard</a>
      <a href="#on-device">On-device Panel</a>
    </nav>
  </header>
  <main>
    <section id="case-entry"></section>
    <section id="memory-explorer"></section>
    <section id="free-question-demo"></section>
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
      if (queryType === 'CHANGE') return memory.status_by_change_query;
      return memory.status_by_current_query;
    }}

    function renderExplorer(selectedCaseId, selectedQueryId) {{
      const cases = showcase.fixed_cases || [showcase.memory_explorer];
      const explorer = cases.find(item => item.case_id === selectedCaseId) || cases[0];
      const query = explorer.queries.find(item => item.query_id === selectedQueryId) || explorer.queries[0];
      const caseButtons = cases.map(item => `
        <button class="${{item.case_id === explorer.case_id ? 'active' : ''}}" onclick="renderExplorer('${{item.case_id}}', '${{item.queries[0].query_id}}')">
          ${{esc(item.label)}}
        </button>
      `).join('');
      const queryButtons = explorer.queries.map(item => `
        <button class="${{item.query_id === query.query_id ? 'active' : ''}}" onclick="renderExplorer('${{explorer.case_id}}', '${{item.query_id}}')">
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
        <div class="notice">${{esc(explorer.label_note || explorer.label)}}</div>
        <div class="case-selector">${{caseButtons}}</div>
        <h3>${{esc(explorer.label)}}</h3>
        <p>${{esc(explorer.thesis || '')}}</p>
        <div class="explain-band">
          <div class="card"><h3>状态变化</h3><p>${{esc(explorer.thesis || '')}}</p></div>
          <div class="card"><h3>旧记忆风险</h3><p>${{esc(explorer.failure_mode || '')}}</p></div>
          <div class="card"><h3>StateBudgetMem</h3><p>${{esc(explorer.why_statebudgetmem || '')}}</p></div>
        </div>
        <div style="margin-bottom: 12px;">${{queryButtons}}</div>
        <p><strong>${{esc(query.query_type)}} query:</strong> ${{esc(query.text)}}</p>
        <div class="timeline">${{timeline}}</div>
        <div class="grid two">
          ${{methodBlock('MemoryBank retrieved memories', query.memorybank_retrieved)}}
          ${{methodBlock('StateBudgetMem retrieved memories', query.statebudgetmem_retrieved)}}
        </div>
        <div class="card" style="margin-top: 14px;">
          <h3>Answer trace</h3>
          <p><strong>如果错误引用旧记忆：</strong>${{esc(query.memorybank_wrong_answer || '')}}</p>
          <p><strong>StateBudgetMem 模板答案：</strong>${{esc(query.statebudgetmem_answer)}}</p>
          <p class="small">answerer=${{esc(query.answerer_type)}}; model=${{esc(query.model_name)}}. Highlighted memories are the records cited by the answer.</p>
        </div>
      `;
    }}

    function tokenizeDemo(text) {{
      return String(text || '').toLowerCase().match(/[a-z0-9]+|[\u4e00-\u9fff]/g) || [];
    }}

    function estimateTokenProxyDemo(text) {{
      return tokenizeDemo(text).length;
    }}

    function classifyFreeQuery(text) {{
      const q = String(text || '').toLowerCase();
      const changeHints = ['变化', '改变', '更换', '替代', '为什么', '怎么变', 'change', 'changed', 'replace', 'replaced', 'switch'];
      const historyHints = ['以前', '过去', '当时', '历史', '之前', '那时', 'history', 'historical', 'before', 'previously', 'past'];
      const currentHints = ['现在', '当前', '今天', '目前', '这次', 'now', 'current', 'today', 'currently'];
      if (changeHints.some(hint => q.includes(hint))) {{
        return {{ queryType: 'CHANGE', reason: 'change/update keywords matched' }};
      }}
      if (historyHints.some(hint => q.includes(hint))) {{
        return {{ queryType: 'HISTORICAL', reason: 'historical-time keywords matched' }};
      }}
      if (currentHints.some(hint => q.includes(hint))) {{
        return {{ queryType: 'CURRENT', reason: 'current-state keywords matched' }};
      }}
      return {{ queryType: 'CURRENT', reason: 'demo fallback: personal question treated as CURRENT' }};
    }}

    function scoreMemoryDemo(query, memory) {{
      const queryTokens = new Set(tokenizeDemo(query));
      const memoryTokens = tokenizeDemo([memory.memory_id, memory.text, memory.operation].join(' '));
      let score = 0;
      memoryTokens.forEach(token => {{
        if (queryTokens.has(token)) score += 1;
      }});
      if (String(memory.text || '').toLowerCase().includes(String(query || '').toLowerCase())) score += 3;
      return score;
    }}

    function rankMemoriesDemo(query, memories, topK) {{
      return memories
        .map((memory, index) => ({{ memory, index, score: scoreMemoryDemo(query, memory) }}))
        .sort((a, b) => b.score - a.score || String(b.memory.time).localeCompare(String(a.memory.time)) || a.index - b.index)
        .slice(0, topK)
        .map((item, index) => ({{ ...item.memory, rank: index + 1, demo_score: item.score }}));
    }}

    function retrieveMemoryBankDemo(query, explorer, topK) {{
      const started = performance.now();
      const retrieved = rankMemoriesDemo(query, explorer.memories, topK);
      return {{
        method: 'MemoryBank flat retrieval demo',
        queryType: 'N/A',
        queryTypeReason: 'No query-type scoping; all demo memories are eligible.',
        retrieved,
        latencyMs: performance.now() - started,
        tokenProxy: retrieved.reduce((total, memory) => total + Number(memory.token_cost || estimateTokenProxyDemo(memory.text)), 0),
      }};
    }}

    function retrieveStateBudgetMemDemo(query, explorer, topK, heuristic) {{
      const started = performance.now();
      const eligible = explorer.memories.filter(memory => {{
        const status = statusFor(memory, heuristic.queryType);
        if (heuristic.queryType === 'CURRENT') return status === 'CURRENT';
        if (heuristic.queryType === 'HISTORICAL') return status === 'CURRENT' || status === 'HISTORICAL';
        if (heuristic.queryType === 'CHANGE') return status === 'CURRENT' || status === 'HISTORICAL';
        return status !== 'STALE';
      }});
      const retrieved = rankMemoriesDemo(query, eligible, topK);
      return {{
        method: showcase.free_question_demo.statebudgetmem_demo_name,
        queryType: heuristic.queryType,
        queryTypeReason: heuristic.reason,
        retrieved,
        latencyMs: performance.now() - started,
        tokenProxy: retrieved.reduce((total, memory) => total + Number(memory.token_cost || estimateTokenProxyDemo(memory.text)), 0),
      }};
    }}

    function retrieveNoMemoryDemo(query, heuristic) {{
      const started = performance.now();
      return {{
        method: 'No Memory baseline',
        queryType: heuristic.queryType,
        queryTypeReason: heuristic.reason,
        retrieved: [],
        latencyMs: performance.now() - started,
        tokenProxy: estimateTokenProxyDemo(query),
      }};
    }}

    function renderTemplateAnswerDemo(query, result, mode) {{
      if (!result.retrieved.length) {{
        return '模板回答：没有使用个人长期记忆上下文，因此只能给出保守回答，无法确认用户当前或历史状态。';
      }}
      const stale = result.retrieved.filter(memory => statusFor(memory, result.queryType) === 'STALE');
      const first = result.retrieved[0];
      if (mode === 'memorybank' && stale.length) {{
        return `模板回答：检索到了 ${{stale.map(memory => memory.memory_id).join(', ')}} 等过期/不适用记忆，回答可能错误引用旧状态。Top memory: ${{first.text}}`;
      }}
      if (mode === 'statebudgetmem') {{
        const current = result.retrieved.find(memory => statusFor(memory, result.queryType) === 'CURRENT') || first;
        if (result.queryType === 'CURRENT') {{
          return `模板回答：根据 ${{current.memory_id}}，当前有效状态是：${{current.text}}`;
        }}
        if (result.queryType === 'HISTORICAL') {{
          return `模板回答：根据 ${{result.retrieved.map(memory => memory.memory_id).join(', ')}}，该问题应从历史视图取证据：${{first.text}}`;
        }}
        if (result.queryType === 'CHANGE') {{
          return `模板回答：根据 ${{result.retrieved.map(memory => memory.memory_id).join(', ')}}，变化查询需要同时查看旧状态和新状态。`;
        }}
      }}
      return `模板回答：根据 ${{result.retrieved.map(memory => memory.memory_id).join(', ')}} 构造上下文；优先使用与 ${{result.queryType}} 查询匹配且非 STALE 的记忆。`;
    }}

    function renderContextItems(result) {{
      if (!result.retrieved.length) {{
        return '<div class="context-item small">No memory context used.</div>';
      }}
      return result.retrieved.map(memory => {{
        const status = result.queryType === 'N/A' ? statusFor(memory, 'CURRENT') : statusFor(memory, result.queryType);
        return `
          <div class="context-item">
            <div><strong>#${{memory.rank}} ${{esc(memory.memory_id)}}</strong> <span class="pill status-${{esc(status)}}">${{esc(status)}}</span> <span class="pill">score=${{esc(memory.demo_score)}}</span></div>
            <div>${{esc(memory.text)}}</div>
            <div class="small">operation=${{esc(memory.operation)}}; token=${{esc(memory.token_cost)}}; valid=${{esc(memory.valid_from)}} to ${{esc(memory.valid_to || 'open')}}</div>
          </div>
        `;
      }}).join('');
    }}

    function renderFreeColumn(title, result, answer, extraClass = '', answerSource = 'template') {{
      const ids = result.retrieved.map(memory => `<span class="pill">${{esc(memory.memory_id)}}</span>`).join(' ') || '<span class="pill">none</span>';
      return `
        <div class="card free-column ${{extraClass}}">
          <h3>${{esc(title)}}</h3>
          <p>${{esc(answer)}}</p>
          <div>
            <span class="pill">answerer: ${{esc(answerSource)}}</span>
            <span class="pill">query type heuristic: ${{esc(result.queryType)}}</span>
            <span class="pill">token proxy: ${{esc(result.tokenProxy)}}</span>
            <span class="pill">latency: ${{fmt(result.latencyMs, 2)}} ms</span>
          </div>
          <p class="small">${{esc(result.queryTypeReason)}}</p>
          <div class="small"><strong>retrieved memory ids:</strong> ${{ids}}</div>
          <div class="context-list">${{renderContextItems(result)}}</div>
        </div>
      `;
    }}

    function columnPayload(columnId, title, result, templateAnswer) {{
      return {{
        column_id: columnId,
        title,
        query_type_heuristic: result.queryType,
        query_type_reason: result.queryTypeReason,
        token_proxy: result.tokenProxy,
        latency_ms: result.latencyMs,
        template_answer: templateAnswer,
        retrieved_memory_ids: result.retrieved.map(memory => memory.memory_id),
        memory_context: result.retrieved.map(memory => ({{
          memory_id: memory.memory_id,
          text: memory.text,
          status: result.queryType === 'N/A' ? statusFor(memory, 'CURRENT') : statusFor(memory, result.queryType),
          operation: memory.operation,
          token_cost: memory.token_cost,
          valid_from: memory.valid_from,
          valid_to: memory.valid_to || 'open',
        }})),
      }};
    }}

    async function fetchDeepSeekAnswers(query, columns) {{
      const apiKey = document.getElementById('deepseek-api-key').value.trim();
      const model = document.getElementById('deepseek-model').value.trim() || showcase.free_question_demo.deepseek_default_model;
      if (!apiKey) {{
        throw new Error('请输入 DeepSeek API key，或切回 Template answerer。');
      }}
      const response = await fetch(showcase.free_question_demo.deepseek_server_endpoint, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          provider: 'deepseek',
          api_key: apiKey,
          model,
          query,
          columns,
        }}),
      }});
      const data = await response.json().catch(() => ({{}}));
      if (!response.ok) {{
        throw new Error(data.error || `DeepSeek demo server returned HTTP ${{response.status}}`);
      }}
      return data;
    }}

    function renderFreeColumns(query, noMemory, memorybank, scoped, answers, sourceLabel) {{
      const columns = [
        renderFreeColumn('No Memory baseline', noMemory, answers.no_memory, 'baseline', sourceLabel),
        renderFreeColumn('MemoryBank flat retrieval demo', memorybank, answers.memorybank, 'baseline', sourceLabel),
        renderFreeColumn(showcase.free_question_demo.statebudgetmem_demo_name, scoped, answers.statebudgetmem, 'proposed', sourceLabel),
      ].join('');
      document.getElementById('free-question-output').innerHTML = `
        <p><strong>Question:</strong> ${{esc(query)}}</p>
        <div class="grid three">${{columns}}</div>
      `;
    }}

    async function runFreeQuestionDemo() {{
      const cases = showcase.fixed_cases || [showcase.memory_explorer];
      const caseId = document.getElementById('free-case-select').value;
      const explorer = cases.find(item => item.case_id === caseId) || cases[0];
      const query = document.getElementById('free-question-input').value.trim() || '我现在应该怎么选择？';
      const topK = Number(showcase.free_question_demo.top_k || 3);
      const heuristic = classifyFreeQuery(query);
      const noMemory = retrieveNoMemoryDemo(query, heuristic);
      const memorybank = retrieveMemoryBankDemo(query, explorer, topK);
      const scoped = retrieveStateBudgetMemDemo(query, explorer, topK, heuristic);
      const templateAnswers = {{
        no_memory: renderTemplateAnswerDemo(query, noMemory, 'none'),
        memorybank: renderTemplateAnswerDemo(query, memorybank, 'memorybank'),
        statebudgetmem: renderTemplateAnswerDemo(query, scoped, 'statebudgetmem'),
      }};
      renderFreeColumns(query, noMemory, memorybank, scoped, templateAnswers, 'template');

      const mode = document.getElementById('free-answer-mode').value;
      const status = document.getElementById('deepseek-status');
      if (mode !== 'deepseek') {{
        status.textContent = 'Template answerer is active. No API call is made.';
        return;
      }}
      status.textContent = 'Calling local demo server for DeepSeek answers...';
      const columnsForApi = [
        columnPayload('no_memory', 'No Memory baseline', noMemory, templateAnswers.no_memory),
        columnPayload('memorybank', 'MemoryBank flat retrieval demo', memorybank, templateAnswers.memorybank),
        columnPayload('statebudgetmem', showcase.free_question_demo.statebudgetmem_demo_name, scoped, templateAnswers.statebudgetmem),
      ];
      try {{
        const data = await fetchDeepSeekAnswers(query, columnsForApi);
        const apiAnswers = {{
          no_memory: data.answers?.no_memory?.answer_text || templateAnswers.no_memory,
          memorybank: data.answers?.memorybank?.answer_text || templateAnswers.memorybank,
          statebudgetmem: data.answers?.statebudgetmem?.answer_text || templateAnswers.statebudgetmem,
        }};
        renderFreeColumns(query, noMemory, memorybank, scoped, apiAnswers, data.answerer || 'deepseek');
        status.textContent = `DeepSeek answers loaded. model=${{data.model || 'n/a'}}; total latency=${{fmt(data.total_latency_ms, 2)}} ms.`;
      }} catch (error) {{
        status.textContent = `DeepSeek unavailable; kept template answers. ${{error.message}}`;
      }}
    }}

    function renderFreeQuestionDemo() {{
      const demo = showcase.free_question_demo;
      const cases = showcase.fixed_cases || [showcase.memory_explorer];
      const options = cases.map(item => `
        <option value="${{esc(item.case_id)}}">${{esc(item.label)}} / ${{esc(item.label_en || item.case_id)}}</option>
      `).join('');
      document.getElementById('free-question-demo').innerHTML = `
        <h2>3. 自由提问三栏对比 / Free Question Three-column Demo</h2>
        <div class="notice">
          <strong>中文：</strong>${{esc(demo.retrieval_boundary_zh)}}<br>
          <strong>English:</strong> Free question demo is illustrative only; it shows retrieval-strategy differences and is not used for formal metrics. Formal conclusions come from results/fair_comparison_v2.
        </div>
        <p class="small">${{esc(demo.llm_policy)}} No memory extraction and no memory update.</p>
        <div class="free-controls">
          <div>
            <label class="small" for="free-case-select">选择固定案例 / Select demo case</label>
            <select id="free-case-select" class="free-input" style="min-height: auto;">${{options}}</select>
          </div>
          <button onclick="runFreeQuestionDemo()">运行对比 / Compare</button>
        </div>
        <div class="grid three" style="margin-bottom: 12px;">
          <div>
            <label class="small" for="free-answer-mode">回答模式 / Answer mode</label>
            <select id="free-answer-mode" class="free-input" style="min-height: auto;">
              <option value="template">Template answerer 默认</option>
              <option value="deepseek">DeepSeek API via local demo server</option>
            </select>
          </div>
          <div>
            <label class="small" for="deepseek-model">DeepSeek model</label>
            <input id="deepseek-model" class="free-input" style="min-height: auto;" value="${{esc(demo.deepseek_default_model || 'deepseek-chat')}}">
          </div>
          <div>
            <label class="small" for="deepseek-api-key">DeepSeek API key（不保存）</label>
            <input id="deepseek-api-key" class="free-input" style="min-height: auto;" type="password" autocomplete="off" placeholder="sk-...">
          </div>
        </div>
        <p id="deepseek-status" class="small">Template answerer is active. To use DeepSeek, start the local demo server and select DeepSeek mode.</p>
        <textarea id="free-question-input" class="free-input">我现在应该怎么选择？</textarea>
        <div id="free-question-output" style="margin-top: 14px;"></div>
      `;
      document.getElementById('free-case-select').value = demo.default_case_id || cases[0].case_id;
      runFreeQuestionDemo();
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
        <h2>4. Experiment Dashboard</h2>
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
    renderExplorer(showcase.memory_explorer.case_id, showcase.memory_explorer.queries[0].query_id);
    renderFreeQuestionDemo();
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

1. Case Entry: a short fixed dialogue used only as a presentation entry.
2. MemoryExplorer: fixed no-free-input cases for temporary invalidation,
   permanent supersede, and historical/change queries.
3. Free Question Demo: browser-side three-column comparison over the same fixed
   demo memories. This is illustrative only and is not used for formal metrics.
4. Experiment Dashboard: formal fair-comparison metrics loaded from
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

## Optional DeepSeek Free-question Answers

The Free Question Demo can optionally ask DeepSeek to organize the three-column
answers. This is demo-only and still not part of formal metrics.

```powershell
.venv\\Scripts\\python.exe tools\\demo\\run_final_showcase_server.py
```

Then open `http://127.0.0.1:8765/index.html`, select `DeepSeek API via local
demo server`, type the DeepSeek API key into the page, and run the comparison.
The API key is sent to the local demo server for that request only; it is not
written into HTML, JSON, logs, or result files.

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

The Case Entry, MemoryExplorer, and Free Question Demo are display and analysis
tools. The free-question area uses browser-side template retrieval/answering
only; it does not call Ollama, does not extract new memories, and does not
update memory state. Formal performance conclusions come from the unified
runner outputs in `results/fair_comparison_v2/`.
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
