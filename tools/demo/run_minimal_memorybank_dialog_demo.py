#!/usr/bin/env python3
"""Run a minimal local-only MemoryBank dialog loop for meeting demos.

This script is evidence for the MemoryBank Core memory-system baseline loop:
write memories, persist locally, reload, retrieve, build an augmented prompt,
and produce a template answer. It is not a full conversational agent and does
not call an LLM or any cloud API.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statebudgetmem.baselines.memorybank.datasets import (
    DEMO_HISTORY,
    DEMO_QUESTIONS,
    load_reproduction_dataset,
)
from statebudgetmem.answering import (
    AnswerRequest,
    AnswerResult,
    LocalLLMAnswerer,
    LocalLLMUnavailable,
    TemplateAnswerer,
)
from statebudgetmem.baselines.memorybank.embeddings import HashEmbeddingModel
from statebudgetmem.baselines.memorybank.system import MemoryBank


DEFAULT_XIAOLIN_OUTPUT = ROOT / "results" / "minimal_memorybank_dialog_demo"
DEFAULT_GROUP_OUTPUT = ROOT / "results" / "minimal_memorybank_dialog_demo_group_dataset"
DEFAULT_GROUP_DATASET = ROOT / "data" / "memorybank_reproduction"
CURRENT_TIME = "2026-07-15 19:00"


@dataclass(frozen=True)
class DemoPayload:
    source_name: str
    source_path: str | None
    output_dir: Path
    query: str
    reference_answer: str
    expected_keywords: list[str]
    dialog_turns: list[dict[str, str]]
    user_portrait: str
    global_summary: str
    note: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("controlled_xiaolin", "new_group_dataset"),
        default="controlled_xiaolin",
        help="Use the stable built-in Xiaolin fixture or the group reproduction dataset.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional custom input. For new_group_dataset, pass a reproduction dataset "
            "directory. For controlled_xiaolin, pass a JSON/JSONL file with role/content/timestamp rows."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--user-id", default="user_001")
    parser.add_argument("--query-id", default="q006")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--current-time", default=CURRENT_TIME)
    parser.add_argument(
        "--answerer",
        choices=("template", "local_llm"),
        default="template",
        help="Answer generator. Template is the default path.",
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
    payload = build_payload(args)
    result = run_demo(
        payload,
        top_k=args.top_k,
        current_time=args.current_time,
        answerer=args.answerer,
        local_llm_model=args.local_llm_model,
        local_llm_endpoint=args.local_llm_endpoint,
        local_llm_timeout_s=args.local_llm_timeout_s,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_payload(args: argparse.Namespace) -> DemoPayload:
    if args.input is not None and args.dataset == "controlled_xiaolin":
        return payload_from_custom_input(args.input, args.output_dir)
    if args.dataset == "new_group_dataset":
        return payload_from_group_dataset(
            args.input or DEFAULT_GROUP_DATASET,
            args.output_dir or DEFAULT_GROUP_OUTPUT,
            user_id=args.user_id,
            query_id=args.query_id,
        )
    return payload_from_xiaolin(args.output_dir or DEFAULT_XIAOLIN_OUTPUT)


def payload_from_xiaolin(output_dir: Path) -> DemoPayload:
    question = next(
        (
            item
            for item in DEMO_QUESTIONS
            if "适合" in item.get("question", "") or "火锅" in item.get("question", "")
        ),
        DEMO_QUESTIONS[6],
    )
    turns = [
        {"role": role, "content": content, "timestamp": timestamp}
        for role, content, timestamp in DEMO_HISTORY
    ]
    query = (
        "我现在适合吃什么类型的食物？ "
        "food recommendation stomach avoid spicy bland food"
    )
    return DemoPayload(
        source_name="controlled_xiaolin",
        source_path="src/statebudgetmem/baselines/memorybank/datasets.py::DEMO_HISTORY",
        output_dir=output_dir,
        query=query,
        reference_answer="应选择清淡饮食，少吃辛辣刺激食物。",
        expected_keywords=list(question.get("expected_keywords", [])),
        dialog_turns=turns,
        user_portrait="Xiaolin is a student with changing food and study preferences.",
        global_summary=(
            "Xiaolin used to like spicy food and hotpot, but later had stomach discomfort. "
            "The current food guidance is to avoid spicy food and choose bland meals."
        ),
        note=(
            "Stable built-in demo fixture for meetings. It is useful for showing the "
            "minimum MemoryBank loop, not for formal evaluation."
        ),
    )


def payload_from_group_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    user_id: str,
    query_id: str,
) -> DemoPayload:
    users, probes = load_reproduction_dataset(dataset_dir)
    user = next((item for item in users if item.user_id == user_id), users[0])
    probe = next(
        (item for item in probes if item.user_id == user.user_id and item.query_id == query_id),
        next(item for item in probes if item.user_id == user.user_id),
    )
    turns: list[dict[str, str]] = []
    for day in user.days:
        for dialog in day.get("dialogues", []):
            turns.append(
                {
                    "role": str(dialog.get("role", "user")),
                    "content": str(dialog.get("content", "")),
                    "timestamp": normalize_timestamp(str(dialog.get("timestamp", ""))),
                }
            )
        if day.get("daily_event_summary"):
            turns.append(
                {
                    "role": "summary",
                    "content": f"Daily event summary: {day['daily_event_summary']}",
                    "timestamp": normalize_timestamp(str(day.get("date", ""))),
                }
            )
    if user.global_summary:
        turns.append(
            {
                "role": "summary",
                "content": f"Global event summary: {user.global_summary}",
                "timestamp": "2026-07-15 18:00",
            }
        )
    if user.user_portrait:
        turns.append(
            {
                "role": "profile",
                "content": f"Global user portrait: {user.user_portrait}",
                "timestamp": "2026-07-15 18:05",
            }
        )
    return DemoPayload(
        source_name="new_group_dataset",
        source_path=str(dataset_dir),
        output_dir=output_dir,
        query=probe.question,
        reference_answer=probe.reference_answer,
        expected_keywords=probe.expected_keywords,
        dialog_turns=turns,
        user_portrait=user.user_portrait,
        global_summary=user.global_summary,
        note=(
            "Group reproduction dataset demo. It is better for showing the later "
            "Phase 1 dataset and gold-labeled probes, but formal metrics should be "
            "reported from the dedicated runners, not this minimal demo."
        ),
    )


def payload_from_custom_input(input_path: Path, output_dir: Path | None) -> DemoPayload:
    rows = read_custom_rows(input_path)
    return DemoPayload(
        source_name="custom_input",
        source_path=str(input_path),
        output_dir=output_dir or DEFAULT_XIAOLIN_OUTPUT,
        query="What should I remember from the recent dialog?",
        reference_answer="Custom input demo has no gold reference answer.",
        expected_keywords=[],
        dialog_turns=rows,
        user_portrait="Custom local demo user.",
        global_summary="Custom local demo input loaded from a user-provided path.",
        note="Custom input path demo; useful for quick local inspection only.",
    )


def read_custom_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if input_path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("dialog_turns", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("custom input must contain a non-empty list of dialog rows")
    normalized = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"custom row {index} must be an object")
        normalized.append(
            {
                "role": str(row.get("role", "user")),
                "content": str(row.get("content", "")),
                "timestamp": normalize_timestamp(
                    str(row.get("timestamp", f"2026-07-15 18:{index:02d}"))
                ),
            }
        )
    return normalized


def run_demo(
    payload: DemoPayload,
    *,
    top_k: int,
    current_time: str,
    answerer: str = "template",
    local_llm_model: str = "qwen2.5:3b",
    local_llm_endpoint: str = "http://localhost:11434/api/generate",
    local_llm_timeout_s: float = 30.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload.output_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = payload.output_dir / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    encoder = HashEmbeddingModel(dim=64)
    bank = MemoryBank(
        embedding_dim=64,
        forgetting_threshold=0.3,
        embedding_model=encoder,
        decay_interval_hours=24.0,
    )
    bank.update_user_portrait(payload.user_portrait)
    bank.update_global_summary(payload.global_summary)

    write_started = time.perf_counter()
    stored = [
        bank.store_dialog(turn["role"], turn["content"], turn["timestamp"])
        for turn in payload.dialog_turns
    ]
    write_latency_ms = elapsed_ms(write_started)

    save_prefix = storage_dir / "memorybank_demo"
    save_started = time.perf_counter()
    bank.save(str(save_prefix))
    save_latency_ms = elapsed_ms(save_started)
    storage_bytes = storage_size(storage_dir)

    loaded = MemoryBank(
        embedding_dim=64,
        forgetting_threshold=0.3,
        embedding_model=encoder,
        decay_interval_hours=24.0,
    )
    load_started = time.perf_counter()
    loaded.load(str(save_prefix))
    load_latency_ms = elapsed_ms(load_started)

    prompt_started = time.perf_counter()
    prompt_context = loaded.build_augmented_prompt(
        query=payload.query,
        top_k=top_k,
        current_time=current_time,
        include_portrait=True,
        exclude_forgotten=False,
    )
    retrieval_and_prompt_latency_ms = elapsed_ms(prompt_started)

    forgetting = loaded.forgetting_log(current_time)
    answer_result = generate_answer(
        prompt_context=prompt_context,
        payload=payload,
        answerer=answerer,
        local_llm_model=local_llm_model,
        local_llm_endpoint=local_llm_endpoint,
        local_llm_timeout_s=local_llm_timeout_s,
    )
    answer = answer_result.answer_text

    result = {
        "status": "success",
        "scope": "MemoryBank Core memory-system baseline minimal dialog demo",
        "dataset_source": payload.source_name,
        "source_path": payload.source_path,
        "note": payload.note,
        "local_only": True,
        "cloud_api_calls": 0,
        "network_calls": 0,
        "answerer_requested": answerer,
        "answerer_type": answer_result.answerer_type,
        "model_name": answer_result.model_name,
        "generation_latency_ms": answer_result.latency_ms,
        "tokens_per_second": answer_result.tokens_per_second,
        "prompt_tokens": answer_result.prompt_tokens,
        "generated_tokens": answer_result.generated_tokens,
        "prompt_token_proxy": answer_result.prompt_tokens,
        "generated_token_proxy": answer_result.generated_tokens,
        "used_memory_ids": answer_result.used_memory_ids,
        "answer_metadata": answer_result.metadata,
        "embedding_backend": "hash",
        "embedding_model": encoder.name,
        "faiss_used": loaded.index is not None,
        "query": payload.query,
        "reference_answer": payload.reference_answer,
        "expected_keywords": payload.expected_keywords,
        "dialog_turn_count": len(payload.dialog_turns),
        "stored_memory_ids": [memory.memory_id for memory in stored],
        "retrieved_memory_ids": prompt_context["retrieved_memory_ids"],
        "retrieved_memories": [
            {
                "memory_id": item["memory_id"],
                "rank": item["retrieval_rank"],
                "content": item["content"],
                "retrieval_score": item["retrieval_score"],
                "composite_score": item["composite_score"],
                "before_strength": item["before_strength"],
                "after_strength": item.get("after_strength"),
                "before_last_accessed": item["before_last_accessed"],
                "after_last_accessed": item.get("after_last_accessed"),
                "retention": item["retention"],
                "is_forgotten": item["is_forgotten"],
            }
            for item in prompt_context["retrieved_memories"]
        ],
        "prompt_token_estimate": prompt_context["prompt_token_estimate"],
        "prompt_template": prompt_context["prompt_template"],
        "answer": answer,
        "answer_result": answer_result.to_dict(),
        "forgetting_threshold": forgetting["threshold"],
        "forgotten_memory_ids": forgetting["forgotten_memory_ids"],
        "latency_ms": {
            "write": write_latency_ms,
            "save": save_latency_ms,
            "load": load_latency_ms,
            "retrieval_and_prompt": retrieval_and_prompt_latency_ms,
            "total": elapsed_ms(started),
        },
        "storage": {
            "directory": str(storage_dir),
            "total_bytes": storage_bytes,
            "files": file_sizes(storage_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cwd": str(ROOT),
        },
        "output_files": {
            "json": str(payload.output_dir / "demo_result.json"),
            "markdown": str(payload.output_dir / "demo_summary.md"),
        },
    }

    write_json(payload.output_dir / "demo_result.json", result)
    write_markdown(payload.output_dir / "demo_summary.md", result)
    return result


def generate_answer(
    *,
    prompt_context: dict[str, Any],
    payload: DemoPayload,
    answerer: str,
    local_llm_model: str,
    local_llm_endpoint: str,
    local_llm_timeout_s: float,
) -> AnswerResult:
    request = AnswerRequest(
        query=payload.query,
        retrieved_memories=prompt_context["retrieved_memories"],
        augmented_prompt=prompt_context["prompt_template"],
        metadata={
            "reference_answer": payload.reference_answer,
            "global_summary": payload.global_summary,
            "source_name": payload.source_name,
        },
    )
    template = TemplateAnswerer()
    if answerer == "template":
        return template.answer(request)

    try:
        return LocalLLMAnswerer(
            model_name=local_llm_model,
            endpoint=local_llm_endpoint,
            timeout_s=local_llm_timeout_s,
            temperature=0.0,
        ).answer(request)
    except (LocalLLMUnavailable, ValueError) as exc:
        fallback = template.answer(request)
        metadata = dict(fallback.metadata)
        metadata.update(
            {
                "requested_answerer": "local_llm",
                "fallback_to": "template",
                "fallback_reason": str(exc),
            }
        )
        return AnswerResult(
            answer_text=fallback.answer_text,
            answerer_type="template_fallback",
            model_name=fallback.model_name,
            prompt_tokens=fallback.prompt_tokens,
            generated_tokens=fallback.generated_tokens,
            latency_ms=fallback.latency_ms,
            tokens_per_second=fallback.tokens_per_second,
            used_memory_ids=fallback.used_memory_ids,
            metadata=metadata,
        )


def template_answer(prompt_context: dict[str, Any], payload: DemoPayload) -> str:
    joined = " ".join(
        [payload.global_summary.lower(), payload.reference_answer.lower()]
        + [item["content"].lower() for item in prompt_context["retrieved_memories"]]
    )
    if "avoid spicy" in joined or "stomach" in joined or "bland" in joined:
        return (
            "模板回答：当前记忆显示饮食偏好发生过变化，现在应优先选择清淡、低辣或不辣的食物；"
            "这证明 retrieved memories 已经被放入回答上下文，但这里没有调用真实 LLM。"
        )
    if payload.reference_answer:
        return (
            "模板回答：根据检索上下文和该数据集的参考答案，"
            f"可回答为：{payload.reference_answer}"
        )
    return "模板回答：当前检索上下文不足以生成更具体的回答。"


def normalize_timestamp(value: str) -> str:
    text = value.strip()
    if not text:
        return "2026-07-15 18:00"
    if "T" in text:
        text = text.replace("T", " ")
    if len(text) == 10:
        return f"{text} 12:00"
    if len(text) == 16:
        return f"{text}:00"
    return text


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def storage_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def file_sizes(path: Path) -> dict[str, int]:
    return {
        str(item.relative_to(path)).replace(os.sep, "/"): item.stat().st_size
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = [
        "| 项目 | 值 |",
        "|---|---|",
        f"| dataset_source | {result['dataset_source']} |",
        f"| local_only | {result['local_only']} |",
        f"| cloud_api_calls | {result['cloud_api_calls']} |",
        f"| answerer | {result['answerer_type']} / {result['model_name']} |",
        f"| generation latency ms | {result['generation_latency_ms']:.3f} |",
        f"| tokens per second | {result['tokens_per_second']} |",
        f"| embedding | {result['embedding_backend']} / {result['embedding_model']} |",
        f"| FAISS used | {result['faiss_used']} |",
        f"| stored memories | {len(result['stored_memory_ids'])} |",
        f"| dialog turns | {result['dialog_turn_count']} |",
        f"| retrieved memories | {len(result['retrieved_memory_ids'])} |",
        f"| prompt token proxy | {result['prompt_token_estimate']} |",
        f"| generated token proxy | {result['generated_token_proxy']} |",
        f"| storage bytes | {result['storage']['total_bytes']} |",
        f"| retrieval + prompt latency ms | {result['latency_ms']['retrieval_and_prompt']:.3f} |",
    ]
    retrieved = "\n".join(
        f"- Rank {item['rank']}: `{item['memory_id']}` - {item['content']}"
        for item in result["retrieved_memories"]
    )
    content = f"""# Minimal MemoryBank Dialog Demo

本 demo 证明 MemoryBank Core memory-system baseline 的最小闭环：写入、本地存储、检索、prompt augmentation、模板回答和基础资源统计。它不调用真实 LLM，不调用云 API，也不代表完整 MemoryBank 论文复现。

## 数据来源

- Source: `{result['dataset_source']}`
- Path: `{result['source_path']}`
- Note: {result['note']}

## 查询

{result['query']}

## 参考答案或预期

{result['reference_answer']}

## 结果概览

{chr(10).join(rows)}

## 检索到的记忆

{retrieved}

## 模板回答

{result['answer']}

## 输出文件

- JSON: `{result['output_files']['json']}`
- Markdown: `{result['output_files']['markdown']}`
- Storage: `{result['storage']['directory']}`
"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
