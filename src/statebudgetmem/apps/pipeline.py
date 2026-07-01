"""
apps.pipeline — Minimal end-to-end memory-augmented agent pipeline.

Wires together:
    query → routing → views → retrieval → (optional LLM) → answer

All components work offline by default.  LLM-based routing is opt-in via
``build_pipeline(use_llm_router=True, llm_api_key=...)``.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from statebudgetmem.core.online import ViewType
from statebudgetmem.retrieval.tfidf import TfidfRetriever
from statebudgetmem.routing import QueryRecord, RuleBasedRouter
from statebudgetmem.routing.router import LLMQueryRouter
from statebudgetmem.versioning.engine import VersioningEngine
from statebudgetmem.versioning.models import StateKey
from statebudgetmem.views.manager import MemoryViewManager, ViewEntry


class PipelineResult:
    """Structured output of one pipeline invocation."""

    __slots__ = (
        "query",
        "query_type",
        "view_type",
        "view_entries",
        "retrieved",
        "context_text",
        "answer",
    )

    def __init__(
        self,
        query: str,
        query_type: str,
        view_type: str,
        view_entries: list[ViewEntry],
        retrieved: list[tuple[int, dict[str, Any], float]],
        context_text: str,
        answer: str = "",
    ) -> None:
        self.query = query
        self.query_type = query_type
        self.view_type = view_type
        self.view_entries = view_entries
        self.retrieved = retrieved
        self.context_text = context_text
        self.answer = answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "view_type": self.view_type,
            "view_size": len(self.view_entries),
            "retrieved": [
                {
                    "rank": i + 1,
                    "memory_id": item.get("memory_id", ""),
                    "content": item.get("text", item.get("content", "")),
                    "score": round(score, 4),
                }
                for i, (_, item, score) in enumerate(self.retrieved)
            ],
            "context_len_chars": len(self.context_text),
            "context_text": self.context_text[:500],
            "answer": self.answer,
        }

    def summary(self) -> str:
        lines = [
            f"Query: {self.query}",
            f"  Type: {self.query_type} → View: {self.view_type}",
            f"  View size: {len(self.view_entries)}, Retrieved: {len(self.retrieved)}",
        ]
        if self.answer:
            lines.append(f"  Answer: {self.answer[:200]}")
        return "\n".join(lines)


class MemoryPipeline:
    """
    Minimal memory-aware agent pipeline (offline by default).

    Usage::

        pipeline = MemoryPipeline()
        pipeline.ingest_controlled("data/controlled/baseline_scenarios.jsonl")
        result = pipeline.ask("我现在适合吃什么？")

    The pipeline is fully deterministic when using ``RuleBasedRouter``
    (default).  Inject ``LLMQueryRouter`` via the *router* parameter to
    enable LLM-based classification (requires an API key).
    """

    def __init__(
        self,
        *,
        router: Any = None,
        retriever: Any = None,
        top_k: int = 5,
        max_context_chars: int = 2000,
    ) -> None:
        self.engine = VersioningEngine()
        self.view_manager = MemoryViewManager(self.engine)
        self.router = router or RuleBasedRouter()
        self.retriever = retriever or TfidfRetriever()
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    # ---- data ingestion -------------------------------------------------

    def ingest_scenarios(self, scenarios: list[Any]) -> int:
        """Ingest a list of MemoryRecord or Scenario objects."""
        count = 0
        for scenario in scenarios:
            memories = getattr(scenario, "memories", [scenario])
            if not isinstance(memories, list):
                memories = [memories]
            for mem in memories:
                self.engine.ingest(mem)
                count += 1
        return count

    def ingest_controlled(self, dataset_path: str) -> int:
        """Ingest controlled JSONL scenarios from *dataset_path*."""
        from statebudgetmem.schemas import MemoryRecord

        count = 0
        with open(dataset_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                scenario_data = json.loads(line)
                memories = scenario_data.get("memories", [])
                for mem_data in memories:
                    mem = MemoryRecord(**mem_data)
                    self.engine.ingest(mem)
                    count += 1
        return count

    # ---- query ----------------------------------------------------------

    def ask(
        self,
        query: str,
        *,
        context: str | None = None,
        reference_time: date | str | None = None,
        state_key: StateKey | None = None,
    ) -> PipelineResult:
        """Run the full pipeline: route → view → retrieve → context."""

        # 1. Route
        qr = QueryRecord(text=query, context=context)
        query_type = self.router.classify(qr)
        view_type = self.router.route(query, query_type)

        # 2. Select view
        entries = self.view_manager.select_view(
            view_type, state_key=state_key, reference_time=reference_time
        )

        # 3. Retrieve
        items = [e.to_dict() for e in entries]
        ranked = self.retriever.retrieve(query, items, top_k=self.top_k)

        # 4. Build context (respect budget)
        context_parts: list[str] = []
        budget = 0
        for _, item, score in ranked:
            text = str(item.get("text", item.get("content", "")))
            if budget + len(text) > self.max_context_chars:
                break
            context_parts.append(f"[{item.get('memory_id','')[:8]}] {text}")
            budget += len(text)
        context_text = "\n".join(context_parts)

        # 5. (Optional) LLM answer — skipped in offline mode
        answer = ""

        return PipelineResult(
            query=query,
            query_type=query_type.value,
            view_type=view_type.value,
            view_entries=entries,
            retrieved=ranked,
            context_text=context_text,
            answer=answer,
        )


def build_pipeline(
    *,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str = "deepseek-ai/DeepSeek-V4-Flash",
    use_llm_router: bool = False,
    top_k: int = 5,
) -> MemoryPipeline:
    """Factory that builds a MemoryPipeline with optional LLM routing.

    If *use_llm_router* is True and *llm_api_key* is provided, uses
    ``LLMQueryRouter``.  Otherwise falls back to ``RuleBasedRouter``.
    """
    if use_llm_router and llm_api_key:
        router = LLMQueryRouter(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
        )
    else:
        router = RuleBasedRouter()

    return MemoryPipeline(router=router, top_k=top_k)
