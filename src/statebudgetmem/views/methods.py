from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from statebudgetmem.baselines.tfidf.retriever import TfidfCosineRetriever
from statebudgetmem.schemas import MemoryRecord, QueryRecord, QueryType
from statebudgetmem.schemas.results import MethodResult, RetrievedMemory as ResultMemory
from statebudgetmem.views.manager import RecordViewManager
from statebudgetmem.views.models import ViewName, ViewPolicy


@dataclass(frozen=True)
class RankedCandidate:
    memory: MemoryRecord
    score: float
    source_view: ViewName
    metadata: dict[str, Any]


class ViewMemoryMethod:
    """MemoryMethod implementation for flat/current/history/dual views."""

    def __init__(
        self,
        *,
        name: str,
        view: ViewName,
        retriever: TfidfCosineRetriever | None = None,
        manager: RecordViewManager | None = None,
        policy: ViewPolicy | None = None,
    ) -> None:
        self._name = name
        self.view = view
        self.retriever = retriever or TfidfCosineRetriever()
        self.manager = manager or RecordViewManager(policy=policy)

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self.manager.reset()

    def ingest(self, memories: list[MemoryRecord]) -> None:
        self.manager.ingest(memories)

    def retrieve(
        self,
        query: QueryRecord,
        *,
        top_k: int,
        token_budget: int | None = None,
        mutate: bool = False,
    ) -> MethodResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        started = time.perf_counter()
        ranked = self._rank(query, top_k=top_k)
        ranked = _apply_token_budget(ranked, token_budget)
        ranked = ranked[:top_k]

        result_memories = [
            ResultMemory(
                memory_id=item.memory.memory_id,
                score=item.score,
                rank=rank,
                token_cost=item.memory.token_cost,
                source_view=item.source_view.value,
                metadata=item.metadata,
            )
            for rank, item in enumerate(ranked, start=1)
        ]

        latency_ms = (time.perf_counter() - started) * 1000.0

        return MethodResult(
            method_name=self.name,
            query_id=query.query_id,
            retrieved_memories=result_memories,
            predicted_query_type=query.query_type,
            total_token_cost=sum(item.token_cost for item in result_memories),
            latency_ms=latency_ms,
            metadata={
                "view": self.view.value,
                "token_budget": token_budget,
                "mutate": mutate,
                "router_source": "oracle_query_type",
                "reference_time": query.reference_time.isoformat(),
            },
        )

    def _rank(self, query: QueryRecord, *, top_k: int) -> list[RankedCandidate]:
        if query.query_type is QueryType.GENERAL:
            return []

        if self.view is ViewName.FLAT:
            memories = self.manager.records_for_query(query, view=ViewName.FLAT)
            return self._rank_one_view(query, memories, ViewName.FLAT, top_k)

        if self.view is ViewName.CURRENT:
            memories = self.manager.records_for_query(query, view=ViewName.CURRENT)
            return self._rank_one_view(query, memories, ViewName.CURRENT, top_k)

        if self.view is ViewName.HISTORY:
            memories = self.manager.records_for_query(query, view=ViewName.HISTORY)
            return self._rank_one_view(query, memories, ViewName.HISTORY, top_k)

        if self.view is ViewName.DUAL:
            return self._rank_dual(query, top_k=top_k)

        raise ValueError(f"unsupported view: {self.view}")

    def _rank_one_view(
        self,
        query: QueryRecord,
        memories: list[MemoryRecord],
        source_view: ViewName,
        top_k: int,
    ) -> list[RankedCandidate]:
        retrieved = self.retriever.retrieve(
            query,
            memories,
            top_k=max(top_k, len(memories)),
        )

        return [
            RankedCandidate(
                memory=item.memory,
                score=item.score,
                source_view=source_view,
                metadata={"raw_rank": item.rank},
            )
            for item in retrieved
        ]

    def _rank_dual(self, query: QueryRecord, *, top_k: int) -> list[RankedCandidate]:
        """Rank all selected-view candidates in one shared TF-IDF space.

        Current and history candidates used to be ranked in separate document
        collections and their cosine scores were then compared directly. This
        method creates one deduplicated candidate pool first, so every score is
        computed with the same IDF statistics.
        """

        decision = self.manager.route(query)
        if not decision.selected_views:
            return []

        candidate_by_id: dict[str, MemoryRecord] = {}
        source_views_by_id: dict[str, list[ViewName]] = {}

        for view in decision.selected_views:
            for memory in self.manager.records_for_query(query, view=view):
                candidate_by_id[memory.memory_id] = memory
                sources = source_views_by_id.setdefault(memory.memory_id, [])
                if view not in sources:
                    sources.append(view)

        candidates = sorted(
            candidate_by_id.values(),
            key=lambda item: (item.event_time, item.memory_id),
        )
        retrieved = self.retriever.retrieve(
            query,
            candidates,
            top_k=max(top_k, len(candidates)),
        )

        ranked: list[RankedCandidate] = []
        for item in retrieved:
            sources = source_views_by_id[item.memory.memory_id]
            primary_source = (
                ViewName.CURRENT if ViewName.CURRENT in sources else sources[0]
            )
            ranked.append(
                RankedCandidate(
                    memory=item.memory,
                    score=item.score,
                    source_view=primary_source,
                    metadata={
                        "raw_rank": item.rank,
                        "source_views": [view.value for view in sources],
                        "dual_route": [
                            view.value for view in decision.selected_views
                        ],
                        "ranking_space": "shared_tfidf_candidate_pool",
                    },
                )
            )

        if query.query_type is QueryType.CHANGE and self.manager.policy.expand_change_lineage:
            ranked = self._expand_lineage(ranked, query=query)

        ranked.sort(
            key=lambda item: (-item.score, item.memory.event_time, item.memory.memory_id)
        )
        return ranked

    def _expand_lineage(
        self,
        ranked: list[RankedCandidate],
        *,
        query: QueryRecord,
    ) -> list[RankedCandidate]:
        by_id = self.manager.memories_by_id
        expanded: list[RankedCandidate] = list(ranked)
        seen = {item.memory.memory_id for item in expanded}
        base_score = ranked[-1].score if ranked else 0.0

        for item in ranked:
            for state in self.manager.version_manager.lineage(item.memory.memory_id):
                if state.memory_id in seen or state.memory_id not in by_id:
                    continue

                seen.add(state.memory_id)
                expanded.append(
                    RankedCandidate(
                        memory=by_id[state.memory_id],
                        score=max(base_score * 0.95, item.score * 0.85),
                        source_view=ViewName.HISTORY,
                        metadata={
                            "expanded_from": item.memory.memory_id,
                            "expansion": "version_lineage",
                            "query_type": query.query_type.value,
                        },
                    )
                )

        return expanded


class FlatViewMemoryMethod(ViewMemoryMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="views_flat", view=ViewName.FLAT, **kwargs)


class CurrentOnlyMemoryMethod(ViewMemoryMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="views_current_only", view=ViewName.CURRENT, **kwargs)


class HistoryOnlyMemoryMethod(ViewMemoryMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="views_history_only", view=ViewName.HISTORY, **kwargs)


class DualViewMemoryMethod(ViewMemoryMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="views_dual", view=ViewName.DUAL, **kwargs)


def _apply_token_budget(
    ranked: list[RankedCandidate],
    token_budget: int | None,
) -> list[RankedCandidate]:
    if token_budget is None:
        return ranked

    if token_budget < 0:
        raise ValueError("token_budget must be non-negative or None")

    kept: list[RankedCandidate] = []
    used = 0

    for item in ranked:
        if used + item.memory.token_cost > token_budget:
            continue

        kept.append(item)
        used += item.memory.token_cost

    return kept
