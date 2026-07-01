from __future__ import annotations

import math
import re
from collections import Counter

from statebudgetmem.schemas import MemoryRecord, QueryRecord, RetrievedMemory

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text deterministically for the baseline."""
    lowered = text.lower()
    tokens = _WORD_RE.findall(lowered)
    cjk_chars = _CJK_RE.findall(lowered)
    tokens.extend(cjk_chars)
    tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
    return tokens


class TfidfCosineRetriever:
    """Small offline TF-IDF + cosine retriever with stable tie-breaking."""

    def retrieve(
        self,
        query: QueryRecord,
        memories: list[MemoryRecord],
        top_k: int,
    ) -> list[RetrievedMemory]:
        if top_k <= 0 or not memories:
            return []

        documents = [memory.text for memory in memories]
        doc_tokens = [tokenize(text) for text in documents]
        query_tokens = tokenize(query.text)
        idf = self._idf(doc_tokens + [query_tokens])
        query_vector = self._tfidf(query_tokens, idf)

        ranked: list[tuple[int, MemoryRecord, float]] = []
        for index, (memory, tokens) in enumerate(zip(memories, doc_tokens)):
            score = self._cosine(query_vector, self._tfidf(tokens, idf))
            ranked.append((index, memory, score))

        ranked.sort(key=lambda item: (-item[2], item[0], item[1].memory_id))
        return [
            RetrievedMemory(memory=memory, score=score, rank=rank)
            for rank, (_, memory, score) in enumerate(ranked[:top_k], start=1)
        ]

    @staticmethod
    def _idf(documents: list[list[str]]) -> dict[str, float]:
        doc_count = len(documents)
        terms = sorted({term for document in documents for term in set(document)})
        idf: dict[str, float] = {}
        for term in terms:
            containing = sum(1 for document in documents if term in set(document))
            idf[term] = math.log((1 + doc_count) / (1 + containing)) + 1.0
        return idf

    @staticmethod
    def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = sum(counts.values())
        return {term: (count / total) * idf.get(term, 0.0) for term, count in counts.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        common = set(left) & set(right)
        numerator = sum(left[term] * right[term] for term in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
