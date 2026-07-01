"""
retrieval.tfidf — Generic offline TF-IDF retriever

Works with any dict-like item that exposes ``text`` (or ``content``).
Used by the views-based pipeline and can replace the baseline-specific
TfidfCosineRetriever when method-agnostic retrieval is needed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    """Deterministic mixed CJK/ASCII tokenizer."""
    lowered = text.lower()
    tokens: list[str] = []
    tokens.extend(_WORD_RE.findall(lowered))
    cjk = _CJK_RE.findall(lowered)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


class TfidfRetriever:
    """Small, deterministic TF-IDF + cosine-similarity retriever.

    Accepts items that are dicts or objects with a ``text`` (or ``content``)
    attribute.  Produces ranked lists of ``(item, score)`` tuples.
    """

    def retrieve(
        self,
        query: str,
        items: list[dict[str, Any]],
        *,
        text_key: str = "text",
        top_k: int = 5,
    ) -> list[tuple[int, dict[str, Any], float]]:
        """Return top-k ``(index, item, score)`` sorted by cosine similarity.

        Parameters
        ----------
        query : str
            Natural-language query.
        items : list[dict]
            Candidates; each must have *text_key*.
        text_key : str
            Key used to access the candidate text.
        top_k : int
            Maximum results to return.
        """
        if top_k <= 0 or not items:
            return []

        documents = [str(_get_text(item, text_key)) for item in items]
        doc_tokens = [_tokenize(doc) for doc in documents]
        query_tokens = _tokenize(query)
        idf = self._idf(doc_tokens + [query_tokens])
        query_vector = self._tfidf(query_tokens, idf)

        ranked: list[tuple[int, dict[str, Any], float]] = []
        for idx, (item, tokens) in enumerate(zip(items, doc_tokens)):
            score = self._cosine(query_vector, self._tfidf(tokens, idf))
            ranked.append((idx, item, score))

        ranked.sort(key=lambda x: (-x[2], x[0], str(x[1].get("memory_id", ""))))
        return ranked[:top_k]

    # ---- internal helpers (mirror baselines/tfidf/retriever.py) ----
    @staticmethod
    def _idf(documents: list[list[str]]) -> dict[str, float]:
        doc_count = len(documents)
        terms = sorted({t for doc in documents for t in set(doc)})
        idf: dict[str, float] = {}
        for term in terms:
            containing = sum(1 for doc in documents if term in set(doc))
            idf[term] = math.log((1 + doc_count) / (1 + containing)) + 1.0
        return idf

    @staticmethod
    def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = sum(counts.values())
        return {t: (c / total) * idf.get(t, 0.0) for t, c in counts.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        common = set(left) & set(right)
        num = sum(left[t] * right[t] for t in common)
        l_norm = math.sqrt(sum(v * v for v in left.values()))
        r_norm = math.sqrt(sum(v * v for v in right.values()))
        if l_norm == 0.0 or r_norm == 0.0:
            return 0.0
        return num / (l_norm * r_norm)


def _get_text(item: dict[str, Any], key: str) -> str:
    """Extract text from an item that may be a dict or have attributes."""
    if isinstance(item, dict):
        return str(item.get(key, item.get("content", "")))
    return str(getattr(item, key, getattr(item, "content", "")))
