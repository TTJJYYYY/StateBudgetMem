"""Deterministic TF-IDF baseline and its controlled-data runner."""

from statebudgetmem.baselines.tfidf.adapter import TfidfMemoryMethod
from statebudgetmem.baselines.tfidf.retriever import TfidfCosineRetriever, tokenize
from statebudgetmem.baselines.tfidf.runner import BaselineConfig, run_baseline

__all__ = [
    "TfidfMemoryMethod",
    "TfidfCosineRetriever",
    "tokenize",
    "BaselineConfig",
    "run_baseline",
]
