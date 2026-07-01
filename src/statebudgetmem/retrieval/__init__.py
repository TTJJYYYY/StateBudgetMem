"""
StateBudgetMem / Retrieval Module

Shared retrieval protocols and implementations.
Method-specific retrievers (e.g. FAISS for MemoryBank) stay in baselines/.
"""

from statebudgetmem.retrieval.interfaces import Embedder, Retriever
from statebudgetmem.retrieval.tfidf import TfidfRetriever

__all__ = ["Embedder", "Retriever", "TfidfRetriever"]
