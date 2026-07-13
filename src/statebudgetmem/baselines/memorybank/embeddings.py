"""Shared embedding construction for dense MemoryBank-based methods."""

from __future__ import annotations

import hashlib

import numpy as np


class HashEmbeddingModel:
    """Offline deterministic encoder used by smoke and contract tests."""

    name = "deterministic_hash_embedding"

    def __init__(self, dim: int = 32) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim!r}")
        self.dimension = dim

    def encode(self, text: str) -> np.ndarray:
        return deterministic_hash_embedding(text, self.dimension)


class SentenceTransformerEmbeddingModel:
    """Thin local-model wrapper with the scalar-text API MemoryBank expects."""

    def __init__(self, model_name: str) -> None:
        if not model_name or model_name == "method_default":
            model_name = "all-MiniLM-L6-v2"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Sentence-transformer embeddings require the memorybank extra: "
                "pip install -e '.[memorybank]'"
            ) from exc
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(
            self._model.encode(text, normalize_embeddings=True), dtype=np.float32
        )


def build_embedding_model(backend: str, model_name: str):
    """Build the shared encoder used by MemoryBank and StateBudgetMem adapters."""
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in {"method_default", "hash", "deterministic_hash"}:
        return HashEmbeddingModel()
    if normalized in {"sentence_transformer", "sentence_transformers", "minilm"}:
        return SentenceTransformerEmbeddingModel(model_name)
    raise ValueError(
        "unsupported embedding_backend for dense MemoryBank methods: "
        f"{backend!r}"
    )


def deterministic_hash_embedding(text: str, dim: int = 32) -> np.ndarray:
    """Return a stable normalized hash embedding without external services."""
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim!r}")

    vector = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = digest[0] % dim
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        vector[index] += sign

    if not vector.any():
        vector[0] = 1.0

    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm > 0 else vector
