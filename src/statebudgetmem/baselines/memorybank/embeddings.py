"""Local deterministic embeddings for MemoryBank smoke runs."""

from __future__ import annotations

import hashlib

import numpy as np


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
