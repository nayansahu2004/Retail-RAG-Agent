"""Text embedding interface and a FastEmbed (ONNX) implementation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Minimal protocol every embedding backend must satisfy."""

    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return an L2-normalised copy of ``matrix`` as ``float32``.

    After normalisation, inner product equals cosine similarity.
    """
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)


class FastEmbedEmbedder:
    """ONNX-backed embedding model loaded via FastEmbed."""

    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of documents and return a normalised ``(n, dim)`` matrix."""
        vectors = np.array(list(self._model.embed(list(texts), batch_size=64)), dtype=np.float32)
        return l2_normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query using the model's query-side formatting."""
        vector = np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)
        return l2_normalize(vector)
