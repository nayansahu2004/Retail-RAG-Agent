"""Cross-encoder reranking stage: interface and FastEmbed implementation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Reranker(Protocol):
    """Minimal protocol for a reranker backend."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    """ONNX-backed cross-encoder reranker loaded via FastEmbed."""

    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return a relevance score for each ``(query, text)`` pair."""
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, list(texts), batch_size=32)]
