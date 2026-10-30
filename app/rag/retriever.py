"""Hybrid dense + sparse retrieval with optional cross-encoder reranking.

The ``HybridRetriever.retrieve`` method accepts a ``RetrievalConfig`` so the
evaluation harness can sweep configurations against the same instance.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings
from app.observability import STAGE_LATENCY
from app.rag.bm25_store import BM25Store
from app.rag.embeddings import Embedder
from app.rag.entities import Chunk, RetrievedChunk
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import Reranker
from app.rag.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)

# Instruction recommended by BAAI for short search queries with bge-*-en-v1.5 models.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class RetrievalConfig:
    """Immutable configuration for a single retrieval call."""

    dense_weight: float = 1.0
    sparse_weight: float = 0.3
    rrf_k: int = 60
    rerank: bool = False
    query_prefix: str = ""

    def __post_init__(self):
        if self.dense_weight <= 0 and self.sparse_weight <= 0:
            raise ValueError("At least one of dense_weight or sparse_weight must be > 0")

    @classmethod
    def from_settings(cls, s: Settings) -> "RetrievalConfig":
        """Build a config from application ``Settings``."""
        return cls(
            dense_weight=s.retrieval_dense_weight,
            sparse_weight=s.retrieval_sparse_weight,
            rrf_k=s.rrf_k,
            rerank=s.retrieval_rerank,
            query_prefix=s.embedding_query_prefix,
        )

    @property
    def label(self) -> str:
        """Short human-readable identifier for logs and evaluation tables."""
        if self.sparse_weight <= 0:
            name = "dense"
        elif self.dense_weight <= 0:
            name = "bm25"
        else:
            name = f"hybrid bm25={self.sparse_weight:g} k={self.rrf_k}"
        if self.query_prefix and self.dense_weight > 0:
            name += " +prefix"
        if self.rerank:
            name += " +rerank"
        return name


class HybridRetriever:
    """Combine FAISS and BM25 results via weighted RRF, with optional rerank."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        vector_store: FaissVectorStore,
        bm25: BM25Store,
        settings: Settings,
        reranker: Reranker | None = None,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.settings = settings
        self.default_config = RetrievalConfig.from_settings(settings)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_type: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        """Retrieve the top chunks for ``query``.

        Args:
            query: Natural-language query.
            top_k: Final number of chunks to return (defaults to settings).
            source_type: Restrict results to a single source type, for
                example ``"product"`` or ``"policy"``.
            config: Overrides the retriever's default fusion configuration.

        Returns:
            A ``(results, timings_ms)`` tuple. ``timings_ms`` includes keys
            such as ``dense_ms``, ``sparse_ms``, ``rerank_ms`` where applicable.
        """
        s = self.settings
        cfg = config or self.default_config
        top_k = top_k or s.final_top_k
        timings: dict[str, float] = {}
        started = time.perf_counter()

        allowed: Callable[[int], bool] = (
            (lambda i: self.chunks[i].source_type == source_type) if source_type else (lambda i: True)
        )
        dense_k = len(self.chunks) if source_type else s.dense_top_k
        sparse_k = len(self.chunks) if source_type else s.sparse_top_k

        dense: list[tuple[int, float]] = []
        sparse: list[tuple[int, float]] = []

        if cfg.dense_weight > 0:
            t = time.perf_counter()
            query_vector = self.embedder.embed_query(cfg.query_prefix + query)
            dense = [hit for hit in self.vector_store.search(query_vector, dense_k) if allowed(hit[0])]
            dense = dense[: s.dense_top_k]
            timings["dense_ms"] = self._record("dense", t)

        if cfg.sparse_weight > 0:
            t = time.perf_counter()
            sparse = [hit for hit in self.bm25.search(query, sparse_k) if allowed(hit[0])]
            sparse = sparse[: s.sparse_top_k]
            timings["sparse_ms"] = self._record("sparse", t)

        dense_rank = {doc: rank for rank, (doc, _) in enumerate(dense, start=1)}
        sparse_rank = {doc: rank for rank, (doc, _) in enumerate(sparse, start=1)}

        if cfg.sparse_weight <= 0:
            ranked = dense
        elif cfg.dense_weight <= 0:
            ranked = sparse
        else:
            ranked = reciprocal_rank_fusion(
                [[d for d, _ in dense], [d for d, _ in sparse]],
                k=cfg.rrf_k,
                weights=[cfg.dense_weight, cfg.sparse_weight],
            )

        rerank_scores: dict[int, float] = {}
        if cfg.rerank and ranked:
            if self.reranker is None:
                logger.warning("Rerank requested but no reranker is loaded; skipping rerank")
            else:
                t = time.perf_counter()
                candidates = [doc for doc, _ in ranked[: s.rerank_candidates]]
                scores = self.reranker.score(query, [self.chunks[d].text for d in candidates])
                rerank_scores = dict(zip(candidates, scores, strict=True))
                ranked = sorted(rerank_scores.items(), key=lambda item: (-item[1], item[0]))
                timings["rerank_ms"] = self._record("rerank", t)

        results = [
            RetrievedChunk(
                chunk=self.chunks[doc],
                score=float(score),
                dense_rank=dense_rank.get(doc),
                sparse_rank=sparse_rank.get(doc),
                rerank_score=rerank_scores.get(doc),
            )
            for doc, score in ranked[:top_k]
        ]
        timings["retrieval_total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return results, timings

    @staticmethod
    def _record(stage: str, started: float) -> float:
        seconds = time.perf_counter() - started
        STAGE_LATENCY.labels(stage=stage).observe(seconds)
        return round(seconds * 1000, 2)


def build_retriever(settings: Settings, with_reranker: bool | None = None) -> HybridRetriever:
    """Load persisted artifacts and local ONNX models into a ``HybridRetriever``.

    Args:
        settings: Application settings.
        with_reranker: Force-load (or skip) the reranker regardless of
            ``settings.retrieval_rerank``. Defaults to the settings value.
    """
    from app.rag.embeddings import FastEmbedEmbedder
    from app.rag.index_store import load_index
    from app.rag.reranker import FastEmbedReranker

    load_reranker = settings.retrieval_rerank if with_reranker is None else with_reranker
    chunks, vector_store, bm25 = load_index(settings.artifacts_dir)
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
    reranker = FastEmbedReranker(settings.reranker_model, settings.model_cache_dir) if load_reranker else None
    return HybridRetriever(chunks, embedder, vector_store, bm25, settings, reranker)
