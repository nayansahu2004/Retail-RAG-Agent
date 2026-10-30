# File: retail-rag-agent/tests/conftest.py
"""
Shared test fixtures.

Unit tests must be fast, offline and deterministic, so real models and real
LLM APIs are replaced by small fakes that follow the same interfaces
(Embedder, Reranker, ChatLLM). This is dependency injection paying off.
"""

import hashlib
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.client import LLMResult
from app.memory import SessionStore
from app.rag.bm25_store import BM25Store
from app.rag.embeddings import l2_normalize
from app.rag.entities import Chunk
from app.rag.pipeline import RAGPipeline
from app.rag.prompts import PLAN_SYSTEM_PROMPT
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import FaissVectorStore


class FakeEmbedder:
    """Hashing-trick bag of words: similar wording -> similar vectors. No model needed."""

    dim = 128

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts):
        return l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text):
        self.last_query = text
        return l2_normalize(self._vec(text))


class FakeReranker:
    """Scores by word overlap with the query."""

    def score(self, query, texts):
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [float(len(q & set(re.findall(r"[a-z0-9]+", t.lower())))) for t in texts]


class FakeLLM:
    """
    Answers like the real models would: planner calls get {"queries": [...]},
    answer calls get {"answer": ..., "coverage": ...}. Every call is recorded.
    """

    def __init__(self, reply: str | None = None, queries: list[str] | None = None, plan_reply: str | None = None):
        self.reply = reply or json.dumps({"answer": "The return window is 10 days [1].", "coverage": "full"})
        self.queries = queries
        self.plan_reply = plan_reply
        self.calls: list[dict] = []

    def generate(self, system, messages, temperature=None, json_mode=False):
        kind = "plan" if system == PLAN_SYSTEM_PROMPT else "answer"
        self.calls.append({"kind": kind, "system": system, "messages": messages, "json_mode": json_mode})
        if kind == "plan":
            question = messages[-1]["content"].split("Latest message: ")[-1]
            text = self.plan_reply or json.dumps({"queries": self.queries or [question]})
        else:
            text = self.reply
        return LLMResult(text=text, provider="fake", model="fake-1", latency_ms=1.0)

    def kinds(self) -> list[str]:
        return [c["kind"] for c in self.calls]


CHUNKS = [
    Chunk(
        "product::SKU-LAP-001",
        "Product: Kestrel Neo 514\nSKU: SKU-LAP-001\nCategory: Laptops\nPrice: ₹54,999\nFanless design that runs completely silent",
        {"source_type": "product", "title": "Kestrel Neo 514", "category": "Laptops", "price_inr": 54999},
    ),
    Chunk(
        "product::SKU-LAP-002",
        "Product: Kestrel Neo 512\nSKU: SKU-LAP-002\nCategory: Laptops\nPrice: ₹89,999\nDedicated NVIDIA RTX graphics",
        {"source_type": "product", "title": "Kestrel Neo 512", "category": "Laptops", "price_inr": 89999},
    ),
    Chunk(
        "product::SKU-SHO-001",
        "Product: Stride Glide 300\nSKU: SKU-SHO-001\nCategory: Running Shoes\nPrice: ₹4,999\nGore-Tex waterproof upper",
        {"source_type": "product", "title": "Stride Glide 300", "category": "Running Shoes", "price_inr": 4999},
    ),
    Chunk(
        "policy::returns::standard-return-window::0",
        "Returns > Standard return window\nMost items can be returned within 10 days of delivery.",
        {"source_type": "policy", "title": "Returns: Standard return window"},
    ),
    Chunk(
        "policy::shipping::shipping-charges::0",
        "Shipping > Shipping charges\nStandard delivery is free on orders of ₹499 or more.",
        {"source_type": "policy", "title": "Shipping: Shipping charges"},
    ),
]


@pytest.fixture
def settings() -> Settings:
    # Reranking is on in tests so every stage of the pipeline is exercised.
    return Settings(
        _env_file=None, dense_top_k=5, sparse_top_k=5, rerank_candidates=5, final_top_k=3, retrieval_rerank=True
    )


@pytest.fixture
def retriever(settings) -> HybridRetriever:
    embedder = FakeEmbedder()
    texts = [c.text for c in CHUNKS]
    return HybridRetriever(
        chunks=list(CHUNKS),
        embedder=embedder,
        vector_store=FaissVectorStore.build(embedder.embed_documents(texts)),
        bm25=BM25Store.build(texts),
        settings=settings,
        reranker=FakeReranker(),
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(retriever, fake_llm, settings) -> RAGPipeline:
    return RAGPipeline(retriever, fake_llm, SessionStore(max_turns=2, ttl_s=60), settings)


@pytest.fixture
def client(retriever, pipeline):
    from app.main import create_app
    from app.services import Services

    services = Services(retriever=retriever, rag=pipeline, sessions=pipeline.sessions, team=None)
    with TestClient(create_app(services)) as test_client:
        yield test_client