import pytest

from app.rag.bm25_store import tokenize
from app.rag.retriever import BGE_QUERY_PREFIX, RetrievalConfig


def test_tokenizer_keeps_skus_together():
    assert "sku-lap-001" in tokenize("Price of SKU-LAP-001?")


def test_bm25_only_matches_exact_model_number(retriever):
    config = RetrievalConfig(dense_weight=0.0, sparse_weight=1.0)
    results, timings = retriever.retrieve("Kestrel Neo 512 price", top_k=1, config=config)
    assert results[0].chunk.id == "product::SKU-LAP-002"
    assert "dense_ms" not in timings


def test_dense_only_skips_bm25(retriever):
    results, timings = retriever.retrieve("silent laptop", config=RetrievalConfig(sparse_weight=0.0))
    assert results and "sparse_ms" not in timings
    assert all(r.sparse_rank is None for r in results)


def test_bm25_finds_bare_sku_in_hybrid(retriever):
    config = RetrievalConfig(sparse_weight=0.3)
    results, _ = retriever.retrieve("Is SKU-SHO-001 in stock?", top_k=1, config=config)
    assert results[0].chunk.id == "product::SKU-SHO-001"


def test_query_prefix_is_applied_to_dense_query_only(retriever):
    retriever.retrieve("silent laptop", config=RetrievalConfig(query_prefix=BGE_QUERY_PREFIX))
    assert retriever.embedder.last_query == BGE_QUERY_PREFIX + "silent laptop"


def test_source_type_filter(retriever):
    results, _ = retriever.retrieve("return delivery price", source_type="policy")
    assert results and all(r.chunk.source_type == "policy" for r in results)


def test_rerank_returns_top_k_with_timings(retriever):
    results, timings = retriever.retrieve("silent fanless laptop", top_k=2, config=RetrievalConfig(rerank=True))
    assert len(results) == 2 and results[0].rerank_score is not None
    assert {"dense_ms", "sparse_ms", "rerank_ms", "retrieval_total_ms"} <= timings.keys()


def test_default_config_comes_from_settings(retriever, settings):
    assert retriever.default_config == RetrievalConfig.from_settings(settings)


def test_config_labels_and_validation():
    assert RetrievalConfig(sparse_weight=0.0).label == "dense"
    assert RetrievalConfig(sparse_weight=0.3, rerank=True).label == "hybrid bm25=0.3 k=60 +rerank"
    with pytest.raises(ValueError):
        RetrievalConfig(dense_weight=0.0, sparse_weight=0.0)
