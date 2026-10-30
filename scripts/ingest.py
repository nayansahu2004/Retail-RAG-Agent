"""Offline indexing pipeline: raw data to FAISS and BM25 artifacts.

Run with::

    python -m scripts.ingest

The script also executes inside ``docker build`` so the shipped image contains
the index and the downloaded model files, allowing the container to start
without network access for retrieval.
"""

import json

from app.config import get_settings
from app.rag.chunking import load_corpus
from app.rag.embeddings import FastEmbedEmbedder
from app.rag.index_store import build_index
from app.rag.reranker import FastEmbedReranker


def main() -> None:
    """Load the corpus, build the artifacts, and warm the reranker cache."""
    s = get_settings()
    chunks = load_corpus(s.data_dir, s.chunk_max_chars, s.chunk_overlap_chars)
    print(f"Loaded {len(chunks)} chunks from {s.data_dir}")

    embedder = FastEmbedEmbedder(s.embedding_model, s.model_cache_dir)
    manifest = build_index(chunks, embedder, s.artifacts_dir, s.embedding_model)

    FastEmbedReranker(s.reranker_model, s.model_cache_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
