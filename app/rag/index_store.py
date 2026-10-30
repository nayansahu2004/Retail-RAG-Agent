"""Build, save, and load the on-disk retrieval artifacts.

Artifact layout::

    artifacts/
        chunks.jsonl     text and metadata of every chunk (row i == vector i)
        faiss.index      dense vectors
        bm25.json        tokenised corpus for keyword search
        manifest.json    what was built, with which models, when
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from app.rag.bm25_store import BM25Store
from app.rag.embeddings import Embedder
from app.rag.entities import Chunk
from app.rag.vector_store import FaissVectorStore

CHUNKS_FILE = "chunks.jsonl"
FAISS_FILE = "faiss.index"
BM25_FILE = "bm25.json"
MANIFEST_FILE = "manifest.json"


def build_index(chunks: list[Chunk], embedder: Embedder, out_dir: Path, embedding_model: str = "") -> dict:
    """Embed ``chunks`` and write the FAISS, BM25, chunk, and manifest files.

    Returns:
        The manifest dict that was written to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [c.text for c in chunks]

    started = time.perf_counter()
    vectors = embedder.embed_documents(texts)
    embed_seconds = time.perf_counter() - started

    FaissVectorStore.build(vectors).save(out_dir / FAISS_FILE)
    BM25Store.build(texts).save(out_dir / BM25_FILE)
    with (out_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    manifest = {
        "chunks": len(chunks),
        "products": sum(c.source_type == "product" for c in chunks),
        "policy_chunks": sum(c.source_type == "policy" for c in chunks),
        "embedding_model": embedding_model,
        "embedding_dim": int(vectors.shape[1]),
        "embed_seconds": round(embed_seconds, 2),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_chunks(path: Path) -> list[Chunk]:
    """Load chunks from a JSONL file written by ``build_index``."""
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def load_index(artifacts_dir: Path) -> tuple[list[Chunk], FaissVectorStore, BM25Store]:
    """Load chunks and both indexes from an artifacts directory.

    Raises:
        FileNotFoundError: If any required artifact file is missing.
        ValueError: If the chunk, FAISS, and BM25 counts disagree.
    """
    missing = [n for n in (CHUNKS_FILE, FAISS_FILE, BM25_FILE) if not (artifacts_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"Index files missing in {artifacts_dir}: {missing}. Run: python -m scripts.ingest")
    chunks = load_chunks(artifacts_dir / CHUNKS_FILE)
    vector_store = FaissVectorStore.load(artifacts_dir / FAISS_FILE)
    bm25 = BM25Store.load(artifacts_dir / BM25_FILE)
    if not (len(chunks) == len(vector_store) == len(bm25)):
        raise ValueError("Index files are out of sync. Re-run: python -m scripts.ingest")
    return chunks, vector_store, bm25
