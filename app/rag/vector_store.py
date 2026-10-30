"""Dense vector index backed by FAISS ``IndexFlatIP`` (exact cosine search)."""

from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    """Thin wrapper around a FAISS inner-product index."""

    def __init__(self, index: faiss.Index):
        self.index = index

    @classmethod
    def build(cls, vectors: np.ndarray) -> "FaissVectorStore":
        """Build a flat inner-product index over the given row-vectors."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return the ``k`` nearest neighbours as ``(row_id, score)`` pairs."""
        k = min(k, self.index.ntotal)
        if k <= 0:
            return []
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, ids = self.index.search(query, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        return cls(faiss.read_index(str(path)))

    def __len__(self) -> int:
        return self.index.ntotal
