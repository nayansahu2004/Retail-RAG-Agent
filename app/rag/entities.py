"""Core data structures shared across the retrieval pipeline."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single retrievable unit of text and the metadata attached to it."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.metadata.get("title", self.id)

    @property
    def source_type(self) -> str:
        return self.metadata.get("source_type", "unknown")


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, together with the signals that produced it."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
