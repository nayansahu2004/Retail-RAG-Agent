"""Information-retrieval metrics for ranked chunk ids.

``ids`` are the ranked chunk identifiers produced by the retriever. ``relevant``
is a list of chunk ids or ``"<prefix>::"`` entries that count as correct.
"""

import math
from collections.abc import Sequence

import numpy as np


def matched_entry(chunk_id: str, relevant: Sequence[str]) -> str | None:
    """Return the ``relevant`` entry this chunk satisfies, or ``None``.

    A chunk matches when its id equals an entry, or when it starts with
    ``"<entry>::"`` (treating ``relevant`` entries as id prefixes).
    """
    for entry in relevant:
        if chunk_id == entry or chunk_id.startswith(entry + "::"):
            return entry
    return None


def _first_hits(ids: Sequence[str], relevant: Sequence[str], k: int) -> list[int]:
    """1-based ranks of the first chunk matching each distinct relevant entry."""
    seen: set[str] = set()
    ranks: list[int] = []
    for rank, chunk_id in enumerate(ids[:k], start=1):
        entry = matched_entry(chunk_id, relevant)
        if entry and entry not in seen:
            seen.add(entry)
            ranks.append(rank)
    return ranks


def hit_at_k(ids: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return 1.0 if any relevant entry appears in the top ``k``, else 0.0."""
    return 1.0 if _first_hits(ids, relevant, k) else 0.0


def recall_at_k(ids: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of distinct relevant entries that appear in the top ``k``."""
    return len(_first_hits(ids, relevant, k)) / len(relevant) if relevant else 0.0


def mrr_at_k(ids: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Mean reciprocal rank of the first relevant entry within the top ``k``."""
    ranks = _first_hits(ids, relevant, k)
    return 1.0 / ranks[0] if ranks else 0.0


def ndcg_at_k(ids: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised discounted cumulative gain at ``k`` for binary relevance."""
    ranks = _first_hits(ids, relevant, k)
    dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile of ``values`` (``0.0`` when empty)."""
    return float(np.percentile(values, q)) if values else 0.0
