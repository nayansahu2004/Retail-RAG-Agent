"""Weighted Reciprocal Rank Fusion for combining ranked result lists."""

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse several ranked lists into a single ranking.

    The fusion score for a document is ``sum(weight / (k + rank))`` across
    every list it appears in. Lists with weight ``<= 0`` are skipped.

    Args:
        ranked_lists: One sequence per retriever, each containing document ids
            in ranked order.
        k: Reciprocal-rank damping constant.
        weights: One weight per list, or ``None`` to weight every list equally.

    Returns:
        ``(doc_id, score)`` pairs sorted by descending score (ties broken by
        ascending doc id for determinism).

    Raises:
        ValueError: If ``weights`` is provided and its length does not match
            ``ranked_lists``.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have one value per ranked list")
    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(ranked_lists, weights, strict=True):
        if weight <= 0:
            continue
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
