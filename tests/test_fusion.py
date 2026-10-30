import pytest

from app.rag.fusion import reciprocal_rank_fusion


def test_doc_found_by_both_retrievers_ranks_first():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 5]], k=60)
    assert fused[0][0] == 3


def test_scores_follow_formula():
    fused = dict(reciprocal_rank_fusion([[7]], k=60))
    assert fused[7] == 1 / 61


def test_empty_and_deterministic_ties():
    assert reciprocal_rank_fusion([[], []]) == []
    assert [d for d, _ in reciprocal_rank_fusion([[2], [1]])] == [1, 2]


def test_low_weight_list_cannot_overrule_strong_list():
    dense, bm25 = [10, 11, 12], [99, 10]
    equal = [d for d, _ in reciprocal_rank_fusion([dense, bm25], k=60)]
    weighted = [d for d, _ in reciprocal_rank_fusion([dense, bm25], k=60, weights=[1.0, 0.1])]
    assert equal.index(99) < equal.index(11)
    assert weighted.index(99) > weighted.index(12)


def test_zero_weight_ignores_list_and_bad_weights_rejected():
    assert [d for d, _ in reciprocal_rank_fusion([[1], [2]], weights=[1.0, 0.0])] == [1]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])
