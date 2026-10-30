import pytest

from eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k

IDS = ["a", "policy::r::s::0", "policy::r::s::1", "b"]


def test_prefix_matching_and_hit():
    assert hit_at_k(IDS, ["policy::r::s"], 2) == 1.0
    assert hit_at_k(IDS, ["policy::r::s"], 1) == 0.0


def test_mrr_uses_first_hit():
    assert mrr_at_k(IDS, ["b"], 10) == 0.25


def test_recall_counts_each_relevant_entry_once():
    assert recall_at_k(IDS, ["policy::r::s", "zzz"], 4) == 0.5


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(["x", "a"], ["a"], 2) < 1
