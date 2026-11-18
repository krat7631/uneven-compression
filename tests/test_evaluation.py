import numpy as np
import pytest

from uneven_compression.evaluation import (classification_scores, mteb_style_classification,
                                           ndcg_at_k, undersample)


def _two_clusters(n_per=20, seed=0):
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(-3, 0.1, (n_per, 4)), rng.normal(3, 0.1, (n_per, 4))])
    y = [0] * n_per + [1] * n_per
    return x, y


def test_classification_separable_data_is_perfect():
    x, y = _two_clusters()
    s = classification_scores(x, y, x, y, num_labels=2)
    assert s["accuracy"] == 100.0
    assert s["macro_f1"] == 100.0
    assert s["per_intent_f1"] == [100.0, 100.0]


def test_per_intent_f1_has_one_entry_per_label_even_if_unseen_in_test():
    x, y = _two_clusters()
    s = classification_scores(x, y, x[:5], y[:5], num_labels=3)
    assert len(s["per_intent_f1"]) == 3
    assert s["per_intent_f1"][2] == 0.0


def test_undersample_caps_each_label():
    labels = [0] * 10 + [1] * 3
    idx = undersample(labels, per_label=4, rng=np.random.default_rng(0))
    picked = [labels[i] for i in idx]
    assert picked.count(0) == 4 and picked.count(1) == 3


def test_mteb_style_is_reproducible_with_a_seed():
    x, y = _two_clusters()
    a = mteb_style_classification(x, y, x, y, num_labels=2, n_experiments=3, per_label=2)
    b = mteb_style_classification(x, y, x, y, num_labels=2, n_experiments=3, per_label=2)
    assert a == b


def test_ndcg_perfect_and_worst_ranking():
    docs = np.eye(3)                        # doc i points along axis i
    q = np.array([[1.0, 0.0, 0.0]])         # the query is closest to doc 0
    ids = ["d0", "d1", "d2"]
    assert ndcg_at_k(q, docs, ["q"], ids, {"q": {"d0": 1.0}}, k=3) == 100.0
    # only d2 is relevant; it ties with d1 at the bottom, so it lands at rank 2 or 3
    worst = ndcg_at_k(q, docs, ["q"], ids, {"q": {"d2": 1.0}}, k=3)
    assert worst == pytest.approx(100 / np.log2(3)) or worst == pytest.approx(100 / np.log2(4))


def test_ndcg_hand_computed():
    # ranking d0, d1, d2; relevant: d1 (rank 2). DCG = 1/log2(3); ideal DCG = 1/log2(2) = 1
    docs = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    q = np.array([[1.0, 0.0]])
    got = ndcg_at_k(q, docs, ["q"], ["d0", "d1", "d2"], {"q": {"d1": 1.0}}, k=10)
    assert got == pytest.approx(100 / np.log2(3))


def test_ndcg_relevant_doc_outside_top_k_scores_zero():
    docs = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    q = np.array([[1.0, 0.0]])
    assert ndcg_at_k(q, docs, ["q"], ["d0", "d1", "d2"], {"q": {"d2": 1.0}}, k=2) == 0.0
