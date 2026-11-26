import numpy as np
import pytest

from uneven_compression.analysis import (bootstrap_gap, count_groups, flips_by_margin, group_drops,
                                         margins, mcnemar_exact, per_intent_f1, reach)


def test_count_groups_takes_fewest_and_most():
    rare, middle, common = count_groups([50, 10, 30, 20, 40, 60], k=2)
    assert sorted(rare) == [1, 3] and sorted(common) == [0, 5] and sorted(middle) == [2, 4]


def test_count_groups_refuses_a_tie_at_the_cut():
    with pytest.raises(ValueError, match="rare cut"):
        count_groups([10, 20, 20, 40, 50, 60], k=2)
    with pytest.raises(ValueError, match="common cut"):
        count_groups([10, 20, 30, 50, 50, 60], k=2)


def test_per_intent_f1_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 6, 200), rng.integers(0, 7, 200)    # label 6 is never true
    expected = 100 * sklearn.f1_score(y, p, labels=list(range(8)), average=None, zero_division=0)
    np.testing.assert_allclose(per_intent_f1(y, p, 8), expected)


def test_group_drops_sign_positive_means_worse():
    y = np.array([0, 0, 1, 1])
    base = np.array([0, 0, 1, 1])                 # perfect
    worse = np.array([0, 1, 1, 1])                # one label-0 message now wrong
    d = group_drops(y, base, worse, {"zero": np.array([0]), "one": np.array([1])}, 2)
    assert d["zero"] > 0 and d["one"] > 0         # both F1s fall: 0 loses recall, 1 loses precision


def test_bootstrap_gap_is_zero_with_identical_predictions():
    y = np.arange(40) % 4
    r = bootstrap_gap(y, y, y, np.array([0, 1]), np.array([2, 3]), 4, n_boot=50)
    assert r["gap"] == r["ci_low"] == r["ci_high"] == 0


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)      # 2 x (1/2)^6
    assert mcnemar_exact(13, 14) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 9) == mcnemar_exact(9, 3)       # two-sided: symmetric


def test_margins_top1_minus_top2():
    np.testing.assert_allclose(margins(np.array([[0.7, 0.2, 0.1], [0.4, 0.45, 0.15]])), [0.5, 0.05])


def test_flips_by_margin_and_reach():
    m = np.array([0.01, 0.02, 0.08, 0.3, 0.9])
    f = np.array([True, False, True, False, False])
    rows = flips_by_margin(m, f, [0, 0.05, 0.1, 1.01])
    assert [(r["messages"], r["flips"]) for r in rows] == [(2, 1), (1, 1), (2, 0)]
    assert rows[0]["rate"] == 50.0
    assert reach(m, f) == 0.08
    assert reach(m, np.zeros(5, bool)) is None
