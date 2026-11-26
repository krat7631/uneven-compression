"""Analysis: where does compression's quality cost land?

Two questions, asked of the test predictions at each precision against FP32:

Confirmatory (fixed before the INT8/NF4 results were seen)
  Do rare intents (the 20 with the fewest training examples) lose more F1 than
  common intents (the 20 with the most)?  gap = rare mean drop - common mean drop,
  with a bootstrap interval over test messages, and FP16 as the noise reference.

Exploratory (proposed after seeing the results)
  Are the messages that flip the ones FP32 was least sure about?  margin =
  FP32 probability of the top intent - probability of the runner-up.  A
  precision's "reach" is the largest margin among the messages it flipped.

Only numpy: F1 is counted directly, and McNemar's test is the exact binomial.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np


def count_groups(counts: Sequence[int], k: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split label indices into (rare, middle, common) by training count.

    rare = the k labels with the fewest examples, common = the k with the most.
    Raises ValueError if a tie straddles either cut, because the group would then
    depend on how the tie was broken rather than on the data.
    """
    counts = np.asarray(counts)
    if 2 * k > len(counts):
        raise ValueError(f"need at least {2 * k} labels for two groups of {k}, got {len(counts)}")
    order = np.argsort(counts, kind="stable")
    s = counts[order]
    if s[k - 1] == s[k]:
        raise ValueError(f"tie at the rare cut: {k}th and {k + 1}th labels both have {s[k]} examples")
    if s[-k] == s[-k - 1]:
        raise ValueError(f"tie at the common cut: both sides have {s[-k]} examples")
    return order[:k], order[k:-k], order[-k:]


def per_intent_f1(y_true, y_pred, num_labels: int) -> np.ndarray:
    """F1 for each label, in percent; 0 where a label never appears in truth or prediction.

    Same definition as sklearn's f1_score(average=None, zero_division=0).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.bincount(y_true[y_true == y_pred], minlength=num_labels)
    true_n = np.bincount(y_true, minlength=num_labels)
    pred_n = np.bincount(y_pred, minlength=num_labels)
    denom = true_n + pred_n                       # = 2tp + fp + fn
    return np.divide(200.0 * tp, denom, out=np.zeros(num_labels), where=denom > 0)


def group_drops(y_true, base_pred, pred, groups: dict[str, np.ndarray], num_labels: int) -> dict[str, float]:
    """Mean F1 drop (base - compressed, percent points) for each named group of labels."""
    drop = per_intent_f1(y_true, base_pred, num_labels) - per_intent_f1(y_true, pred, num_labels)
    return {name: float(drop[idx].mean()) for name, idx in groups.items()}


def bootstrap_gap(y_true, base_pred, pred, rare, common, num_labels: int,
                  n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """rare mean drop - common mean drop, with a 95% percentile interval.

    Each resample draws test messages with replacement and scores the SAME draw
    under both precisions (a paired bootstrap), so the interval reflects which
    messages happened to be in the test set, not a second source of noise.
    """
    y_true, base_pred, pred = map(np.asarray, (y_true, base_pred, pred))
    groups = {"rare": rare, "common": common}

    def gap(idx):
        d = group_drops(y_true[idx], base_pred[idx], pred[idx], groups, num_labels)
        return d["rare"] - d["common"]

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = np.array([gap(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"gap": gap(np.arange(n)), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": n_boot, "seed": seed}


def mcnemar_exact(right_to_wrong: int, wrong_to_right: int) -> float:
    """Two-sided exact McNemar p-value: is one kind of flip more common than the other?

    Under "no difference" each of the n flips is a fair coin, so the count of
    right->wrong flips is Binomial(n, 1/2).
    """
    n = right_to_wrong + wrong_to_right
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(right_to_wrong, wrong_to_right) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def margins(proba: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability for each row: 0 = a coin toss, 1 = certain."""
    top2 = np.sort(np.asarray(proba), axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def flips_by_margin(margin, flipped, edges: Sequence[float]) -> list[dict]:
    """Messages, flips and flip rate (percent) in each margin band [edges[i], edges[i+1])."""
    margin, flipped = np.asarray(margin), np.asarray(flipped, dtype=bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (margin >= lo) & (margin < hi)
        n, f = int(band.sum()), int(flipped[band].sum())
        rows.append({"low": lo, "high": hi, "messages": n, "flips": f,
                     "rate": 100 * f / n if n else 0.0})
    return rows


def reach(margin, flipped) -> float | None:
    """The largest FP32 margin among flipped messages; None if nothing flipped."""
    margin, flipped = np.asarray(margin), np.asarray(flipped, dtype=bool)
    return float(margin[flipped].max()) if flipped.any() else None
