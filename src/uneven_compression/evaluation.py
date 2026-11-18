"""Evaluation: quality metrics per precision, efficiency metrics per precision x batch size.

Quality is measured on frozen embeddings:
  Banking77  logistic regression trained on train-set embeddings, scored on the
             test set: accuracy, macro-F1, and F1 for each of the 77 intents.
  SciFact    rank all documents by cosine similarity for each query: nDCG@10.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def classification_scores(train_x: np.ndarray, train_y: Sequence[int],
                          test_x: np.ndarray, test_y: Sequence[int],
                          num_labels: int, max_iter: int = 1000, seed: int = 0) -> dict:
    """Fit logistic regression on train embeddings; score it on test embeddings.

    Returns accuracy and macro-F1 in percent, per_intent_f1 (a list of
    num_labels F1 scores in percent, position i = intent i), and predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    labels = list(range(num_labels))
    per_intent = f1_score(test_y, pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": 100 * accuracy_score(test_y, pred),
        "macro_f1": 100 * f1_score(test_y, pred, labels=labels, average="macro", zero_division=0),
        "per_intent_f1": [100 * float(f) for f in per_intent],
        "predictions": [int(p) for p in pred],
    }


def undersample(labels: Sequence[int], per_label: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of a random subset with at most `per_label` examples of each label."""
    counts: dict[int, int] = {}
    keep = []
    for i in rng.permutation(len(labels)):
        y = labels[i]
        if counts.get(y, 0) < per_label:
            keep.append(i)
            counts[y] = counts.get(y, 0) + 1
    return np.array(sorted(keep))


def mteb_style_classification(train_x: np.ndarray, train_y: Sequence[int],
                              test_x: np.ndarray, test_y: Sequence[int], num_labels: int,
                              n_experiments: int = 10, per_label: int = 8, seed: int = 42) -> dict:
    """MTEB's classification recipe (mteb 1.1.0 defaults), for calibration only.

    Repeat n_experiments times: take per_label random training examples of each
    label, fit logistic regression (max_iter=100), score the full test set.
    Report the mean and standard deviation over the repeats.
    """
    rng = np.random.default_rng(seed)
    train_y = np.asarray(train_y)
    accs, f1s = [], []
    for _ in range(n_experiments):
        idx = undersample(train_y, per_label, rng)
        s = classification_scores(train_x[idx], train_y[idx], test_x, test_y,
                                  num_labels, max_iter=100)
        accs.append(s["accuracy"])
        f1s.append(s["macro_f1"])
    return {"accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "n_experiments": n_experiments,
            "per_label": per_label}


def ndcg_at_k(query_vecs: np.ndarray, doc_vecs: np.ndarray,
              query_ids: Sequence[str], doc_ids: Sequence[str],
              qrels: Mapping[str, Mapping[str, float]], k: int = 10) -> float:
    """Mean nDCG@k (percent) over the queries, ranking documents by dot product.

    Vectors must be unit length, so the dot product is the cosine similarity.
    DCG@k = sum over the top k of rel / log2(rank + 1), rank starting at 1;
    nDCG@k = DCG@k / (the DCG@k of the ideal ordering).
    """
    scores = query_vecs @ doc_vecs.T
    top = np.argsort(-scores, axis=1)[:, :k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    total = 0.0
    for qi, qid in enumerate(query_ids):
        judged = qrels.get(qid, {})
        gains = np.array([judged.get(doc_ids[d], 0.0) for d in top[qi]])
        ideal = np.sort(np.array(list(judged.values()), dtype=float))[::-1][:k]
        idcg = float(ideal @ discounts[:len(ideal)])
        total += float(gains @ discounts[:len(gains)]) / idcg if idcg > 0 else 0.0
    return 100 * total / len(query_ids)
