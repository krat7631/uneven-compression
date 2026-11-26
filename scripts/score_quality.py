"""Score saved embeddings: quality per precision, and change relative to FP32.

Reads a run written by scripts/embed_precisions.py; runs on any machine (no GPU).

Per precision:
  Banking77  logistic regression (all training data): accuracy, macro-F1,
             per-intent F1, every test prediction
  SciFact    nDCG@10 (queries carry bge's retrieval instruction)
Against FP32 (same run, so same machine and libraries):
  cosine between each test message's vector and its FP32 vector (min, mean)
  predictions changed, split into right->wrong and wrong->right
  per-intent F1 change (precision minus FP32)

Writes results/quality/<run id>.json.

Usage:  python scripts/score_quality.py results/embeddings/<run id>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from uneven_compression.evaluation import classification_scores, ndcg_at_k
from uneven_compression.ingestion import BANKING77_NUM_LABELS, load_banking77, load_scifact
from uneven_compression.serving import PRECISIONS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text())
    limit = run["limit"]
    precisions = [pr for pr in PRECISIONS if (args.run_dir / pr / "banking77_test.npy").exists()]

    train_y = [r.label for r in load_banking77("train")][:limit]
    test_y = [r.label for r in load_banking77("test")][:limit]
    corpus, queries, qrels = load_scifact("test")
    doc_ids, query_ids = [d.request_id for d in corpus], [q.request_id for q in queries]
    assert len(train_y) == run["rows"]["banking77_train"] and len(test_y) == run["rows"]["banking77_test"]

    result = {"run": run, "precisions": {},
              "train_count_per_intent": [Counter(train_y)[i] for i in range(BANKING77_NUM_LABELS)],
              "test_count_per_intent": [Counter(test_y)[i] for i in range(BANKING77_NUM_LABELS)]}
    vecs = {}
    for pr in precisions:
        load = lambda name: np.load(args.run_dir / pr / f"{name}.npy")
        vecs[pr] = load("banking77_test")
        scores = classification_scores(load("banking77_train"), train_y, vecs[pr], test_y, BANKING77_NUM_LABELS)
        # nDCG over a truncated corpus would be meaningless, so only on full runs
        scores["scifact_ndcg_at_10"] = None if limit else ndcg_at_k(
            load("scifact_queries"), load("scifact_corpus"), query_ids, doc_ids, qrels, k=10)
        result["precisions"][pr] = scores

    if "fp32" in vecs:
        base = result["precisions"]["fp32"]
        truth = np.array(test_y)
        base_right = np.array(base["predictions"]) == truth
        for pr in precisions:
            s = result["precisions"][pr]
            cos = np.sum(vecs[pr] * vecs["fp32"], axis=1)
            right = np.array(s["predictions"]) == truth
            s["vs_fp32"] = {
                "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                "predictions_changed": int((np.array(s["predictions"]) != np.array(base["predictions"])).sum()),
                "right_to_wrong": int((base_right & ~right).sum()),
                "wrong_to_right": int((~base_right & right).sum()),
                "per_intent_f1_change": [a - b for a, b in zip(s["per_intent_f1"], base["per_intent_f1"])],
            }

    out = Path("results/quality") / f"{args.run_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))

    print(f"{'':>4} {'acc':>6} {'macroF1':>8} {'nDCG@10':>8} {'min cos':>8} {'changed':>8} {'R->W':>5} {'W->R':>5}")
    for pr in precisions:
        s = result["precisions"][pr]
        v = s.get("vs_fp32", {})
        ndcg = f"{s['scifact_ndcg_at_10']:8.2f}" if s["scifact_ndcg_at_10"] is not None else f"{'-':>8}"
        print(f"{pr:>4} {s['accuracy']:6.2f} {s['macro_f1']:8.2f} {ndcg} {v.get('min_cosine', float('nan')):8.4f} "
              f"{v.get('predictions_changed', 0):8d} {v.get('right_to_wrong', 0):5d} {v.get('wrong_to_right', 0):5d}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
