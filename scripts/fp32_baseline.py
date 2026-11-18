"""FP32 baseline: quality of bge-small-en-v1.5 at full precision, batch size 32.

Runs on any machine (CUDA, Apple MPS or CPU). Writes results/fp32_baseline.json.

Three measurements:
  1. Our baseline: Banking77 (mteb/banking77, current revision, 9,993 / 3,076),
     logistic regression on all train embeddings -> accuracy, macro-F1, per-intent F1.
  2. Calibration: MTEB's own classification recipe on the dataset revision the
     published bge-small score used (10,003 / 3,080). Should land near 85.74.
  3. SciFact nDCG@10 (published: 71.275), with and without the query instruction
     bge recommends for retrieval.

Usage:  python scripts/fp32_baseline.py
"""
from __future__ import annotations

import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from uneven_compression.batching import make_batches
from uneven_compression.evaluation import classification_scores, mteb_style_classification, ndcg_at_k
from uneven_compression.ingestion import BANKING77, BANKING77_NUM_LABELS, load_banking77, load_scifact
from uneven_compression.serving import MODEL_NAME, embed_batches, load_model

BATCH_SIZE = 32
PUBLISHED = {"banking77_accuracy": 85.74025974025975, "banking77_macro_f1": 85.71493566466381,
             "scifact_ndcg_at_10": 71.275}
PUBLISHED_BANKING77_REVISION = "0fd18e25b25c072e09e0d92ab615fda904d66300"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
OUT = Path("results/fp32_baseline.json")


def embed(tok, model, device, texts):
    t0 = time.perf_counter()
    vecs = embed_batches(tok, model, device, make_batches(list(texts), BATCH_SIZE))
    assert vecs.shape == (len(texts), model.config.hidden_size), vecs.shape
    return vecs, time.perf_counter() - t0


def main() -> None:
    import torch
    import transformers
    from datasets import load_dataset

    tok, model, device = load_model("fp32")
    print(f"model={MODEL_NAME} precision=fp32 device={device} batch_size={BATCH_SIZE}")
    result = {"model": MODEL_NAME, "precision": "fp32", "device": device, "batch_size": BATCH_SIZE,
              "machine": platform.platform(), "torch": torch.__version__,
              "transformers": transformers.__version__, "published": PUBLISHED, "seconds": {}}

    # 1. Our baseline on our pipeline's data
    train, test = load_banking77("train"), load_banking77("test")
    train_x, result["seconds"]["embed_banking77_train"] = embed(tok, model, device, [r.text for r in train])
    test_x, result["seconds"]["embed_banking77_test"] = embed(tok, model, device, [r.text for r in test])
    train_y, test_y = [r.label for r in train], [r.label for r in test]
    ours = classification_scores(train_x, train_y, test_x, test_y, BANKING77_NUM_LABELS)
    ours["mteb_recipe"] = mteb_style_classification(train_x, train_y, test_x, test_y, BANKING77_NUM_LABELS)
    ours.update(n_train=len(train), n_test=len(test),
                train_count_per_intent=[Counter(train_y)[i] for i in range(BANKING77_NUM_LABELS)],
                test_count_per_intent=[Counter(test_y)[i] for i in range(BANKING77_NUM_LABELS)])
    result["banking77_ours"] = ours
    print(f"Banking77 (ours, {len(train)}/{len(test)}): full-train LR acc={ours['accuracy']:.2f} "
          f"macro-F1={ours['macro_f1']:.2f} | MTEB recipe acc={ours['mteb_recipe']['accuracy']:.2f}")

    # 2. Calibration on the revision the published score used
    old = load_dataset(BANKING77, revision=PUBLISHED_BANKING77_REVISION)
    old_train, old_test = load_banking77("train", rows=old["train"]), load_banking77("test", rows=old["test"])
    old_train_x, _ = embed(tok, model, device, [r.text for r in old_train])
    old_test_x, _ = embed(tok, model, device, [r.text for r in old_test])
    calib = mteb_style_classification(old_train_x, [r.label for r in old_train],
                                      old_test_x, [r.label for r in old_test], BANKING77_NUM_LABELS)
    calib.update(revision=PUBLISHED_BANKING77_REVISION, n_train=len(old_train), n_test=len(old_test))
    result["banking77_calibration"] = calib
    print(f"Banking77 calibration (rev {PUBLISHED_BANKING77_REVISION[:8]}, {len(old_train)}/{len(old_test)}): "
          f"MTEB recipe acc={calib['accuracy']:.2f} ± {calib['accuracy_std']:.2f} "
          f"(published {PUBLISHED['banking77_accuracy']:.2f})")

    # 3. SciFact retrieval
    corpus, queries, qrels = load_scifact("test")
    doc_texts = [f"{d.title} {d.text}".strip() if d.title else d.text for d in corpus]
    doc_x, result["seconds"]["embed_scifact_corpus"] = embed(tok, model, device, doc_texts)
    doc_ids, query_ids = [d.request_id for d in corpus], [q.request_id for q in queries]
    sci = {"n_docs": len(corpus), "n_queries": len(queries)}
    for name, prefix in [("no_instruction", ""), ("with_instruction", QUERY_INSTRUCTION)]:
        q_x, _ = embed(tok, model, device, [prefix + q.text for q in queries])
        sci[f"ndcg_at_10_{name}"] = ndcg_at_k(q_x, doc_x, query_ids, doc_ids, qrels, k=10)
    result["scifact"] = sci
    print(f"SciFact ({len(corpus)} docs, {len(queries)} queries): nDCG@10 "
          f"no-instruction={sci['ndcg_at_10_no_instruction']:.2f} "
          f"with-instruction={sci['ndcg_at_10_with_instruction']:.2f} (published {PUBLISHED['scifact_ndcg_at_10']})")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
