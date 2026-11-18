"""Preliminary checks that run without a CUDA GPU (Apple MPS or CPU).

  1. Batch invariance: FP32 embeddings of the Banking77 test set at batch size
     1 vs 32. Batch size should change speed only, never the vectors.
  2. FP16 vs FP32 quality on Banking77 (same classifier recipe as the baseline).
     INT8 and NF4 need bitsandbytes on CUDA and run on the V100 instead.

Writes results/mac_preliminary.json. Not a timing study.

Usage:  python scripts/mac_preliminary_checks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uneven_compression.batching import make_batches
from uneven_compression.evaluation import classification_scores
from uneven_compression.ingestion import BANKING77_NUM_LABELS, load_banking77
from uneven_compression.serving import embed_batches, load_model

OUT = Path("results/mac_preliminary.json")


def main() -> None:
    train, test = load_banking77("train"), load_banking77("test")
    train_t, test_t = [r.text for r in train], [r.text for r in test]
    train_y, test_y = [r.label for r in train], [r.label for r in test]
    result = {}

    tok, model, device = load_model("fp32")
    fp32_test_b32 = embed_batches(tok, model, device, make_batches(test_t, 32))
    fp32_test_b1 = embed_batches(tok, model, device, make_batches(test_t, 1))
    cos = np.sum(fp32_test_b1 * fp32_test_b32, axis=1)
    result["batch_invariance_fp32"] = {
        "device": device, "n": len(test_t),
        "min_cosine_b1_vs_b32": float(cos.min()),
        "max_abs_diff": float(np.abs(fp32_test_b1 - fp32_test_b32).max()),
    }
    fp32_train = embed_batches(tok, model, device, make_batches(train_t, 32))
    fp32 = classification_scores(fp32_train, train_y, fp32_test_b32, test_y, BANKING77_NUM_LABELS)
    del model

    tok, model, device = load_model("fp16", device=device)
    fp16_train = embed_batches(tok, model, device, make_batches(train_t, 32))
    fp16_test = embed_batches(tok, model, device, make_batches(test_t, 32))
    fp16 = classification_scores(fp16_train, train_y, fp16_test, test_y, BANKING77_NUM_LABELS)
    drift = np.sum(fp16_test * fp32_test_b32, axis=1)
    agree = np.mean(np.array(fp16["predictions"]) == np.array(fp32["predictions"]))
    per_intent_drop = np.array(fp32["per_intent_f1"]) - np.array(fp16["per_intent_f1"])
    result["fp16_vs_fp32"] = {
        "device": device,
        "fp32_accuracy": fp32["accuracy"], "fp16_accuracy": fp16["accuracy"],
        "fp32_macro_f1": fp32["macro_f1"], "fp16_macro_f1": fp16["macro_f1"],
        "min_cosine_fp16_vs_fp32": float(drift.min()), "mean_cosine_fp16_vs_fp32": float(drift.mean()),
        "prediction_agreement": float(agree),
        "max_abs_per_intent_f1_change": float(np.abs(per_intent_drop).max()),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
