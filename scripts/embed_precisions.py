"""Embed the evaluation data at every precision and save the vectors (the storage stage).

GPU work only: scoring happens afterwards, anywhere, with scripts/score_quality.py.
Batch size is fixed at 32: batch size does not change the vectors (batch 1 vs 32
differ by at most 2.7e-7), so quality needs one run per precision, not one per setting.

Writes to --out (default results/embeddings/<SLURM job id or timestamp>/):
  <precision>/banking77_train.npy   (9,993 x 384 float32, rows in dataset order)
  <precision>/banking77_test.npy    (3,076 x 384)
  <precision>/scifact_corpus.npy    (5,183 x 384; text = title + " " + abstract)
  <precision>/scifact_queries.npy   (300 x 384; with bge's retrieval instruction)
  run.json                          environment, row counts, seconds per precision

Usage:
  python scripts/embed_precisions.py                          # all four precisions (V100)
  python scripts/embed_precisions.py --precisions fp32 fp16   # what a Mac can run
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np

from uneven_compression.batching import make_batches
from uneven_compression.ingestion import load_banking77, load_scifact
from uneven_compression.serving import PRECISIONS, embed_batches, load_model, pick_device

BATCH_SIZE = 32
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def texts_to_embed(limit: int | None) -> dict[str, list[str]]:
    """Every text set we score, in the loaders' order (the row-count checks run here)."""
    train, test = load_banking77("train"), load_banking77("test")
    corpus, queries, _ = load_scifact("test")
    sets = {
        "banking77_train": [r.text for r in train],
        "banking77_test": [r.text for r in test],
        "scifact_corpus": [f"{d.title} {d.text}".strip() if d.title else d.text for d in corpus],
        "scifact_queries": [QUERY_INSTRUCTION + q.text for q in queries],
    }
    return {k: v[:limit] for k, v in sets.items()} if limit else sets


def main() -> None:
    import torch
    import transformers

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--limit", type=int, default=None, help="dry run: only the first N texts of each set")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    sets = texts_to_embed(args.limit)
    device = pick_device()
    run_id = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("results/embeddings") / run_id
    out.mkdir(parents=True, exist_ok=True)
    run = {
        "node": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "batch_size": BATCH_SIZE, "limit": args.limit, "query_instruction": QUERY_INSTRUCTION,
        "rows": {k: len(v) for k, v in sets.items()}, "seconds": {},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"{run['gpu']} on {run['node']} -> {out}  rows={run['rows']}", flush=True)

    for precision in args.precisions:
        tok, model, device = load_model(precision, device=device)
        (out / precision).mkdir(exist_ok=True)
        t0 = time.perf_counter()
        for name, texts in sets.items():
            vecs = embed_batches(tok, model, device, make_batches(texts, BATCH_SIZE))
            assert vecs.shape == (len(texts), model.config.hidden_size), (name, vecs.shape)
            assert np.isfinite(vecs).all(), f"{precision}/{name}: non-finite values"
            np.save(out / precision / f"{name}.npy", vecs)
        run["seconds"][precision] = round(time.perf_counter() - t0, 1)
        print(f"{precision:>4}: saved {len(sets)} sets in {run['seconds'][precision]} s", flush=True)
        (out / "run.json").write_text(json.dumps(run, indent=1))   # after every precision
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"done: {len(args.precisions)} precisions -> {out}", flush=True)


if __name__ == "__main__":
    main()
