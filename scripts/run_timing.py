"""Timing study: every (precision, batch size) setting on the Banking77 test set.

For each precision the model is loaded once; every batch size is then timed with
the harness in uneven_compression.timing (warm-up discarded, repeated trials).

Writes to --out (default results/timing/<SLURM job id or timestamp>/):
  summary.parquet   one row per setting: latencies, throughput, memory
  samples.parquet   one row per timed full batch: total_ms, model_ms (for re-analysis)
  run.json          environment: GPU, node, library versions, arguments

Usage:
  python scripts/run_timing.py                        # all 16 settings
  python scripts/run_timing.py --quick                # 2 trials on 256 texts: a dry run
  python scripts/run_timing.py --precisions fp32 fp16 # a subset (e.g. on a Mac)
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

import pandas as pd

from uneven_compression.ingestion import load_banking77
from uneven_compression.serving import PRECISIONS, load_model, pick_device
from uneven_compression.timing import time_setting

BATCH_SIZES = (1, 8, 32, 128)


def environment(device: str, args: argparse.Namespace) -> dict:
    import torch
    import transformers
    env = {
        "node": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    try:
        import bitsandbytes
        env["bitsandbytes"] = bitsandbytes.__version__
    except ImportError:
        env["bitsandbytes"] = None
    return env


def free(model) -> None:
    """Release a model before loading the next precision, so peak memory starts clean."""
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCH_SIZES))
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="use only the first N test texts")
    p.add_argument("--quick", action="store_true", help="dry run: 2 trials on 256 texts")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.quick:
        args.trials, args.limit = 2, args.limit or 256

    texts = [r.text for r in load_banking77("test")]      # row-count check runs here
    if args.limit:
        texts = texts[:args.limit]

    device = pick_device()
    run_id = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("results/timing") / run_id
    out.mkdir(parents=True, exist_ok=True)
    env = environment(device, args)
    (out / "run.json").write_text(json.dumps(env, indent=1))
    print(f"{env['gpu']} on {env['node']} -> {out}", flush=True)

    summary, samples = [], []
    for precision in args.precisions:
        tok, model, device = load_model(precision, device=device)
        for batch_size in args.batch_sizes:
            r = time_setting(tok, model, device, texts, batch_size,
                             trials=args.trials, warmup_batches=args.warmup_batches)
            summary.append({
                "precision": precision, "batch_size": batch_size, "device": device,
                "gpu": env["gpu"], "node": env["node"], "n_texts": r["n_texts"],
                "trials": r["trials"], "warmup_batches": r["warmup_batches"],
                "full_batches_per_trial": r["full_batches_per_trial"],
                "total_p50_ms": r["latency_total"]["p50_ms"], "total_p95_ms": r["latency_total"]["p95_ms"],
                "model_p50_ms": r["latency_model"]["p50_ms"], "model_p95_ms": r["latency_model"]["p95_ms"],
                "throughput_median": r["throughput_median"],
                "throughput_min": min(r["throughput_per_trial"]),
                "throughput_max": max(r["throughput_per_trial"]),
                "peak_memory_mb": r["peak_memory_mb"], "weights_mb": r["weights_mb"],
                "activations_mb": r["activations_mb"],
            })
            samples += [{"precision": precision, "batch_size": batch_size, "sample": i,
                         "total_ms": t * 1000, "model_ms": m * 1000}
                        for i, (t, m) in enumerate(zip(r["samples_total_s"], r["samples_model_s"]))]
            s = summary[-1]
            print(f"{precision:>4} b{batch_size:<3} total p50 {s['total_p50_ms']:8.2f} ms  "
                  f"model p50 {s['model_p50_ms']:8.2f} ms  {s['throughput_median']:8.0f} msg/s", flush=True)
            # write after every setting, so a job killed by its time limit keeps what it finished
            pd.DataFrame(summary).to_parquet(out / "summary.parquet", index=False)
            pd.DataFrame(samples).to_parquet(out / "samples.parquet", index=False)
        free(model)

    print(f"done: {len(summary)} settings -> {out}", flush=True)


if __name__ == "__main__":
    main()
