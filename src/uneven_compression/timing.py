"""Timing harness: latency, throughput and peak memory for one (precision, batch size) setting.

One setting runs as:
  warm-up   the first `warmup_batches` batches, timed and thrown away (GPU waking up)
  trials    `trials` full passes over the texts, batch by batch
Every batch records two clocks:
  total     tokenize + move to device + model + copy vectors back (what a user waits for)
  model     the forward pass alone (the part precision changes)
Latency statistics use full batches only: the smaller last batch is not a typical
batch of `batch_size`. Throughput counts every text: all of them were embedded.

GPU work is asynchronous: the CPU queues it and moves on. Every clock reading is
preceded by synchronize(), so a clock never stops while the GPU is still working.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .batching import make_batches
from .serving import cls_pool_normalize, run_model, tokenize_batch

MB = 1024 ** 2


def synchronize(device: str) -> None:
    """Block until all queued work on `device` has finished (no-op on the CPU)."""
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_batch(tokenizer, model, device: str, texts: Sequence[str]) -> tuple[float, float]:
    """Embed one batch and return (total seconds, model-only seconds)."""
    synchronize(device)
    t0 = time.perf_counter()
    enc = tokenize_batch(tokenizer, device, texts)
    synchronize(device)
    m0 = time.perf_counter()
    hidden = run_model(model, enc)
    synchronize(device)
    m1 = time.perf_counter()
    cls_pool_normalize(hidden)          # copies vectors back to the CPU
    synchronize(device)
    t1 = time.perf_counter()
    return t1 - t0, m1 - m0


def _summary(seconds: list[float]) -> dict[str, float]:
    ms = np.array(seconds) * 1000
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95))}


def time_setting(tokenizer, model, device: str, texts: Sequence[str], batch_size: int,
                 trials: int = 5, warmup_batches: int = 5) -> dict:
    """Time one batch size for an already-loaded model. See the module docstring for the protocol."""
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials!r}")
    batches = make_batches(texts, batch_size)
    if not any(len(b) == batch_size for b in batches):
        raise ValueError(f"{len(texts)} texts give no full batch of {batch_size}")

    for b in batches[:warmup_batches]:
        time_batch(tokenizer, model, device, b)

    if device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()

    total_s: list[float] = []
    model_s: list[float] = []
    throughput: list[float] = []
    for _ in range(trials):
        synchronize(device)
        trial_start = time.perf_counter()
        for b in batches:
            t, m = time_batch(tokenizer, model, device, b)
            if len(b) == batch_size:
                total_s.append(t)
                model_s.append(m)
        throughput.append(len(texts) / (time.perf_counter() - trial_start))

    peak_mb = weights_mb = None
    if device == "cuda":
        import torch
        peak_mb = torch.cuda.max_memory_allocated() / MB
    if hasattr(model, "get_memory_footprint"):
        weights_mb = model.get_memory_footprint() / MB

    return {
        "device": device,
        "batch_size": batch_size,
        "n_texts": len(texts),
        "trials": trials,
        "warmup_batches": min(warmup_batches, len(batches)),
        "full_batches_per_trial": sum(len(b) == batch_size for b in batches),
        "latency_total": _summary(total_s),
        "latency_model": _summary(model_s),
        "throughput_per_trial": throughput,
        "throughput_median": float(np.median(throughput)),
        "peak_memory_mb": peak_mb,
        "weights_mb": weights_mb,
        "activations_mb": None if peak_mb is None or weights_mb is None else peak_mb - weights_mb,
        "samples_total_s": total_s,
        "samples_model_s": model_s,
    }
