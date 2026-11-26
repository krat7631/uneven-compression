"""Classify new messages into Banking77 intents, and flag the ones compression may change.

This is the service applied to data it has never seen: a CSV of messages goes in,
one row per message comes out with

    intent       the predicted Banking77 intent (its name, e.g. "card_swallowed")
    confidence   the classifier's probability for that intent
    runner_up    the second most likely intent
    margin       confidence minus the runner-up's probability (0 = coin toss, 1 = certain)
    at_risk      margin < risk_margin: a message this close to the decision border
                 may be answered differently once the model is compressed

The flag comes from the quality study: under NF4, every changed prediction had an
FP32 margin below 0.083, and none of the 2,749 test messages with margin >= 0.10
changed (results/analysis/12145703.json). DEFAULT_RISK_MARGIN is that 0.10 edge.
"""
from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .ingestion import BANKING77, BANKING77_NUM_LABELS, Request, ValidationError, _ensure_unique, validate_record

DEFAULT_RISK_MARGIN = 0.10
OUTPUT_COLUMNS = ["id", "text", "intent", "confidence", "runner_up", "margin", "at_risk"]


def read_messages(path: str | Path) -> list[Request]:
    """Load a CSV of messages: a `text` column is required, an `id` column is optional.

    Every row goes through the same validate_record rules as the datasets; the first
    bad row stops the load with an error naming its line. Rows without an id get
    "line-<n>", the line number in the file, so any output row can be traced back.
    """
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"{path}: no such file")
    with path.open(newline="", encoding="utf-8-sig") as f:     # -sig: tolerate Excel's BOM
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValidationError(f"{path}: needs a 'text' column, found {reader.fieldnames}")
        requests = []
        for row in reader:
            line = reader.line_num
            validate_record(f"{path.name} line {line}", row["text"])   # raises, naming the line
            rid = (row.get("id") or "").strip() or f"line-{line}"
            requests.append(Request(request_id=rid, text=row["text"]))
    if not requests:
        raise ValidationError(f"{path}: no messages (header only?)")
    return _ensure_unique(requests)


def intent_names(rows=None) -> list[str]:
    """Banking77 intent names, position i = intent id i (from the dataset's label_text)."""
    if rows is None:
        from datasets import load_dataset
        rows = load_dataset(BANKING77, split="train")
    names: dict[int, str] = {}
    for r in rows:
        if names.setdefault(r["label"], r["label_text"]) != r["label_text"]:
            raise ValidationError(f"intent {r['label']} has two names: {names[r['label']]!r}, {r['label_text']!r}")
    if sorted(names) != list(range(BANKING77_NUM_LABELS)):
        raise ValidationError(f"expected names for intents 0-{BANKING77_NUM_LABELS - 1}, got {len(names)}")
    return [names[i] for i in range(BANKING77_NUM_LABELS)]


def train_vectors(embed: Callable[[Sequence[str]], np.ndarray], texts: Sequence[str],
                  cache: Path | None) -> np.ndarray:
    """Embed the training texts, or reuse a cached copy if it has the right shape."""
    if cache is not None and cache.is_file():
        vecs = np.load(cache)
        if vecs.shape[0] == len(texts):
            return vecs
    vecs = embed(texts)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vecs)
    return vecs


def classify(proba: np.ndarray, names: Sequence[str], risk_margin: float = DEFAULT_RISK_MARGIN) -> list[dict]:
    """Turn class probabilities (one row per message) into predictions with a risk flag."""
    if not 0 <= risk_margin <= 1:
        raise ValueError(f"risk_margin must be between 0 and 1, got {risk_margin}")
    proba = np.asarray(proba)
    order = np.argsort(proba, axis=1)
    top, second = order[:, -1], order[:, -2]
    rows = []
    for i in range(len(proba)):
        conf, runner = float(proba[i, top[i]]), float(proba[i, second[i]])
        # flag on the rounded margin the user sees: unrounded, 0.6 - 0.4 = 0.19999999999999996
        # would be flagged "below 0.2" while the output shows 0.2000
        margin = round(conf - runner, 4)
        rows.append({"intent": names[top[i]], "confidence": round(conf, 4),
                     "runner_up": names[second[i]], "margin": margin,
                     "at_risk": margin < risk_margin})
    return rows


def write_predictions(path: str | Path, requests: Sequence[Request], results: Sequence[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for req, res in zip(requests, results, strict=True):
            w.writerow({"id": req.request_id, "text": req.text, **res})
