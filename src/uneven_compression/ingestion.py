"""Data ingestion: load a labeled task and turn it into a queue of requests.

Supported tasks: Banking77 (intent classification) and SciFact (retrieval),
both as published by MTEB on the Hugging Face Hub.

Ingestion checks records; it never edits them. Text reaches the model exactly
as the dataset has it, so results stay comparable to published MTEB scores.
Any invalid record raises ValidationError and stops the load: bad data must
not silently change the number of examples a run measures.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

BANKING77 = "mteb/banking77"
SCIFACT = "mteb/scifact"
BANKING77_NUM_LABELS = 77


class ValidationError(ValueError):
    """A record failed validation. The message names the record and the rule."""


@dataclass(frozen=True)
class Request:
    """One text to embed.

    request_id: unique within a load, e.g. "banking77-test-00042", "doc:4983", "query:1"
    text:       exactly what the dataset contains
    label:      Banking77 intent id (0-76); None for SciFact texts
    title:      SciFact document title; None otherwise
    """

    request_id: str
    text: str
    label: int | None = None
    title: str | None = None


def validate_record(request_id: str, text: Any, label: Any = None,
                    num_labels: int | None = None, title: Any = None) -> Request:
    """Check one raw record and return it as a Request, or raise ValidationError.

    Rules:
      1. text must be a str containing at least one non-whitespace character
      2. if num_labels is given (classification): label must be an int with
         0 <= label < num_labels. bool is not an acceptable int here.
      3. if num_labels is None (retrieval): label must be None
      4. title must be None or a str

    Error messages must include request_id, so a failure in a 10,000-row load
    points at the exact row.
    """
    # Rule 1: isinstance first, so non-strings fail here instead of crashing on .strip()
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"{request_id}: text must be a non-blank string, got {text!r}")

    if num_labels is not None:
        # Rule 2: bool is a subclass of int in Python, so True would pass isinstance(label, int)
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < num_labels:
            raise ValidationError(f"{request_id}: label must be an int in [0, {num_labels}), got {label!r}")
    elif label is not None:
        # Rule 3
        raise ValidationError(f"{request_id}: retrieval record must not have a label, got {label!r}")

    # Rule 4
    if title is not None and not isinstance(title, str):
        raise ValidationError(f"{request_id}: title must be a str or None, got {title!r}")

    return Request(request_id=request_id, text=text, label=label, title=title)


def _ensure_unique(requests: list[Request]) -> list[Request]:
    seen: set[str] = set()
    for r in requests:
        if r.request_id in seen:
            raise ValidationError(f"{r.request_id}: duplicate request_id")
        seen.add(r.request_id)
    return requests


def load_banking77(split: str, rows: Iterable[Mapping[str, Any]] | None = None) -> list[Request]:
    """Banking77 requests for one split ("train" or "test").

    `rows` lets tests pass hand-written records; by default the split is
    downloaded from the Hub.
    """
    if rows is None:
        from datasets import load_dataset
        rows = load_dataset(BANKING77, split=split)
    requests = [
        validate_record(f"banking77-{split}-{i:05d}", row["text"], row["label"],
                        num_labels=BANKING77_NUM_LABELS)
        for i, row in enumerate(rows)
    ]
    return _ensure_unique(requests)


def load_scifact(split: str,
                 corpus_rows: Iterable[Mapping[str, Any]] | None = None,
                 query_rows: Iterable[Mapping[str, Any]] | None = None,
                 qrel_rows: Iterable[Mapping[str, Any]] | None = None,
                 ) -> tuple[list[Request], list[Request], dict[str, dict[str, float]]]:
    """SciFact corpus, the queries judged in `split`, and that split's relevance labels.

    The Hub's queries table mixes every split (1,109 rows); only queries that
    appear in this split's qrels are returned (300 for "test").

    Returns (corpus, queries, qrels) where qrels maps
    "query:<id>" -> {"doc:<id>": relevance score}.
    """
    if corpus_rows is None or query_rows is None or qrel_rows is None:
        from datasets import load_dataset
        corpus_rows = load_dataset(SCIFACT, "corpus", split="corpus")
        query_rows = load_dataset(SCIFACT, "queries", split="queries")
        qrel_rows = load_dataset(SCIFACT, "default", split=split)

    qrels: dict[str, dict[str, float]] = {}
    for row in qrel_rows:
        qrels.setdefault(f"query:{row['query-id']}", {})[f"doc:{row['corpus-id']}"] = float(row["score"])

    corpus = _ensure_unique([
        validate_record(f"doc:{row['_id']}", row["text"], title=row["title"])
        for row in corpus_rows
    ])
    queries = _ensure_unique([
        validate_record(f"query:{row['_id']}", row["text"])
        for row in query_rows
        if f"query:{row['_id']}" in qrels
    ])

    # Every judged query and judged document must exist, or the metric is wrong.
    query_ids = {q.request_id for q in queries}
    doc_ids = {d.request_id for d in corpus}
    missing_queries = qrels.keys() - query_ids
    if missing_queries:
        raise ValidationError(f"qrels reference {len(missing_queries)} unknown queries, e.g. {sorted(missing_queries)[:3]}")
    missing_docs = {d for judged in qrels.values() for d in judged} - doc_ids
    if missing_docs:
        raise ValidationError(f"qrels reference {len(missing_docs)} unknown documents, e.g. {sorted(missing_docs)[:3]}")

    return corpus, queries, qrels
