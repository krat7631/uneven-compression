# The Uneven Cost of Compression

Quantization and batching trade-offs in an embedding inference service.

CS 5130 (Programming and Data Processing for AI), Fall 2026 · Theme 5: Model Optimization & Efficient Inference Systems
Devansh Jignesh Nayak · Venkatashivasai Muppidi

## Purpose

An embedding inference service (text in, vectors out) built around
`BAAI/bge-small-en-v1.5`. Numeric precision (FP32, FP16, INT8, 4-bit NF4) and
batch size (1, 8, 32, 128) are configuration of the running service. The
project measures latency, throughput and memory across those settings, and
asks whether compression's quality loss is spread evenly or concentrated on
rare and easily confused inputs (Banking77, SciFact).

## Code organisation

```
src/uneven_compression/
    ingestion.py    load a labeled task, validate it, produce a request queue
    batching.py     group requests into batches of a configured size
    serving.py      load the model at a configured precision, embed batches
    storage.py      persist embeddings and timing logs by configuration
    evaluation.py   quality and efficiency metrics
tests/              unit tests (pytest)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core + tests
pip install -e ".[dev,model]"    # also the GPU model stack
```

Requires Python 3.10 or newer.

## Running the tests

```bash
pytest
```

## Usage

```python
from uneven_compression.ingestion import load_banking77, load_scifact
from uneven_compression.batching import make_batches

test = load_banking77("test")                 # 3,076 validated Requests
batches = make_batches(test, batch_size=32)   # 97 batches: 96 x 32, then 1 x 4

corpus, queries, qrels = load_scifact("test") # 5,183 docs, 300 judged queries
```

Any invalid record raises `ValidationError` and stops the load, so bad data
can never silently change the number of examples a run measures.

Banking77 is MTEB's copy (`mteb/banking77`: 9,993 train / 3,076 test), which
differs from the original release by 14 rows; it is used so baselines are
directly comparable to published MTEB scores.
