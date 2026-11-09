"""Data ingestion: load a labeled task and turn it into a queue of requests.

Supported tasks: Banking77 (intent classification) and SciFact (retrieval).
Each record is validated before it enters the queue, so later modules can
assume well-formed input.
"""
