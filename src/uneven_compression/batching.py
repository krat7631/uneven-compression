"""Request batching: group requests into batches of a configured size.

Batches keep the original request order. The last batch is smaller when the
number of requests is not a multiple of batch_size: every request is served
exactly once, so batching never changes how many examples a run measures.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def make_batches(requests: Sequence[T], batch_size: int) -> list[list[T]]:
    """Split requests into consecutive batches of batch_size (the last may be smaller).

    Raises ValueError for a batch_size that is not a positive int.
    """
    # bool is an int subclass: batch_size=True would otherwise mean batches of 1
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    return [list(requests[i:i + batch_size]) for i in range(0, len(requests), batch_size)]
