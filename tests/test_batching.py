import pytest

from uneven_compression.batching import make_batches


def test_exact_multiple():
    assert make_batches([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_last_batch_is_smaller_and_nothing_is_dropped():
    batches = make_batches(list(range(3076)), 32)
    assert len(batches) == 97
    assert [len(b) for b in batches[-2:]] == [32, 4]
    assert sum(len(b) for b in batches) == 3076


def test_order_is_preserved():
    items = list(range(10))
    assert [x for b in make_batches(items, 3) for x in b] == items


def test_batch_size_larger_than_input():
    assert make_batches([1, 2], 128) == [[1, 2]]


def test_batch_size_one():
    assert make_batches(["a", "b"], 1) == [["a"], ["b"]]


def test_empty_input():
    assert make_batches([], 8) == []


@pytest.mark.parametrize("bad", [0, -1, 2.0, "8", None, True])
def test_bad_batch_size_fails_loudly(bad):
    with pytest.raises(ValueError, match="batch_size"):
        make_batches([1, 2, 3], bad)


def test_works_on_ingested_requests():
    from uneven_compression.ingestion import load_banking77
    reqs = load_banking77("test", rows=[{"text": f"t{i}", "label": i % 77} for i in range(5)])
    batches = make_batches(reqs, 2)
    assert [[r.request_id for r in b] for b in batches] == [
        ["banking77-test-00000", "banking77-test-00001"],
        ["banking77-test-00002", "banking77-test-00003"],
        ["banking77-test-00004"],
    ]
