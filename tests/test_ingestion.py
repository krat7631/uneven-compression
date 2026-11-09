import pytest

from uneven_compression.ingestion import (
    Request,
    ValidationError,
    load_banking77,
    load_scifact,
    validate_record,
)


# ---- validate_record: one test per rule ------------------------------------

def test_valid_classification_record():
    r = validate_record("b-1", "my card payment was declined abroad", 25, num_labels=77)
    assert r == Request("b-1", "my card payment was declined abroad", 25)


def test_valid_retrieval_record_with_title():
    r = validate_record("doc:1", "abstract text", title="A title")
    assert r.label is None and r.title == "A title"


def test_text_is_not_modified():
    raw = "\nWhere can I withdraw money from?  "
    assert validate_record("b-2", raw, 3, num_labels=77).text == raw


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None, 42])
def test_rule1_text_must_be_nonblank_str(text):
    with pytest.raises(ValidationError, match="b-3"):
        validate_record("b-3", text, 0, num_labels=77)


@pytest.mark.parametrize("label", [-1, 77, 100, "5", 3.0, True, None])
def test_rule2_classification_label_must_be_int_in_range(label):
    with pytest.raises(ValidationError, match="b-4"):
        validate_record("b-4", "some text", label, num_labels=77)


def test_rule2_boundaries_are_accepted():
    assert validate_record("lo", "t", 0, num_labels=77).label == 0
    assert validate_record("hi", "t", 76, num_labels=77).label == 76


def test_rule3_retrieval_record_must_not_have_label():
    with pytest.raises(ValidationError, match="doc:9"):
        validate_record("doc:9", "text", label=3)


@pytest.mark.parametrize("title", [5, ["t"]])
def test_rule4_title_must_be_str_or_none(title):
    with pytest.raises(ValidationError, match="doc:7"):
        validate_record("doc:7", "text", title=title)


# ---- loaders, on hand-written rows (no download) ----------------------------

def test_load_banking77_builds_ids_and_keeps_order():
    rows = [{"text": "a", "label": 1}, {"text": "b", "label": 2}]
    reqs = load_banking77("test", rows=rows)
    assert [r.request_id for r in reqs] == ["banking77-test-00000", "banking77-test-00001"]
    assert [r.label for r in reqs] == [1, 2]


def test_load_banking77_fails_loudly_on_one_bad_row():
    rows = [{"text": "fine", "label": 1}, {"text": "  ", "label": 2}]
    with pytest.raises(ValidationError, match="banking77-test-00001"):
        load_banking77("test", rows=rows)


def _scifact_rows():
    corpus = [{"_id": "10", "title": "T10", "text": "doc ten"},
              {"_id": "20", "title": "T20", "text": "doc twenty"}]
    queries = [{"_id": "1", "text": "judged query"},
               {"_id": "2", "text": "query from another split"}]
    qrels = [{"query-id": "1", "corpus-id": "20", "score": 1.0}]
    return corpus, queries, qrels


def test_load_scifact_keeps_only_judged_queries():
    corpus, queries, qrels = load_scifact("test", *_scifact_rows())
    assert [q.request_id for q in queries] == ["query:1"]
    assert len(corpus) == 2
    assert qrels == {"query:1": {"doc:20": 1.0}}


def test_load_scifact_rejects_qrels_pointing_at_missing_document():
    corpus, queries, _ = _scifact_rows()
    bad_qrels = [{"query-id": "1", "corpus-id": "999", "score": 1.0}]
    with pytest.raises(ValidationError, match="unknown documents"):
        load_scifact("test", corpus, queries, bad_qrels)


def test_load_scifact_rejects_duplicate_document_ids():
    corpus, queries, qrels = _scifact_rows()
    corpus.append({"_id": "10", "title": "dup", "text": "again"})
    with pytest.raises(ValidationError, match="duplicate"):
        load_scifact("test", corpus, queries, qrels)
