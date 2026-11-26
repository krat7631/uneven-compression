import csv

import numpy as np
import pytest

from uneven_compression import cli
from uneven_compression.classify import classify, intent_names, read_messages, train_vectors, write_predictions
from uneven_compression.ingestion import ValidationError


def write(tmp_path, content, name="msgs.csv"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# --- reading new messages -------------------------------------------------

def test_read_messages_uses_id_column_or_line_number(tmp_path):
    reqs = read_messages(write(tmp_path, "id,text\na1,lost my card\n,where is my refund\n"))
    assert [(r.request_id, r.text) for r in reqs] == [("a1", "lost my card"), ("line-3", "where is my refund")]


def test_read_messages_text_column_alone_is_enough(tmp_path):
    reqs = read_messages(write(tmp_path, "text\nhello\n"))
    assert reqs[0].request_id == "line-2"


def test_blank_message_names_its_line(tmp_path):
    with pytest.raises(ValidationError, match="line 3"):
        read_messages(write(tmp_path, "text\nfine\n   \n"))


@pytest.mark.parametrize("content, match", [
    ("message\nhello\n", "needs a 'text' column"),
    ("", "needs a 'text' column"),
    ("text\n", "no messages"),
    ("id,text\nx,a\nx,b\n", "duplicate"),
])
def test_bad_files_are_rejected(tmp_path, content, match):
    with pytest.raises(ValidationError, match=match):
        read_messages(write(tmp_path, content))


def test_missing_file(tmp_path):
    with pytest.raises(ValidationError, match="no such file"):
        read_messages(tmp_path / "nope.csv")


def test_excel_byte_order_mark_is_tolerated(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbftext\nhello\n")      # the UTF-8 byte-order mark Excel writes
    assert read_messages(p)[0].text == "hello"


# --- intent names ---------------------------------------------------------

def test_intent_names_need_all_77_and_one_name_each():
    rows = [{"label": i, "label_text": f"intent_{i}"} for i in range(77)]
    assert intent_names(rows)[5] == "intent_5"
    with pytest.raises(ValidationError, match="two names"):
        intent_names(rows + [{"label": 5, "label_text": "other"}])
    with pytest.raises(ValidationError, match="expected names"):
        intent_names(rows[:-1])


# --- the risk flag --------------------------------------------------------

def test_classify_reports_top_two_and_flags_small_margins():
    names = ["a", "b", "c"]
    proba = np.array([[0.70, 0.20, 0.10],      # margin 0.50 -> safe
                      [0.40, 0.05, 0.55]])     # margin 0.15 -> at risk at 0.2
    rows = classify(proba, names, risk_margin=0.2)
    assert rows[0] == {"intent": "a", "confidence": 0.7, "runner_up": "b", "margin": 0.5, "at_risk": False}
    assert rows[1]["intent"] == "c" and rows[1]["runner_up"] == "a" and rows[1]["at_risk"] is True


def test_margin_exactly_at_threshold_is_not_at_risk():
    assert classify(np.array([[0.6, 0.4]]), ["a", "b"], risk_margin=0.2)[0]["at_risk"] is False


def test_risk_margin_out_of_range_rejected():
    with pytest.raises(ValueError):
        classify(np.array([[0.6, 0.4]]), ["a", "b"], risk_margin=1.5)


# --- training-vector cache ------------------------------------------------

def test_train_vectors_are_cached_and_reused(tmp_path):
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    cache = tmp_path / "c" / "train.npy"
    train_vectors(embed, ["a", "b"], cache)
    train_vectors(embed, ["a", "b"], cache)
    assert calls == [2]                                   # second call read the cache
    train_vectors(embed, ["a", "b", "c"], cache)          # wrong size: re-embed
    assert calls == [2, 3]


# --- the command line, end to end with a fake model -----------------------

class FakeClassifier:
    def predict_proba(self, x):
        # message length decides the answer: short -> confident "a", long -> near tie "a"/"b"
        return np.array([[0.9, 0.05, 0.05] if v[0] < 10 else [0.48, 0.46, 0.06] for v in x])


@pytest.fixture
def fake_service(monkeypatch):
    def build(precision, batch_size, cache_dir):
        embed = lambda texts: np.array([[len(t)] for t in texts], dtype=float)
        return embed, FakeClassifier(), ["a", "b", "c"], "cpu"
    monkeypatch.setattr(cli, "build_service", build)


def test_cli_classifies_and_writes_csv(tmp_path, fake_service, capsys):
    src = write(tmp_path, "id,text\nq1,short\nq2,a much longer message\n")
    out = tmp_path / "pred.csv"
    assert cli.main(["classify", str(src), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert [r["at_risk"] for r in rows] == ["False", "True"]
    assert "1 of 2 messages at risk" in capsys.readouterr().out


def test_cli_bad_input_exits_2_without_traceback(tmp_path, fake_service, capsys):
    assert cli.main(["classify", str(write(tmp_path, "text\n  \n"))]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_rejects_bad_settings_before_loading_the_model(tmp_path, monkeypatch, capsys):
    def must_not_load(*a):
        raise AssertionError("model loaded before the settings were checked")
    monkeypatch.setattr(cli, "build_service", must_not_load)
    src = str(write(tmp_path, "text\nhi\n"))
    assert cli.main(["classify", src, "--risk-margin", "2"]) == 2
    assert cli.main(["classify", src, "--batch-size", "0"]) == 2
    assert cli.main(["classify", str(tmp_path / "missing.csv")]) == 2


def test_quantized_precision_fails_cleanly_without_cuda(tmp_path, capsys):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this machine can run nf4")
    assert cli.main(["classify", str(write(tmp_path, "text\nhi\n")), "--precision", "nf4"]) == 2
    assert "needs a CUDA GPU" in capsys.readouterr().err


def test_write_predictions_rejects_mismatched_lengths(tmp_path):
    reqs = read_messages(write(tmp_path, "text\na\nb\n"))
    with pytest.raises(ValueError):
        write_predictions(tmp_path / "o.csv", reqs, [{"intent": "x"}])
