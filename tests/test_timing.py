from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from uneven_compression.timing import time_setting  # noqa: E402


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return FakeEncoding(input_ids=torch.zeros(len(texts), 3))


class FakeModel:
    """Stands in for BGE-small: counts calls, returns unit CLS vectors."""

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, input_ids):
        self.batch_sizes.append(len(input_ids))
        return SimpleNamespace(last_hidden_state=torch.ones(len(input_ids), 3, 4))


def run(n_texts=10, batch_size=4, trials=3, warmup_batches=2):
    model = FakeModel()
    texts = [f"t{i}" for i in range(n_texts)]
    result = time_setting(FakeTokenizer(), model, "cpu", texts, batch_size,
                          trials=trials, warmup_batches=warmup_batches)
    return result, model


def test_partial_batch_is_excluded_from_latency():
    result, _ = run()                       # 10 texts at 4 -> batches of 4, 4, 2
    assert result["full_batches_per_trial"] == 2
    assert len(result["samples_total_s"]) == 3 * 2        # trials x full batches
    assert len(result["samples_model_s"]) == 3 * 2


def test_warmup_runs_but_is_not_recorded():
    result, model = run()
    # 2 warm-up batches, then 3 trials x 3 batches each
    assert model.batch_sizes == [4, 4] + [4, 4, 2] * 3
    assert len(result["samples_total_s"]) == 6


def test_throughput_is_one_number_per_trial_over_all_texts():
    result, _ = run()
    assert result["n_texts"] == 10
    assert len(result["throughput_per_trial"]) == 3
    assert all(t > 0 for t in result["throughput_per_trial"])


def test_model_time_never_exceeds_total_time():
    result, _ = run()
    for total, model_only in zip(result["samples_total_s"], result["samples_model_s"]):
        assert 0 <= model_only <= total


def test_peak_memory_only_reported_on_cuda():
    result, _ = run()
    assert result["peak_memory_mb"] is None and result["activations_mb"] is None


def test_no_full_batch_fails_loudly():
    with pytest.raises(ValueError, match="no full batch"):
        run(n_texts=3, batch_size=4)


def test_zero_trials_rejected():
    with pytest.raises(ValueError, match="trials"):
        run(trials=0)
