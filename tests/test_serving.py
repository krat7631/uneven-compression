import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uneven_compression.serving import check_precision, cls_pool_normalize  # noqa: E402


@pytest.mark.parametrize("bad", ["fp8", "FP32", "int4", "", None])
def test_unknown_precision_fails_loudly(bad):
    with pytest.raises(ValueError, match="precision"):
        check_precision(bad, "cuda")


@pytest.mark.parametrize("precision", ["int8", "nf4"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_quantized_precisions_need_cuda(precision, device):
    with pytest.raises(ValueError, match="CUDA"):
        check_precision(precision, device)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8", "nf4"])
def test_all_precisions_allowed_on_cuda(precision):
    check_precision(precision, "cuda")


def test_cls_pooling_takes_first_token_and_normalizes():
    h = torch.zeros(2, 3, 2)
    h[0, 0] = torch.tensor([3.0, 4.0])      # CLS of row 0 -> (0.6, 0.8)
    h[1, 0] = torch.tensor([0.0, 2.0])      # CLS of row 1 -> (0, 1)
    h[:, 1:] = 99.0                         # other tokens must be ignored
    out = cls_pool_normalize(h)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)


def test_cls_pooling_upcasts_fp16_to_float32():
    h = torch.ones(1, 2, 4, dtype=torch.float16)
    out = cls_pool_normalize(h)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], rtol=1e-6)
