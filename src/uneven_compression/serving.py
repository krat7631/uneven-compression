"""Model serving: load the embedding model at a configured precision and embed batches.

Precisions: fp32, fp16, int8 (bitsandbytes LLM.int8()), nf4 (bitsandbytes 4-bit).
INT8 requires a Volta-or-newer GPU; it fails on the Pascal P100.

The embedding of a text is the final hidden state of its first token ([CLS]),
scaled to unit length. That is how bge-small-en-v1.5 is meant to be used, so
cosine similarity between two embeddings is just their dot product.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PRECISIONS = ("fp32", "fp16", "int8", "nf4")
QUANTIZED = ("int8", "nf4")       # need bitsandbytes and a CUDA GPU
MAX_LENGTH = 512                  # the model's position limit


def pick_device() -> str:
    """The best available device: a CUDA GPU, then Apple's MPS, then the CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_precision(precision: str, device: str) -> None:
    """Raise ValueError for an unknown precision or one the device cannot run."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision in QUANTIZED and device != "cuda":
        raise ValueError(f"{precision} needs a CUDA GPU (bitsandbytes); device is {device!r}")


def load_model(precision: str = "fp32", device: str | None = None, model_name: str = MODEL_NAME):
    """Load (tokenizer, model, device) with the model's weights at `precision`.

    fp32 and fp16 set the weight dtype directly. int8 and nf4 replace the
    Linear layers with bitsandbytes layers; everything else (including the
    word-embedding table) stays fp16, and nf4 computes in fp16.
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    device = device or pick_device()
    check_precision(precision, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if precision in ("fp32", "fp16"):
        dtype = torch.float32 if precision == "fp32" else torch.float16
        # transformers 5 renamed torch_dtype to dtype; Explorer's container may be on 4.x
        key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        model = AutoModel.from_pretrained(model_name, **{key: dtype}).to(device)
    else:
        from transformers import BitsAndBytesConfig
        if precision == "int8":
            cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
        model = AutoModel.from_pretrained(model_name, quantization_config=cfg, device_map={"": 0})
    model.eval()
    return tokenizer, model, device


def cls_pool_normalize(last_hidden_state) -> np.ndarray:
    """(batch, tokens, dim) hidden states -> (batch, dim) unit-length float32 vectors.

    Takes token 0 ([CLS]) of every row and divides it by its L2 norm.
    """
    import torch
    cls = last_hidden_state[:, 0].float()
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
    return cls.cpu().numpy()


def tokenize_batch(tokenizer, device: str, texts: Sequence[str]):
    """Steps 1-2 of embedding: texts -> padded token ids on `device`."""
    return tokenizer(list(texts), padding=True, truncation=True,
                     max_length=MAX_LENGTH, return_tensors="pt").to(device)


def run_model(model, enc):
    """Step 3 of embedding: the model's forward pass. Returns the last hidden state."""
    import torch
    with torch.inference_mode():
        return model(**enc).last_hidden_state


def embed_texts(tokenizer, model, device: str, texts: Sequence[str]) -> np.ndarray:
    """Embed one batch of texts. Returns a (len(texts), 384) float32 array, rows in input order.

    The timing harness runs these same three steps, so what it times is what we serve.
    """
    enc = tokenize_batch(tokenizer, device, texts)
    hidden = run_model(model, enc)
    return cls_pool_normalize(hidden)


def embed_batches(tokenizer, model, device: str, batches: Sequence[Sequence[str]]) -> np.ndarray:
    """Embed every batch in order and stack the results: one row per text."""
    if not batches:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack([embed_texts(tokenizer, model, device, b) for b in batches])
