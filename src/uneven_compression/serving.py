"""Model serving: load the embedding model at a configured precision and embed batches.

Precisions: fp32, fp16, int8 (bitsandbytes LLM.int8()), nf4 (bitsandbytes 4-bit).
INT8 requires a Volta-or-newer GPU; it fails on the Pascal P100.
"""
