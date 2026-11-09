"""uneven_compression: an embedding inference service.

Text goes in, embedding vectors come out. Numeric precision and batch size are
configuration of the running service, so the same pipeline can be measured at
every setting.

Modules, in data-flow order:
    ingestion   load a labeled task and turn it into a queue of requests
    batching    group requests into batches of a configured size
    serving     load the model at a configured precision and embed batches
    storage     persist embeddings and timing logs, keyed by configuration
    evaluation  compute quality and efficiency metrics from stored runs
"""

__version__ = "0.1.0"
