import importlib

import pytest

import uneven_compression

MODULES = ["ingestion", "batching", "serving", "storage", "evaluation"]


def test_version_is_set():
    assert uneven_compression.__version__ == "0.1.0"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"uneven_compression.{name}")
