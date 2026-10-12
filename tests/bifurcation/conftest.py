"""Bifurcation tests may import the example scripts unchanged (like the gallery)."""
import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="session")
def example():
    def _load(relpath: str):
        path = EXAMPLES / f"{relpath}.py"
        spec = importlib.util.spec_from_file_location(f"bif_example_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return _load
