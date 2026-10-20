"""GPU tests: collected everywhere, skipped with a reason when no CUDA device is visible."""
import pytest

# the C++ core itself is required (never a skip reason, review finding 98): an import error here
# fails the collection; only the absence of a CUDA build or of a device skips, with that reason
from conatus.runtime import cuda as _cuda

_AVAILABLE = _cuda.available()
_WHY = _cuda.unavailable_reason()


def pytest_collection_modifyitems(config, items):
    for item in items:
        # CPU tests here: dedalus/ parity, and test_fused_rhs.py tests without an explicit gpu mark
        cpu_only = "dedalus" in str(item.fspath) or (
            "test_fused_rhs" in str(item.fspath) and item.get_closest_marker("gpu") is None)
        if "cross_backend" in str(item.fspath) and not cpu_only:
            item.add_marker(pytest.mark.gpu)
            if not _AVAILABLE:
                item.add_marker(pytest.mark.skip(reason=_WHY))
