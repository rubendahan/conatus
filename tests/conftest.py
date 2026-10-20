import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # shared test helpers (rounding_bounds.py)

# Tornado's AsyncHTTPTestCase waits 5 s per request by default; under 8 xdist workers on a busy
# node a Studio page fetch can exceed that and fail spuriously. An explicit value wins.
os.environ.setdefault("ASYNC_TEST_TIMEOUT", "60")

# Margin probe (ci/assert_margins.py rule): with CN_ASSERT_MARGINS=<dir> and
# `-o enable_assertion_pass_hook=true`, every passing literal-bounded numeric assert is recorded
# with its margin. Off (no hook) otherwise.
if os.environ.get("CN_ASSERT_MARGINS"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
    from assert_margins import record as pytest_assertion_pass  # noqa: E402,F401


@pytest.fixture
def rng(request):
    """A random generator of the test's own, seeded from its node id (review finding 100). A module-level generator made every draw depend on which tests of the module ran
    before in the same process (xdist distribution, ``-k``), so a failure did not reproduce alone;
    tests/core/test_test_hygiene.py refuses module-level generators."""
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


@pytest.fixture(autouse=True, scope="session")
def _experimental_already_warned():
    """Experimental features warn once per process (``conatus._experimental``): in a test
    process the order of the tests (xdist) would decide which test sees the warning, and a test that
    turns warnings into errors would fail or pass by chance. They are marked as warned for the whole
    session; the development test suite checks the warning itself, with a fresh state."""
    from conatus import _experimental
    _experimental._warned.update(_experimental.EXPERIMENTAL)
    yield
