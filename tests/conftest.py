import os
import sys
from pathlib import Path

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
