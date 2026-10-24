"""F606 (audit of 0.1.2, F06): a cached function that draws from NumPy's global generator is
recomputed after reseeding, never served from the cache.

Oracle: np.random.RandomState(s).random_sample() is the first draw after np.random.seed(s)."""
from _acceptance import check, passed, start

start(606, "function cache and a library's global random state")
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ["CN_RUNS"] = tempfile.mkdtemp(prefix="cn_runs_f606_")
import numpy as np  # noqa: E402

import conatus.runs as R  # noqa: E402


@R.track("f606-rng", cache=True)
def draw():
    return float(np.random.random())


np.random.seed(1)
a = draw()
np.random.seed(2)
b = draw()
ea, eb = np.random.RandomState(1).random_sample(), np.random.RandomState(2).random_sample()
check(a == ea, f"first draw {a} != {ea}")
check(b == eb, f"draw after seed(2) is {b}, expected {eb} (cached {a}?)")
passed(f"draws {a:.10f}, {b:.10f} match the reseeded generator")
