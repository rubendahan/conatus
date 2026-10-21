"""F522 (external review of 0.1.1, F5): a dynamic configuration read is never a stale cache hit.

``getattr(cfg, 'factor')`` names the attribute by a string constant, which the identity walker
did not see: after ``cfg.factor`` went from 1 to 3, ``f(2)`` returned the cached 2. Expected:
6 (the call is computed again, the cache refused with a warning naming the read).
"""
import sys
import tempfile
import types
import warnings
from pathlib import Path

from _acceptance import check, passed, start

start(522, "a getattr() read refuses the cache instead of a stale hit")
import conatus as cn  # noqa: E402

cfg = types.ModuleType("review_cfg_f522")
cfg.factor = 1.0
sys.modules[cfg.__name__] = cfg


@cn.runs.track("dynamic_attr", cache=True, root=Path(tempfile.mkdtemp()))
def f(x):
    return getattr(cfg, "factor") * x  # noqa: B009 -- the reviewed read


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    first = f(2.0)
    cfg.factor = 3.0
    second = f(2.0)
msgs = [str(m.message) for m in w]
print(f"  f(2) = {first}, then {second}")
check(first == 2.0 and second == 6.0, f"f(2) after cfg.factor = 3 returned {second}, not 6")
check(any("getattr" in m for m in msgs), "no warning names the dynamic read")
passed("6 after the change; the cache refusal names getattr")
