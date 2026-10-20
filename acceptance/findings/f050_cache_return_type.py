"""F050: a cache hit must return the same value and type as the computation.

A decorated function returning the string "nan": 0.1.0 returned float('nan') on the cache hit.
Expected: the hit returns the str "nan" (or the value is refused for the cache up front).
"""
import warnings
from pathlib import Path

from _acceptance import check, passed, refused, start

import conatus as cn

start(50, "cache hits preserve the return type")
root = Path("registry").resolve()


@cn.runs.track("text", cache=True, root=root)
def label(x):
    return "nan"


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    first = label(1)
    was_refused, second = refused(label, 1)
if was_refused:
    passed("cache refused")
check(isinstance(first, str), f"computed value {first!r}")
check(type(second) is str and second == "nan", f"cache hit returned {second!r} ({type(second).__name__})")
passed("hit returns 'nan' as str")
