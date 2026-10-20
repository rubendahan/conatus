"""F104 (new, critic 2026-10-19, review of the identity lot): a cached function that reads a
global NumPy array mutated in place must not return a stale value.

W = ones(3), f(x) = W.sum() * x; then W[:] = 2 in place: the second call must be 12, not 6
(a key built from the object's identity or from a copy taken at decoration would hit).
"""
import warnings
from pathlib import Path

import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(104, "in-place changes of a global array are part of the cache identity")
root = Path("registry").resolve()
W = np.ones(3)


@cn.runs.track("globarr", cache=True, root=root)
def total(x):
    return float(W.sum() * x)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    first = total(2.0)
    W[:] = 2.0
    was_refused, second = refused(total, 2.0)
if was_refused:
    passed("cache refused")
check(first == 6.0, f"first call returned {first}")
check(second == 12.0, f"after W[:] = 2 the call returned {second}, expected 12")
passed("in-place change misses the cache")
