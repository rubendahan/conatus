"""F049: two closures from the same factory must not share a cache entry.

make(k) returns a decorated function returning k * x; make(1)(2) then make(3)(2) with
cache=True: 0.1.0 returned 2 for the second call (same source, same arguments). Expected: 6, or
a typed refusal / a cache miss because the closure's captured values are not hashed.
"""
import warnings
from pathlib import Path

from _acceptance import check, passed, refused, start

import conatus as cn

start(49, "closure variables are part of the cache identity")
root = Path("registry").resolve()


def make(k):
    @cn.runs.track("closure", cache=True, root=root)
    def scaled(x):
        return k * x
    return scaled


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    first = make(1)(2.0)
    was_refused, second = refused(make(3), 2.0)
if was_refused:
    passed("cache refused for a closure")
check(first == 2.0, f"first call returned {first}")
check(second == 6.0, f"second closure (k=3) returned {second}, expected 6 (cache shared)")
passed("distinct closures, distinct results")
