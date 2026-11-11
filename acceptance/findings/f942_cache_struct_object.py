"""F942 (review H9): a structured array with an object field is not hashed by its pointers.

Oracle: a direct Python call. The function reads a list stored in an object field; the list is
changed between two calls, so the second value differs (1 then 2). Control: a structured array
of numbers only is a cache hit.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(942, "structured arrays with object fields are not cache keys")
import conatus as cn  # noqa: E402

root = "registry"


def call(f, *a):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(*a)
    return out, any("cache hit" in str(w.message) for w in caught)


def first(x):
    return float(x["o"][0][0])


tracked = cn.runs.track("struct_obj", cache=True, root=root)(first)
box = [1.0]
a = np.empty(1, dtype=[("f", float), ("o", object)])
a["f"][0], a["o"][0] = 0.0, box
v1, _ = call(tracked, a)
box[0] = 2.0
v2, hit = call(tracked, a)
check((v1, v2) == (1.0, 2.0) == (1.0, first(a)), f"values {v1}, {v2} (hit {hit}); direct call gives 1 then {first(a)}")


@cn.runs.track("struct_num", cache=True, root=root)
def total(x):
    return float(x["f"].sum() + x["g"].sum())


s = np.array([(1.0, 2), (3.0, 4)], dtype=[("f", float), ("g", "i4")])
call(total, s)
v, hit = call(total, s)
check(hit and v == 10.0, f"control: numeric structured array (hit {hit}, value {v})")
passed("object-field arrays are recomputed; numeric structured arrays reused")
