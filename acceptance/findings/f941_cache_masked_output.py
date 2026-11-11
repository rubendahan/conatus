"""F941 (review G12): a masked return is never read back from the cache as a plain array.

Oracle: numpy.ma. The fresh return has mean 1.0 (the entry 100 is masked); a cached plain
array has mean 50.5. Inputs: two masked arguments with the same data and another mask are two
computations (sum 1.0 and 101.0). Control: a plain-array return is a cache hit, equal bytes.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(941, "masks survive the cache or the call is not cached")
import conatus as cn  # noqa: E402

root = "registry"


def call(f, *a):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(*a)
    return out, any("cache hit" in str(w.message) for w in caught)


@cn.runs.track("masked_out", cache=True, root=root)
def make(n):
    return np.ma.array([1.0, 100.0], mask=[False, True])


oracle = np.ma.array([1.0, 100.0], mask=[False, True]).mean()
for _ in range(2):
    out, hit = call(make, 2)
    check(isinstance(out, np.ma.MaskedArray), f"return of type {type(out).__name__} (hit {hit})")
    check(float(out.mean()) == float(oracle), f"mean {float(out.mean())}, numpy.ma oracle {oracle}")


@cn.runs.track("masked_in", cache=True, root=root)
def masked_sum(x):
    return float(x.sum())


x = np.ma.array([1.0, 100.0], mask=[False, True])
y = np.ma.array([1.0, 100.0], mask=False)
check(call(masked_sum, x)[0] == float(x.sum()), "first masked sum")
check(call(masked_sum, y)[0] == float(y.sum()) == 101.0, "a second mask read the first result")


@cn.runs.track("plain_out", cache=True, root=root)
def plain(n):
    return np.arange(float(n))


a, _ = call(plain, 3)
b, hit = call(plain, 3)
check(hit and type(b) is np.ndarray and a.tobytes() == b.tobytes(), "control: plain array not reused")
passed("masked return kept (not cached), masked inputs distinct, plain return reused")
