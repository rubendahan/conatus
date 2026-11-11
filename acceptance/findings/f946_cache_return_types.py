"""F946 (review H12): a cache hit returns what the computed call returned, in the stated details.

Oracle: the first (computed) return. A 0-d array stays a 0-d array (it came back as a NumPy
scalar), a dict keeps its key order (it came back sorted), a Fortran-ordered array stays
Fortran-ordered. Control: values, dtypes and shapes equal.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(946, "a cache hit keeps 0-d arrays, dict order and Fortran order")
import conatus as cn  # noqa: E402

root = "registry"


def call(f, *a):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(*a)
    return out, any("cache hit" in str(w.message) for w in caught)


@cn.runs.track("types", cache=True, root=root)
def make(n):
    return {"zeta": np.array(3.5), "alpha": np.asfortranarray(np.arange(6.0).reshape(2, 3)), "mid": n}


a, _ = call(make, 1)
b, hit = call(make, 1)
check(hit, "the second call is not a cache hit")
check(list(b) == list(a) == ["zeta", "alpha", "mid"], f"key order {list(b)}, computed {list(a)}")
check(type(b["zeta"]) is np.ndarray and b["zeta"].ndim == 0, f"0-d array came back as {type(b['zeta']).__name__}")
check(b["alpha"].flags.f_contiguous and not b["alpha"].flags.c_contiguous, "Fortran order lost")
check(all(np.array_equal(a[k], b[k]) and np.asarray(a[k]).dtype == np.asarray(b[k]).dtype for k in a),
      "values or dtypes differ")
passed("0-d array, key order and Fortran order kept on a hit")
