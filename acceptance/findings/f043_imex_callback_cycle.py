"""F043: an ImexIntegrator whose callback refers to itself must be garbage-collectable.

F closes over the integrator (a common pattern: read its iteration count). 0.1.0 kept the
Python closure in the C++ object, a cycle invisible to the GC: the integrator was never freed.
"""
import gc
import weakref

import numpy as np
import scipy.sparse as sp
from _acceptance import check, need, passed, start

start(43, "self-referencing IMEX callback does not leak")
need("conatus.runtime")
from conatus.runtime.imex import ImexIntegrator  # noqa: E402

n = 3


def build():
    box = []                                  # the closure's reference to its own integrator

    def F(x, t):
        return np.zeros(n) * box[0].iteration

    integ = ImexIntegrator(sp.identity(n, format="csr"), sp.identity(n, format="csr"), [], "RK222",
                           0.1, F=F)
    box.append(integ)
    return integ


integ = build()
integ.run(np.ones(n), 0.0, 2)
ref = weakref.ref(integ)
del integ
gc.collect()
gc.collect()
check(ref() is None, "the integrator is still alive after del + gc.collect()")
passed("collected")
