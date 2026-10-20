"""F304 (new, audit "state" 2026-10-20): a user pointwise function's value at zero is part of its
identity in the process-wide registry.

pointwise("g", f, value_at_zero=1.0) then pointwise("g", f, value_at_zero=None) (the same notebook
cell edited and run again, or a second model) took the same registry key and overwrote f(0) for
the first declaration too: the symmetry checks of the first model then read "f(0) undefined".
Expected: each declaration keeps its own f(0).
"""
import numpy as np
from _acceptance import check, passed, start

import conatus.expert as X
from conatus.ir import pointwise

start(304, "two declarations of one function with different f(0) do not overwrite each other")


def f(u):
    return np.cosh(u)


cs = X.CartesianCoordinates("x")
D = X.Domain(cs, [X.RealFourier(cs["x"], 8, (0, 1), dealias=1.5)])
u = X.Field("u", D)
first = pointwise("acc_g", f, value_at_zero=1.0)(u)
second = pointwise("acc_g", f, value_at_zero=None)(u)
check(first.value_at_zero == 1.0, f"first declaration's f(0) is now {first.value_at_zero!r}")
check(second.value_at_zero is None, f"second declaration's f(0) is {second.value_at_zero!r}")
passed("each declaration keeps its f(0)")
