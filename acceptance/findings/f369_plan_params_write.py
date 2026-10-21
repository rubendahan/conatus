"""F369: plan.params writes are validated, and a change after an assembly is refused (review of
the inputs audit).

plan.params was a plain dict: plan.params["Re"] = nan, "2000" or a typo key went through, and a
value written after the first assembly reached F and later groups but not the cached group: the
eigenvalues of the old Re were returned silently. Before any assembly the new value is used.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn
from conatus.expert import eigenvalues

start(369, "plan.params validated on write; stale assembly refused")


def model(Re):
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=Re, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    return m


plan = model(1e4).discretize(32)
for key, val in (("Re", float("nan")), ("Re", "2000"), ("re", 2000.0)):
    r, _ = refused(plan.params.__setitem__, key, val)
    check(r, f"plan.params[{key!r}] = {val!r} accepted")
plan.params["Re"] = 2000.0
a = np.sort_complex(eigenvalues(plan).values)
b = np.sort_complex(eigenvalues(model(2000.0).discretize(32)).values)
c = np.sort_complex(eigenvalues(model(1e4).discretize(32)).values)
# non-empty, and the spectrum of Re = 2000, not of the Re = 1e4 the plan was built with (review
# 2026-10-21, F14: allclose of two empty spectra passed)
check(a.size == b.size > 0 and np.allclose(a, b), "a value written before the assembly was not the one used")
check(c.size != a.size or not np.allclose(a, c), "the spectra at Re = 2000 and Re = 1e4 coincide")
plan.params["Re"] = 3000.0
r, _ = refused(eigenvalues, plan)
check(r, "a change after the assembly was used with the cached group")
passed("validated; used before the assembly; refused after")
