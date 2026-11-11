"""F940 (review G2): a run with a callback is never a cache source.

Oracle: dt(u) = 0 keeps u = 1 exactly. A first run with a callback that adds 1 to the state
ends at u = 4; the same computation without a callback must end at u = 1, computed or read from
a run without a callback. Control: two runs without a callback, the second a cache hit, u = 1.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(940, "a run with a callback does not feed the cache")
import conatus as cn  # noqa: E402


def build():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations("dt(u)=0")
    s = m.simulate(8, scheme="RK443", dt=.1)
    s.initial(u=1)
    return s.solver


def kick(s):
    s.X[0] += 1


def run(solver, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = solver.run(3, **kw)
    hit = any("cache hit" in str(w.message) for w in caught)
    return r, hit


root = "registry"
a = build()
run(a, callback=kick, every=1, record={"root": root})
check(abs(a.grid("u").mean() - 4.0) < 1e-12, f"the callback did not act: {a.grid('u').mean()}")
b = build()
r, hit = run(b, record={"root": root}, cache=True)
check(not hit, "the run without a callback was read from the run with a callback")
check(abs(b.grid("u").mean() - 1.0) < 1e-12, f"u = {b.grid('u').mean()} after a run of dt(u) = 0, expected 1")
c = build()
r, hit = run(c, record={"root": root}, cache=True)
check(hit, "control: the second run without a callback is not a cache hit")
check(abs(c.grid("u").mean() - 1.0) < 1e-12, f"control: hit gives u = {c.grid('u').mean()}")
passed("the callback run is not a source; a plain run is reused with u = 1")
