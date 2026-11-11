"""F944 (review G13): a restart never treats an incomparable dependency as equal.

The user function ``force`` closes over an object (``cfg.rate``), which has no content hash. A
checkpoint written at rate 1 is restarted by a run at rate 2. Oracle: the two right-hand sides
differ (rate * u at u = 1: 1 against 2), so a silent restart continues another computation.
Outcomes: refused with "cannot be compared", or accepted only with an explicit allow= that
explain() records. Control: a closure over a float is compared by value (same value restarts,
another value is refused as a difference).
"""
from types import SimpleNamespace

import numpy as np
from _acceptance import check, passed, refused, start

start(944, "restart: incomparable dependencies are refused unless allowed")
import conatus as cn  # noqa: E402


def build_obj(value):
    cfg = SimpleNamespace(rate=value)

    def force(u):
        return cfg.rate * u
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.function("force", force)
    m.equations("dt(u)=force(u)")
    s = m.simulate(8, scheme="RK443", dt=.01)
    s.initial(u=1)
    return s


def build_float(value):
    rate = float(value)

    def force(u):
        return rate * u
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.function("force", force)
    m.equations("dt(u)=force(u)")
    s = m.simulate(8, scheme="RK443", dt=.01)
    s.initial(u=1)
    return s


a = build_obj(1.0)
a.run(steps=10)
a.save("obj.npz")
ok, exc = refused(build_obj(2.0).restart, "obj.npz")
check(ok, "a changed opaque dependency was accepted silently")
check("cannot be compared" in str(exc), f"the refusal does not say why: {exc}")
b = build_obj(2.0)
b.restart("obj.npz", allow=("force",))
check("cannot be compared" in b.explain(full=True), "allow=('force',) is not recorded by explain()")

c = build_float(1.0)
c.run(steps=10)
c.save("float.npz")
same = build_float(1.0)
same.restart("float.npz")
check(np.array_equal(same.grid("u"), c.grid("u")), "control: same float closure restart")
ok, exc = refused(build_float(2.0).restart, "float.npz")
check(ok and "cannot be compared" not in str(exc), f"control: a changed float is not a plain difference: {exc}")
passed("incomparable refused, allowed with a record; comparable closures compared by value")
