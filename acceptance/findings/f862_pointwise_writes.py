"""F862: a user pointwise function that writes into its argument cannot change another
term through the shared grid cache of the right-hand side.

Oracle: the same model with a function that computes the same values without writing (sin(2 u)),
whose trajectory must be identical bit for bit; the grid of u is shared with u*dx(u).
"""
import numpy as np
from _acceptance import check, passed, start

start(862, "user pointwise function writing into its input: no effect on other terms")
import conatus as cn  # noqa: E402


def writes(a):
    a *= 2.0
    return np.sin(a)


def clean(a):
    return np.sin(2.0 * a)


X = []
for name, f in (("w862", writes), ("c862", clean)):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="p", nu=0.1)
    m.scalar("u")
    m.function(name, f)
    m.equations(f"dt(u) - nu*lap(u) = -u*dx(u) + 0.01*{name}(u)")
    sim = m.simulate(N=32, scheme="RK443", dt=1e-3)
    sim.initial(u=lambda x: np.sin(x) + 0.2 * np.cos(3 * x))
    sim.run(steps=20)
    X.append(sim.solver.X.copy())
check(np.array_equal(X[0], X[1]), "the writing function changed the trajectory")
passed("same trajectory bit for bit as the non-writing function")
