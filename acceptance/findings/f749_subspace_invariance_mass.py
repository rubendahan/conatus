"""F749 (audit of 0.2.1, N35): invariance of a subspace is a statement on M^-1 G.

Oracle: M = [[1, 0], [c, 1]], G = (x, -y): M^-1 G(1, 0) = (1, -c), not invariant unless c = 0; G = (x, c x - y): invariant for every c.
"""
from _acceptance import check, passed, start

start(749, "restricted subspace with a mass matrix")
import numpy as np  # noqa: E402


from conatus.bifurcation import ir_system  # noqa: E402
from conatus.errors import OptionError  # noqa: E402
from conatus.ir import CartesianCoordinates, Domain, Field, IVP, dt  # noqa: E402
from conatus.ir.bases import RealFourier  # noqa: E402
from conatus.manifolds import restrict_system  # noqa: E402


def system(c, invariant):
    cs = CartesianCoordinates("s")
    D = Domain(cs, [RealFourier(cs["s"], 2, (0., 1.), dealias=1)])
    x, y = Field("x", D), Field("y", D)
    p = IVP([x, y])
    p.add_equation((dt(x), x))
    p.add_equation((c * dt(x) + dt(y), c * x - y if invariant else -y))
    return ir_system(p, [])


for c in (0., .25, 1., 2.):
    for invariant in (False, True):
        ds = system(c, invariant)
        idx = np.arange(ds.n // 2)
        if invariant or c == 0:
            r = restrict_system(ds, idx, name="x", invariance_tol=1e-12)
            y = np.ones(idx.size)
            check(np.allclose(r.G(y), y), f"c = {c}: reduced field")
        else:
            try:
                restrict_system(ds, idx, name="x", invariance_tol=1e-12)
                check(False, f"c = {c}: leaking subspace accepted")
            except OptionError:
                pass
passed("leak measured on the dynamics")
