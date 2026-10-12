"""FlowMap tangent/adjoint dot-product and Taylor tests on every Cartesian non-periodic basis. The existing tests cover Fourier, Fourier x Chebyshev and the spherical shell;
the transposed transforms of Legendre, Jacobi, Hermite, Laguerre and compound Chebyshev (dense
family matrices) were not exercised by an adjoint identity.

Problem on each basis: u_t - nu u_xx = -u u_x (+ taus for the walls the basis has), RK443 and
SBDF2, 6 steps; |<J v, w> - <v, J^T w>| within the rounding of this computation (rule,
``tests/rounding_bounds.py``: measured linearity defects; w on u, the tau outputs carry the rounding
of the bordered solves) and a Taylor remainder of order 2. Before this pass the polynomial transpose of F refused every family basis at the first
vjp (after a successful jvp): their dense transforms are now transposed as they are.
"""
import numpy as np
import pytest
from rounding_bounds import adjoint_dot_check

from conatus.adjoint import FlowMap
from conatus.discretize import discretize
from conatus.ir import (IVP, CartesianCoordinates, Chebyshev, CompoundChebyshev, Domain, Field,
                        Hermite, Jacobi, Laguerre, Legendre, add, diff, dt, interp, lap, lift)
from conatus.timestep import IVPSolver

cs = CartesianCoordinates("x")
x = cs["x"]
NU = 0.1


def _problem(name):
    b = {"chebyshev": lambda: Chebyshev(x, 16, (-1, 1), dealias=1.5),
         "legendre": lambda: Legendre(x, 16, (-1, 1), dealias=1.5),
         "jacobi": lambda: Jacobi(x, 16, (-1, 1), a=0.5, b=-0.25, dealias=1.5),
         "compound": lambda: CompoundChebyshev(x, (8, 8), (-1, 0.2, 1), dealias=1.5),
         "hermite": lambda: Hermite(x, 16, scale=1.0, dealias=1.5),
         "laguerre": lambda: Laguerre(x, 16, scale=0.5, dealias=1.5)}[name]()
    D = Domain(cs, [b])
    u = Field("u", D)
    rhs = -u * diff(u, x)
    if name == "hermite":                                         # no wall
        pb = IVP([u], name="burgers line")
        pb.add_equation((dt(u) - NU * lap(u), rhs))
        return pb, u, lambda X: 0.5 * np.exp(-X ** 2) * (1 + X)
    if name == "laguerre":                                        # one wall, one condition
        t1 = Field("t1", D.without(x))
        pb = IVP([u, t1], name="burgers half line")
        pb.add_equation((dt(u) - NU * lap(u) + lift(t1, b, -1), rhs))
        pb.add_equation((interp(u, x, "left"), 0))
        return pb, u, lambda X: 0.5 * X * np.exp(-X)
    if name == "compound":
        P = len(b.sizes)
        taus = [Field(f"t{p}{j}", D.without(x)) for p in range(P) for j in (1, 2)]
        pb = IVP([u] + taus, name="burgers compound")
        pb.add_equation((dt(u) - NU * lap(u)
                         + add(*[lift(taus[2 * p + j - 1], b, -j, piece=p)
                                 for p in range(P) for j in (1, 2)]), rhs))
        for c in b.interfaces:
            for q in (u, diff(u, x)):
                pb.add_equation((interp(q, x, c, side="above") - interp(q, x, c, side="below"), 0))
    else:
        t1, t2 = Field("t1", D.without(x)), Field("t2", D.without(x))
        pb = IVP([u, t1, t2], name="burgers interval")
        pb.add_equation((dt(u) - NU * lap(u) + lift(t1, b, -1) + lift(t2, b, -2), rhs))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    return pb, u, lambda X: 0.5 * np.sin(np.pi * X) + 0.2 * (1 - X ** 2)


@pytest.mark.parametrize("scheme", ["RK443", "SBDF2"])
@pytest.mark.parametrize("name", ["chebyshev", "legendre", "jacobi", "compound", "hermite",
                                  "laguerre"])
def test_flowmap_dot_product_and_taylor(name, scheme):
    pb, u, u0 = _problem(name)
    s = IVPSolver(discretize(pb), scheme, 1e-2)
    s.set_state({u: u0})
    X0 = s.X.copy()
    fm = FlowMap(s, 6)
    rng = np.random.default_rng(3)
    V = rng.standard_normal(fm.n).astype(fm.dtype)
    # W on the state u only: the tau outputs of the last stage solve carry the rounding of the
    # bordered solve (cond(M + dt L / 2) = 4e4 Chebyshev, 2e5 Laguerre at N = 16: |J^T - (J)^T|
    # up to 2e-12 / 8e-10 in the tau rows, 4e-15 / 1e-12 on the u block, measured densely)
    sys_ = s.system
    def on_u():
        return sys_.gather_state([rng.standard_normal(c.shape) if v is u else np.zeros(c.shape)
                                  for v, c in zip(sys_.variables, sys_.scatter(X0), strict=True)]
                                 ).astype(fm.dtype)
    W, W2, V2 = on_u(), on_u(), rng.standard_normal(fm.n).astype(fm.dtype)
    # bound from the rounding of this computation (tests/rounding_bounds.py; was a fixed 1e-13 of the
    # scale, met at 8.6e-14 by Laguerre SBDF2 on genoa, margin 1.2: run).
    # cap: eps x cond(M + dt L / 2) <= 1.1e-16 x 2e5 per solve, 6 steps of up to 4 solves -> 1e-9
    chk = adjoint_dot_check(lambda v: fm.jvp(X0, v), lambda w: fm.vjp(X0, w), V, W, V2, W2,
                            cap=1e-9, dims=(int(np.count_nonzero(W)), fm.n))
    JV = chk.JV
    print(f"\n{name} {scheme}: {chk}")
    P0 = fm(X0)
    errs = [np.linalg.norm(fm(X0 + e * V) - P0 - e * JV) for e in (1e-3, 5e-4, 2.5e-4)]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert min(orders) > 1.9, (name, scheme, errs, orders)
