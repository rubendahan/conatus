"""Levels C and D (§71–72): spectral convergence and IR-driven MMS for LBVPs."""
import numpy as np
import pytest
import sympy as sy

from conatus.discretize import discretize
from conatus.ir import (LBVP, CartesianCoordinates, Chebyshev, ComplexFourier, Domain, Field,
                        Parameter, RealFourier, diff, interp, lap, lift)
from conatus.reference import transforms as T
from conatus.verify.mms import manufacture, observed_order


def _poisson(N, bc="dirichlet"):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, N, (-1, 2), dealias=1)
    D = Domain(cs, [b])
    u, f = Field("u", D), Field("f", D)
    t1, t2 = Field("tau1", D.without(x)), Field("tau2", D.without(x))
    gl, gr = Field("gl", D.without(x)), Field("gr", D.without(x))
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    if bc == "dirichlet":
        pb.add_equation((interp(u, x, "left"), gl))
        pb.add_equation((interp(u, x, "right"), gr))
    else:  # Neumann left, Robin right
        pb.add_equation((interp(diff(u, x), x, "left"), gl))
        pb.add_equation((interp(diff(u, x), x, "right") + 3 * interp(u, x, "right"), gr))
    return pb, b, u, x, (f, gl, gr)


@pytest.mark.parametrize("bc", ["dirichlet", "robin"])
@pytest.mark.slow                   # 2.6 s in the fast suite (qual_durations0)
def test_spectral_convergence_poisson_1d(bc):
    X = sy.Symbol("x", real=True)
    ue = sy.exp(sy.sin(2 * X)) / (2 + X)
    Ns, errs = [8, 12, 16, 20, 24, 28, 32], []
    for N in Ns:
        pb, b, u, x, data_fields = _poisson(N, bc)
        man = manufacture(pb, {u: ue})
        data = {fld: man.callable(r, [x]) for fld, r in zip(data_fields, man.rhs)}
        sol = discretize(pb).solve(data)
        assert sol.max_residual < 1e-12
        Xg = T.grid(b, dealias=False)
        errs.append(np.max(abs(sol.grid("u") - sy.lambdify(X, ue)(Xg))))
    fit = observed_order(Ns[:4], errs[:4])
    assert fit["rate"] > 0.3, (errs, fit["report"])     # exponential decay before round-off
    assert errs[-1] < 1e-8 and errs[-1] < 1e-7 * errs[0], errs


@pytest.mark.parametrize("fourier", [RealFourier, ComplexFourier])
def test_mms_helmholtz_2d_fourier_cheb(fourier):
    cs = CartesianCoordinates("x", "z")
    x, z = cs["x"], cs["z"]
    bx = fourier(x, 16, (0, 2 * np.pi), dealias=3 / 2)
    bz = Chebyshev(z, 24, (0, 1), dealias=3 / 2)
    D = Domain(cs, [bx, bz])
    Dx = D.without(z)
    u, f = Field("u", D), Field("f", D)
    t1, t2 = Field("t1", Dx), Field("t2", Dx)
    g0, g1 = Field("g0", Dx), Field("g1", Dx)
    k2 = Parameter("k2", 3.0)
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) - k2 * u + lift(t1, bz, -1) + lift(t2, bz, -2), f))
    pb.add_equation((interp(u, z, 0), g0))
    pb.add_equation((interp(diff(u, z), z, 1) + 2 * interp(u, z, 1), g1))
    X, Z = sy.symbols("x z", real=True)
    ue = sy.sin(2 * X) * sy.exp(Z) + sy.cos(X) * Z ** 3 + 1
    man = manufacture(pb, {u: ue})
    plan = discretize(pb)
    sol = plan.solve({f: man.callable(man.rhs[0], [x, z]), g0: man.callable(man.rhs[1], [x, z]),
                      g1: man.callable(man.rhs[2], [x, z])})
    assert len(plan.groups) == (8 if fourier is RealFourier else 15)
    assert sol.max_residual < 1e-13
    Xg, Zg = T.grid(bx, False)[:, None], T.grid(bz, False)[None, :]
    assert np.max(abs(sol.grid("u") - sy.lambdify((X, Z), ue)(Xg, Zg))) < 1e-12
    assert "residual" in sol.explain()
