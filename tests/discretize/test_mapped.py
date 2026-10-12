"""Mapped Chebyshev bases on infinite / semi-infinite intervals (ir/maps.py; gallery gap, BACKLOG
wave 1 "domaine infini"). Chain rule d/dy = mu(xi) d/dxi with the metric multiplied as an NCC:
banded exactly for polynomial metrics (tanh, rational TL maps), expanded to machine precision
otherwise (tan map), always reported."""
import numpy as np
import pytest

from conatus.expert import (EVP, IVP, LBVP, CartesianCoordinates, Domain, Field, Parameter, diff,
                            discretize, dt, eigenvalues, grid, integ, interp)
from conatus.discretize.compiler import DiscretizationError
from conatus.ir.maps import MappedChebyshev, RationalMap, TanhMap, TanMap
from conatus.reference import evaluate


def _dom(N, mp):
    cs = CartesianCoordinates("y")
    y = cs["y"]
    return y, Domain(cs, [MappedChebyshev(y, N, mp, dealias=1)])


@pytest.mark.parametrize("mp", [TanhMap(2.0), TanMap(2.0), RationalMap(1.0)])
def test_reference_derivative_chain_rule(mp):
    y, D = _dom(64, mp)
    U = Field("U", D)
    Y = grid(D.bases[0], dealias=False)
    f = lambda Z: np.exp(-(Z - 0.5) ** 2)                                  # noqa: E731
    got = evaluate(diff(U, y), {U: f(Y)})
    assert np.abs(got - (-2 * (Y - 0.5) * f(Y))).max() < 3e-6     # C-infinity flat at xi = +-1: subgeometric


def test_semi_infinite_lbvp_exponential():
    """-u'' + u = 0 on [0, inf), u(0) = 1, u(inf) = 0 -> e^{-y} (rational Chebyshev TL, banded)."""
    errs = []
    for N in (16, 32, 48):
        y, D = _dom(N, RationalMap(2.0))
        u = Field("u", D)
        pb = LBVP([u])
        pb.add_equation((-diff(diff(u, y), y) + u, 0))
        pb.add_bc((interp(u, y, 0.0), 1.0), interp(u, y, np.inf))
        plan = discretize(pb)
        sol = plan.solve()
        Y = grid(D.bases[0], dealias=False)
        errs.append(np.abs(sol.grid("u") - np.exp(-Y)).max())
        gm = plan.matrices()
        assert gm.L.nnz < 12 * N                       # banded: polynomial metric (1 - xi)^2 / 2L
    assert errs[-1] < 1e-10 and errs[2] < errs[1] < errs[0], errs    # 2e-5, 7e-9, 4e-11
    assert "rational Chebyshev TL" in plan.explain()


def _rayleigh(N, mp, alpha=0.4446):
    y, D = _dom(N, mp)
    v, U = Field("v", D), Field("U", D)
    c = Parameter("c", dtype=complex)
    L2 = lambda f: diff(diff(f, y), y) - alpha ** 2 * f                    # noqa: E731
    pb = EVP([v], eigenvalue=c)
    pb.add_equation(((U - c) * L2(v) - diff(diff(U, y), y) * v, 0))
    pb.add_bc(interp(v, y, -np.inf), interp(v, y, np.inf))
    # tanh and its U'' on the tanh map: tail ~3e-8 at N = 64 (the drift test below measures it)
    return discretize(pb, ncc={U: np.tanh}, ncc_tail_tol=1e-7)


def _kh(N, mp):
    r = eigenvalues(_rayleigh(N, mp))
    ok = np.isfinite(r.values) & (np.abs(r.values) < 1)                     # Howard semicircle
    i = np.flatnonzero(ok)[np.argmax(r.values[ok].imag)]
    return r.values[i], r.residuals[i]


def test_kelvin_helmholtz_on_the_whole_line():
    """Michalke (1964): omega_i = 0.1897 at alpha = 0.4446. Tanh map L = 4: exponent alpha L / 2
    = 0.89 of the far field; N = 64 -> 96 drift 2e-6 (truncated [-20, 20] needed N = 384)."""
    c1, _ = _kh(64, TanhMap(4.0))
    c2, res = _kh(96, TanhMap(4.0))
    assert abs(c2 - c1) < 3e-6 and res < 1e-12, (c1, c2)
    assert abs(0.4446 * c2.imag - 0.1897) < 1e-4
    c3, _ = _kh(96, TanhMap(4.0, contour=0.3))                               # path-independent
    assert abs(c3 - c2) < 3e-6, (c2, c3)


def test_refusals():
    y, D = _dom(16, TanhMap(1.0))
    u = Field("u", D)
    ivp = IVP([u])
    ivp.add_equation((dt(u) - diff(diff(u, y), y), 0))
    ivp.add_bc(interp(u, y, -np.inf), interp(u, y, np.inf))
    with pytest.raises(DiscretizationError, match="IVP"):
        discretize(ivp)
    pb = LBVP([u])
    pb.add_equation((integ(u) + u, 1.0))
    with pytest.raises(DiscretizationError, match="not integrable"):
        discretize(pb).matrices()
    y, D = _dom(16, TanMap(1.0, contour=0.2))
    u, S = Field("u", D), Field("S", D)
    pb = LBVP([u])
    pb.add_equation((diff(diff(u, y), y) - S * u, 0))
    pb.add_bc((interp(u, y, 0.0), 1.0), interp(u, y, np.inf))
    with pytest.raises(ValueError, match="deformed contour"):
        discretize(pb, ncc={S: lambda Y: 1 + 0 * Y}).matrices()
    y, D = _dom(64, TanMap(1.0, contour=0.2))          # complex y up to ~1e3: cosh overflows
    u, S = Field("u", D), Field("S", D)
    pb = LBVP([u])
    pb.add_equation((diff(diff(u, y), y) - S * u, 0))
    pb.add_bc(interp(u, y, -np.inf), interp(u, y, np.inf))
    with pytest.raises(DiscretizationError, match="non-finite"):
        discretize(pb, ncc={S: lambda Y: 1 / np.cosh(Y) ** 2}).matrices()
