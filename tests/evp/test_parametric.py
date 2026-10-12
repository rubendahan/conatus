"""Parametric pencil: exact Laurent reconstruction, tracking, Re_c and dc/dRe (speed)."""
import numpy as np
import pytest

from conatus.ir import (EVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, ProblemError,
                        diff, interp, lift)
from conatus.solvers.parametric import ParametricEVP, critical, laurent_exponents

from cases import ALPHA_C, ORSZAG_C, RE_C


def os_problem(N=64):
    cs = CartesianCoordinates("y")
    y = cs["y"]
    b = Chebyshev(y, N, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    v, U, Upp = Field("v", D), Field("U", D), Field("Upp", D)
    taus = [Field(f"tau{i}", D.without(y)) for i in range(1, 5)]
    c, Re, a = Parameter("c", dtype=complex), Parameter("Re", 1e4), Parameter("alpha", 1.0)

    def L2(f):
        return diff(diff(f, y), y) - a ** 2 * f
    lhs = (1 / (1j * a * Re) * L2(L2(v)) - U * L2(v) + Upp * v + c * L2(v)
           + sum(lift(t, b, -(i + 1)) for i, t in enumerate(taus)))
    pb = EVP([v] + taus, eigenvalue=c)
    pb.add_equation((lhs, 0))
    for side in (-1, 1):
        pb.add_equation((interp(v, y, side), 0))
        pb.add_equation((interp(diff(v, y), y, side), 0))
    return pb, Re, a, {U: lambda Y: 1 - Y ** 2, Upp: lambda Y: -2 + 0 * Y}


def test_laurent_exponents():
    cs = CartesianCoordinates("y")
    D = Domain(cs, [Chebyshev(cs["y"], 8, (-1, 1), dealias=1)])
    Re, a, u = Parameter("Re", 2.0), Parameter("a", 1.0), Field("u", D)
    e = 1 / (1j * a * Re) * (a ** 2 + u) - a ** 3
    assert laurent_exponents(e, Re) == frozenset({-1, 0})
    assert laurent_exponents(e, a) == frozenset({-1, 1, 3})
    assert laurent_exponents(1 / (a + 1) * u, a) is None


def test_non_laurent_parameter_refused():
    pb, Re, a, ncc = os_problem(16)
    w = Parameter("w", 1.0)
    pb2 = EVP(pb.variables, eigenvalue=pb.eigenvalue)
    pb2.add_equation((pb.equations[0].lhs + 1 / (w + 1) * pb.variables[0], 0))
    for eq in pb.equations[1:]:
        pb2.add_equation((eq.lhs, 0))
    with pytest.raises(ProblemError, match="Laurent"):
        ParametricEVP(pb2, [w], ncc=ncc)


def test_os_parametric_matches_orszag_rec_and_sensitivity():
    pb, Re, a, ncc = os_problem()
    P = ParametricEVP(pb, [Re, a], ncc=ncc)
    assert P.recon_error < 1e-12
    r = P.solve(Re=1e4, alpha=1.0)
    ph = np.abs(r.values) < 2
    i = np.nonzero(ph)[0][np.argmax(r.values[ph].imag)]
    assert abs(r.values[i] - ORSZAG_C) < 1e-8
    m = P.track(r.values[i], r.vectors[:, i], Re=1e4, alpha=1.0)
    d = P.sensitivity(m, "Re")
    assert abs(d - (-4.2132186e-6 + 3.4277279e-7j)) < 1e-13
    cp = critical(P, "Re", bracket=(5000, 7000), growth=lambda s: s.imag,
                  select=lambda s: np.abs(s) < 2, over={"alpha": (0.98, 1.06)}, xtol=1e-9)
    assert abs(cp.value - RE_C) / RE_C < 1e-6 and abs(cp.over["alpha"] - ALPHA_C) < 2e-5
    assert "verification" in cp.explain()
