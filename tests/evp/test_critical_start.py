"""critical(): explicit starting point, bracket centred on 0, sensitivity at p = 0 (gallery gaps). Swift–Hohenberg growth pencil s = r - (1 - q^2)^2: neutral r = (1 - q^2)^2, ds/dr = 1."""
import numpy as np
import pytest

from conatus.expert import (CartesianCoordinates, ComplexFourier, Domain, Field, IVP, Parameter,
                            ParametricEVP, critical, diff, dt, linearize)
from conatus.ir import Zero


def sh_pencil(q):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [ComplexFourier(x, 8, (0, 2 * np.pi / q), dealias=1)])
    u, r = Field("u", D), Parameter("r", 0.1)
    d2 = lambda f: diff(diff(f, x), x)                        # noqa: E731
    ivp = IVP([u])
    ivp.add_equation((dt(u) - r * u + (u + d2(u)) + d2(u + d2(u)), -u * u * u))
    evp = linearize(ivp, around={u: Zero(())}, eigenvalue=Parameter("s", dtype=complex))
    return ParametricEVP(evp, [r], group=lambda g: g[0].k > 0)


def test_bracket_centred_on_zero():
    """Midpoint 0: brentq was called with xtol = 0 (ValueError), Newton's relative test never ended."""
    for q in (1.0, 0.9):
        cp = critical(sh_pencil(q), "r", bracket=(-1.0, 1.0), xtol=1e-9)
        assert abs(cp.value - (1 - q * q) ** 2) < 1e-12, cp.value
        assert "bracket midpoint (default)" in cp.explain()


def test_sensitivity_at_zero_parameter_is_finite():
    P = sh_pencil(1.0)
    m = P.least_stable(r=0.0)
    assert abs(P.sensitivity(m, "r") - 1.0) < 1e-12        # was NaN (0/0)


def test_explicit_start_is_used_and_declared():
    P = sh_pencil(0.9)
    cp = critical(P, "r", bracket=(-0.5, 5.0), start=0.2, xtol=1e-9)
    assert abs(cp.value - 0.19 ** 2) < 1e-12
    assert "r = 0.2 (given)" in cp.explain()
    with pytest.raises(ValueError, match="outside the bracket"):
        critical(P, "r", bracket=(-0.5, 5.0), start=7.0, xtol=1e-9)
