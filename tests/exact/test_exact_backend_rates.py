"""Exact decay rates on every time-stepping path: CPU, fused F, CUDA, fused F on the GPU, CUDA graph
(hole 1: the GPU paths had only ever been compared with the CPU, never with an exact value).

The cases are those of tests/discretize/test_exact_rates.py that reach a different part of the
device code: Fourier rate with a mass coefficient and an explicit RHS (dense spectral solve, F on
the device), the same inside a product c * (dt(u) + u) (nested form family), Chebyshev
Dirichlet with lifts (bordered-banded device solve), the reflection subspace (reduced system),
an NCC in the mass matrix. RK443 and SBDF2 for each. Every LHS keeps a *bare* unknown ``+ u``
with the coefficient a declared as a Parameter (u + 0.4 u would fold into 1.4 u), so each
test fails when is reintroduced (checked by mutation, docs/AUDIT_J2a.md section 5).
"""
import numpy as np
import pytest

from exact_backends import MODES, advance, make_solver

import conatus.expert as C
from conatus.discretize import discretize
from conatus.reference import transforms as T

K = 3                              # Fourier wavenumber on (0, 2 pi)
A, B, DIFF, CM = 0.4, -0.25, 0.3, 2.0   # a u (L), b u (explicit RHS), diffusivity, mass coefficient
SCHEMES = [("RK443", 2e-3, 1e-8), ("SBDF2", 1e-3, 5e-6)]   # (scheme, dt, |rate error| tolerance)


def _a():
    """a as a Parameter, so that u + a u stays two terms."""
    return C.Parameter("a", A)


def _rate(s, name, u0_max):
    """Observed exponential rate log(max|u(t)| / max|u(0)|) / t."""
    return np.log(np.abs(s.grid(name)).max() / u0_max) / s.t


# ------------------------------------------------------------------ Fourier, flat and nested forms
@pytest.mark.parametrize("form", ["c*dt(u)", "c*(dt(u)+u)"])
@pytest.mark.parametrize("scheme,dt,tol", SCHEMES)
@pytest.mark.parametrize("mode", MODES)
def test_fourier_rate_every_backend(form, scheme, dt, tol, mode):
    """c dt(u) + u + a u - D u_xx = b u        -> rate (b - 1 - a - D k^2) / c
       c (dt(u) + u) + a u - D u_xx = b u      -> rate (b - c - a - D k^2) / c   (nested bare u)"""
    cs = C.CartesianCoordinates("x")
    x = cs["x"]
    bx = C.RealFourier(x, 16, (0, 2 * np.pi), dealias=3 / 2)
    u = C.Field("u", C.Domain(cs, [bx]))
    c = C.Parameter("c", CM)
    pb = C.IVP([u])
    if form == "c*dt(u)":
        pb.add_equation((c * C.dt(u) + u + _a() * u - DIFF * C.lap(u), B * u))
        exact = (B - 1 - A - DIFF * K ** 2) / CM
    else:
        pb.add_equation((c * (C.dt(u) + u) + _a() * u - DIFF * C.lap(u), B * u))
        exact = (B - CM - A - DIFF * K ** 2) / CM
    s = make_solver(discretize(pb), scheme, dt, mode)
    s.set_state({u: lambda X: np.cos(K * X)})
    advance(s, int(round(0.5 / dt)), mode)
    assert abs(_rate(s, "u", 1.0) - exact) < tol


# ------------------------------------------------------------------ Chebyshev Dirichlet (lifts, taus)
def _cheb_ivp(symmetric: bool):
    """c dt(u) + u + a u - D u'' + lifts = 0, u(+-1) = 0 (optionally in the even subspace)."""
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    b = C.Chebyshev(z, 24, (-1, 1), dealias=3 / 2)
    D = C.Domain(cs, [b])
    u, t1, t2 = C.Field("u", D), C.Field("t1", D.without(z)), C.Field("t2", D.without(z))
    G = C.SymmetrySubspace(C.Reflection(z, about=0), u=C.EVEN, t1=C.ODD, t2=C.EVEN) if symmetric else None
    pb = C.IVP([u, t1, t2], symmetry=G)
    pb.add_equation((CM * C.dt(u) + u + _a() * u - DIFF * C.lap(u) + C.lift(t1, b, -1) + C.lift(t2, b, -2),
                     0))
    pb.add_equation((C.interp(u, z, "left"), 0))
    pb.add_equation((C.interp(u, z, "right"), 0))
    return discretize(pb), u


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("scheme,dt,tol", SCHEMES)
@pytest.mark.parametrize("mode", MODES)
def test_chebyshev_dirichlet_rate_every_backend(symmetric, scheme, dt, tol, mode):
    """u = cos(pi z / 2) = sin(pi (z + 1) / 2): rate -(1 + a + D (pi/2)^2) / c."""
    plan, u = _cheb_ivp(symmetric)
    s = make_solver(plan, scheme, dt, mode)
    if symmetric:
        assert s.system.reduced
    s.set_state({u: lambda Z: np.cos(np.pi * Z / 2)})
    u0 = np.abs(s.grid("u")).max()            # grid maximum (z = 0 is not a Gauss point)
    advance(s, int(round(0.5 / dt)), mode)
    exact = -(1 + A + DIFF * (np.pi / 2) ** 2) / CM
    assert abs(_rate(s, "u", u0) - exact) < tol


# ------------------------------------------------------------------ NCC in the mass matrix
@pytest.mark.parametrize("scheme,dt,tol", [("RK443", 2e-3, 1e-9), ("SBDF2", 1e-3, 2e-6)])
@pytest.mark.parametrize("mode", MODES)
def test_ncc_mass_pointwise_every_backend(scheme, dt, tol, mode):
    """(2 + z) dt(u) + u = 0 pointwise: u = exp(-t / (2 + z)) (NCC in M, bare u in L)."""
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    b = C.Chebyshev(z, 16, (-1, 1), dealias=1)
    D = C.Domain(cs, [b])
    u, f = C.Field("u", D), C.Field("f", D)
    pb = C.IVP([u])
    pb.add_equation((f * C.dt(u) + u, 0))
    s = make_solver(discretize(pb, ncc={f: lambda Z: 2 + Z}), scheme, dt, mode)
    s.set_state({u: lambda Z: 1.0 + 0 * Z})
    advance(s, int(round(1.0 / dt)), mode)
    Z = T.grid(b, dealias=False)
    assert np.abs(s.grid("u") - np.exp(-s.t / (2 + Z))).max() < tol
