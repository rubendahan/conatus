"""Native layer on the whole line (cn.line, Hermite), the half line (cn.half_line, Laguerre) and
compound intervals (cn.walls(a, c, b)): parity lot Dedalus.

The user writes the equations and the wall conditions only; continuity at the interfaces, the
taus of each piece and the scale of the unbounded bases are declared by explain(). Exact solutions
(Dedalus 3 has none of these bases)."""
import numpy as np
import pytest

import conatus as cn
from conatus.discretize.families import family_of
from conatus.errors import InputTypeError, ResolutionError, UnsupportedError
from conatus.reference import transforms as T


def _grid(m, N, c):
    return T.grid(m.compile(N).bases[c], dealias=False)


def test_line_harmonic_oscillator():
    m = cn.Model(cn.Box(x=cn.line(scale=1.0)), name="QHO")
    m.scalar("u")
    m.eigenvalue("E", growth="real")
    m.given(V=lambda x: x ** 2)
    m.equations("E*u + lap(u) - V*u = 0")
    ev = np.sort(np.real(m.eigenvalues(N=40).values))
    assert np.max(abs(ev[:20] - (2 * np.arange(20) + 1))) < 1e-12
    txt = m.explain(N=40, full=True)
    assert "Hermite functions" in txt and "declared scale 1" in txt and "No boundary" in txt


def test_half_line_heat_with_wall_condition():
    """u_t = u_yy - a u, u(0) = 0: u = y (1+t)^-3/2 exp(-y^2/(4(1+t)) - a t), CNAB2."""
    m = cn.Model(cn.Box(y=cn.half_line(scale=0.5)), name="heat on the half line", a=0.7)
    m.scalar("u")
    m.equations("dt(u) - lap(u) = -a*u")
    m.at("wall", u=0)

    def ex(y, t):
        return y / (1 + t) ** 1.5 * np.exp(-y ** 2 / (4 * (1 + t)) - 0.7 * t)
    sim = m.simulate(N=40, scheme="CNAB2", dt=1e-3)
    sim.initial(u=lambda y: ex(y, 0))
    sim.run(until=0.5)
    assert np.max(abs(sim.grid("u") - ex(sim.coordinate("y"), sim.t))) < 3e-7
    txt = m.explain(N=40, full=True)
    assert "Laguerre lift" in txt and "wall at y = 0" in txt


def _layer(geo, N, k=20.0):
    m = cn.Model(geo, name="layer")
    m.scalar("u")
    m.given(f=lambda x: -2 * k * k * np.tanh(k * x) / np.cosh(k * x) ** 2)
    m.equations("lap(u) = f")
    m.at("bottom", u=float(np.tanh(-k)))
    m.at("top", u=float(np.tanh(k)))
    sol = m.solve(N=N)
    x = sol.coordinate("x")
    assert np.array_equal(x, _grid(m, N, "x"))
    return np.max(abs(sol.grid("u") - np.tanh(k * x))), m


def test_compound_interval_native_layer():
    """Same total N = 192: one interval 1.3e-7, three pieces refined at the layer 2.5e-14. The
    interface conditions and per-piece taus are automatic and listed."""
    e_one, _ = _layer(cn.Box(x=cn.walls(-1, 1)), 192)
    e_cmp, m = _layer(cn.Box(x=cn.walls(-1, -0.15, 0.15, 1)), (64, 64, 64))
    assert e_one > 1e-8 and e_cmp < 1e-12, (e_one, e_cmp)
    txt = m.explain(N=(64, 64, 64), full=True)
    assert "u continuous at interface x=-0.15" in txt and "dx(u) continuous" in txt
    assert "of piece 2" in txt and "compound interval [-1, -0.15], [-0.15, 0.15], [0.15, 1]" in txt
    assert txt.count("lifted into") == 6                   # 2 walls + 2 x 2 interface conditions


@pytest.mark.slow
def test_compound_fourier_ivp_matches_single_interval():
    """2D (Fourier x compound), nonlinear: (24, 24) pieces vs a 128-mode single interval."""
    def run(zax, N):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=zax), name="2D", nu=0.05)
        m.scalar("T")
        m.equations("dt(T) - nu*lap(T) = -T*dz(T) - T*dx(T)")
        m.at("walls", T=0)
        sim = m.simulate(N=N, scheme="RK443", dt=2e-3)
        sim.initial(T=lambda x, z: np.sin(np.pi * z) * (1 + 0.3 * np.cos(x)))
        sim.run(until=0.2)
        zs = np.linspace(-0.97, 0.97, 24)
        c, F = sim.coeffs("T"), family_of(sim.compiled.bases["z"])
        if F is None:
            return np.array([np.polynomial.chebyshev.chebval(z, c.T) for z in zs])
        return np.array([F.evaluation_row(0, z) @ c.T for z in zs])
    ref = run(cn.walls(-1, 1), (16, 128))
    assert np.max(abs(run(cn.walls(-1, 0, 1), (16, (24, 24))) - ref)) < 5e-6


def test_fourier_ivp_with_two_and_three_interfaces():
    """Fourier x compound with 2+ interfaces (each real Fourier group has a cos
    and a sin half) failed with "banded part is singular"; the pinned columns now keep the banded
    part square on every piece. Same linear run as one interval, to spatial round-off."""
    def run(zax, N):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=zax), name="heat2D")
        m.scalar("T")
        m.equations("dt(T) - lap(T) = 0")
        m.at("walls", T=0)
        sim = m.simulate(N=N, scheme="SBDF2", dt=1e-3)
        sim.initial(T=lambda x, z: np.sin(np.pi * z) * (1 + 0.3 * np.cos(x) + 0.2 * np.sin(2 * x)))
        sim.run(steps=5)
        zs = np.linspace(-0.95, 0.95, 8)               # no interface among them
        c, F = sim.coeffs("T"), family_of(sim.compiled.bases["z"])
        if F is None:
            return np.array([np.polynomial.chebyshev.chebval(z, c.T) for z in zs])
        return np.array([F.evaluation_row(0, z) @ c.T for z in zs])
    ref = run(cn.walls(-1, 1), (8, 40))
    for zax, N in ((cn.walls(-1, -0.3, 0.4, 1), (8, (16, 20, 16))),
                   (cn.walls(-1, -0.5, 0, 0.5, 1), (8, (12, 12, 12, 12)))):
        assert np.max(abs(run(zax, N) - ref)) < 1e-9


def test_refusals_say_what_to_write():
    with pytest.raises(InputTypeError, match="needs scale"):
        cn.line()
    with pytest.raises(InputTypeError, match="must increase"):
        cn.walls(-1, 1, 0)
    with pytest.raises(InputTypeError, match="Chebyshev per piece"):
        cn.walls(0, 0.5, 1, basis="sincos")
    m = cn.Model(cn.Box(x=cn.walls(-1, 0, 1)), name="r")
    m.scalar("u")
    m.equations("dt(u) - lap(u) = 0")
    m.at("walls", u=0)
    with pytest.raises(ResolutionError, match="one integer >= 2 per piece"):
        m.simulate(N=32, scheme="SBDF2", dt=1e-3)
    m2 = cn.Model(cn.Box(x=cn.line(scale=1.0)), name="r2")
    m2.scalar("u")
    m2.equations("dt(u) - lap(u) = 0")
    with pytest.raises(Exception, match="no boundary"):
        m2.at("walls", u=0)
        m2.compile(16)
    m3 = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.half_line(scale=1.0)), name="NS")
    m3.velocity("u")
    m3.pressure("p")
    m3.equations("dt(u) + grad(p) - lap(u) = 0", "div(u) = 0")
    m3.at("wall", u=0)
    with pytest.raises(UnsupportedError, match="incompressible"):
        m3.compile((8, 16))


def test_one_dimensional_compound_resolution_forms():
    m = cn.Model(cn.Box(z=cn.walls(-1, 0, 1)), name="heat")
    m.scalar("u")
    m.equations("dt(u) - lap(u) = 0")
    m.at("walls", u=0)
    for N in ((8, 12), ((8, 12),), {"z": (8, 12)}):
        assert m.compile(N).bases["z"].sizes == (8, 12)
