"""ParametricEVP: Fourier wavenumber as a continuous parameter, declared functions of parameters
(gallery gaps: Swift–Hohenberg rebuilt one pencil per q; rotating convection one
pencil per Ta because sqrt(Ta) is not a Laurent polynomial)."""
import numpy as np
import pytest
import scipy.optimize as so

from conatus.expert import (EVP, IVP, CartesianCoordinates, ComplexFourier, Domain, Field, Parameter,
                            ParametricEVP, ProblemError, Zero, channel, critical, diff, dt, interp,
                            linearize, sqrt)


def sh_pencil(N=8):
    """Linearized Swift–Hohenberg on a period 2 pi (the period does not matter: k is a parameter)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [ComplexFourier(x, N, (0, 2 * np.pi), dealias=1)])
    u, r = Field("u", D), Parameter("r", 0.1)
    d2 = lambda f: diff(diff(f, x), x)                        # noqa: E731
    ivp = IVP([u])
    ivp.add_equation((dt(u) - r * u + (u + d2(u)) + d2(u + d2(u)), -u * u * u))
    evp = linearize(ivp, around={u: Zero(())}, eigenvalue=Parameter("s", dtype=complex))
    q = Parameter("q", 1.0)
    return ParametricEVP(evp, [r], wavenumbers={"x": q})


def test_wavenumber_parameter_dispersion_relation():
    P = sh_pencil()
    for q in (0.3, 0.9, 1.0, 1.7):
        s = P.solve(r=0.1, q=q).values
        assert np.abs(s - (0.1 - (1 - q * q) ** 2)).min() < 1e-12
    assert "wavenumber parameters k_x = q" in P.explain() and P.exponents["q"] == (0, 1, 2, 3, 4)


def test_critical_minimizes_over_the_wavenumber_without_rebuilding():
    """q_c = 1, r_c = 0 (Cross & Hohenberg 1993) with one pencil; d s/d q exact (Taylor test)."""
    P = sh_pencil()
    n0 = P.n_plans
    cp = critical(P, "r", bracket=(-0.5, 1.0), over={"q": (0.5, 1.5)}, xtol=1e-10)
    assert abs(cp.over["q"] - 1) < 1e-4 and abs(cp.value) < 1e-8
    assert P.n_plans == n0                                            # no rediscretization
    m = P.least_stable(r=0.1, q=0.8)
    h = 1e-6
    fd = (P.track(m.sigma, m.x, r=0.1, q=0.8 + h).sigma - P.track(m.sigma, m.x, r=0.1, q=0.8 - h).sigma) / (2 * h)
    assert abs(P.sensitivity(m, "q") - fd) < 1e-7 and abs(fd - 4 * 0.8 * (1 - 0.64)) < 1e-7


def rotating_layer(N=16):
    """Rotating RB, free walls, Pr = 1: Coriolis parameter sqrt(Ta) declared as a derived coordinate."""
    ch = channel(N, base=None, interval=(0.0, 1.0))
    z = ch.y
    u, v, w, p, T = (ch.field(n) for n in ("u", "v", "w", "p", "T"))
    sigma = Parameter("sigma", dtype=complex)
    Ra, k, Ta = Parameter("Ra", 1e3), Parameter("k", 3.0), Parameter("Ta", 1e3)
    f = sqrt(Ta)
    Dz = lambda q: diff(q, z)                                  # noqa: E731
    lap = lambda q: Dz(Dz(q)) - k ** 2 * q                     # noqa: E731
    pb = EVP([u, v, w, p, T], eigenvalue=sigma)
    pb.add_equation((sigma * u - lap(u) + 1j * k * p - f * v, 0))
    pb.add_equation((sigma * v - lap(v) + f * u, 0))
    pb.add_equation((sigma * w - lap(w) + Dz(p) - Ra * T, 0))
    pb.add_equation((sigma * T - lap(T) - w, 0))
    pb.add_equation((1j * k * u + Dz(w), 0))
    for i, cond in enumerate((Dz(u), Dz(v), w, T)):
        pb.add_bc(interp(cond, z, 0.0), interp(cond, z, 1.0), into=i)
    return pb, (Ra, k, Ta), f


def test_undeclared_function_of_a_parameter_is_refused():
    pb, params, _ = rotating_layer()
    with pytest.raises(ProblemError, match="derived"):
        ParametricEVP(pb, list(params))


def test_declared_sqrt_ta_one_pencil_for_all_ta():
    """Chandrasekhar (1961, §26): Ra(k) = ((pi^2 + k^2)^3 + pi^2 Ta) / k^2 on free walls."""
    pb, params, f = rotating_layer()
    P = ParametricEVP(pb, list(params), derived=[f])
    assert "declared functions of parameters" in P.explain()
    for Ta in (10.0, 1e3, 1e4):
        x = so.brentq(lambda x: 2 * x ** 3 + 3 * np.pi ** 2 * x ** 2 - np.pi ** 6 - np.pi ** 2 * Ta,
                      1e-6, 1e4, xtol=1e-15)
        ex, kx = ((np.pi ** 2 + x) ** 3 + np.pi ** 2 * Ta) / x, np.sqrt(x)
        cp = critical(P, "Ra", bracket=(0.8 * ex, 1.25 * ex), over={"k": (0.8 * kx, 1.25 * kx)},
                      values={"Ta": Ta}, xtol=1e-12)
        assert abs(cp.value / ex - 1) < 1e-10, (Ta, cp.value, ex)
    m = P.least_stable(Ra=2000.0, k=3.0, Ta=500.0)
    h = 1e-3
    fd = (P.track(m.sigma, m.x, Ra=2000.0, k=3.0, Ta=500.0 + h).sigma
          - P.track(m.sigma, m.x, Ra=2000.0, k=3.0, Ta=500.0 - h).sigma) / (2 * h)
    an = P.sensitivity(m, "Ta")                                  # chain rule through sqrt(Ta)
    assert abs(an - fd) < 1e-6 * max(abs(an), 1e-3), (an, fd)
