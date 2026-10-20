"""Equation scaling and unit rescaling: an equivalent statement of the problem gives the same answer.

Relations:
  * an equation (or a boundary condition) multiplied by a constant c: the same solution, the same
    eigenvalues, the same modes kept by the spurious-mode filters, the same Newton convergence;
  * the eigenvalue written in another unit (s -> s / tau): eigenvalues scaled by 1 / tau, the
    same modes kept (the drift filter is a statement about modes, not about units);
  * a linear problem with data multiplied by a: the solution multiplied by a (no absolute
    threshold on the amplitude);
  * space and time rescaled together (x -> a x, t -> a^2 t for a diffusion): the same field.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import conatus as cn
from conatus._tolerances_internal import INTERNAL

RTOL = INTERNAL["INVARIANCE_ROUNDING"].value
ETOL = INTERNAL["INVARIANCE_EIGENVALUE"].value
ROW = INTERNAL["INVARIANCE_ROW_SCALING"].value


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), np.finfo(float).tiny))


# ----------------------------------------------------------------------------------- IVP
def _burgers(c: float, scheme: str):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="burgers", nu=0.05)
    m.scalar("u")
    m.equations(f"{c!r}*dt(u) - {c!r}*nu*dx(dx(u)) = -{c!r}*u*dx(u)")
    sim = m.simulate(N=32, scheme=scheme, dt=2e-3)
    sim.initial(u=lambda x: np.sin(x) + 0.3 * np.cos(2 * x))
    sim.run(steps=50)
    return sim.grid("u")


@pytest.mark.parametrize("scheme", ["RK443", "SBDF2", "CNAB2"])
@pytest.mark.parametrize("c", [1e-6, 7.0, 1e6])
def test_ivp_equation_scaling(scheme, c):
    """dt(u) = ... and c dt(u) = c ... are one equation."""
    assert _rel(_burgers(c, scheme), _burgers(1.0, scheme)) < RTOL


def _rb(c: float):
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="RB", Ra=2500.0, Pr=1.0)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations(f"""
        momentum:   {c!r}*dt(u) + {c!r}*grad(p) - {c!r}*Pr*lap(u) = {c!r}*Pr*Ra*T*ez - {c!r}*u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=0.5)
    m.at("top", u=0, T=-0.5)
    sim = m.simulate(N=(16, 12), scheme="RK222", dt=1e-3)
    sim.initial(T=lambda x, z: -z + 1e-2 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sim.run(steps=40)
    # T and u together, relative to the largest value (T; the velocity is 1e-3 of it); the tau
    # unknowns of the scaled equation are scaled with it and are left out
    return np.concatenate([sim.grid("T").ravel(), sim.grid("u").ravel()])


@pytest.mark.parametrize("c", [1e-2, 1e2, 1e5])
def test_ivp_equation_scaling_rb2d(c):
    """Rayleigh-Benard with the momentum equation multiplied by c (pressure gauge, taus). The
    bordered banded solve does not equilibrate its rows: the relative scale of the equations
    enters the rounding (fragile spot of docs/audits/invariance.md, bound ROW_SCALING)."""
    a, b = _rb(c), _rb(1.0)
    assert _rel(a, b) < ROW, _rel(a, b)


@pytest.mark.parametrize("a", [1e-12, 1e8])
def test_ivp_amplitude_scaling(a):
    """A linear problem: data times a gives the solution times a, bit for bit up to rounding."""
    def run(amp):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), y=cn.walls(0, 1)), name="heat", kappa=0.3)
        m.scalar("u")
        m.equations("dt(u) - kappa*lap(u) = 0")
        m.at("walls", u=0)
        sim = m.simulate(N=(16, 16), scheme="SBDF2", dt=1e-2)
        sim.initial(u=lambda x, y: amp * np.sin(np.pi * y) * (1 + 0.5 * np.cos(x)))
        sim.run(steps=30)
        return sim.grid("u")
    assert _rel(run(a) / a, run(1.0)) < RTOL


# ----------------------------------------------------------------------------------- EVP
def _os(c_eq: float = 1.0, c_bc: float = 1.0, unit: float = 1.0, N: int = 48, drift=64):
    """Orr-Sommerfeld at Re = 1e4, alpha = 1: equation times c_eq, the derivative condition times
    c_bc, eigenvalue c' = c / unit (the equation holds unit * c')."""
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations(f"OS: {c_eq!r}*(L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + "
                f"{unit!r}*c*L2(v)) = 0")
    m.at("walls", "v = 0", f"{c_bc!r}*dy(v) = 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.eigenvalues(N=N, drift=drift, drift_tol=1e-6)


def _match(a: np.ndarray, b: np.ndarray) -> float:
    """Largest relative distance from a value of a to the nearest of b (same count asserted)."""
    assert a.size == b.size, f"{a.size} modes vs {b.size}"
    return max(float(np.min(np.abs(b - x)) / max(abs(x), 1e-300)) for x in a)


@pytest.mark.parametrize("c_eq,c_bc", [(1e-8, 1.0), (1e8, 1.0), (1.0, 1e-6), (1.0, 1e6)])
def test_evp_equation_scaling(c_eq, c_bc):
    """The same modes survive the drift filter, with the same eigenvalues."""
    ref = _os()
    r = _os(c_eq=c_eq, c_bc=c_bc)
    assert _match(r.values, ref.values) < ETOL
    assert ref.n_infinite == r.n_infinite


@pytest.mark.parametrize("unit", [1e-4, 1e4])
def test_evp_eigenvalue_unit(unit):
    """c = unit * c': the eigenvalues are divided by unit and the drift filter keeps the same
    modes (it compares modes, whatever the unit of the eigenvalue)."""
    ref = _os()
    r = _os(unit=unit)
    assert _match(r.values * unit, ref.values) < ETOL


def test_generalized_eigenvalues_scaling():
    """(J, M) and (a J, b M): eigenvalues a / b times, the same finite / infinite split."""
    from conatus.bifurcation.equilibria import generalized_eigenvalues
    rng = np.random.default_rng(3)
    J = rng.standard_normal((6, 6))
    M = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    ref = generalized_eigenvalues(J, M)
    for a, b in ((1e-9, 1.0), (1.0, 1e-9), (1e9, 1e9), (1e-12, 1e12)):
        w = generalized_eigenvalues(a * J, b * M)
        assert _match(w * b / a, ref) < ETOL, (a, b)


# --------------------------------------------------------------------------------- NLBVP
def _bratu(c: float):
    from conatus.expert import NewtonSolver
    m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
    m.scalar("u")
    m.equations(f"bratu: {c!r}*dx(dx(u)) + {c!r}*lam*exp(u) = 0")
    m.at("walls", u=0)
    pb = m.compile(48, dealias=1)
    newton = NewtonSolver(pb.problem, tol=1e-12, max_iterations=20, params={"lam": 0.3})
    return newton.solve({pb.fields["u"]: lambda x: 0 * x})


@pytest.mark.parametrize("c", [1e-8, 1e8])
def test_nlbvp_equation_scaling(c):
    """Newton on c F(u) = 0: the same iterates, the same relative residuals, the same verdict."""
    ref, sol = _bratu(1.0), _bratu(c)
    assert sol.converged == ref.converged
    assert len(sol.iterations) == len(ref.iterations)
    assert _rel(sol.grid("u"), ref.grid("u")) < RTOL
    for a, b in zip(sol.iterations, ref.iterations, strict=True):
        assert abs(a.rel_residual - b.rel_residual) <= 1e-3 * b.rel_residual + 1e-15


# ------------------------------------------------------------------------- units: space/time
def test_units_heat_space_time():
    """u_t = kappa u_xx on [0, L], and on [0, a L] with kappa a^2: the same field at the same
    points, the same number of steps (time unchanged; space stretched by a)."""
    def run(a):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi * a), y=cn.walls(0, a)), name="heat",
                     kappa=0.2 * a * a)
        m.scalar("u")
        m.equations("dt(u) - kappa*lap(u) = -u*u")
        m.at("walls", u=0)
        sim = m.simulate(N=(16, 16), scheme="RK443", dt=1e-2)
        sim.initial(u=lambda x, y: np.sin(np.pi * y / a) * (1 + 0.5 * np.cos(x / a)))
        sim.run(steps=40)
        return sim.grid("u")
    ref = run(1.0)
    for a in (1e-3, 37.0):
        assert _rel(run(a), ref) < RTOL, a
