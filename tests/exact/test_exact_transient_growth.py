"""Exact transient growth of a non-normal 2 x 2 system, on every time-stepping path (audit hole 2).

Per Fourier mode e^{ikx}, the IR problem

    c1 dt(u) + u + a u - D u_xx - beta v = 0
    c2 (dt(v) + v) + a v - D v_xx = 0                  (nested form: bare v inside a product)

is x' = A x with the non-normal A = [[l1, g], [0, l2]], l1 = -(1 + a + D k^2) / c1,
l2 = -(c2 + a + D k^2) / c2, g = beta / c1. Closed form of the propagator:

    Phi(t) = [[e^{l1 t}, g (e^{l1 t} - e^{l2 t}) / (l1 - l2)], [0, e^{l2 t}]]

and of the energy gain (norm 1/2 int |u|^2 + |v|^2, equal weights per mode):
G(t) = sigma_max(Phi)^2 = (s + sqrt(s^2 - 4 det^2)) / 2, s = ||Phi||_F^2, det = e^{(l1 + l2) t}.
Both eigenvalues decay while G reaches ~ 40: a threshold or eigenvalue test cannot see a wrong
off-diagonal or a wrong mass (c1 -> c1 + 1 changes every entry of Phi).

Two Conatus paths, each against the closed form: (1) IVPSolver on every backend (exact_backends:
CPU, fused F, CUDA, fused F on the GPU, CUDA graph), Phi read column by column from two runs;
(2) conatus.growth.propagator (dense IMEX propagator of the transient-growth toolkit, CPU and
CuPy) fed with the plan's M and L of the mode, then conatus.growth.gain (SVD).
"""
import numpy as np
import pytest

from exact_backends import _GPU, MODES, make_solver

import conatus.expert as C
from conatus.discretize import discretize
from conatus.growth import gain, propagator
from conatus.reference import transforms as T

K, A, DIFF, C1, C2, BETA = 1, 0.4, 0.3, 2.0, 1.5, 40.0
L1 = -(1 + A + DIFF * K ** 2) / C1
L2 = -(C2 + A + DIFF * K ** 2) / C2
G_OFF = BETA / C1


def phi_exact(t: float) -> np.ndarray:
    """Closed-form propagator of the mode (see module docstring)."""
    e1, e2 = np.exp(L1 * t), np.exp(L2 * t)
    return np.array([[e1, G_OFF * (e1 - e2) / (L1 - L2)], [0.0, e2]])


def gain_exact(t: float) -> float:
    """sigma_max(Phi)^2 for a 2 x 2 matrix, in closed form."""
    s = np.sum(phi_exact(t) ** 2)
    det = np.exp((L1 + L2) * t)
    return 0.5 * (s + np.sqrt(s * s - 4 * det * det))


def _problem():
    cs = C.CartesianCoordinates("x")
    x = cs["x"]
    bx = C.RealFourier(x, 8, (0, 2 * np.pi), dealias=3 / 2)
    D = C.Domain(cs, [bx])
    u, v = C.Field("u", D), C.Field("v", D)
    a = C.Parameter("a", A)                  # a Parameter: u + a u stays two terms
    pb = C.IVP([u, v])
    pb.add_equation((C1 * C.dt(u) + u + a * u - DIFF * C.lap(u) - BETA * v, 0))
    pb.add_equation((C2 * (C.dt(v) + v) + a * v - DIFF * C.lap(v), 0))
    return discretize(pb), u, v, bx


def _project(s, name: str, X: np.ndarray) -> float:
    """Amplitude of cos(K x) in the grid values of ``name`` (discrete orthogonality on the grid)."""
    g = s.grid(name)
    return 2.0 * np.mean(g * np.cos(K * X))


def test_closed_form_is_consistent():
    """The closed form itself: Phi' = A Phi (finite difference) and G(0) = 1, G peaks above 30."""
    Am = np.array([[L1, G_OFF], [0.0, L2]])
    h, t = 1e-6, 0.7
    dPhi = (phi_exact(t + h) - phi_exact(t - h)) / (2 * h)
    assert np.abs(dPhi - Am @ phi_exact(t)).max() < 1e-7
    assert abs(gain_exact(0.0) - 1) < 1e-14
    assert max(gain_exact(t) for t in np.linspace(0, 3, 61)) > 30


@pytest.mark.parametrize("scheme,dt,tol", [("RK443", 2e-3, 1e-8), ("SBDF2", 1e-3, 2e-4)])
@pytest.mark.parametrize("mode", MODES)
def test_ivp_propagator_and_gain_every_backend(scheme, dt, tol, mode):
    """Phi(T) column by column (u = cos kx, then v = cos kx), T = 0.5 and 1.5 (rising and near the
    peak of G): relative error of Phi and of G(T) below the time-stepping tolerance."""
    plan, u, v, bx = _problem()
    X = T.grid(bx, dealias=False)
    cols = {}
    for j, f in enumerate((u, v)):
        s = make_solver(plan, scheme, dt, mode)
        s.set_state({f: lambda x: np.cos(K * x)})
        done = 0
        for T_end in (0.5, 1.5):
            n = int(round(T_end / dt)) - done
            advance_chunk(s, n, mode)
            done += n
            cols[(j, T_end)] = np.array([_project(s, "u", X), _project(s, "v", X)])
    for T_end in (0.5, 1.5):
        Phi = np.column_stack([cols[(0, T_end)], cols[(1, T_end)]])
        ref = phi_exact(T_end)
        assert np.abs(Phi - ref).max() < tol * np.abs(ref).max()
        G = gain(Phi, np.eye(2), None).G
        assert abs(G / gain_exact(T_end) - 1) < 2 * tol


def advance_chunk(s, n: int, mode: str) -> None:
    """Advance n more steps (graph mode: one captured-graph replay per chunk)."""
    if mode == "cuda-graph":
        s.run_graph(n)
    else:
        s.run(n)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=_GPU)])
def test_growth_propagator_on_plan_matrices(backend):
    """conatus.growth.propagator + gain on the plan's (M, L) of mode k: the transient-growth toolkit
    (dense IMEX RK443 propagator) against the closed form G(T)."""
    plan = _problem()[0]
    grp = next(g for g in plan.groups if abs(g[0].k - K) < 1e-12)
    mats = plan.matrices(grp)

    class _Op:                                 # the duck-typed operator propagator() reads
        M = mats.M.toarray().astype(complex)
        L0 = mats.L.toarray().astype(complex)
        n = M.shape[0]
        state_cols = np.arange(n)
        Bp = np.zeros((n, n), complex)        # autonomous: no base-flow harmonics
        Bm = Bp

    for T_end in (0.5, 1.5):
        Phi = propagator(_Op, 0.0, T_end, int(round(T_end / 2e-3)), "RK443", backend)
        G = gain(Phi, np.eye(_Op.n), None).G
        assert abs(G / gain_exact(T_end) - 1) < 1e-8
