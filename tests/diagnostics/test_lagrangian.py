"""Lagrangian diagnostics against closed-form flows.

Solid rotation (exact circles), cellular flow psi = sin x sin y (psi conserved along paths,
spectral interpolation = analytic velocity), observed Lagrange order, Maxey–Riley in linear flows
(matrix exponential), steady-flow streaklines = pathlines = streamlines, timelines, FTLE of the
hyperbolic saddle (= a exactly), boundary policy, CPU = GPU."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from conatus.diagnostics.grid import Chebyshev, Grid, Periodic
from conatus.diagnostics.lagrangian import (AnalyticVelocity, Emitter, GridVelocity, Interpolator,
                                            MaxeyRiley, Particles, PathRecorder, Streaklines,
                                            Timelines, ftle)
from conatus.errors import CoordinateError, OptionError

TWO_PI = 2 * np.pi
FREE = [Chebyshev(-10, 10, 4), Chebyshev(-10, 10, 4)]     # unbounded in practice (no clipping)


def rotation(x, t):
    """Solid-body rotation, angular velocity 1."""
    return np.c_[-x[:, 1], x[:, 0]]


def cellular(x, t):
    """u = (dpsi/dy, -dpsi/dx) for psi = sin x sin y."""
    return np.c_[np.sin(x[:, 0]) * np.cos(x[:, 1]), -np.cos(x[:, 0]) * np.sin(x[:, 1])]


def test_solid_rotation_exact_circle():
    x0 = np.array([[1.0, 0.0], [0.3, -0.7], [2.0, 1.5]])
    p = Particles(x0, AnalyticVelocity(rotation, 2), axes=FREE, scheme="rk4", boundary="raise")
    p.advance(TWO_PI / 2000, 2000)                      # one full turn
    assert np.max(np.abs(p.x - x0)) < 1e-10
    assert abs(p.t - TWO_PI) < 1e-12 and "RK4" in p.explain()


def test_spectral_interpolation_is_exact_and_cellular_paths_keep_psi():
    g = Grid([Periodic(TWO_PI, 16), Periodic(TWO_PI, 16)])
    X, Y = g.mesh()
    U = np.stack([np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y)])
    it = Interpolator(g, "spectral")
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, TWO_PI, (200, 2))
    assert np.max(np.abs(it(U, pts).T - cellular(pts, 0))) < 1e-13
    # a grid velocity (exact interpolant) gives the analytic trajectories
    gv = GridVelocity(it)
    gv.set(U, 0.0)
    x0 = rng.uniform(0.2, 3.0, (20, 2))
    pa = Particles(x0, AnalyticVelocity(cellular, 2), axes=g.axes, scheme="rk4", boundary="raise")
    pg = Particles(x0, gv, axes=g.axes, scheme="rk4", boundary="raise")
    pa.advance(0.01, 300)
    pg.advance(0.01, 300)
    assert np.max(np.abs(pa.x - pg.x)) < 1e-12
    psi0 = np.sin(x0[:, 0]) * np.sin(x0[:, 1])
    psi = np.sin(pa.x[:, 0]) * np.sin(pa.x[:, 1])
    assert np.max(np.abs(psi - psi0)) < 1e-8          # RK4 error at dt = 0.01, T = 3


def test_spectral_interpolation_chebyshev_axis_exact():
    g = Grid([Periodic(TWO_PI, 12), Chebyshev(-1.0, 2.0, 10)])
    X, Y = g.mesh()
    f = np.cos(3 * X) * Y**7 + np.sin(X) - Y**2
    pts = np.c_[np.linspace(0, TWO_PI, 37), np.linspace(-1, 2, 37)]
    ex = np.cos(3 * pts[:, 0]) * pts[:, 1] ** 7 + np.sin(pts[:, 0]) - pts[:, 1] ** 2
    assert np.max(np.abs(Interpolator(g, "spectral")(f, pts) - ex)) < 1e-11


@pytest.mark.parametrize("p", [2, 3, 4, 5, 6])
def test_lagrange_observed_order(p):
    """Periodic 2D, smooth field: the error falls like h^p (observed order within 0.35)."""
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, TWO_PI, (400, 2))
    ex = np.exp(np.sin(pts[:, 0])) * np.cos(pts[:, 1])
    errs = []
    for n in (32, 64):
        g = Grid([Periodic(TWO_PI, n), Periodic(TWO_PI, n)])
        X, Y = g.mesh()
        errs.append(np.max(np.abs(Interpolator(g, "lagrange", p)(np.exp(np.sin(X)) * np.cos(Y), pts) - ex)))
    order = np.log2(errs[0] / errs[1])
    assert order > p - 0.35, (errs, order)


def test_lagrange_chebyshev_polynomial_exact_and_3d():
    """A degree-(p-1) polynomial in y is reproduced exactly by p-point stencils, walls included."""
    g = Grid([Periodic(TWO_PI, 8), Chebyshev(-1, 1, 20), Periodic(2.0, 6)])
    X, Y, Z = g.mesh()
    f = np.cos(X) * (Y**3 - 2 * Y) * np.cos(np.pi * Z)
    pts = np.c_[np.zeros(9), np.linspace(-1, 1, 9), np.zeros(9)]
    v = Interpolator(g, "lagrange", 4)(f, pts)
    assert np.max(np.abs(v - (pts[:, 1] ** 3 - 2 * pts[:, 1]))) < 1e-12


def test_maxey_riley_linear_flows_exact():
    # uniform flow U: v relaxes exponentially, x follows the closed form
    tau, Uc = 0.3, np.array([1.0, -0.5])
    p = Particles([[0.0, 0.0]], AnalyticVelocity(lambda x, t: np.tile(Uc, (len(x), 1)), 2), axes=FREE,
                  scheme="rk4", boundary="raise", inertia=MaxeyRiley(tau), v0=np.zeros((1, 2)))
    T = 1.0
    p.advance(T / 400, 400)
    v_ex = Uc * (1 - np.exp(-T / tau))
    x_ex = Uc * (T - tau * (1 - np.exp(-T / tau)))
    assert np.max(np.abs(p.v[0] - v_ex)) < 1e-10 and np.max(np.abs(p.x[0] - x_ex)) < 1e-10
    # linear flow u = A x (rotation + strain): state (x, v) obeys a linear ODE -> expm
    A = np.array([[0.2, -1.0], [1.0, -0.2]])
    tau = 0.5
    x0, v0 = np.array([0.7, 0.1]), np.array([0.0, 0.3])
    p = Particles(x0[None], AnalyticVelocity(lambda x, t: x @ A.T, 2), axes=FREE, scheme="rk4",
                  boundary="raise", inertia=MaxeyRiley(tau), v0=v0[None])
    p.advance(0.005, 400)
    M = np.block([[np.zeros((2, 2)), np.eye(2)], [A / tau, -np.eye(2) / tau]])
    s = expm(2.0 * M) @ np.r_[x0, v0]
    assert np.max(np.abs(np.r_[p.x[0], p.v[0]] - s)) < 1e-9
    assert "Basset" in p.explain()
    # settling: terminal velocity (1 - beta) g tau
    g = np.array([0.0, -9.81])
    p = Particles([[0.0, 0.0]], AnalyticVelocity(lambda x, t: np.zeros_like(x), 2), axes=FREE,
                  scheme="rk4", boundary="raise", inertia=MaxeyRiley(0.1, beta=0.5, g=g), v0="fluid")
    p.advance(0.01, 300)
    assert np.max(np.abs(p.v[0] - 0.5 * g * 0.1)) < 1e-10


def test_time_interpolated_snapshots_second_order():
    """u = (1 + t, 0) sampled at t = 0 and t = 1: linear in time is exact for this flow."""
    g = Grid([Periodic(TWO_PI, 8), Periodic(TWO_PI, 8)])
    gv = GridVelocity(Interpolator(g, "spectral"))
    gv.set(np.stack([np.ones(g.shape), np.zeros(g.shape)]), 0.0)
    gv.set(np.stack([2 * np.ones(g.shape), np.zeros(g.shape)]), 1.0)
    p = Particles([[1.0, 1.0]], gv, axes=g.axes, scheme="rk2", boundary="raise")
    p.advance(0.1, 10)
    assert abs(p.x[0, 0] - 2.5) < 1e-13                  # 1 + int_0^1 (1 + t) dt
    with pytest.raises(OptionError):
        p.advance(0.1)                                     # beyond the last snapshot: refused
    assert "linear in time" in gv.explain()


def test_steady_streakline_equals_pathline_equals_streamline():
    """Solid rotation: particles released at (1, 0) every step lie on the unit circle, at the
    angles of the pathline of the first particle (steady flow)."""
    v = AnalyticVelocity(rotation, 2)
    dt = TWO_PI / 400
    s = Streaklines([[1.0, 0.0]], v, axes=FREE, emit_every=1, capacity=100, scheme="rk4", boundary="raise")
    s.advance(dt, 99)
    L = s.polylines()[0]
    assert L.shape == (100, 2) and not np.isnan(L).any()
    ang = np.arctan2(L[:, 1], L[:, 0])
    assert np.max(np.abs(np.hypot(L[:, 0], L[:, 1]) - 1)) < 1e-10   # RK4 radius drift 1.0e-11
    assert np.max(np.abs(ang - dt * np.arange(100))) < 2e-9   # newest first, age k dt; RK4 phase 7.9e-10
    p = Particles([[1.0, 0.0]], v, axes=FREE, scheme="rk4", boundary="raise")
    rec = PathRecorder(p, every=1, max_points=200)
    rec.record()
    for _ in range(99):
        p.advance(dt)
        rec.record()
    path = rec.polylines(wrap=False)[0]
    assert np.max(np.abs(path - L)) < 1e-12   # steady: streak point of age k = pathline at k dt
    assert "smoke" in s.explain()


def test_timelines_and_capacity():
    v = AnalyticVelocity(lambda x, t: np.c_[np.ones(len(x)), np.zeros(len(x))], 2)
    wire = np.c_[np.zeros(5), np.linspace(-1, 1, 5)]
    tl = Timelines(wire, v, axes=FREE, emit_every=10, capacity=3, scheme="rk4", boundary="raise")
    tl.advance(0.1, 25)
    L = tl.polylines()                                    # (capacity, n_wire, 2)
    assert L.shape == (3, 5, 2)
    assert np.allclose(L[0, :, 0], 0.5) and np.allclose(L[1, :, 0], 1.5) and np.allclose(L[2, :, 0], 2.5)
    tl.advance(0.1, 10)                                   # 4th emission: oldest line dropped
    assert tl.dropped == 5 and "hydrogen" in tl.explain()
    e = Emitter(wire, v, axes=FREE, emit_every=10, capacity=4, scheme="rk4", boundary="raise", mode="streak")
    assert np.isnan(e.polylines()[:, 1:]).all()           # not emitted yet: NaN


def test_ftle_hyperbolic_saddle_exact():
    a = 0.8
    v = AnalyticVelocity(lambda x, t: np.c_[a * x[:, 0], -a * x[:, 1]], 2)
    F, info = ftle(v, [np.linspace(-1, 1, 7), np.linspace(-1, 1, 5)], axes=FREE, t0=0.0, T=1.5,
                   dt=0.01, delta=1e-2)
    assert F.shape == (7, 5) and np.max(np.abs(F - a)) < 1e-10
    Fb, _ = ftle(v, [np.linspace(-1, 1, 3)] * 2, axes=FREE, t0=0.0, T=-1.5, dt=0.01, delta=1e-2)
    assert np.max(np.abs(Fb - a)) < 1e-10                 # backward: the other eigendirection
    # 3D saddle (a, b, -a-b): the largest rate
    v3 = AnalyticVelocity(lambda x, t: x * np.array([0.3, 0.5, -0.8]), 3)
    F3, _ = ftle(v3, [np.linspace(0, 1, 3)] * 3, axes=FREE + [Chebyshev(-10, 10, 4)], t0=0.0, T=1.0,
                 dt=0.01, delta=1e-2)
    assert np.max(np.abs(F3 - 0.5)) < 1e-10
    with pytest.raises(OptionError):
        ftle(v, [np.zeros(1)] * 2, axes=FREE, t0=0.0, T=1.0, dt=0.3, delta=1e-2)


def test_bounded_axis_policy_counted_or_refused():
    axes = [Periodic(TWO_PI, 8), Chebyshev(-1, 1, 8)]
    up = AnalyticVelocity(lambda x, t: np.c_[np.zeros(len(x)), np.ones(len(x))], 2)
    p = Particles([[0.0, 0.95], [6.2, 0.0]], up, axes=axes, scheme="rk4", boundary="clip")
    p.advance(0.1)
    assert p.n_clipped > 0 and p.x[0, 1] == 1.0 and "clipped" in p.explain()
    q = Particles([[0.0, 0.95]], up, axes=axes, scheme="rk4", boundary="raise")
    with pytest.raises(CoordinateError):
        q.advance(0.1)
    # periodic wrap: positions wrapped, unwrapped kept
    r = Particles([[6.2, 0.0]], AnalyticVelocity(lambda x, t: np.c_[np.ones(len(x)), np.zeros(len(x))], 2),
                  axes=axes, scheme="rk4", boundary="raise")
    r.advance(0.1, 2)
    assert abs(r.x[0, 0] - 6.4) < 1e-13 and abs(r.positions[0, 0] - (6.4 - TWO_PI)) < 1e-13
    with pytest.raises(CoordinateError):
        Interpolator(Grid(axes), "spectral")(np.zeros((8, 8)), np.array([[0.0, 1.5]]))
    with pytest.raises(OptionError):
        Interpolator(Grid(axes), "lagrange")


@pytest.mark.gpu
def test_cpu_gpu_agree():
    cp = pytest.importorskip("cupy")
    g = Grid([Periodic(TWO_PI, 24), Chebyshev(-1, 1, 20), Periodic(3.0, 16)])
    X, Y, Z = g.mesh()
    U = np.stack([np.sin(X) * (1 - Y**2), np.cos(Z) * (1 - Y**2) ** 2, np.cos(X) * np.sin(2 * np.pi * Z / 3)])
    rng = np.random.default_rng(5)
    pts = np.c_[rng.uniform(0, TWO_PI, 500), rng.uniform(-0.9, 0.9, 500), rng.uniform(0, 3, 500)]
    gg = g.on(cp)
    for meth, order in (("spectral", None), ("lagrange", 4)):
        a = Interpolator(g, meth, order)(U, pts)
        b = cp.asnumpy(Interpolator(gg, meth, order)(cp.asarray(U), cp.asarray(pts)))
        assert np.max(np.abs(a - b)) < 1e-12
    gv, gvg = GridVelocity(Interpolator(g, "lagrange", 4)), GridVelocity(Interpolator(gg, "lagrange", 4))
    gv.set(U, 0.0)
    gvg.set(cp.asarray(U), 0.0)
    pc = Particles(pts[:50], gv, axes=g.axes, scheme="rk4", boundary="clip")
    pg = Particles(cp.asarray(pts[:50]), gvg, axes=g.axes, scheme="rk4", boundary="clip")
    pc.advance(0.01, 20)
    pg.advance(0.01, 20)
    assert np.max(np.abs(pc.x - cp.asnumpy(pg.x))) < 1e-12 and pc.n_clipped == pg.n_clipped


def test_nonfinite_velocity_is_refused_not_dropped():
    """A NaN velocity must stop the particles (ConvergenceError), never turn them into 'not yet
    emitted' slots of an emitter (seen when the GPU KH run diverged); an empty
    point set interpolates to an empty result."""
    from conatus.diagnostics import AnalyticVelocity, Grid, Interpolator, Periodic, Streaklines
    from conatus.errors import ConvergenceError, CoordinateError

    def bad(x, t):
        return np.full_like(x, np.nan)
    g = Grid([Periodic(2 * np.pi, 16), Periodic(2 * np.pi, 16)])
    em = Streaklines(np.array([[1.0, 1.0], [2.0, 2.0]]), AnalyticVelocity(bad, 2), axes=g.axes,
                     emit_every=1, capacity=4, scheme="rk4", boundary="clip")
    with pytest.raises(ConvergenceError):
        em.advance(0.1)
    it = Interpolator(g, "lagrange", order=4)
    assert it(np.zeros(g.shape), np.zeros((0, 2))).shape == (0,)
    with pytest.raises(CoordinateError):
        it(np.zeros(g.shape), np.array([[np.nan, 0.0]]))


def test_snapshot_clock_roundoff_at_large_time():
    """The particle clock is a sum of dt: at t ~ 229 it differs from the solver time by ~1e-12.
    That must not be refused as extrapolation, while a real gap still is."""
    from conatus.diagnostics import GridVelocity, Interpolator
    from conatus.errors import OptionError
    g = Grid([Periodic(2 * np.pi, 8), Periodic(2 * np.pi, 8)])
    gv = GridVelocity(Interpolator(g, "lagrange", order=2))
    u = np.ones((2,) + g.shape)
    gv.set(u, 229.48)
    gv.set(u, 229.5)
    x = np.zeros((3, 2))
    assert np.allclose(gv(x, 229.50000000000102), 1.0)
    with pytest.raises(OptionError):
        gv(x, 229.51)


def test_fractional_step_counts_are_refused():
    """advance(dt, 2.5) took 2 steps without a word (as IVPSolver.run did)."""
    from conatus.errors import OptionError
    p = Particles([[0.5, 0.0]], AnalyticVelocity(rotation, 2), axes=FREE, scheme="rk4",
                  boundary="raise")
    with pytest.raises(OptionError, match="nsteps=2.5"):
        p.advance(0.01, 2.5)
    p.advance(0.01, 2)
    assert abs(p.t - 0.02) < 1e-15


def test_review_v011_counts_and_snapshot_copies():
    """Findings 66 and 68: counts are never truncated; a velocity snapshot
    does not follow later in-place changes of the caller's array."""
    import pytest as _pt

    from conatus.diagnostics.lagrangian import AnalyticVelocity, Emitter, GridVelocity
    from conatus.errors import OptionError as _OE
    with _pt.raises(_OE, match="integer"):
        AnalyticVelocity(lambda x, t: x, 2.5)
    with _pt.raises(_OE, match="integer"):
        Emitter(np.zeros((1, 1)), AnalyticVelocity(lambda x, t: 0 * x, 1), axes=None,
                emit_every=1.5, capacity=4, scheme="rk4", boundary="raise", mode="streak")
    gv = GridVelocity.__new__(GridVelocity)
    gv.xp, gv.d, gv._snaps = np, 1, []

    class _G:
        shape = (4,)
    gv.grid = _G()
    u = np.ones((1, 4))
    gv.set(u, 0.0)
    u[...] = 5.0
    assert np.all(gv._snaps[0][1] == 1.0)
