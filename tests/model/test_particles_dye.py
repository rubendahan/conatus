"""Lagrangian particles and numerical dye of a native simulation (fast CPU checks).

* the exact spectral evaluation at points reproduces the grid values (Fourier x Chebyshev, complex
  layouts refused) to rounding;
* tracers in steady Taylor–Green (dt(u) = 0) against the trajectory integrated to 1e-13 by
  DOP853: order 4 (RK4); in decaying Taylor–Green (exact Navier–Stokes solution, u = e^{-2 nu t}
  u_0, trajectories = the steady ones at the time s(t) = (1 - e^{-2 nu t}) / (2 nu)): order 3,
  set by the RK443 flow states (exact past states at the start lower the error, not the order);
* plane Couette (exact steady solution u = y): x(t) = x0 + y0 t to rounding, particles on the
  walls move with them; the HDF5 file reads back with dims (t, particle, component);
* walls: a tracer pushed through a wall is refused, or clipped and counted when declared;
* dye: m.dye declares the equation and the no-flux walls; mass conserved to rounding in a
  periodic box, to the measured level with walls; kappa = 0 is pure advection without a wall
  condition; sim.dye writes the diagnostics;
* vector initial data whose components have different broadcast shapes.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import conatus as cn
from conatus.errors import InputTypeError, NumericalError, OptionError, UnsupportedError
from conatus.model.particles import SpectralProbe

P2 = 2 * np.pi


def tg(nu, dt, N=16):
    m = cn.Model(cn.Box(x=cn.periodic(P2), y=cn.periodic(P2)), name="Taylor-Green", nu=nu)
    m.velocity("u")
    if nu:
        m.pressure("p")
        m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    else:
        m.equations("dt(u) = 0")
    sim = m.simulate(N=(N, N), scheme="RK443", dt=dt)
    sim.initial(u=lambda x, y: (np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)))
    return sim


def tg_exact(X0, s):
    """Steady Taylor–Green trajectories at time s (DOP853, tolerances 1e-13)."""
    P = len(X0)

    def f(_t, Z):
        x, y = Z[:P], Z[P:]
        return np.concatenate([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)])
    r = solve_ivp(f, (0, s), np.concatenate([X0[:, 0], X0[:, 1]]), method="DOP853",
                  rtol=1e-13, atol=1e-13)
    return np.stack([r.y[:P, -1], r.y[P:, -1]], axis=1)


def couette(N=(16, 24), dt=0.01, kappa=0.01, dye=True):
    m = cn.Model(cn.Box(x=cn.periodic(4.0), y=cn.walls(-1, 1)), name="Couette", nu=0.1)
    m.velocity("u"), m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    m.at("bottom", u=(-1, 0))
    m.at("top", u=(1, 0))
    if dye:
        m.dye("c", velocity="u", kappa=kappa)
    return m, m.simulate(N=N, scheme="RK443", dt=dt)


def test_probe_reproduces_grid_values():
    _, sim = couette(dye=False)
    sim.initial(u=lambda x, y: (y + 0.1 * np.sin(P2 * x / 4) * (1 - y ** 2), 0.05 * np.cos(P2 * x / 2) * (1 - y ** 2) ** 2))
    pr = SpectralProbe(sim, "u")
    X, Y = np.meshgrid(sim.coordinate("x"), sim.coordinate("y"), indexing="ij")
    v = pr.evaluate(pr.coefficients(), np.stack([X.ravel(), Y.ravel()], 1)).reshape(2, *X.shape)
    g = sim.grid("u")
    assert np.abs(v - g).max() <= 1e-13 * np.abs(g).max()
    assert "exact evaluation" in pr.explain() and "x: real fourier" in str(pr) and "y: chebyshev" in str(pr)
    m = cn.Model(cn.Box(x=cn.periodic(P2, complex=True), y=cn.walls(0, 1)), name="cplx")
    m.velocity("u")
    m.equations("dt(u) = 0")
    s2 = m.simulate(N=(8, 8), scheme="RK443", dt=0.1)
    with pytest.raises(UnsupportedError, match="complex-valued"):
        SpectralProbe(s2, "u")


def test_taylor_green_orders():
    X0 = np.random.default_rng(0).uniform(0, P2, (12, 2))
    for nu, order, ratio in ((0.0, 4, 12.0), (0.5, 3, 6.0)):
        err = []
        for dt in (0.1, 0.05):
            sim = tg(nu, dt)
            pt = sim.particles(X0)
            sim.run(until=2.0)
            s = 2.0 if not nu else (1 - np.exp(-2 * nu * 2.0)) / (2 * nu)
            err.append(np.abs(pt.positions(unwrapped=True) - tg_exact(X0, s)).max())
        # halving dt divides the error by 2^order (measured 15.6 and 7.7)
        assert err[0] / err[1] > ratio, (nu, err)
        assert f"order 4 in a steady flow, {3} in an unsteady flow" in pt.explain()


def test_couette_exact_file_and_walls(tmp_path):
    _, sim = couette(dye=False)
    sim.initial(u=lambda x, y: (y, 0 * x))
    X0 = np.array([[0.1, -1.0], [0.5, 1.0], [1.0, 0.3], [3.9, -0.7]])
    pt = sim.particles(X0, path=tmp_path / "p.h5", every=50)
    sim.run(until=3.0)
    ex = X0 + np.stack([3.0 * X0[:, 1], 0 * X0[:, 1]], 1)
    assert np.abs(pt.positions(unwrapped=True) - ex).max() < 1e-12
    w = pt.positions()
    assert np.all((w[:, 0] >= 0) & (w[:, 0] < 4.0))
    d = cn.io.read(tmp_path / "p.h5")
    assert d.dims["position"] == ["t", "particle", "component"]
    assert d.dims["wall_clips"] == ["t", "particle"]
    assert len(d.sim_time) == 7 and d.sim_time[0] == 0.0          # initial row + 6
    assert np.allclose(d.tasks["position_unwrapped"][-1], ex, rtol=0, atol=1e-12)
    assert np.allclose(d.tasks["velocity"][-1][:, 0], X0[:, 1], rtol=0, atol=1e-12)
    ds = d.to_xarray()
    assert ds["position"].dims == ("t", "particle", "component")
    assert list(ds["component"].values) == ["x", "y"]


def test_walls_refuse_and_clip():
    m = cn.Model(cn.Box(x=cn.periodic(2.0), y=cn.walls(0, 1)), name="uniform up")
    m.velocity("u")
    m.equations("dt(u) = 0")
    sim = m.simulate(N=(8, 8), scheme="RK443", dt=0.1)
    sim.initial(u=lambda x, y: (0 * x, 1 + 0 * y))       # crosses the top wall (kinematic)
    pt = sim.particles(np.array([[0.5, 0.95], [1.0, 0.2]]))
    with pytest.raises(NumericalError, match="left \\[0, 1\\] along y"):
        sim.run(steps=1)
    sim2 = m.simulate(N=(8, 8), scheme="RK443", dt=0.1)
    sim2.initial(u=lambda x, y: (0 * x, 1 + 0 * y))
    pc = sim2.particles(np.array([[0.5, 0.95], [1.0, 0.2]]), at_walls="clip")
    sim2.run(steps=3)
    assert np.allclose(pc.positions()[:, 1], [1.0, 0.5], rtol=0, atol=1e-14)
    assert list(pc.wall_clips()) == [3, 0] and "3 clip(s)" in pc.explain()
    del pt


def test_particle_refusals(tmp_path):
    _, sim = couette(dye=False)
    sim.initial(u=lambda x, y: (y, 0 * x))
    with pytest.raises(InputTypeError, match="outside the walls"):
        sim.particles(np.array([[0.1, 1.5]]))
    with pytest.raises(OptionError, match="at_walls"):
        sim.particles(np.array([[0.1, 0.5]]), at_walls="bounce")
    with pytest.raises(OptionError, match="both or neither"):
        sim.particles(np.array([[0.1, 0.5]]), path=tmp_path / "a.h5")
    with pytest.raises(InputTypeError, match="vector unknown"):
        sim.particles(np.array([[0.1, 0.5]]), velocity="p")
    with pytest.raises(InputTypeError, match="missing"):
        sim.particles({"x": [0.1]})
    pd = sim.particles({"x": [0.1, 0.2], "y": [0.0, 0.5]})
    assert pd.positions().shape == (2, 2)
    m = cn.Model(cn.Disk(radius=1.0), name="disk")
    m.velocity("u")
    m.equations("dt(u) = 0")
    sd = m.simulate(N=(8, 8), scheme="RK443", dt=0.1)
    with pytest.raises(UnsupportedError, match="particles on"):
        sd.particles(np.array([[0.1, 0.2]]))


def test_dye_mass_periodic_and_walls(tmp_path):
    # periodic: mass conserved to rounding (div form, Fourier)
    m = cn.Model(cn.Box(x=cn.periodic(P2), y=cn.periodic(P2)), name="TG dye", nu=0.05)
    m.velocity("u"), m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    m.dye("c", velocity="u", kappa=0.01)
    sim = m.simulate(N=(32, 32), scheme="RK443", dt=0.02)
    sim.initial(u=lambda x, y: (np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)),
                c=lambda x, y: np.exp(np.cos(x - 1.0) + np.cos(y)))
    out = sim.dye("c", tmp_path / "d.h5", every=25)
    sim.run(until=2.0)
    d = cn.io.read(tmp_path / "d.h5")
    mass = d.tasks["mass"]
    assert np.abs(mass - mass[0]).max() <= 1e-13 * abs(mass[0])
    assert np.all(np.diff(d.tasks["variance"]) < 0)          # diffusion destroys variance
    assert d.tasks["c"].shape[1:] == sim.grid("c").shape
    e = out.explain()
    assert "dt(c) - 0.01*lap(c) = -div(c*u)" in e and "no walls" in e
    # walls: no-flux condition written; the mass drift is the measured level (1.8e-10 at
    # N = 16 x 24, dt = 0.01, a Gaussian blob sheared for 3 time units)
    m2, s2 = couette()
    assert "dy(c) = 0" in m2.explain()
    s2.initial(u=lambda x, y: (y, 0 * x), c=lambda x, y: np.exp(-((x - 2) ** 2 + y ** 2) / 0.3))
    s2.dye("c", tmp_path / "w.h5", every=50, field=False)
    s2.run(until=1.0)
    mw = cn.io.read(tmp_path / "w.h5").tasks["mass"]
    assert np.abs(mw - mw[0]).max() < 1e-8 * mw[0]
    assert "c" not in cn.io.read(tmp_path / "w.h5").tasks


def test_dye_pure_advection_and_refusals(tmp_path):
    m, sim = couette(N=(16, 16), kappa=0.0)
    e = m.explain()
    assert "dt(c) = -div(c*u)" in e and "dy(c)" not in e
    sim.initial(u=lambda x, y: (y, 0 * x), c=lambda x, y: np.cos(P2 * x / 4) * (1 - y ** 2))
    f = sim.dye("c", tmp_path / "a.h5", every=5)
    sim.run(steps=10)
    assert "no diffusion and no filter" in f.explain()
    m3 = cn.Model(cn.Box(x=cn.periodic(1.0)), name="x")
    m3.velocity("u")
    with pytest.raises(InputTypeError, match="diffusivity"):
        m3.dye("c", velocity="u", kappa=-1.0)
    with pytest.raises(InputTypeError, match="not a vector unknown"):
        m3.dye("c", velocity="w", kappa=1.0)
    with pytest.raises(InputTypeError, match="not a declared parameter"):
        m3.dye("c", velocity="u", kappa="kap")
    m4 = cn.Model(cn.Disk(radius=1.0), name="d")
    m4.velocity("u")
    with pytest.raises(UnsupportedError, match="m.dye"):
        m4.dye("c", velocity="u", kappa=1.0)


def test_vector_initial_mixed_shapes():
    # (y, 0*x) has components of shapes (1, Ny) and (Nx, 1): each is broadcast
    # to the grid (a bare ValueError before)
    _, sim = couette(dye=False)
    sim.initial(u=lambda x, y: (y, 0 * x))
    g = sim.grid("u")
    assert np.allclose(g[0], np.broadcast_to(sim.coordinate("y"), g[0].shape), rtol=0, atol=1e-14)
    assert np.all(g[1] == 0)


def test_dye_kappa_parameter_values_checked_at_compilation():
    # kappa given as a declared parameter was taken as diffusive whatever its
    # value; kappa = 0 kept the no-flux wall condition (ill-posed: the dye grew 14-fold in half a
    # time unit) and kappa < 0 ran as anti-diffusion, both silently
    def model(kv):
        m = cn.Model(cn.Box(x=cn.periodic(4.0), y=cn.walls(-1, 1)), name="C", nu=0.1, kappa=kv)
        m.velocity("u"), m.pressure("p")
        m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
        m.at("bottom", u=(-1, 0))
        m.at("top", u=(1, 0))
        m.dye("c", velocity="u", kappa="kappa")
        return m
    with pytest.raises(OptionError, match="ill-posed"):
        model(0.0).simulate(N=(8, 8), scheme="RK443", dt=0.01)
    with pytest.raises(OptionError, match="anti-diffusion"):
        model(-0.01).simulate(N=(8, 8), scheme="RK443", dt=0.01)
    with pytest.raises(OptionError, match="anti-diffusion"):
        model(0.01).simulate(N=(8, 8), scheme="RK443", dt=0.01, kappa=-1.0)
    m = model(0.01)
    m.simulate(N=(8, 8), scheme="RK443", dt=0.01)
    # periodic box: no wall condition, kappa = 0 by name is plain advection
    mp = cn.Model(cn.Box(x=cn.periodic(P2), y=cn.periodic(P2)), name="P", kap=0.0)
    mp.velocity("u")
    mp.equations("dt(u) = 0")
    mp.dye("c", velocity="u", kappa="kap")
    mp.simulate(N=(8, 8), scheme="RK443", dt=0.01)


def test_unsteady_order_three_is_the_flow_scheme():
    # in decaying Taylor-Green the tracers see the RK443 flow states, whose own
    # error is O(dt^3); with the start of the run given the exact past states (so the first
    # step interpolates with four states, not two) the error drops but the order stays 3
    X0 = np.random.default_rng(0).uniform(0, P2, (12, 2))
    nu, T = 0.5, 2.0
    ref = tg_exact(X0, (1 - np.exp(-2 * nu * T)) / (2 * nu))
    err = {}
    for seeded in (False, True):
        e = []
        for dt in (0.1, 0.05):
            sim = tg(nu, dt)
            pt = sim.particles(X0)
            if seeded:
                c0 = pt.probe.coefficients()
                hist = [(-2 * dt, np.exp(4 * nu * dt) * c0), (-dt, np.exp(2 * nu * dt) * c0),
                        (0.0, c0)]
                pt.begin_run = lambda pt=pt, hist=hist: setattr(pt, "_hist", list(hist))
            sim.run(until=T)
            e.append(np.abs(pt.positions(unwrapped=True) - ref).max())
        err[seeded] = e
    assert err[True][0] < 0.5 * err[False][0]               # the start contributes ...
    assert 6.0 < err[True][0] / err[True][1] < 10.0         # ... but the order stays 3
