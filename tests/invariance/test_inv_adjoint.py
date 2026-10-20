"""Tangent and adjoint of the flow map: the dot-product identity, Taylor tests and equivalences.

For every scheme (and both start-ups of the multistep schemes):
  * FlowMap(X0) is the solver's own run from X0 (the map that is differentiated is the one
    that is run), bit for bit;
  * <J V, W> = <V, J^T W> (bilinear pairing of the FlowMap) to rounding;
  * Taylor: |Phi(X0 + e V) - Phi(X0) - e J V| = O(e^2);
  * Revolve with a small checkpoint budget = full storage, bit for bit;
  * the polynomial (matrix-free) transpose of F' = the dense-Jacobian oracle to rounding;
and on a time-dependent problem (forcing and wall data in t) the same identities from t0 != 0.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import conatus as cn
from conatus._tolerances_internal import INTERNAL
from conatus.adjoint import FlowMap
from conatus.timestep.integrator import SCHEMES, scheme_family

DOT = INTERNAL["INVARIANCE_ADJOINT"].value
TAYLOR = INTERNAL["INVARIANCE_TAYLOR_ORDER"].value

CASES = [(s, st) for s in SCHEMES for st in (("lower_order", "rk443")
                                             if scheme_family(s) != "rk" else ("lower_order",))]


def _sim(scheme, startup, forced=False):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), y=cn.walls(0, 1)), name="adv-diff", nu=0.1)
    m.scalar("u")
    m.equations("dt(u) - nu*lap(u) = -u*dx(u)" + (" + sin(2*t)*u*u/4" if forced else ""))
    m.at("bottom", u="cos(t)/5" if forced else 0)
    m.at("top", u=0)
    kw = {} if scheme_family(scheme) == "rk" else {"startup": startup}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = m.simulate(N=(8, 8), scheme=scheme, dt=1e-2, **kw)
        sim.initial(u=lambda x, y: (1 - y) * (0.2 * forced + 0.0) + y * (1 - y) * (np.sin(x) + 0.3))
    return sim


def _rand(rng, n, dtype):
    v = rng.standard_normal(n)
    if np.iscomplexobj(np.zeros(1, dtype)):
        v = v + 1j * rng.standard_normal(n)
    return v.astype(dtype)


def _identities(fm, X0, rng):
    V, W = _rand(rng, fm.n, fm.dtype), _rand(rng, fm.n, fm.dtype)
    JV, JTW = fm.jvp(X0, V), fm.vjp(X0, W)
    lhs, rhs = fm.pair(JV, W), fm.pair(V, JTW)
    dot = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    P0 = fm(X0)
    errs = [np.linalg.norm(fm(X0 + e * V) - P0 - e * JV) for e in (1e-3, 5e-4, 2.5e-4)]
    orders = [float(np.log2(errs[i] / errs[i + 1])) for i in range(2)]
    return dot, orders, W, JTW


@pytest.mark.parametrize("scheme,startup", CASES, ids=[f"{s}-{st}" for s, st in CASES])
def test_flowmap_identities_every_scheme(scheme, startup, rng):
    sim = _sim(scheme, startup)
    X0 = sim.solver.X.copy()
    n = 9
    fm = sim.flowmap(n * sim.dt)
    ref = _sim(scheme, startup)
    ref.run(steps=n)
    assert np.array_equal(fm(X0), ref.solver.X), "the flow map is not the solver's run"
    dot, orders, W, JTW = _identities(fm, X0, rng)
    assert dot < DOT, f"dot-product identity off by {dot:.2e}"
    assert min(orders) > TAYLOR, f"Taylor orders {orders}"
    rev = FlowMap(sim.solver, n, checkpoints=2)
    assert np.array_equal(rev.vjp(X0, W), JTW), "Revolve differs from full storage"


@pytest.mark.parametrize("scheme,startup", [("RK443", "lower_order"), ("SBDF2", "lower_order"),
                                            ("CNAB2", "rk443"), ("RKSMR", "lower_order")])
def test_flowmap_time_dependent_from_t0(scheme, startup, rng):
    sim = _sim(scheme, startup, forced=True)
    sim.run(steps=3)                                  # t0 = 0.03, not a multiple of anything else
    X0, t0 = sim.solver.X.copy(), sim.t
    fm = sim.flowmap(7 * sim.dt, t0=t0)
    dot, orders, W, JTW = _identities(fm, X0, rng)
    assert dot < DOT and min(orders) > TAYLOR, (dot, orders)
    # the map's reference is a run from X0 at t0 with the scheme's own start-up (fresh multistep
    # history, iteration 0), not the continuation of the simulation, whose history holds the
    # steps before t0: for SBDF2/CNAB2 the two differ by ~1e-3. The test used to compare with
    # sim.run(steps=7) and passed only because fm(X0) left the simulation's history cleared;
    # since the state audit sim.flowmap restores it (critic, fix of master f57617f1)
    ref = _sim(scheme, startup, forced=True)
    s = ref.solver
    s.X = X0.copy()
    s.t, s.iteration, s._t_origin, s._it_origin = t0, 0, t0, 0
    s._reset_history()
    ref.run(steps=7)
    assert np.array_equal(fm(X0), s.X), "the flow map is not the solver's run from X0 at t0"
    # and the simulation itself was left as it was (restored), so it continues its own run
    before = sim.solver.X.copy()
    assert np.array_equal(before, X0) and sim.t == t0


@pytest.mark.parametrize("scheme", ["RK222", "SBDF2"])
def test_flowmap_polynomial_vs_dense_transpose(scheme, rng):
    sim = _sim(scheme, "lower_order")
    X0 = sim.solver.X.copy()
    W = _rand(rng, sim.solver.X.size, sim.solver.X.dtype)
    a = FlowMap(sim.solver, 4).vjp(X0, W)
    b = FlowMap(sim.solver, 4, f_transpose="dense").vjp(X0, W)
    c = FlowMap(sim.solver, 4, solves="superlu").vjp(X0, W)
    scale = np.abs(a).max()
    assert np.abs(a - b).max() < DOT * scale and np.abs(a - c).max() < DOT * scale
