"""Exact discrete tangent/adjoint of the IMEX flow map: dot-product and Taylor tests."""
import numpy as np
import pytest

from conatus.adjoint import FlowMap, revolve_schedule
from conatus.timestep import IVPSolver
from ivp_cases import burgers_1d, ks_1d, rayleigh_benard_2d


def _state(solver, fun):
    solver.set_state({solver.system.variables[0]: fun})
    return solver.X.copy()


def _rand(rng, fm):
    v = rng.standard_normal(fm.n)
    if np.iscomplexobj(np.zeros(1, fm.dtype)):
        v = v + 1j * rng.standard_normal(fm.n)
    # stay in the admissible space: project through one solve-free map (zero tau rows are fine)
    return v.astype(fm.dtype)


def _dot_and_taylor(rng, fm, X0, eps0=1e-3):
    V, W = _rand(rng, fm), _rand(rng, fm)
    JV = fm.jvp(X0, V)
    JTW = fm.vjp(X0, W)
    lhs, rhs = np.sum(JV * W), np.sum(V * JTW)
    dot_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    P0 = fm(X0)
    errs = []
    for eps in (eps0, eps0 / 2, eps0 / 4):
        errs.append(np.linalg.norm(fm(X0 + eps * V) - P0 - eps * JV))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    return dot_err, orders


@pytest.mark.parametrize("scheme", ["RK222", "RK443"])
def test_burgers_rk(scheme, rng):
    plan, u, _ = burgers_1d(N=16, nu=0.1)
    s = IVPSolver(plan, scheme, 1e-2)
    X0 = _state(s, lambda x: 0.5 * np.sin(x) + 0.2 * np.cos(2 * x))
    fm = FlowMap(s, 10)
    dot_err, orders = _dot_and_taylor(rng, fm, X0)
    assert dot_err < 1e-11
    assert min(orders) > 1.9


def test_burgers_complex(rng):
    plan, u, _ = burgers_1d(N=16, nu=0.1, basis="complex")
    s = IVPSolver(plan, "RK443", 1e-2)
    X0 = _state(s, lambda x: 0.5 * np.sin(x) + 0j)
    fm = FlowMap(s, 5)
    dot_err, orders = _dot_and_taylor(rng, fm, X0)
    assert dot_err < 1e-11 and min(orders) > 1.9


@pytest.mark.parametrize("scheme,startup", [("SBDF2", "lower_order"), ("SBDF3", "rk443")])
def test_ks_sbdf(scheme, startup, rng):
    plan, u = ks_1d(N=32, L=22.0)
    s = IVPSolver(plan, scheme, 1e-2, startup=startup)
    X0 = _state(s, lambda x: 0.3 * np.cos(2 * np.pi * x / 22) + 0.1 * np.sin(4 * np.pi * x / 22))
    fm = FlowMap(s, 12)
    dot_err, orders = _dot_and_taylor(rng, fm, X0, eps0=1e-2)
    assert dot_err < 1e-10
    assert min(orders) > 1.9


def _central_difference_error(fm, X0, V, JV, h):
    """|(Phi(X0 + h V) - Phi(X0 - h V)) / 2h - JV| / |JV|: an independent derivative of the map."""
    fd = (fm(X0 + h * V) - fm(X0 - h * V)) / (2 * h)
    return float(np.linalg.norm(fd - JV) / np.linalg.norm(JV))


def test_jvp_matches_central_difference(rng):
    """Tangent is the derivative of the *same* map IVPSolver computes (not of the PDE): jvp
    against a central difference of the map (the former test only compared fm(X0) with the end
    of its trajectory and never called jvp: axis 11)."""
    plan, u, _ = burgers_1d(N=16)
    s = IVPSolver(plan, "RK443", 1e-2)
    X0 = _state(s, lambda x: np.sin(x))
    fm = FlowMap(s, 3)
    assert np.array_equal(fm(X0), fm.trajectory(X0)[-1])
    V = _rand(rng, fm)
    JV = fm.jvp(X0, V)
    assert np.linalg.norm(JV) > 0
    err = _central_difference_error(fm, X0, V, JV, 1e-4)
    assert err < 1e-7, err
    # mutation check: a tangent off by 1e-4 (relative) is detected by the same oracle
    assert _central_difference_error(fm, X0, V, JV * (1 + 1e-4), 1e-4) > 1e-5


@pytest.mark.slow
def test_rayleigh_benard_rk222(rng):
    plan, f = rayleigh_benard_2d(8, 8, 2000.0)
    s = IVPSolver(plan, "RK222", 1e-3)
    s.set_state({f["T"]: lambda x, z: 0.1 * np.cos(np.pi * x) * np.sin(np.pi * (z + 1) / 2)
                 + 0.5 * (1 - z)})
    X0 = s.X.copy()
    fm = FlowMap(s, 2)
    dot_err, orders = _dot_and_taylor(rng, fm, X0)
    assert dot_err < 1e-9 and min(orders) > 1.8


def test_revolve_equals_full_storage(rng):
    plan, u, _ = burgers_1d(N=16)
    s = IVPSolver(plan, "RK443", 1e-2)
    X0 = _state(s, lambda x: np.sin(x))
    W = _rand(rng, FlowMap(s, 1))
    full = FlowMap(s, 13).vjp(X0, W)
    rev = FlowMap(s, 13, checkpoints=2)
    out = rev.vjp(X0, W)
    assert np.array_equal(out, full)              # same arithmetic, bit-for-bit
    assert rev.n_recompute > 0
    assert all(1 <= revolve_schedule(t, c) < t for t in range(2, 40) for c in range(1, 5))


def test_physical_adjoint_and_refusal(rng):
    plan, u, _ = burgers_1d(N=16)
    s = IVPSolver(plan, "RK222", 1e-2)
    X0 = _state(s, lambda x: np.sin(x))
    fm = FlowMap(s, 4)
    V, W = _rand(rng, fm), _rand(rng, fm)
    G = np.diag(1.0 + np.arange(fm.n))
    lhs = V @ G @ fm.jvp(X0, W)
    rhs = (fm.vjp_physical(X0, V, G)) @ G @ W
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)
    Gs = G.copy()
    Gs[0, 0] = 0.0
    with pytest.raises(ValueError):
        fm.vjp_physical(X0, V, Gs)


def test_refusals():
    plan, u, _ = burgers_1d(N=16)
    s = IVPSolver(plan, "RK443", 1e-2)
    fm = FlowMap(s, 2, f_transpose="dense", max_dense=4)
    with pytest.raises(ValueError):
        fm.vjp(np.zeros(fm.n), np.ones(fm.n))
    with pytest.raises(ValueError):
        FlowMap(s, 2, f_transpose="auto")
    with pytest.raises(ValueError):
        FlowMap(s, 2, solves="lu")

    assert "Revolve" in FlowMap(IVPSolver(plan, "RK443", 1e-2), 4, checkpoints=1).explain()
