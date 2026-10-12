"""Fused polynomial RHS (f_backend="fused") vs the pseudospectral evaluator (oracle)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ivp"))
from ivp_cases import burgers_1d, ks_1d, rayleigh_benard_2d  # noqa: E402

from conatus.ir import expr as E  # noqa: E402
from conatus.timestep import IVPSolver  # noqa: E402
from conatus.timestep.fused import expand_polynomial  # noqa: E402
from conatus.timestep.spectral import SpectralError  # noqa: E402

CASES = {
    "rb2d": lambda: rayleigh_benard_2d(32, 16, Ra=1e5, Lx=4.0)[0],
    "rb2d_sym": lambda: rayleigh_benard_2d(32, 16, Ra=1e5, Lx=4.0, symmetric=True)[0],
    "burgers_real": lambda: burgers_1d(32, basis="real")[0],
    "burgers_complex": lambda: burgers_1d(32, basis="complex")[0],
    "ks": lambda: ks_1d(64)[0],
}


def _random_state(s, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(s.system.n) * 0.1
    if np.iscomplexobj(s.X):
        X = X + 1j * rng.standard_normal(s.system.n) * 0.1
    return X.astype(s.X.dtype)


@pytest.mark.parametrize("name", sorted(CASES))
def test_fused_matches_evaluator_cpu(name):
    plan = CASES[name]()
    a = IVPSolver(plan, "RK443", 1e-4)
    b = IVPSolver(plan, "RK443", 1e-4, f_backend="fused")
    X = _random_state(a)
    Fa, Fb = a.F(X, 0.0), b.F(X, 0.0)
    assert np.abs(Fa - Fb).max() <= 1e-13 * max(1.0, np.abs(Fa).max())
    assert "FusedRHS" in b._fused.explain()


def test_fused_trajectory_cpu():
    plan, f = rayleigh_benard_2d(32, 16, Ra=1e5, Lx=4.0)
    ss = [IVPSolver(plan, "RK443", 1e-4, f_backend=fb) for fb in ("evaluator", "fused")]
    for s in ss:
        s.set_state({f["T"]: lambda x, z: 1e-2 * np.cos(np.pi * z) * np.cos(2 * np.pi * x / 4.0)})
        s.run(20)
    assert np.abs(ss[0].X - ss[1].X).max() <= 1e-12 * np.abs(ss[0].X).max()


def test_derivative_of_product_refused():
    plan, f = rayleigh_benard_2d(16, 8, Ra=1e5, Lx=4.0)
    x, z = plan.domain.coordsys.coords
    T = f["T"]
    with pytest.raises(SpectralError, match="derivative of a product"):
        expand_polynomial(E.diff(T * T, x), [x, z], {})


def test_unknown_f_backend_refused():
    plan = CASES["ks"]()
    with pytest.raises(ValueError, match="f_backend"):
        IVPSolver(plan, "RK443", 1e-4, f_backend="auto")


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(CASES))
def test_fused_matches_evaluator_gpu(name):
    cp = pytest.importorskip("cupy")
    plan = CASES[name]()
    a = IVPSolver(plan, "RK443", 1e-4)
    b = IVPSolver(plan, "RK443", 1e-4, backend="cuda", f_backend="fused")
    X = _random_state(a)
    Fa, Fb = a.F(X, 0.0), cp.asnumpy(b.F(cp.asarray(X), 0.0))
    assert np.abs(Fa - Fb).max() <= 1e-13 * max(1.0, np.abs(Fa).max())


@pytest.mark.gpu
def test_fused_graph_gpu():
    cp = pytest.importorskip("cupy")
    plan, f = rayleigh_benard_2d(64, 32, Ra=1e5, Lx=4.0)
    init = {f["T"]: lambda x, z: 1e-2 * np.cos(np.pi * z) * np.cos(2 * np.pi * x / 4.0)}
    a = IVPSolver(plan, "RK443", 1e-4)
    b = IVPSolver(plan, "RK443", 1e-4, backend="cuda", f_backend="fused")
    for s in (a, b):
        s.set_state(init)
    a.run(10)
    b.run_graph(10)
    assert np.abs(a.X - cp.asnumpy(b.X)).max() <= 1e-11 * np.abs(a.X).max()


def test_fused_parity_reduction_is_used_and_exact_cpu():
    """In the x -> -x subspace F is evaluated parity-reduced (half grid) and declared so;
    same F as the full-grid fused path and as the evaluator."""
    plan = CASES["rb2d_sym"]()
    ev = IVPSolver(plan, "RK443", 1e-4)
    red = IVPSolver(plan, "RK443", 1e-4, f_backend="fused")
    assert red._fused.reflection is not None and not red._fused_note
    assert "parity-reduced along x" in red.explain(full=True)
    X = _random_state(ev, seed=3)
    Fe, Fr = ev.F(X, 0.0), red.F(X, 0.0)
    assert np.abs(Fe - Fr).max() <= 1e-13 * np.abs(Fe).max()


@pytest.mark.gpu
def test_fused_parity_reduced_graph_gpu():
    cp = pytest.importorskip("cupy")
    from conatus.runtime import cuda
    if not cuda.available():
        pytest.skip("no CUDA device on this node")
    plan, f = rayleigh_benard_2d(64, 32, Ra=1e5, Lx=4.0, symmetric=True)
    init = {f["T"]: lambda x, z: 1e-2 * np.cos(np.pi * z) * np.cos(2 * np.pi * x / 4.0)}
    a = IVPSolver(plan, "SBDF2", 1e-4)
    b = IVPSolver(plan, "SBDF2", 1e-4, backend="cuda", f_backend="fused")
    assert b._fused.reflection is not None
    for s in (a, b):
        s.set_state(init)
    a.run(12)
    b.run_graph(12)
    assert np.abs(a.X - cp.asnumpy(b.X)).max() <= 1e-11 * np.abs(a.X).max()
