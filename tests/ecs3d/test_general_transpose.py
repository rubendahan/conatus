"""Linearized RHS of the poloidal–toroidal channel (general form: derivatives of products, the
same representation as the fused forward F): tangent = directional derivative of the forward F,
transpose = exact bilinear transpose (dot test), FlowMap Taylor test.."""
import numpy as np
from rounding_bounds import adjoint_dot_check

from ch3d_cases import channel_3d_pt
from conatus.adjoint import FlowMap
from conatus.adjoint.rhs_transpose import LinearizedRHS
from conatus.timestep import IVPSolver


def _solver(f_backend="fused", couette=False):
    kw = dict(pressure_gradient=0.0, wall_velocity=(-1.0, 1.0)) if couette else \
        dict(pressure_gradient=0.01)
    plan, _ = channel_3d_pt(6, 11, 6, 150.0, 2 * np.pi, np.pi, **kw)
    return IVPSolver(plan, "RK443", 0.01, f_backend=f_backend)


def _state(s, rng, amp=0.1):
    X = amp * rng.standard_normal(s.system.n)
    return X


def test_general_form_tangent_and_transpose():
    rng = np.random.default_rng(0)
    s = _solver()
    lin = LinearizedRHS(s)
    assert any(g is not None for g in lin.generals)
    X, V, W = _state(s, rng), _state(s, rng), _state(s, rng)
    JV = lin.apply(X, V)
    # quadratic RHS: central difference is exact up to rounding
    h = 1.0
    fd = (s.F(X + h * V, 0.0) - s.F(X - h * V, 0.0)) / (2 * h)
    assert np.linalg.norm(JV - fd) < 1e-12 * np.linalg.norm(fd)
    lhs, rhs = float(W @ JV), float(lin.transpose(X, W) @ V)
    assert abs(lhs - rhs) < 1e-12 * max(abs(lhs), 1e-300) * 10 + 1e-14 * np.linalg.norm(W) * np.linalg.norm(JV)


def test_flowmap_dot_and_taylor_on_pt_channel():
    rng = np.random.default_rng(1)
    s = _solver()
    fm = FlowMap(s, 3)
    X, V, W = _state(s, rng, 0.05), _state(s, rng), _state(s, rng)
    # rounding-derived bound (tests/rounding_bounds.py rule; was a fixed 1e-11 |lhs|, met at
    # 4.9e-11 / 2.8e-10 on genoa: run); cap as test_matrix_free
    chk = adjoint_dot_check(lambda v: fm.jvp(X, v), lambda w: fm.vjp(X, w), V, W, _state(s, rng),
                            _state(s, rng), cap=1e-8, dims=(fm.n, fm.n))
    JV = chk.JV
    P0 = fm(X)
    errs = [np.linalg.norm(fm(X + e * V) - P0 - e * JV) for e in (1e-2, 5e-3)]
    assert 3.5 < errs[0] / errs[1] < 4.5          # second order: the tangent is exact
