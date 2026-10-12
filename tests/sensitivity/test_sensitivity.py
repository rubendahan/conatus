"""Sensitivity toolkit: wavemaker, base-flow and forcing sensitivity, parameter sensitivity.

Closed-form oracle: the linear Ginzburg–Landau equation with μ(x) = μ_c + μ₂x²/2 on the whole line
(Chomaz 2005; Bagheri et al. 2009 parameters ν = 2 + 0.4i, γ = 1 − i, μ₂ = −0.01), truncated to
[−60, 60] (tail 1e-86). Finite-difference oracle: the exact eigenvalue of the perturbed discrete
problem. The Orr–Sommerfeld (plane Poiseuille) convergence study is in the slow tests.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy

from conatus.errors import InputTypeError, OptionError
from conatus.nonmodal import Y, parallel_shear_flow
from conatus.sensitivity import (base_flow_sensitivity, eigenvalue_sensitivity, feedback_system,
                                 ginzburg_landau_exact, linear_ginzburg_landau, mode_pair,
                                 parameter_sensitivity, steady_forcing_sensitivity,
                                 structural_sensitivity)

NU, GAMMA, MUC, MU2 = 2 + 0.4j, 1 - 1j, 0.34, -0.01
BOUNDS = (-60.0, 60.0)
N = 140
TOL_INF = 1e-10
ADJ_BCS = [("q", 0, "left"), ("q", 0, "right")]


def gl(mu):
    return linear_ginzburg_landau(nu=NU, gamma=GAMMA, mu=mu, bounds=BOUNDS)


MU = MUC + MU2 * Y ** 2 / 2


@pytest.fixture(scope="module", params=[0, pytest.param(1, marks=pytest.mark.slow)])
def exact(request):
    return ginzburg_landau_exact(nu=NU, gamma=GAMMA, mu_c=MUC, mu2=MU2, n=request.param)


@pytest.fixture(scope="module")
def pairs(exact):
    return {k: mode_pair(gl(MU), N=N, target=exact.eigenvalue, adjoint=k, infinite_tol=TOL_INF,
                         adjoint_bcs=ADJ_BCS) for k in ("discrete", "continuous")}


def test_truncation_is_negligible(exact):
    assert exact.truncation(BOUNDS) < 1e-60


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_eigenvalue_and_residuals(exact, pairs, kind):
    p = pairs[kind]
    assert abs(p.sigma - exact.eigenvalue) < 1e-12
    assert p.residual_right < 1e-12 and p.residual_left < 1e-12 and p.residual_adjoint < 1e-12
    assert abs(p.adjoint_eigenvalue - np.conj(exact.eigenvalue)) < 1e-10
    assert abs(p.inner(p.adjoint, p.direct) - 1) < 1e-12
    assert abs(p.inner(p.direct, p.direct) - 1) < 1e-12
    assert "adjoint" in p.explain()


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_wavemaker_matches_chomaz_closed_form(exact, pairs, kind):
    """S(x) = |e^{−χ²x²}| |H_n(χx)|² |χ| / (√π 2ⁿ n!) (Giannetti–Luchini wavemaker of GL)."""
    st = structural_sensitivity(pairs[kind])
    x = np.linspace(-30, 30, 601)
    S, Se = st.wavemaker(x), exact.wavemaker(x)
    assert np.max(np.abs(S - Se)) / Se.max() < 1e-9
    y0, smax = st.maximum(points=601)
    assert abs(smax - exact.wavemaker(np.array([y0]))[0]) / Se.max() < 1e-9
    assert smax >= Se.max() * (1 - 1e-12)     # the refined maximum is above every sample
    if exact.n == 0:                          # maximum at the origin: S(0) = |χ|/√π
        assert abs(y0) < 1e-5 and abs(smax - abs(exact.chi) / np.sqrt(np.pi)) < 1e-10
    T = st.tensor(x)
    assert T.shape == (x.size, 1, 1)
    assert np.max(np.abs(np.abs(T[:, 0, 0]) - Se)) / Se.max() < 1e-9


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_gradient_wrt_mu_matches_closed_form(exact, pairs, kind):
    """δλ/δμ(x) = ψ̄φ/⟨ψ,φ⟩ (complex density) and dλ/dμ₂, dλ/dμ_c in closed form."""
    bfs = base_flow_sensitivity(gl, MU, N=N, target=exact.eigenvalue, adjoint=kind,
                                infinite_tol=TOL_INF, adjoint_bcs=ADJ_BCS)
    x = np.linspace(-30, 30, 601)
    G, Ge = bfs.gradient(x), exact.gradient_mu(x)
    assert np.max(np.abs(G - Ge)) / np.abs(Ge).max() < 1e-9
    assert bfs.boundary == {}                 # μ enters without derivatives: no boundary term
    d = exact.d_eigenvalue()
    assert abs(bfs.predict(Y ** 2 / 2) - d["mu2"]) < 1e-9 * abs(d["mu2"])
    assert abs(bfs.predict(1) - 1) < 1e-12
    # discrete (pencil) derivative: same number
    assert abs(bfs.discrete_shift(Y ** 2 / 2) - d["mu2"]) < 1e-9 * abs(d["mu2"])
    assert "Marquet" in bfs.explain()


def test_parameter_sensitivity_nu(exact):
    """dλ/dν = −ν/(2γ) for every n; Richardson FD of the descriptor, rel_h declared."""
    r = parameter_sensitivity(lambda v: linear_ginzburg_landau(nu=v + 1j * NU.imag, gamma=GAMMA,
                                                               mu=MU, bounds=BOUNDS),
                              NU.real, parameter="Re(nu)", N=N, target=exact.eigenvalue,
                              infinite_tol=TOL_INF, rel_h=1e-3)
    # the factory varies Re ν only: dλ/d(Re ν) = dλ/dν (holomorphic in ν)
    assert abs(r.value - exact.d_eigenvalue()["nu"]) < 1e-8
    assert r.check < 1e-8 and "Richardson" in r.explain()


def test_feedback_shift_against_finite_differences(pairs):
    """Localized feedback g(y) q, g = exp(−(y−3)²/16): continuous formula, pencil formula and the
    central difference of the discrete eigenvalue agree."""
    p = pairs["discrete"]
    g = sympy.exp(-(Y - 3) ** 2 / 16)
    st = structural_sensitivity(p)
    ds, info = st.feedback_shift(g, {("q", "q"): 1.0})
    disc = p.shift_towards(feedback_system(p.system, g, {("q", "q"): 1.0}, epsilon=1.0))
    eps = 1e-3

    def sig(e):
        return mode_pair(feedback_system(p.system, g, {("q", "q"): 1.0}, epsilon=e), N=N,
                         target=p.sigma, adjoint="discrete", infinite_tol=TOL_INF).sigma
    fd = (8 * (sig(eps) - sig(-eps)) - (sig(2 * eps) - sig(-2 * eps))) / (12 * eps)
    assert abs(disc - fd) < 1e-9 * abs(fd)
    assert abs(ds - disc) < 1e-9 * abs(disc)
    assert info["g_degree"] >= 16


def test_unified_eigenvalue_sensitivity_dispatch(pairs):
    p = pairs["discrete"]
    d2 = gl(MU + Y ** 2 / 2).discretize(N)
    r = eigenvalue_sensitivity(p, d2.L - p.descriptor.L, parameter="mu2")
    assert abs(r.value - p.shift_towards(gl(MU + Y ** 2 / 2))) < 1e-14
    from conatus.solvers import solve_dense
    res = solve_dense(p.descriptor.L, p.descriptor.M, left=True)
    i = int(np.argmin(np.abs(res.values - p.sigma)))
    r2 = eigenvalue_sensitivity(res, i, p.descriptor.M, d2.L - p.descriptor.L, parameter="mu2")
    assert abs(r2.value - r.value) < 1e-9 * abs(r.value)
    with pytest.raises(InputTypeError):
        eigenvalue_sensitivity(3.0)


def test_declared_choices_are_mandatory():
    with pytest.raises(OptionError):
        mode_pair(gl(MU), N=40, target=0, adjoint="auto", infinite_tol=TOL_INF)
    with pytest.raises(OptionError):
        mode_pair(gl(MU), N=40, target=0, adjoint="continuous", infinite_tol=TOL_INF)
    with pytest.raises(OptionError):
        linear_ginzburg_landau(nu=1, gamma=-1, mu=0, bounds=(-1, 1))


# ---------------------------------------------------------------------- plane Poiseuille
RE = 1e4
ABCS = [(f, 0, s) for f in ("u", "v", "w") for s in ("left", "right")]
C_ORSZAG = 0.23752648882 + 0.00373967062j


def lns(U):
    return parallel_shear_flow(U, kx=1.0, kz=0.0, Re=RE, walls="no-slip")


@pytest.mark.slow                   # 2.7 s in the fast suite (qual_durations0)
def test_poiseuille_base_flow_and_forcing_sensitivity_small():
    """N = 48: pencil derivative = Richardson FD of the eigenvalue; the continuous-adjoint
    density agrees to the discretization error; forcing sensitivity closes (δF = 1 gives
    δU = Re(1 − y²)/2 in closed form)."""
    Nn = 48
    target = -1j * C_ORSZAG
    bfs = base_flow_sensitivity(lns, 1 - Y ** 2, N=Nn, target=target, adjoint="continuous",
                                infinite_tol=TOL_INF, adjoint_bcs=ABCS)
    dU = (1 - Y ** 2) ** 2 * sympy.sin(Y + sympy.Rational(3, 10))
    disc = bfs.discrete_shift(dU)
    h = 1e-3

    def sig(e):
        return mode_pair(lns(1 - Y ** 2 + e * dU), N=Nn, target=bfs.sigma, adjoint="discrete",
                         infinite_tol=TOL_INF).sigma
    fd = (8 * (sig(h) - sig(-h)) - (sig(2 * h) - sig(-2 * h))) / (12 * h)
    assert abs(fd - disc) < 1e-8 * abs(disc)
    assert abs(bfs.predict(dU) - disc) < 1e-5 * abs(disc)
    assert all(abs(B) < 1e-13 for B in bfs.boundary.values())     # v = 0 at the walls
    fs = steady_forcing_sensitivity(bfs, operator={2: 1 / RE}, bcs=[(0, "left"), (0, "right")],
                                    N=Nn)
    yy = np.linspace(-1, 1, 21)
    assert np.max(np.abs(fs.base_change(1)(yy) - RE * (1 - yy ** 2) / 2)) < 1e-10 * RE
    assert abs(fs.predict(1) - bfs.predict(RE * (1 - Y ** 2) / 2)) < 1e-12 * abs(fs.predict(1))
    F2 = sympy.cos(2 * Y)
    assert abs(fs.predict(F2) - bfs.predict(fs.base_change(F2))) < 1e-10 * abs(fs.predict(F2))
    assert "Steady-forcing" in fs.explain()


@pytest.mark.slow
def test_poiseuille_density_converges_spectrally():
    """‖predict(δU) − discrete_shift(δU)‖ decreases spectrally with N (continuous adjoint)."""
    target = -1j * C_ORSZAG
    dU = (1 - Y ** 2) ** 2 * sympy.sin(Y + sympy.Rational(3, 10))
    errs = []
    for Nn in (48, 64, 80):
        bfs = base_flow_sensitivity(lns, 1 - Y ** 2, N=Nn, target=target, adjoint="continuous",
                                    infinite_tol=TOL_INF, adjoint_bcs=ABCS)
        d = bfs.discrete_shift(dU)
        errs.append(abs(bfs.predict(dU) - d) / abs(d))
    assert errs[2] < 1e-10 and errs[1] < errs[0] and errs[2] < errs[1]


def test_api_exports_both_j3_toolkits_without_clash():
    """conatus.expert flattens conatus.sensitivity and conatus.datadriven; none of their names is
    shadowed by (or shadows) an earlier export."""
    import conatus.expert as api
    import conatus.datadriven as dd
    import conatus.sensitivity as sens
    for mod in (sens, dd):
        for n in mod.__all__:
            assert getattr(api, n) is getattr(mod, n), n
            assert n not in api.SHADOWED, n
