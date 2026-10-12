"""Adjoint gradients against closed forms (coverage hole 4: analytic adjoint gradient).

Equation (RealFourier on (0, 2 pi), one unknown u, parameters carried as frozen fields):

    c dt(u) + u = D u_xx - a u          (LHS form "c*dt(u) + u", bare unknown: the case)
    c (dt(u) + u) = D u_xx - a u        (LHS form "c*(dt(u) + u)", nested bare unknown)

Mode k of u decays as e^{sigma_k t}, sigma_k = -(g + a + D k^2) / c with g = 1 (first form) or
g = c (second form). With the L^2 functional

    J(u0) = 1/2 ||u(T)||^2 = 1/2 sum_j G_j X_j(T)^2,   G_j = 2 pi (mean) or pi (cos k x, sin k x),

(Fourier norms in closed form, not taken from Conatus) the gradients are, mode by mode,

    bilinear gradient  dJ/dX0_j = G_j e^{2 sigma_k T} X0_j          (FlowMap.vjp, pairing sum a b, D8)
    physical gradient  G^{-1} dJ/dX0 = e^{2 sigma_k T} X0_j         (FlowMap.vjp_physical: the adjoint
                                                                    for <a, b>_G = a^T G b is
                                                                    Phi^dagger = G^{-1} Phi^T G)
    dJ/da = sum_j G_j (-T / c) e^{2 sigma_k T} X0_j^2,  dJ/dD = sum_j G_j (-k^2 T / c) e^{2 sigma_k T} X0_j^2.

FlowMap has no parameter-gradient API (it differentiates X0 -> X_N only). dJ/da and dJ/dD are
obtained with the standard augmented state: a and D become unknowns with dt(a) = dt(D) = 0, and
the terms -a u, D u_xx (products of unknowns) move to the explicit RHS, whose transpose FlowMap
builds matrix-free (LinearizedRHS). The mean coefficient of the frozen field *is* the parameter
(coefficient of the constant mode = its value, checked below), so dJ/da is the vjp entry of that
coefficient. The reference remains the closed form; Conatus is never compared with itself.

Discretization: RK443 / SBDF3 at dt = 4e-3 over T = 0.4; the discrete flow map differs from
e^{sigma T} by O(dt^3). Measured relative gradient errors (dt = 1e-2 / 4e-3 / 2e-3): RK443
4.9e-8 / 3.1e-9 / 3.9e-10, SBDF3 5.6e-7 / 3.7e-8 / 4.7e-9, dJ/dD 4.2e-7 / 2.6e-8 / 3.3e-9 (ratio 8:
third order). Tolerance 3e-7; an-type error changes 2 sigma T by ~0.2 (a 20 % gradient error).

Mutations (script outside the repository): with reintroduced (bare unknown in M) the
rates become -(g + a + D k^2)/(c + 1) or -(...)/(2c): every test fails. (eigenvalue lost in a
nested product) lives in the EVP split, which an IVP flow map never calls: not reachable here; the
nested IVP form c*(dt(u) + u) is its IVP analogue and is covered by the mutation.
"""
import numpy as np
import pytest

import conatus.expert as C
from conatus.adjoint import FlowMap
from conatus.discretize import discretize
from conatus.runtime import cuda

N = 8                              # RealFourier size: modes k = 0..3
A, DIFF, CVAL = 0.4, 0.3, 2.0      # a, D, c
DT, NSTEPS = 4e-3, 100             # T = 0.4
TOL = 3e-7                         # relative; measured <= 9e-8 (SBDF3), third order in dt
# Parameter gradients with SBDF3: the frozen-field products are on the explicit RHS, extrapolated at
# order 3; measured relative errors dJ/da 1.3e-7 / 2.4e-7, dJ/dD 3.9e-7 / 6.1e-7 (flat / nested form)
# at dt = 4e-3, and 4.8e-8 / 7.7e-8 at dt = 2e-3 (ratio 8: truncation, identical on CPU and GPU,
# run). An-type error is a ~20 % error.
TOL_PARAM = {"RK443": TOL, "SBDF3": 1.5e-6}
T_END = DT * NSTEPS


def _problem(form: str, frozen: bool):
    """IVP for u (and frozen a, D fields when ``frozen``); returns (plan, u, a, d)."""
    cs = C.CartesianCoordinates("x")
    b = C.RealFourier(cs["x"], N, (0, 2 * np.pi), dealias=3 / 2)
    dom = C.Domain(cs, [b])
    u = C.Field("u", dom)
    c = C.Parameter("c", CVAL)
    lhs = c * C.dt(u) + u if form == "c*dt(u) + u" else c * (C.dt(u) + u)
    if not frozen:
        # a, D as Parameters in L (kept as separate terms: a*u + u does not fold)
        pb = C.IVP([u])
        pb.add_equation((lhs + C.Parameter("a", A) * u - C.Parameter("D", DIFF) * C.lap(u), 0))
        return discretize(pb), u, None, None
    a, d = C.Field("a", dom), C.Field("d", dom)
    pb = C.IVP([u, a, d])
    pb.add_equation((lhs, d * C.lap(u) - a * u))       # products of unknowns: explicit RHS
    pb.add_equation((C.dt(a), 0))
    pb.add_equation((C.dt(d), 0))
    return discretize(pb), u, a, d


def _g(form: str) -> float:
    return 1.0 if form == "c*dt(u) + u" else CVAL


def _modes():
    """Wavenumber and closed-form L^2 weight of each per-variable RealFourier coefficient slot
    (layout [mean, (unused sin 0), cos 1, sin 1, ...]); gather_state drops the unused slot."""
    k = np.arange(N) // 2
    w = np.where(k == 0, 2 * np.pi, np.pi)
    return k, w


def _x0(rng):
    """Random u coefficients (all resolved modes)."""
    c = rng.standard_normal(N)
    c[1] = 0.0                     # sin(0 x) slot: not part of the state
    return c


def _solver(plan, scheme, backend="cpu"):
    kw = dict(startup="rk443") if scheme.startswith("SBDF") else {}
    return C.IVPSolver(plan, scheme, DT, backend=backend, **kw)


FORMS = ["c*dt(u) + u", "c*(dt(u) + u)"]


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("scheme", ["RK443", "SBDF3"])
def test_gradient_wrt_initial_condition_is_exact(form, scheme):
    """dJ/dX0 = G e^{2 sigma_k T} X0 (bilinear) and e^{2 sigma_k T} X0 (physical)."""
    plan, u, _, _ = _problem(form, frozen=False)
    s = _solver(plan, scheme)
    sysm = s.system
    k, w = _modes()
    rng = np.random.default_rng(4)
    cu = _x0(rng)
    X0 = sysm.gather_state([cu])
    G = sysm.gather_state([w])                        # diagonal weight on the state vector
    fm = FlowMap(s, NSTEPS)
    XN = fm(X0)
    sig = -(_g(form) + A + DIFF * k ** 2) / CVAL
    grad = sysm.scatter(fm.vjp(X0, G * XN))[0]         # dJ/dX0 = Phi^T G Phi X0
    ref = w * np.exp(2 * sig * T_END) * cu
    ref[1] = 0.0
    assert np.abs(grad - ref).max() < TOL * np.abs(ref).max()
    phys = sysm.scatter(fm.vjp_physical(X0, XN, np.diag(G)))[0]
    ref_p = np.exp(2 * sig * T_END) * cu
    ref_p[1] = 0.0
    assert np.abs(phys - ref_p).max() < TOL * np.abs(ref_p).max()


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("scheme", ["RK443", "SBDF3"])
def test_gradient_wrt_parameters_is_exact(form, scheme):
    """dJ/da and dJ/dD through the frozen-field augmented state, against the closed forms."""
    tol = TOL_PARAM[scheme]
    plan, u, a, d = _problem(form, frozen=True)
    s = _solver(plan, scheme)
    sysm = s.system
    k, w = _modes()
    rng = np.random.default_rng(5)
    cu = _x0(rng)
    s.set_state({a: lambda X: A + 0 * X, d: lambda X: DIFF + 0 * X})
    ca, cd = s.coeffs("a"), s.coeffs("d")
    assert abs(ca[0] - A) < 1e-14 and abs(cd[0] - DIFF) < 1e-14   # mean coefficient = value
    X0 = sysm.gather_state([cu, ca, cd])
    zero = np.zeros(N)
    G = sysm.gather_state([w, zero, zero])            # J measures u only
    fm = FlowMap(s, NSTEPS)
    XN = fm(X0)
    gu, ga, gd = sysm.scatter(fm.vjp(X0, G * XN))
    sig = -(_g(form) + A + DIFF * k ** 2) / CVAL
    e2 = w * np.exp(2 * sig * T_END) * cu ** 2
    dJda = np.sum(-T_END / CVAL * e2)
    dJdD = np.sum(-k ** 2 * T_END / CVAL * e2)
    assert abs(ga[0] - dJda) < tol * abs(dJda)
    assert abs(gd[0] - dJdD) < tol * abs(dJdD)
    ref_u = w * np.exp(2 * sig * T_END) * cu
    assert np.abs(gu - ref_u).max() < tol * np.abs(ref_u).max()


# ------------------------------------------------------------------------------------------------
# GPU (FlowMap on the CUDA backend: same factors as the forward map; vjp_physical is CPU-only)
# ------------------------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.skipif(not cuda.available(), reason="no CUDA device on this node")
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("scheme", ["RK443", "SBDF3"])
def test_gpu_gradients_are_exact(form, scheme):
    """Same closed forms on the GPU: dJ/dX0 (u only) and dJ/da, dJ/dD (augmented state)."""
    import cupy as cp
    k, w = _modes()
    sig = -(_g(form) + A + DIFF * k ** 2) / CVAL
    rng = np.random.default_rng(6)
    cu = _x0(rng)
    # initial-condition gradient
    plan, u, _, _ = _problem(form, frozen=False)
    s = _solver(plan, scheme, backend="cuda")
    sysc = _solver(plan, scheme).system                # host layout (gather/scatter on NumPy)
    X0, G = sysc.gather_state([cu]), sysc.gather_state([w])
    fm = FlowMap(s, NSTEPS)
    XN = fm(cp.asarray(X0))
    grad = sysc.scatter(cp.asnumpy(fm.vjp(cp.asarray(X0), cp.asarray(G) * XN)))[0]
    ref = w * np.exp(2 * sig * T_END) * cu
    ref[1] = 0.0
    assert np.abs(grad - ref).max() < TOL * np.abs(ref).max()
    # parameter gradients
    plan, u, a, d = _problem(form, frozen=True)
    s = _solver(plan, scheme, backend="cuda")
    sysc = _solver(plan, scheme).system
    ca, cd = np.zeros(N), np.zeros(N)
    ca[0], cd[0] = A, DIFF
    X0 = sysc.gather_state([cu, ca, cd])
    G = sysc.gather_state([w, np.zeros(N), np.zeros(N)])
    fm = FlowMap(s, NSTEPS)
    XN = fm(cp.asarray(X0))
    gu, ga, gd = sysc.scatter(cp.asnumpy(fm.vjp(cp.asarray(X0), cp.asarray(G) * XN)))
    e2 = w * np.exp(2 * sig * T_END) * cu ** 2
    tol = TOL_PARAM[scheme]
    assert abs(ga[0] - np.sum(-T_END / CVAL * e2)) < tol * abs(np.sum(-T_END / CVAL * e2))
    assert abs(gd[0] - np.sum(-k ** 2 * T_END / CVAL * e2)) < tol * abs(np.sum(-k ** 2 * T_END / CVAL * e2))
