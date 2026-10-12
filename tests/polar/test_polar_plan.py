"""Polar problems through the standard API ("Polar plans").

EVP, LBVP and IVP on Domain(PolarCoordinates, [Fourier(phi), DiskBasis | AnnulusBasis]) go through
``discretize`` -> ``PolarPlan`` (groups m), ``solvers.eigenvalues``, ``plan.solve`` and ``IVPSolver``.
References: zeros of J_m (scipy.special.jn_zeros; Abramowitz & Stegun Table 9.5); annulus Dirichlet
eigenvalues = roots of J_m(l a) Y_m(l b) - J_m(l b) Y_m(l a) (cross-product Bessel equation,
Abramowitz & Stegun 9.5.27); manufactured solutions otherwise.
"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import jn, jn_zeros, yn

import conatus.expert as C
from conatus.discretize import DiscretizationError, discretize
from conatus.discretize.polar_plan import PolarPlan
from conatus.discretize.polar_radial import DiskOps, jacobi_values

PC = C.PolarCoordinates("phi", "r")
PHI, R = PC["phi"], PC["r"]


def _dom(Np=16, N=16, geom="disk", fourier=C.RealFourier, dealias=1, radii=(0.5, 1.5)):
    bp = fourier(PHI, Np, (0, 2 * np.pi), dealias=dealias)
    br = C.DiskBasis(R, N, 1.0, dealias=dealias) if geom == "disk" else C.AnnulusBasis(R, N, radii, dealias=dealias)
    return bp, br, C.Domain(PC, [bp, br])


def _cart(P, Rr):
    return Rr * np.cos(P), Rr * np.sin(P)


def _phys(P, vx, vy):
    """Cartesian components -> physical polar components (phi, r)."""
    return np.array([-vx * np.sin(P) + vy * np.cos(P), vx * np.cos(P) + vy * np.sin(P)])


# ----------------------------------------------------------------------------------------------
# IR: domains
# ----------------------------------------------------------------------------------------------

def test_polar_domain_accepted_and_refusals():
    bp, br, D = _dom()
    assert "DiskBasis" in D.explain() and D.separable_coords == (PHI,)
    with pytest.raises(ValueError, match="DiskBasis"):
        C.Domain(PC, [bp, C.Chebyshev(R, 8, (0, 1), dealias=1)])
    with pytest.raises(ValueError, match="2 pi"):
        C.Domain(PC, [C.RealFourier(PHI, 8, (0, 3), dealias=1), br])
    with pytest.raises(ValueError, match="0 < ri < ro"):
        C.AnnulusBasis(R, 8, (0.0, 1.0), dealias=1)
    SC = C.SphericalCoordinates("p", "t", "r")       # spherical systems: no Chebyshev in theta
    with pytest.raises(ValueError, match="SphereBasis"):
        C.Domain(SC, [C.Chebyshev(SC["t"], 8, (0, 3), dealias=1)])


# ----------------------------------------------------------------------------------------------
# radial toolkit
# ----------------------------------------------------------------------------------------------

def test_disk_ops_identities_and_transforms():
    d = DiskOps(24, 1.0)
    for k in (-3, 0, 1, 4):
        n = d.size(k)
        L = d.L(k, 0, 0, n)[0]
        GpGm = d.G(-1, k + 1, 1, 0, n)[0] @ d.G(+1, k, 0, 0, n)[0]      # lap = G_- G_+ = G_+ G_-
        GmGp = d.G(+1, k - 1, 1, 0, n)[0] @ d.G(-1, k, 0, 0, n)[0]
        scale = abs(L).max()
        assert abs(GpGm - L).max() < 1e-13 * scale and abs(GmGp - L).max() < 1e-13 * scale
        for M in (24, 36):                                              # projection exact on the space (M >= N)
            c = np.random.default_rng(k + 5).standard_normal(n)
            assert np.abs(d.forward(k, M, n) @ (d.backward(k, M, 0, 0, n) @ c) - c).max() < 1e-12
    z = np.linspace(-0.9, 0.9, 5)
    from scipy.special import eval_jacobi
    V = jacobi_values(6, 2, 7, z)
    assert max(np.abs(V[:, n] - eval_jacobi(n, 2, 7, z)).max() for n in range(6)) < 1e-12


# ----------------------------------------------------------------------------------------------
# EVP
# ----------------------------------------------------------------------------------------------

def _bessel_evp(D, br, lam_lift=2, extra_bc=None):
    u, t1 = C.Field("u", D, dtype=complex), C.Field("t1", D.without(R), dtype=complex)
    lam = C.Parameter("lam")
    pb = C.EVP([u, t1], eigenvalue=lam)
    pb.add_equation((C.lap(u) + lam * u + C.lift(t1, br, -1, lam=lam_lift), 0))
    pb.add_equation((C.interp(u, R, "right"), 0))
    return pb


@pytest.mark.parametrize("m", [0, 1, 3, -2, 9])
def test_disk_laplacian_eigenvalues_standard_api(m):
    bp, br, D = _dom(Np=24, N=32, fourier=C.ComplexFourier)
    plan = discretize(_bessel_evp(D, br))
    assert isinstance(plan, PolarPlan) and len(plan.groups) == 23
    res = C.eigenvalues(plan, plan.group(m))
    w = np.sort(res.values.real)
    w = w[w > 0][:4]
    assert np.abs(w / jn_zeros(abs(m), 4) ** 2 - 1).max() < 1e-12
    assert res.residuals.max() < 1e-13
    assert plan.rank_report(plan.group(m)).ok


def _annulus_ref(m, a, b, n):
    f = lambda l: jn(m, l * a) * yn(m, l * b) - jn(m, l * b) * yn(m, l * a)      # noqa: E731
    xs = np.linspace(0.5, 40, 4000)
    v = f(xs)
    roots = [brentq(f, xs[i], xs[i + 1]) for i in range(len(xs) - 1) if v[i] * v[i + 1] < 0]
    return np.array(roots[:n]) ** 2


@pytest.mark.parametrize("m", [0, 2])
def test_annulus_laplacian_eigenvalues_standard_api(m):
    bp, br, D = _dom(Np=8, N=32, geom="annulus")
    u = C.Field("u", D, dtype=complex)
    t1, t2 = C.Field("t1", D.without(R), dtype=complex), C.Field("t2", D.without(R), dtype=complex)
    lam = C.Parameter("lam")
    pb = C.EVP([u, t1, t2], eigenvalue=lam)
    pb.add_equation((C.lap(u) + lam * u + C.lift(t1, br, -1, lam=2) + C.lift(t2, br, -2, lam=2), 0))
    pb.add_equation((C.interp(u, R, "left"), 0))
    pb.add_equation((C.interp(u, R, "right"), 0))
    plan = discretize(pb)
    res = C.eigenvalues(plan, plan.group(m))
    w = np.sort(res.values.real)
    w = w[w > 0][:4]
    assert np.abs(w / _annulus_ref(m, 0.5, 1.5, 4) - 1).max() < 1e-11
    text = plan.explain()
    assert "r^2" in text or "x r^2" in text           # the declared r-multiplier is shown


# ----------------------------------------------------------------------------------------------
# LBVP
# ----------------------------------------------------------------------------------------------

A_G = 4.0          # Gaussian exp(-A ((x - 0.3)^2 + 2 y^2)): needs ~30 modes in r and phi


def _gauss(P, Rr):
    X, Y = _cart(P, Rr)
    return np.exp(-A_G * ((X - 0.3) ** 2 + 2 * Y ** 2))


def _lap_gauss(P, Rr):
    X, Y = _cart(P, Rr)
    g = _gauss(P, Rr)
    return g * ((4 * A_G ** 2 * (X - 0.3) ** 2 - 2 * A_G) + (16 * A_G ** 2 * Y ** 2 - 4 * A_G))


@pytest.mark.parametrize("geom", ["disk", "annulus"])
def test_poisson_mms_spectral_convergence(geom):
    errs = []
    for N in (16, 24, 32):
        bp, br, D = _dom(Np=64, N=N, geom=geom)
        u, f = C.Field("u", D), C.Field("f", D)
        Db = D.without(R)
        t1, t2, gi, go = (C.Field(n, Db) for n in ("t1", "t2", "gi", "go"))
        if geom == "disk":
            pb = C.LBVP([u, t1])
            pb.add_equation((C.lap(u) + C.lift(t1, br, -1, lam=2), f))
            pb.add_equation((C.interp(u, R, "right"), go))
            data = {f: _lap_gauss, go: lambda P, Rr: _gauss(P, 1.0 + 0 * Rr)}
        else:     # Dirichlet inside, Neumann outside (d/dr of the Gaussian, analytic)
            pb = C.LBVP([u, t1, t2])
            pb.add_equation((C.lap(u) + C.lift(t1, br, -1, lam=2) + C.lift(t2, br, -2, lam=2), f))
            pb.add_equation((C.interp(u, R, "left"), gi))
            pb.add_equation((C.interp(C.diff(u, R), R, "right"), go))

            def dr(P, Rr):
                X, Y = _cart(P, Rr)
                g = _gauss(P, Rr)
                return g * A_G * (-2 * (X - 0.3) * np.cos(P) - 4 * Y * np.sin(P))
            data = {f: _lap_gauss, gi: lambda P, Rr: _gauss(P, 0.5 + 0 * Rr),
                    go: lambda P, Rr: dr(P, 1.5 + 0 * Rr)}
        plan = discretize(pb)
        sol = plan.solve(data)
        assert sol.max_residual < 1e-12
        Pg, Rg = plan.space.grids(False)
        errs.append(np.abs(sol.grid("u") - _gauss(Pg[:, None], Rg[None, :])).max())
    assert errs[-1] < 1e-9 and errs[-1] < 1e-4 * errs[0], errs


def test_vector_poisson_disk_exact_and_rank():
    bp, br, D = _dom(Np=32, N=12)
    u, f = C.VectorField("u", D), C.VectorField("f", D)
    tu, g = C.VectorField("tu", D.without(R)), C.VectorField("g", D.without(R))
    pb = C.LBVP([u, tu])
    pb.add_equation((C.lap(u) + C.lift(tu, br, -1, lam=2), f))
    pb.add_equation((C.interp(u, R, "right"), g))

    def ue(P, Rr):
        X, Y = _cart(P, Rr)
        return _phys(P, np.cos(Y) * X, X * X - Y * Y * X)

    def fe(P, Rr):
        X, Y = _cart(P, Rr)
        return _phys(P, -np.cos(Y) * X, 2 - 2 * X)
    plan = discretize(pb)
    reps = plan.check()
    assert len(reps) == 16 and all(r.ok for r in reps)
    sol = plan.solve({f: fe, g: lambda P, Rr: ue(P, 1.0 + 0 * Rr)})
    Pg, Rg = plan.space.grids(False)
    assert np.abs(sol.grid("u") - ue(Pg[:, None], Rg[None, :])).max() < 1e-12


# ----------------------------------------------------------------------------------------------
# evaluator (nonlinear terms): operators against closed forms, disk and annulus
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("geom", ["disk", "annulus"])
def test_polar_evaluator_operators(geom):
    bp, br, D = _dom(Np=48, N=20, geom=geom, dealias=3 / 2)
    f, u = C.Field("f", D), C.VectorField("u", D)
    ev_plan = discretize(_bessel_evp(D, br)) if geom == "disk" else None
    from conatus.discretize.polar_eval import PolarEvaluator, PolarSpace
    S = PolarSpace(D) if ev_plan is None else ev_plan.space
    ev = PolarEvaluator(S)
    P, Rr = S.grids(False)
    P, Rr = P[:, None], Rr[None, :]
    X, Y = _cart(P, Rr)
    fv = X ** 3 * Y + np.cos(X) * Y
    ux, uy = np.cos(Y) * X, X * X - Y
    cf = S.forward(fv, D, 0, False)
    cu = S.forward(_phys(P, ux, uy), D, 1, False)

    def grid(e):
        return S.backward(ev.evaluate(e, {f: cf, u: cu}, {}), e.domain, e.rank, False)
    fx, fy = 3 * X * X * Y - np.sin(X) * Y, X ** 3 + np.cos(X)
    # round-off of the transform (~5e-15) amplified like (2N)^(2 x order): measured at N = 20:
    # lap 1.3e-9, first derivatives 1e-11 (polar_radial.DiskOps)
    tol = 1e-10
    assert np.abs(grid(C.lap(f)) - (6 * X * Y - np.cos(X) * Y)).max() < 1e-8
    assert np.abs(grid(C.grad(f)) - _phys(P, fx, fy)).max() < tol
    assert np.abs(grid(C.div(u)) - (np.cos(Y) - 1)).max() < tol
    assert np.abs(grid(C.curl(u)) - (2 * X + X * np.sin(Y))).max() < tol           # d_x u_y - d_y u_x
    assert np.abs(grid(C.curl(f)) - _phys(P, fy, -fx)).max() < tol                 # (d_y f, -d_x f)
    ax, ay = ux * np.cos(Y) - uy * X * np.sin(Y), ux * 2 * X - uy                  # (u . grad) u
    assert np.abs(grid(C.dot(u, C.grad(u))) - _phys(P, ax, ay)).max() < tol
    assert np.abs(grid(f * f) - fv ** 2).max() < 1e-12
    if geom == "annulus":         # u_r is a regular scalar only away from the axis
        assert np.abs(grid(C.comp(u, 1)) - (ux * np.cos(P) + uy * np.sin(P))).max() < 1e-12
    else:
        with pytest.raises(DiscretizationError, match="not a regular function"):
            grid(C.comp(u, 1))


# ----------------------------------------------------------------------------------------------
# IVP (nonlinear, manufactured solution) and Navier-Stokes invariants
# ----------------------------------------------------------------------------------------------

def _heat_mms(step, scheme="SBDF2", T=0.2, N=16, Np=32):
    bp, br, D = _dom(Np=Np, N=N, dealias=3 / 2)
    u, H, t1, Ho = C.Field("u", D), C.Field("H", D), C.Field("t1", D.without(R)), C.Field("Ho", D.without(R))
    t = C.Parameter("t", 0.0)
    a = 1 + 0.5 * C.sin(t)
    pb = C.IVP([u, t1])            # u = a(t) h(x, y), h = e^x cos 2y (lap h = -3 h)
    pb.add_equation((C.dt(u) - C.lap(u) + C.lift(t1, br, -1, lam=2),
                     (0.5 * C.cos(t) + 3 * a) * H + a * a * H * H - u * u))
    pb.add_equation((C.interp(u, R, "right"), a * Ho))

    def h(P, Rr):
        X, Y = _cart(P, Rr)
        return np.exp(X) * np.cos(2 * Y)
    plan = discretize(pb, ncc={H: h, Ho: lambda P, Rr: h(P, 1.0 + 0 * Rr)})
    s = C.IVPSolver(plan, scheme, step, time_parameter="t")
    s.set_state({u: h})
    s.run(int(round(T / step)))
    Pg, Rg = plan.space.grids(False)
    return np.abs(s.grid("u") - (1 + 0.5 * np.sin(s.t)) * h(Pg[:, None], Rg[None, :])).max(), s


def test_nonlinear_heat_disk_mms_second_order():
    e1, _ = _heat_mms(0.02)
    e2, s = _heat_mms(0.01)
    assert 1.8 < np.log2(e1 / e2) < 2.2 and e2 < 2e-4, (e1, e2)
    assert type(s).__name__ == "PolarIVPSolver" and "polar" in s.explain()


def test_navier_stokes_disk_divergence_free_and_dissipative():
    bp, br, D = _dom(Np=16, N=12, dealias=3 / 2)
    u, p = C.VectorField("u", D), C.Field("p", D)
    tu, tp = C.VectorField("tu", D.without(R)), C.Field("tp", C.Domain(PC, []))
    pb = C.IVP([u, p, tu, tp])
    pb.add_equation((C.dt(u) + C.grad(p) - 0.05 * C.lap(u) + C.lift(tu, br, -1, lam=2), -C.dot(u, C.grad(u))))
    pb.add_equation((C.div(u) + tp, 0))
    pb.add_equation((C.interp(u, R, "right"), 0))
    pb.add_equation((C.integ(p), 0))
    plan = discretize(pb)
    assert all(r.ok for r in plan.check())

    def u0(P, Rr):                   # u = curl((1 - r^2)^2 x): no slip at r = 1
        X, Y = _cart(P, Rr)
        s_ = 1 - Rr * Rr
        px, py = -4 * X * s_ * X + s_ * s_, -4 * Y * s_ * X
        return _phys(P, py, -px)
    s = C.IVPSolver(plan, "RK443", 5e-3)
    s.set_state({u: u0})
    e0 = np.abs(s.coeffs("u")).max()
    s.run(20)
    dv = plan.evaluator.evaluate(C.div(u), {u: s.coeffs("u")}, {})
    assert np.abs(dv).max() < 1e-12
    assert abs(s.coeffs("tp")).max() < 1e-12            # the pressure gauge tau stays zero
    assert np.abs(s.coeffs("u")).max() < e0


# ----------------------------------------------------------------------------------------------
# refusals and reports
# ----------------------------------------------------------------------------------------------

def test_polar_refusals_are_explicit():
    bp, br, D = _dom(Np=8, N=8)
    # azimuthal modes without radial coefficients (|m| >= 2N): refused, never emptied silently
    bq, brq, Dq = _dom(Np=40, N=8)
    with pytest.raises(DiscretizationError, match="no radial coefficient"):
        discretize(_bessel_evp(Dq, brq))
    Db = D.without(R)
    s, t, u = C.Field("s", D), C.Field("t", Db), C.VectorField("u", D)
    # a physical component is not regular on the disk (u_r mixes rho^|m+1| and rho^|m-1|)
    q = C.LBVP([s, t, u])
    q.add_equation((C.lap(s) + C.comp(u, 1) + C.lift(t, br, -1), 0))
    with pytest.raises(DiscretizationError, match="not a regular function on the disk"):
        discretize(q).matrices()
    # d/dr alone in the bulk (only allowed inside interp for boundary conditions)
    q = C.LBVP([s, t])
    q.add_equation((C.lap(s) + C.diff(s, R) + C.lift(t, br, -1), 0))
    q.add_equation((C.interp(s, R, "right"), 0))
    with pytest.raises(DiscretizationError, match="d/dr alone is not regular"):
        discretize(q).matrices()
    # a phi-only unknown in a disk bulk equation at m != 0 would be e^{i m phi} without rho^|m|
    q = C.LBVP([s, t])
    q.add_equation((C.lap(s) + t, 0))
    q.add_equation((C.interp(s, R, "right"), 0))
    plan = discretize(q)
    with pytest.raises(DiscretizationError, match="not regular at the axis"):
        plan.matrices(plan.group(1))
    SC = C.SphericalCoordinates("a", "b", "c")       # spherical systems: regular radial bases only
    with pytest.raises(ValueError, match="BallBasis"):
        C.Domain(SC, [C.Chebyshev(SC["c"], 8, (0, 1), dealias=1)])


def test_polar_plan_explain_and_methods():
    bp, br, D = _dom(Np=8, N=8)
    plan = discretize(_bessel_evp(D, br))
    text = plan.explain()
    for key in ("groups m = 0..3", "P^(2, 0)", "PASS", "spin components"):
        assert key in text, key
    m = plan.methods(eigen_solver="dense")
    assert "Zernike" in m.text or "disk" in m.text
    assert "vasil2016" in m.citations


def test_real_field_imaginary_tolerance_declared():
    """Audit R9 (polar part): the tolerance on the imaginary part of real fields on the grid is a
    declared, visible value (plan.space.real_tol = 1e-12, printed by explain()) -- it used to be a
    hidden 1e-9 default argument of polar_eval._real. An inconsistent m = 0 pair (u_+ != conj(u_-))
    is refused above it, rounding-level imaginary parts are accepted."""
    bp, br, D = _dom(Np=8, N=6)
    u = C.VectorField("u", D)
    pb = C.LBVP([u])
    pb.add_equation((u, 0))
    plan = discretize(pb)
    tol = plan.space.real_tol
    assert tol <= 1e-12
    assert f"{tol:g}" in plan.explain() and f"{tol:g}" in plan.space.explain()
    c = np.zeros(plan.space.coeff_shape(D, 1), complex)
    c[0, 0, 0], c[1, 0, 0] = 1.0, 1.0 + 1e-10j                # m = 0: u_+ != conj(u_-) at 1e-10
    with pytest.raises(DiscretizationError, match="imaginary part"):
        plan.space.backward(c, D, 1, False)
    c[1, 0, 0] = 1.0 + 1e-15j                                 # rounding level: accepted
    plan.space.backward(c, D, 1, False)
