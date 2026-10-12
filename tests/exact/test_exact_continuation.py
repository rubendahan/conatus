"""Continuation with a nonzero dynamic quantity against closed forms (audit coverage hole 6).

The continuation tests of validated folds (stationary quantities) and Hopf points of toy maps:
a wrong mass matrix (bare unknown in M: eigenvalue term lost inside a product) moves
no fold and no threshold. Here a branch of an IR problem is continued by ``NLBVPContinuation`` and
the growth rates along it come from the *linearized IVP* (``linearize`` -> EVP -> ``eigenvalues``),
compared at every branch point with their closed form (nonzero away from the event).

Problems on z in (-1, 1), Chebyshev, Neumann walls (single Fourier group, as the NLBVP adapter
requires), the mass coefficient c != 1 sits *inside a product with a bare-free sum*
``c*(dt(u) + a*u)`` (the form after dt -> sigma) and every LHS carries a *bare* unknown
``+ u`` (the form; ``a`` is a Parameter so the IR cannot fold the terms):

  fold:  c*(dt(u) + a u) + u - D u'' = c a u + r - u^2
         steady uniform states u^2 + u - r = 0, u = (-1 +- sqrt(1 + 4 r)) / 2, fold at r = -1/4;
         growth rates of cos(n pi (z + 1) / 2):  sigma_n = -(1 + 2 u + D (n pi / 2)^2) / c.
  Hopf:  c*(dt(x) + a x) + x - D x'' + w y = c a x + r x - (x^2 + y^2) x
         c*(dt(y) + a y) + y - D y'' - w x = c a y + r y - (x^2 + y^2) y
         trivial branch x = y = 0 for all r; sigma_n = (r - 1 - D (n pi / 2)^2 +- i w) / c,
         Hopf at r = 1 with frequency w / c.

The two branch tests fail with reintroduced (a bare top-level unknown in the sigma part: M
gets c + 1, e.g. Hopf frequency 0.5 instead of 0.75; with the audit plugin's split mutation, a loud
"degree 2" refusal) and with (the fold rates of u vanish *silently*, only the passive -1
remains; Hopf: "M is identically zero"). Checked by mutation plugins, not committed.

Two defects of the continuation module found here: ``NLBVPContinuation(eigen=True)`` counted the
eigenvalues of the steady Newton Jacobian as growth rates (refused now, a callback is required),
and the crossing locator read the wrong eigenvalue once the secant came within marginal_tol of the
crossing (fixed). An open one is kept as a strict xfail: the relative residual scale vanishes at
the fold of a uniform state.
"""
import numpy as np
import pytest

import conatus.expert as C
from conatus.continuation import ContinuationSettings, NLBVPContinuation
from conatus.discretize import discretize
from conatus.reference.evaluate import FieldData
from conatus.solvers.nlbvp import NewtonSolver

CM, A, DIFF, W = 2.0, 0.7, 0.3, 1.5      # mass coefficient c, folded-away a, diffusivity, frequency
N = 16                                    # Chebyshev modes
NEXACT = 3                                # eigenvalues compared per branch point (n = 0, 1, 2)


def _cfg(**kw):
    """Explicit continuation settings (no defaults in the library)."""
    base = dict(ds=0.1, ds_min=1e-8, ds_max=0.3, adapt=1.5, fast_iterations=3, tolerance=1e-10,
                max_newton=15, max_points=40, dp_fd=1e-6)
    base.update(kw)
    return ContinuationSettings(**base)


def _domain():
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    b = C.Chebyshev(z, N, (-1, 1), dealias=3 / 2)
    return z, b, C.Domain(cs, [b])


def _taus(t1, t2, b):
    """Lifted tau terms of one Neumann unknown."""
    return C.lift(t1, b, -1) + C.lift(t2, b, -2)


def _neumann(pb, u, z):
    """u'(+-1) = 0 (the uniform mode and cos(n pi (z + 1) / 2) are the eigenfunctions)."""
    pb.add_equation((C.interp(C.diff(u, z), z, "left"), 0))
    pb.add_equation((C.interp(C.diff(u, z), z, "right"), 0))


# ----------------------------------------------------------------------------------------------
# Builders: the steady NLBVP (continued) and the IVP (linearized for the growth rates)
# ----------------------------------------------------------------------------------------------

def _fold_problems(passive=True):
    """Fold normal form; ``passive`` adds a decoupled uniform unknown v (dt(v) + v = 1, v = 1,
    sigma = -1 exactly, bare term): see test_uniform_fold_relative_residual_scale_vanishes."""
    z, b, D = _domain()
    D0 = D.without(z)
    a, c, r = C.Parameter("a", A), C.Parameter("c", CM), C.Parameter("r", 0.5)
    u, t1, t2, v = C.Field("u", D), C.Field("t1", D0), C.Field("t2", D0), C.Field("v", D)
    unknowns = [u, t1, t2] + ([v] if passive else [])
    out = []
    for dyn in (False, True):
        pb = (C.IVP if dyn else C.NLBVP)(unknowns, name="fold normal form")
        # c*(dt(u) + a u) + u - D u'' = c a u + r - u^2 (steady problem: the same terms, dt dropped)
        mass = c * (C.dt(u) + a * u) if dyn else c * (a * u)
        pb.add_equation((mass + u - DIFF * C.lap(u) + _taus(t1, t2, b), c * a * u + r - u * u))
        _neumann(pb, u, z)
        if passive:
            pb.add_equation(((C.dt(v) + v) if dyn else v, 1))
        out.append(pb)
    return out[0], out[1], dict(u=u, t1=t1, t2=t2, v=v, z=z, D=D, D0=D0)


def _hopf_problems():
    z, b, D = _domain()
    D0 = D.without(z)
    a, c, r = C.Parameter("a", A), C.Parameter("c", CM), C.Parameter("r", 0.6)
    x, y = C.Field("x", D), C.Field("y", D)
    tx1, tx2, ty1, ty2 = (C.Field(n, D0) for n in ("tx1", "tx2", "ty1", "ty2"))
    rho = x * x + y * y
    out = []
    for dyn in (False, True):
        pb = (C.IVP if dyn else C.NLBVP)([x, y, tx1, tx2, ty1, ty2], name="Hopf normal form")
        # c*(dt(x) + a x) + x - D x'' + w y = c a x + r x - |z|^2 x (and y, with -w x)
        mx = c * (C.dt(x) + a * x) if dyn else c * (a * x)
        my = c * (C.dt(y) + a * y) if dyn else c * (a * y)
        pb.add_equation((mx + x - DIFF * C.lap(x) + W * y + _taus(tx1, tx2, b),
                         c * a * x + r * x - rho * x))
        pb.add_equation((my + y - DIFF * C.lap(y) - W * x + _taus(ty1, ty2, b),
                         c * a * y + r * y - rho * y))
        _neumann(pb, x, z)
        _neumann(pb, y, z)
        out.append(pb)
    return out[0], out[1], dict(x=x, y=y, z=z, D=D, D0=D0)


def _growth_rates(ivp, o, names):
    """Callback (coeffs, p) -> growth rates of the linearized IVP at a branch point: linearize()
    turns c*(dt(u) + a u) into c*(s du + a du) (the nested eigenvalue form of)."""
    base = {o[n]: C.Field("B" + n, o["D"]) for n in names}      # taus enter linearly: no base
    evp = C.linearize(ivp, base, C.Parameter("s", dtype=complex))
    # r survives the linearization only where it multiplies an unknown (Hopf: r x; fold: r alone)
    uses_r = any(p.name == "r" for eq in evp.equations for p in eq.lhs.free_parameters())

    def rates(coeffs, p):
        ncc = {base[o[n]]: FieldData(base[o[n]], coeffs[n], "coeff") for n in names}
        plan = discretize(evp, ncc=ncc, params={"r": p} if uses_r else None)
        res = C.eigenvalues(plan, plan.groups[0])
        vals = res.values[np.isfinite(res.values)]
        return vals[np.argsort(-vals.real)], res
    return rates


def _uniform(coeffs, name):
    """Value of a spatially uniform Chebyshev field (T_0 coefficient), after checking uniformity."""
    a = np.asarray(coeffs[name]).real
    assert np.abs(a[1:]).max() < 1e-11 * max(1.0, abs(a[0])), "branch state is not uniform"
    return a[0]


def _kn(n):
    return DIFF * (np.arange(n) * np.pi / 2) ** 2


# ----------------------------------------------------------------------------------------------
# Fold (saddle-node) branch: states, growth rates along the branch, fold and crossings located
# ----------------------------------------------------------------------------------------------

R1 = ((DIFF * np.pi ** 2 / 4) ** 2 - 1) / 4     # sigma_1 = 0 on the lower branch: sqrt(1 + 4 r) = D pi^2 / 4


@pytest.fixture(scope="module")
def fold_run():
    """Upper branch from r = 0.1, through the fold r = -1/4, up the lower branch to r = 0; the
    stability count uses the growth rates of the linearized IVP (mass c, bare u)."""
    st, ivp, o = _fold_problems()
    newton = NewtonSolver(st, tolerance=1e-13, damping=1.0, max_iterations=30, params={"r": 0.1})
    sol = newton.solve({o["u"]: lambda Z: 0.1 + 0 * Z, o["v"]: lambda Z: 1.0 + 0 * Z})
    rates = _growth_rates(ivp, o, ["u"])
    cont = NLBVPContinuation(newton, "r", _cfg(tolerance=1e-12, ds=0.15, ds_max=0.3), marginal_tol=1e-10,
                             eigen=lambda coeffs, p: rates(coeffs, p)[0])
    return cont, sol, rates, cont.run(sol.coeffs, 0.1, -1, p_bounds=(-0.3, 0.0))


@pytest.mark.slow                   # 3.0 s in the fast suite (qual_durations0)
def test_fold_branch_growth_rates_and_events_are_exact(fold_run):
    """u(r) exact at every point; sigma_n = -(1 + 2 u + D (n pi/2)^2) / c exact at every point
    (nonzero except at the fold); fold located at r = -1/4 (converged); stability 0 -> 1 -> 2 with
    the real crossings located at r = -1/4 (sigma_0) and r = R1 (sigma_1)."""
    cont, sol, rates, br = fold_run
    (fold,) = br.folds()
    assert fold.converged and abs(fold.p + 0.25) < 1e-9, br.explain()
    i_fold = fold.between[1]
    sgn_seen = set()
    for k, q in enumerate(br.points):
        u = _uniform(cont.coeffs_of(q.x), "u")
        sgn = +1 if k < i_fold else -1                           # upper, then lower branch
        assert abs(u - (-1 + sgn * np.sqrt(1 + 4 * q.p)) / 2) < 1e-10, (k, q.p)
        vals, res = rates(cont.coeffs_of(q.x), q.p)
        exact = np.append(-(1 + 2 * u + _kn(NEXACT)) / CM, -1.0)   # u modes n = 0..2, passive v
        for e in exact:                                          # each exact rate is present
            assert np.min(np.abs(vals - e)) < 1e-9, (k, q.p, e, vals[:6])
        assert res.residuals.max() < 1e-9
        assert q.unstable == int(np.sum(exact > 0)) == int(np.sum(vals.real > 1e-10)), (k, q.p)
        sgn_seen.add(np.sign(exact[0]))
    assert sgn_seen == {-1.0, 1.0}                               # stable and unstable points met
    cross = sorted((e for e in br.events if e.kind == "real crossing"), key=lambda e: e.p)
    assert [e.converged for e in cross] == [True, True], br.explain()
    assert abs(cross[0].p + 0.25) < 1e-9 and abs(cross[1].p - R1) < 1e-9, br.explain()


def test_nlbvp_continuation_refuses_the_steady_jacobian_as_stability():
    """eigen=True used the eigenvalues of the steady Jacobian A = F'(x) (F = LHS - RHS): opposite
    sign to growth rates, no mass matrix, tau/BC rows mixed in; on the stable upper branch it
    counted every mode as unstable. A growth-rate callback must now be given."""
    st = _fold_problems()[0]
    newton = NewtonSolver(st, tolerance=1e-13, damping=1.0, max_iterations=30, params={"r": 0.1})
    with pytest.raises(TypeError, match="growth rates"):
        NLBVPContinuation(newton, "r", _cfg(), eigen=True, marginal_tol=1e-10)


# ----------------------------------------------------------------------------------------------
# Hopf on the trivial branch: complex growth rates, Hopf located at r = 1 with frequency w / c
# ----------------------------------------------------------------------------------------------

def test_hopf_rates_and_frequency_on_the_trivial_branch():
    """sigma_n = (r - 1 - D (n pi/2)^2 +- i w) / c at every point; one Hopf (r <= 1.3 < 1 + D pi^2/4)
    located at r = 1 with Im lambda = w / c (the frequency carries the mass coefficient)."""
    st, ivp, o = _hopf_problems()
    newton = NewtonSolver(st, tolerance=1e-13, damping=1.0, max_iterations=30, params={"r": 0.7})
    sol = newton.solve({o["x"]: lambda Z: 0 * Z, o["y"]: lambda Z: 0 * Z})
    rates = _growth_rates(ivp, o, ["x", "y"])
    cont = NLBVPContinuation(newton, "r", _cfg(ds=0.25, ds_max=0.4), marginal_tol=1e-10,
                             eigen=lambda coeffs, p: rates(coeffs, p)[0])
    br = cont.run(sol.coeffs, 0.7, +1, p_bounds=(0.0, 1.3))
    for q in br.points:
        vals, res = rates(cont.coeffs_of(q.x), q.p)
        re = (q.p - 1 - _kn(NEXACT)) / CM
        for e in np.concatenate([re + 1j * W / CM, re - 1j * W / CM]):   # each exact rate present
            assert np.min(np.abs(vals - e)) < 1e-9, (q.p, e, vals[:6])
        assert res.residuals.max() < 1e-9
    (hopf,) = [e for e in br.events if e.kind == "hopf"]
    assert hopf.converged and abs(hopf.p - 1.0) < 1e-9, br.explain()
    im = float(hopf.detail.split("Im λ = ")[1])
    assert abs(abs(im) - W / CM) < 1e-5, hopf.detail            # detail prints 6 digits


def test_crossing_located_when_the_secant_lands_on_it():
    """Re λ = p exactly (linear): the first secant iterate lands on p = 0, where the threshold pair
    is marginal. The locator used to read the *next* eigenvalue (-3) there and never converged."""
    from conatus.continuation import PseudoArclength
    pa = PseudoArclength(lambda x, p: np.array([x[0] - p]), lambda x, p: np.array([[1.0]]),
                         _cfg(dp_fd=None, tolerance=1e-12), dGdp=lambda x, p: np.array([-1.0]),
                         eigen=lambda x, p: np.array([p + 1j, p - 1j, -3.0]), marginal_tol=1e-10)
    br = pa.run(np.array([-0.7]), -0.7, +1, p_bounds=(-1, 0.5))
    (hopf,) = br.events
    assert hopf.kind == "hopf" and hopf.converged and abs(hopf.p) < 1e-12, br.explain()


@pytest.mark.xfail(strict=True, reason=" (open): NLBVPContinuation's relative residual scale "
                   "max(|A x|, |b|) vanishes at the fold of a uniform state (A x = 0 there): the "
                   "fold is found (p to 1e-9) but its location never converges")
def test_uniform_fold_relative_residual_scale_vanishes():
    """Same fold without the passive unknown: at r = -1/4 the state u = -1/2 spans the kernel of
    A = 1 + 2u - D d^2, so A x = 0 and b = u^2 + r = 0 as assembled vectors; |G| ~ 1e-17 is
    divided by a scale ~ 1e-8 and the corrector cannot reach its tolerance."""
    st, _, o = _fold_problems(passive=False)
    newton = NewtonSolver(st, tolerance=1e-13, damping=1.0, max_iterations=30, params={"r": -0.2})
    sol = newton.solve({o["u"]: lambda Z: -0.2 + 0 * Z})
    br = NLBVPContinuation(newton, "r", _cfg(tolerance=1e-12, max_points=3, event_max_iterations=4)).run(
        sol.coeffs, -0.2, -1, p_bounds=(-0.3, 0.0))
    (fold,) = br.folds()
    assert abs(fold.p + 0.25) < 1e-8                            # found ...
    assert fold.converged, br.explain()                         # ... but flagged not converged
