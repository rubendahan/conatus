"""Resolution report (conatus.resolution): under-resolved profiles are flagged, resolved ones
pass, the recommended N is estimated from the fitted decay and never applied, real IVP states."""
import copy

import numpy as np
import pytest

import conatus.resolution as R
from conatus.discretize import discretize
from conatus.errors import InputTypeError, OptionError, UnsupportedError
from conatus.ir import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter,
                        RealFourier, VectorField, div, dot, dt, grad, integ, interp, lap,
                        lift, unit_vector)
from conatus.reference import transforms as T
from conatus.reference.evaluate import FieldData
from conatus.timestep import IVPSolver


def _heat_plan(N):
    """1D heat on (-1, 1), Chebyshev N, tau lifts (a plan to attach states to)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, N, (-1, 1), dealias=1.5)
    D = Domain(cs, [b])
    u, t1, t2 = Field("u", D), Field("t1", D.without(x)), Field("t2", D.without(x))
    pb = IVP([u, t1, t2], name="heat 1D")
    pb.add_equation((dt(u) - lap(u) + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    return discretize(pb), u


def _fourier_coeffs(f, N):
    b = RealFourier(CartesianCoordinates("x")["x"], N, (0, 2 * np.pi), dealias=1.5)
    return T.forward(f(T.grid(b, dealias=False)), b, dealias=False), b


def _snapshot(plan):
    """Everything a report could plausibly change on a plan."""
    return (copy.copy(plan.__dict__), [(v.name, [(b.size, b.dealias) for b in v.domain.bases])
                                       for v in plan.variables])


def test_sharp_tanh_under_resolved_and_smooth_resolved():
    plan, u = _heat_plan(24)
    bad = plan.resolution_report({u: lambda X: np.tanh(10 * X)})
    e = bad.entry("u", "x")
    assert e.verdict == "under-resolved" and e.tail > R.TAIL_MARGINAL, e.tail   # 1.6e-2
    assert not bad.ok and bad.recommend() and "increase Nx from 24 to ~" in bad.recommend()[0]
    good = _heat_plan(32)[0]
    rep = good.resolution_report({good.variables[0]: lambda X: np.sin(np.pi * X)})
    assert rep.ok and rep.entry("u", "x").tail < 1e-14 and rep.recommend() == []
    assert "t1: tau variable" in rep.explain() or "t1: not in the state" in rep.explain()


def test_fourier_recommendation_reaches_target():
    """exp(sin x) with N=8 is under-resolved; the N estimated from the fitted decay reaches the
    target (checked by projecting at that N), and the estimate states its assumption."""
    f = lambda X: np.exp(np.sin(X))  # noqa: E731
    c, b = _fourier_coeffs(f, 8)
    est = R.estimate_modes(c, b)
    assert est.direction.verdict == "under-resolved" and est.N is not None and est.N > 8
    assert "exponential decay" in est.explain()
    c2, b2 = _fourier_coeffs(f, est.N)
    got = R.estimate_modes(c2, b2).direction.tail
    assert got <= R.TAIL_TARGET, (est.N, got)


def test_recommendation_never_modifies_plan_or_solver():
    plan, u = _heat_plan(16)
    before = _snapshot(plan)
    s = IVPSolver(plan, "SBDF2", 1e-3)
    s.set_state({u: lambda X: np.tanh(20 * X) * (1 - X ** 2)})
    X0, t0 = s.X.copy(), s.t
    rep = plan.resolution_report(s)
    assert not rep.ok and rep.recommend()
    rep.explain(), rep.to_dict(), rep._repr_html_()
    assert _snapshot(plan)[1] == before[1] and plan.__dict__.keys() == before[0].keys()
    assert all(plan.__dict__[k] is before[0][k] for k in before[0])
    assert np.array_equal(s.X, X0) and s.t == t0 and s.plan is plan


def test_ivp_solver_state_rb2d_with_cfl():
    """A real IVPSolver state (2D RB, 16 x 8, a few steps): per-direction entries, taus skipped,
    the advective CFL number reported as an indicator."""
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx, bz = RealFourier(x, 16, (0, 2.0), dealias=1.5), Chebyshev(z, 8, (-0.5, 0.5), dealias=1.5)
    D = Domain(cs, [bx, bz])
    Dx = D.without(z)
    u, Tf, p = VectorField("u", D), Field("T", D), Field("p", D)
    tp, tT1, tT2 = Field("tp", Domain(cs, [])), Field("tT1", Dx), Field("tT2", Dx)
    tu1, tu2 = VectorField("tu1", Dx), VectorField("tu2", Dx)
    Ra, ez = Parameter("Ra", 1e5), unit_vector(z)
    pb = IVP([u, Tf, p, tp, tT1, tT2, tu1, tu2], name="RB 2D")
    pb.add_equation((div(u) + dot(ez, lift(tu2, bz, -1)) + tp, 0))
    pb.add_equation((dt(Tf) - lap(Tf) + lift(tT1, bz, -1) + lift(tT2, bz, -2), -dot(u, grad(Tf))))
    pb.add_equation((dt(u) - lap(u) + grad(p) - Ra * Tf * ez + lift(tu1, bz, -1)
                     + lift(tu2, bz, -2), -dot(u, grad(u))))
    pb.add_equation((interp(Tf, z, "left"), 0.5))
    pb.add_equation((interp(Tf, z, "right"), -0.5))
    pb.add_equation((interp(u, z, "left"), 0))
    pb.add_equation((interp(u, z, "right"), 0))
    pb.add_equation((integ(p), 0))
    plan = discretize(pb)
    s = IVPSolver(plan, "RK222", 2e-4)
    s.set_state({Tf: lambda X, Z: -Z + 1e-2 * np.cos(np.pi * X) * np.cos(np.pi * Z)})
    s.run(5)
    rep = plan.resolution_report(s, velocity="u")
    assert {(e.variable, e.coord) for e in rep.entries} >= {("u", "x"), ("u", "z"), ("T", "x"),
                                                            ("T", "z"), ("p", "z")}
    assert rep.entry("T", "x").dealias == 1.5 and rep.entry("T", "z").N == 8
    skipped = dict(rep.skipped)
    assert "tau" in skipped["tT1"] and "constant" in skipped["tp"]
    assert rep.cfl is not None and rep.cfl["dt"] == 2e-4 and np.isfinite(rep.cfl["cfl"])
    assert "not an error control" in rep.explain()
    # T: resolved along x (16 Fourier modes), not to 1e-8 along z with Nz = 8 (tail 7e-6)
    assert rep.entry("T", "x").verdict == "resolved"
    assert rep.entry("T", "z").tail > R.TAIL_TARGET and any("T/z" in r for r in rep.recommend())
    d = rep.to_dict()
    assert d["entries"] and d["definition"].startswith("tail = ")


def test_inputs_refused_by_name():
    plan, u = _heat_plan(16)
    with pytest.raises(InputTypeError, match="no declared layout"):
        plan.resolution_report({u: np.zeros(16)})
    c = T.forward(np.sin(np.pi * T.grid(plan.domain.bases[0], dealias=False)),
                  plan.domain.bases[0], dealias=False)
    assert plan.resolution_report({"u": c}, layout="coeff").ok
    assert plan.resolution_report({u: FieldData(u, c, "coeff")}).ok
    with pytest.raises(OptionError, match="not a variable"):
        plan.resolution_report({"w": c}, layout="coeff")
    with pytest.raises(OptionError, match="target <= marginal"):
        plan.resolution_report({"u": c}, layout="coeff", target=1e-2, marginal=1e-4)


def test_curvilinear_plan_refused():
    import conatus.expert as C
    PC = C.PolarCoordinates("phi", "r")
    PHI, Rr = PC["phi"], PC["r"]
    bp = C.RealFourier(PHI, 8, (0, 2 * np.pi), dealias=1)
    br = C.DiskBasis(Rr, 8, 1.0, dealias=1)
    D = C.Domain(PC, [bp, br])
    u, t1 = C.Field("u", D, dtype=complex), C.Field("t1", D.without(Rr), dtype=complex)
    lam = C.Parameter("lam")
    pb = C.EVP([u, t1], eigenvalue=lam)
    pb.add_equation((C.lap(u) + lam * u + C.lift(t1, br, -1, lam=2), 0))
    pb.add_equation((C.interp(u, Rr, "right"), 0))
    plan = discretize(pb)
    with pytest.raises(UnsupportedError, match="Cartesian Plan only"):
        R.resolution_report(plan, {})


def test_constant_on_few_fourier_modes_is_not_judged():
    """F101 (-10-16), same rule as the run-time tail check: with N = 4 Fourier modes
    (M = 2 mode numbers) the tail window held the mean, so a constant was "under-resolved,
    tail 1.0"; it is now "too few modes" (not judged, not resolved)."""
    c, b = _fourier_coeffs(lambda x: 1.0 + 0 * x, 4)
    d = R._analyse("u", c, 0, b, R.TAIL_TARGET, R.TAIL_MARGINAL)
    assert d.verdict == "too few modes" and np.isnan(d.tail)
    assert "not judged" in d.recommendation()
    c, b = _fourier_coeffs(lambda x: 1.0 + 0 * x, 16)
    assert R._analyse("u", c, 0, b, R.TAIL_TARGET, R.TAIL_MARGINAL).verdict == "resolved"
