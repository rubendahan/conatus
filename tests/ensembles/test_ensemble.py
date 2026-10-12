"""Ensembles: member-by-member agreement with single IVPSolver runs.

* rhs="loop" (IVPSolver.F member by member, batched M/L products and stacked solves): every
  member **bit-identical** to its single run (checks the batched step, products and solves);
* rhs="batched" (vectorized polynomial/general form): agreement to rounding, bound declared per
  case below; a member of a batch is bit-identical to the same state run alone (B = 1) through
  the same batched path;
* parameter ensembles, with the parameter in F (Lorenz r) or in L (Burgers nu: per-member factors).
"""
import numpy as np
import pytest

from conatus.discretize import discretize
from conatus.ensembles import Ensemble
from conatus.errors import OptionError, UnsupportedError
from conatus.timestep import IVPSolver
from ens_cases import lorenz_problem, run_single
from ivp_cases import burgers_1d, ks_1d, rayleigh_benard_2d

rng = np.random.default_rng(11)


def _members(ens_kind, solvers, X0, nsteps, rhs):
    e = (Ensemble(solvers[0], len(X0), rhs=rhs) if ens_kind == "ic"
         else Ensemble.from_solvers(solvers, rhs=rhs))
    e.set_states(X0)
    e.run(nsteps)
    return e


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("scheme,kw", [("RK443", {}), ("RK222", {}),
                                       ("SBDF3", dict(startup="rk443")),
                                       ("SBDF2", dict(startup="lower_order"))])
def test_ks_initial_condition_ensemble(scheme, kw):
    plan, u = ks_1d(N=32, L=22.0)
    s = IVPSolver(plan, scheme, 0.02, **kw)
    X0 = 0.3 * rng.standard_normal((4, s.system.n))
    ref = np.array([run_single(s, x, 15) for x in X0])
    loop = _members("ic", [s], X0, 15, "loop")
    assert np.array_equal(loop.X, ref)                          # bit for bit
    bat = _members("ic", [s], X0, 15, "batched")
    assert _rel(bat.X, ref) < 1e-13
    one = _members("ic", [s], X0[2:3], 15, "batched")
    assert np.array_equal(one.X[0], bat.X[2])                   # member = B=1, same path
    assert "polynomial form" in bat.explain()


def test_rb2d_chebyshev_tau_rows():
    """Chebyshev direction, tau rows, constant boundary rows, C^(lam) conversion of the RHS."""
    plan, _ = rayleigh_benard_2d(8, 8, 2e3)
    s = IVPSolver(plan, "RK443", 1e-3)
    X0 = 0.05 * rng.standard_normal((3, s.system.n))
    ref = np.array([run_single(s, x, 5) for x in X0])
    assert np.array_equal(_members("ic", [s], X0, 5, "loop").X, ref)
    bat = _members("ic", [s], X0, 5, "batched")
    assert _rel(bat.X, ref) < 1e-12
    assert "constant RHS rows" in bat.rhs.explain()


@pytest.mark.slow                   # 2.4 s in the fast suite (qual_durations0)
def test_lorenz_parameter_ensemble_in_F():
    rs = [0.5, 15.0, 28.0, 99.96]
    ss = [IVPSolver(discretize(lorenz_problem(r)), "RK443", 5e-3) for r in rs]
    X0 = np.tile([1.0, 1.0, 20.0], (len(rs), 1))
    ref = np.array([run_single(s, x, 200) for s, x in zip(ss, X0)])
    assert np.array_equal(_members("par", ss, X0, 200, "loop").X, ref)
    bat = _members("par", ss, X0, 200, "batched")
    # one time unit of the chaotic members: rounding amplified by at most exp(lambda_1 t) ~ 3
    assert _rel(bat.X, ref) < 1e-12
    assert "1 with per-member coefficients" in bat.rhs.explain()


def test_burgers_parameter_in_L_own_factors():
    nus = [0.05, 0.1, 0.2]
    ss = [IVPSolver(burgers_1d(N=32, nu=nu)[0], "RK443", 1e-2) for nu in nus]
    X0 = 0.4 * rng.standard_normal((3, ss[0].system.n))
    ref = np.array([run_single(s, x, 20) for s, x in zip(ss, X0)])
    loop = _members("par", ss, X0, 20, "loop")
    assert np.array_equal(loop.X, ref)
    assert not loop.sys.shared and "distinct (M, L)" in loop.sys.explain()
    assert _rel(_members("par", ss, X0, 20, "batched").X, ref) < 1e-13


def test_refusals():
    plan, _ = ks_1d(N=16, L=22.0)
    a, b = IVPSolver(plan, "RK443", 0.02), IVPSolver(plan, "RK443", 0.01)
    with pytest.raises(OptionError, match="dt"):
        Ensemble.from_solvers([a, b], rhs="batched")
    with pytest.raises(OptionError, match="rhs"):
        Ensemble(a, 2, rhs="fast")
    e = Ensemble(a, 2, rhs="batched")
    with pytest.raises(OptionError, match="shape"):
        e.set_states(np.zeros((3, a.system.n)))
    with pytest.raises(OptionError, match="together"):
        e.run(2, callback=lambda _: None)
    from conatus.ir import IVP, CartesianCoordinates, Domain, Field, Parameter, RealFourier, dt, lap
    cs = CartesianCoordinates("x")
    D = Domain(cs, [RealFourier(cs["x"], 8, (0, 1), dealias=1.5)])
    u, t = Field("u", D), Parameter("t", 0.0)
    pb = IVP([u])
    pb.add_equation((dt(u) - lap(u), t * u))
    s = IVPSolver(discretize(pb), "RK443", 0.1, time_parameter="t")
    with pytest.raises(UnsupportedError, match="time-dependent"):
        Ensemble(s, 2, rhs="batched")



def test_changing_plan_params_after_construction_is_refused():
    """Audit A23: the members of an ensemble do not pass through IVPSolver.step, so the
    refusal of a plan.params change after construction did not cover them: the change was
    taken by F and not by the factored L, M, silently."""
    ss = [IVPSolver(discretize(lorenz_problem(r)), "RK443", 5e-3) for r in (15.0, 28.0)]
    e = Ensemble.from_solvers(ss, rhs="loop")
    e.set_states(np.tile([1.0, 1.0, 20.0], (2, 1)))
    e.run(2)
    ss[1].plan.params["r"] = 30.0
    with pytest.raises(OptionError, match=r"parameter set 1\): plan.params changed .*r: None -> 30.0"):
        e.step()

@pytest.mark.slow                   # 7.8 s in the fast suite (qual_durations0)
def test_fate_basins_lorenz_and_wilson_interval():
    """Lorenz r = 15 (C+ and C- stable, no chaos): every member ends near C+ or C-; the verdict of
    a member equals that of its single run; Wilson interval of a known proportion."""
    from conatus.ensembles import ensemble_fate, proportion_ci
    r, b = 15.0, 8.0 / 3.0
    c = np.sqrt(b * (r - 1))
    Cp, Cm = np.array([c, c, r - 1]), np.array([-c, -c, r - 1])
    s = IVPSolver(discretize(lorenz_problem(r)), "RK443", 0.01)
    X0 = np.column_stack([rng.uniform(-15, 15, 12), rng.uniform(-15, 15, 12), np.full(12, r - 1)])
    e = Ensemble(s, 12, rhs="batched")
    e.set_states(X0)

    def classify(ens):
        X = ens.states()
        dp, dm = np.linalg.norm(X - Cp, axis=1), np.linalg.norm(X - Cm, axis=1)
        return np.where(dp < 0.5, 1, np.where(dm < 0.5, -1, 0))

    fate = ensemble_fate(e, classify, check_every=100, min_steps=100, max_steps=6000,
                         classifier="|X - C+-| < 0.5")
    assert np.all(fate.labels != 0)
    for i in (0, 5, 11):
        Xs = run_single(s, X0[i], fate.settings["steps_run"])
        assert np.sign(np.linalg.norm(Xs - Cm) - np.linalg.norm(Xs - Cp)) == fate.labels[i]
    assert "undecided" in fate.explain()
    p, lo, hi = proportion_ci(7, 20)
    assert abs(p - 0.35) < 1e-15 and abs(lo - 0.1812) < 1e-3 and abs(hi - 0.5671) < 1e-3


@pytest.mark.parametrize("f_backend", ["evaluator", "fused"])
@pytest.mark.slow                   # 3.3 s in the fast suite (qual_durations0)
def test_couette_general_form_and_reynolds_ensemble(f_backend):
    """Plane Couette, poloidal-toroidal form: the RHS is a divergence of products, so the
    batched RHS uses the general form (products of linear forms, post-operators); Re enters L, so
    a Reynolds-number ensemble has one factorization per member. Reference: single runs with the
    tree evaluator and with the fused general form (the W6 production path)."""
    from couette_cases import BOX_GHC, couette_pt
    res = [300.0, 400.0]
    ss = [IVPSolver(couette_pt(6, 13, 6, re, BOX_GHC)[0], "RK443", 0.02, f_backend=f_backend,
                    decouple=True) for re in res]
    plan, f = couette_pt(6, 13, 6, 400.0, BOX_GHC)
    Lx, Lz = BOX_GHC
    X0 = []
    for b in range(2):        # laminar U = y plus smooth wall-compatible potentials (amplitude 0.1)
        c = rng.standard_normal((2, 2, 2))
        g = lambda X, Z, k, c=c: sum(c[k, m, n] * np.cos(2 * np.pi * (m * X / Lx + n * Z / Lz) + m)
                                     for m in range(2) for n in range(2))  # noqa: E731
        ss[0].set_state({f["U"]: lambda X, Y, Z: Y + 0 * X,
                         f["phi"]: lambda X, Y, Z: 0.1 * (1 - Y**2) ** 2 * g(X, Z, 0),
                         f["psi"]: lambda X, Y, Z: 0.1 * (1 - Y**2) * g(X, Z, 1)})
        X0.append(ss[0].X.copy())
    X0 = np.array(X0)
    ref = np.array([run_single(s, x, 6) for s, x in zip(ss, X0)])
    loop = _members("par", ss, X0, 6, "loop")
    assert np.array_equal(loop.X, ref)
    bat = _members("par", ss, X0, 6, "batched")
    assert bat.rhs.form == "general" and not bat.sys.shared
    assert _rel(bat.X, ref) < 1e-12
