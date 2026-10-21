"""Bifurcation core: every value against a closed form (table).

Systems are written in the IR (``ode_problem``); derivatives are the IR's symbolic Fréchet
derivatives. Closed forms: Brusselator (Hopf at b = 1 + a², cycle amplitude from the
Guckenheimer–Holmes planar formula, derived independently with sympy in the docstring of
``test_brusselator_hopf``), Lorenz (pitchfork r = 1, C± with x² = b(r − 1), Hopf 470/19),
normal forms of Kuznetsov (2004) ch. 3 and 8 (Hopf, Bautin, Bogdanov–Takens, cusp, fold–Hopf,
Hopf–Hopf), and constructed orbits with known multipliers (period doubling, Neimark–Sacker).
"""
import numpy as np
import pytest

from conatus.bifurcation import (CollocationMesh, PeriodicOrbitProblem, branch_switch,
                                 continue_codim2, continue_equilibria, continue_orbits, ir_system,
                                 normal_form, ode_problem)
from conatus.continuation import ContinuationSettings


def cfg(**kw):
    base = dict(ds=0.05, ds_min=1e-9, ds_max=0.3, adapt=1.3, fast_iterations=3, tolerance=1e-11,
                max_newton=12, max_points=40, dp_fd=None)
    base.update(kw)
    return ContinuationSettings(**base)


def brusselator(a=2.0, b=3.0):
    return ode_problem(["u", "v"], {"a": a, "b": b},
                       lambda X, P: [P["a"] - (P["b"] + 1) * X["u"] + X["u"] * X["u"] * X["v"],
                                     P["b"] * X["u"] - X["u"] * X["u"] * X["v"]], "Brusselator")


def lorenz(r=0.5):
    return ode_problem(["x", "y", "z"], {"sigma": 10.0, "r": r, "b": 8 / 3},
                       lambda X, P: [P["sigma"] * (X["y"] - X["x"]),
                                     P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                     X["x"] * X["y"] - P["b"] * X["z"]], "Lorenz")


def test_pointwise_and_evaluator_paths_agree():
    """The compiled 0-D path and the spectral evaluator give the same G, J, Gp, B, C."""
    a = ir_system(lorenz(), ["r", "sigma"], pointwise=True)
    b = ir_system(lorenz(), ["r", "sigma"], pointwise=False)
    rng = np.random.default_rng(0)
    x, u, v, w = rng.standard_normal((4, 3))
    for f in (lambda s: s.G(x), lambda s: s.J(x), lambda s: s.Gp(x, "r"), lambda s: s.Gp(x, "sigma"),
              lambda s: s.B(x, u, v), lambda s: s.C(x, u, v, w), lambda s: s.Jp(x, u, "sigma")):
        assert np.abs(f(a) - f(b)).max() < 1e-13
    # exact B of Lorenz: F'' [u, v] = (0, -(u_x v_z + v_x u_z), u_x v_y + v_x u_y)
    Bex = np.array([0, -(u[0] * v[2] + v[0] * u[2]), u[0] * v[1] + v[0] * u[1]])
    assert np.abs(a.B(x, u, v) - Bex).max() < 1e-14
    assert "pointwise=True" in a.explain() and "spectral evaluator" in b.explain()


def test_brusselator_hopf():
    """Hopf at b = 1 + a² (ω = a); cycle amplitude of u: A_u² = 4a² Δb / (a² + 2).

    Reference derived independently (sympy, Guckenheimer & Holmes 1983 eq. 3.4.11 in the normal
    coordinates of J): a_GH = −(a² + 2) / (8(a² + 1)) for eigenvector columns T; the invariant
    K_u = (T_u1² + T_u2²)/a_GH = −8a²/(a² + 2); with α' = 1/2, A_u² = −α' Δb K_u.
    Ours: K_u = 4|q_u|²/Re c1, and the collocation branch."""
    a = 2.0
    ds = ir_system(brusselator(a), ["b"])
    cv = continue_equilibria(ds, np.array([a, 1.5]), "b", cfg(ds=0.1), marginal_tol=1e-9, bounds=(0, 8))
    (h,) = cv.events_of("hopf")
    assert h.converged and abs(h.point.z[-1] - (1 + a * a)) < 1e-12
    nf = normal_form(ds, h.point.z[:-1], kind="hopf", par={"b": h.point.z[-1]}, parameter="b")
    assert abs(nf.omega - a) < 1e-12 and nf.criticality == "supercritical"
    assert abs(4 * abs(nf.q[0]) ** 2 / nf.c1.real - (-8 * a * a / (a * a + 2))) < 1e-12
    assert abs(nf.dlambda - 0.5) < 1e-12
    oc = branch_switch(cv, h, cfg(ds=0.02, ds_max=0.05, max_points=6, tolerance=1e-10),
                       amplitude=0.02, mesh=CollocationMesh.uniform(20, 4))
    pb = oc.context
    slope = []
    for pt in oc.points:
        U = pb.split(pt.z)[0]
        slope.append(((U[:, 0].max() - U[:, 0].min()) / 2) ** 2 / (pt.z[-1] - (1 + a * a)))
    # A_u² / Δb -> 4a²/(a² + 2) = 8/3 as Δb -> 0 (first point: Δb ~ 1e-4)
    assert abs(slope[0] - 8 / 3) < 2e-3
    assert max(p.data["trivial_error"] for p in oc.points) < 1e-8
    assert "Hopf → cycles" in oc.explain()


def test_lorenz_pitchfork_symmetry_and_hopf():
    """r = 1: symmetry-breaking pitchfork (S = diag(−1, −1, 1)); C±: x² = b (r − 1);
    Hopf at r_H = σ(σ + b + 3)/(σ − b − 1) = 470/19, ω² = b(σ + r_H), subcritical."""
    ds = ir_system(lorenz(), ["r"])
    S = np.diag([-1.0, -1.0, 1.0])
    cv = continue_equilibria(ds, np.zeros(3), "r", cfg(ds=0.1, ds_max=1.0, max_points=15),
                             marginal_tol=1e-9, bounds=(0, 3))
    (bp,) = cv.events_of("branch point")
    assert abs(bp.point.z[-1] - 1.0) < 1e-12
    nfp = normal_form(ds, bp.point.z[:-1], kind="zero", par={"r": 1.0}, parameter="r", symmetry=S)
    assert nfp.kind == "pitchfork" and nfp.a3 < 0
    assert abs(-nfp.mu / nfp.a3 * nfp.q[0] ** 2 - 8 / 3) < 1e-12
    nb = branch_switch(cv, bp, cfg(ds=0.1, ds_max=2.0, max_points=60), step=0.05, symmetry=S,
                       bounds=(0, 30))
    assert nb.switch["classification"] == "pitchfork (symmetry-breaking)"
    x = np.array([p.z[0] for p in nb.points])
    assert np.abs(x ** 2 - 8 / 3 * (nb.column("r") - 1)).max() < 1e-9
    (h,) = nb.events_of("hopf")
    assert abs(h.point.z[-1] - 470 / 19) < 1e-9
    nfh = normal_form(ds, h.point.z[:-1], kind="hopf", par={"r": h.point.z[-1]}, parameter="r")
    assert abs(nfh.omega ** 2 - 8 / 3 * (10 + 470 / 19)) < 1e-9     # ω² = b(σ + r_H)
    assert nfh.criticality == "subcritical"


@pytest.mark.parametrize("a,b", [(-0.7, 0.3), (0.4, -1.2)])
def test_hopf_normal_form(a, b):
    """ż = (μ + iω)z + (a + ib) z|z|² in z = x + iy: with ‖q‖ = 1, c1 = 2(a + ib), l1 = 2a/ω."""
    w = 1.7
    pb = ode_problem(["x", "y"], {"mu": 0.0},
                     lambda X, P: [P["mu"] * X["x"] - w * X["y"] + (a * X["x"] - b * X["y"]) * (X["x"] * X["x"] + X["y"] * X["y"]),
                                   w * X["x"] + P["mu"] * X["y"] + (b * X["x"] + a * X["y"]) * (X["x"] * X["x"] + X["y"] * X["y"])],
                     "Hopf normal form")
    ds = ir_system(pb, ["mu"])
    nf = normal_form(ds, np.zeros(2), kind="hopf", parameter="mu")
    assert abs(nf.c1 - 2 * (a + 1j * b)) < 1e-12
    assert abs(nf.l1 - 2 * a / w) < 1e-12 and abs(nf.dlambda - 1) < 1e-12
    assert "first Lyapunov coefficient" in nf.explain()


def test_transcritical_and_fold():
    """ẋ = p x − x² (transcritical at 0, a2 = −1); ẋ = p − x² (fold at p = 0, β = 1, a2 = −1)."""
    ds = ir_system(ode_problem(["x", "y"], {"p": -0.5},
                               lambda X, P: [P["p"] * X["x"] - X["x"] * X["x"], -X["y"]]), ["p"])
    cv = continue_equilibria(ds, np.zeros(2), "p", cfg(ds=0.1), marginal_tol=1e-9, bounds=(-1, 1))
    (bp,) = cv.events_of("branch point")
    nf = normal_form(ds, bp.point.z[:-1], kind="zero", par={"p": bp.point.z[-1]}, parameter="p")
    assert nf.kind == "transcritical" and abs(nf.a2 * nf.q[0] + 1) < 1e-12 and abs(nf.mu - 1) < 1e-12
    nb = branch_switch(cv, bp, cfg(ds=0.1), step=0.05, bounds=(-1, 1))
    assert nb.switch["classification"] == "transcritical"
    assert np.abs(np.array([p.z[0] for p in nb.points]) - nb.column("p")).max() < 1e-10   # x = p
    ds2 = ir_system(ode_problem(["x"], {"p": 1.0}, lambda X, P: [P["p"] - X["x"] * X["x"]]), ["p"])
    c2 = continue_equilibria(ds2, np.array([1.0]), "p", cfg(ds=0.1), marginal_tol=1e-9, direction=-1,
                             bounds=(-1, 2))
    (f,) = c2.events_of("fold")
    assert abs(f.point.z[-1]) < 1e-12
    nf2 = normal_form(ds2, f.point.z[:-1], kind="zero", par={"p": f.point.z[-1]}, parameter="p")
    assert nf2.kind == "fold" and abs(nf2.beta - 1) < 1e-12 and abs(nf2.a2 + 1) < 1e-12


def test_bogdanov_takens_curves():
    """ẋ = y, ẏ = β1 + β2 x + x² − x y (Kuznetsov 2004, §8.4): fold β1 = β2²/4, Hopf β1 = 0 (β2 < 0),
    both end at the BT point (0, 0)."""
    pb = ode_problem(["x", "y"], {"b1": -0.5, "b2": -1.0},
                     lambda X, P: [X["y"], P["b1"] + P["b2"] * X["x"] + X["x"] * X["x"] - X["x"] * X["y"]], "BT")
    ds = ir_system(pb, ["b1", "b2"])
    cv = continue_equilibria(ds, np.array([(1 - np.sqrt(3)) / 2, 0.0]), "b1", cfg(), marginal_tol=1e-9,
                             bounds=(-2, 2))
    (f,) = cv.events_of("fold")
    (h,) = cv.events_of("hopf")
    fc = continue_codim2(ds, f.point.z[:-1], "fold", ("b1", "b2"), cfg(), par={"b1": f.point.z[-1]},
                         bounds={"b2": (-1.5, 1.5)})
    assert np.abs(fc.column("b1") - fc.column("b2") ** 2 / 4).max() < 1e-10     # corrector tol 1e-11
    (bt,) = fc.events_of("bt")
    assert np.abs(bt.point.z[-2:]).max() < 1e-10
    hc = continue_codim2(ds, h.point.z[:-1], "hopf", ("b1", "b2"), cfg(), par={"b1": h.point.z[-1]},
                         bounds={"b2": (-1.5, 1.5)})
    assert np.abs(hc.column("b1")).max() < 1e-12
    bts = hc.events_of("bt")
    assert len(bts) == 1 and np.abs(bts[0].point.z[-2:]).max() < 1e-11
    assert not hc.events_of("zh")          # det J = 0 at BT is not a fold-Hopf point
    assert "minimally augmented" in hc.explain()


def test_bogdanov_takens_hopf_curve_has_an_imaginary_pair():
    """Defining condition of a Hopf point, from the analytic Jacobian [[0, 1], [b2 + 2x - y, -x]]
    (independent of the augmented system): a pair +-i omega with omega > 0 at every point of the
    Hopf curve away from the BT point (the tutorial checked only b1 = 0)."""
    pb = ode_problem(["x", "y"], {"b1": -0.5, "b2": -1.0},
                     lambda X, P: [X["y"], P["b1"] + P["b2"] * X["x"] + X["x"] * X["x"] - X["x"] * X["y"]], "BT")
    ds = ir_system(pb, ["b1", "b2"])
    cv = continue_equilibria(ds, np.array([(1 - np.sqrt(3)) / 2, 0.0]), "b1", cfg(), marginal_tol=1e-9,
                             bounds=(-2, 2))
    (h,) = cv.events_of("hopf")
    hc = continue_codim2(ds, h.point.z[:-1], "hopf", ("b1", "b2"), cfg(), par={"b1": h.point.z[-1]},
                         bounds={"b2": (-1.5, 1.5)})
    b2 = hc.column("b2")
    checked = 0
    for pt, q in zip(hc.points, b2, strict=True):
        if abs(q) < 1e-8:                              # the BT point itself: double zero
            continue
        x, y = pt.z[:2]
        ev = np.linalg.eigvals(np.array([[0.0, 1.0], [q + 2 * x - y, -x]]))
        assert np.max(np.abs(ev.real)) < 1e-8 and np.min(np.abs(ev.imag)) > 1e-8, (q, ev)
        checked += 1
    assert checked >= 3


def test_cusp():
    """ẋ = β1 + β2 x − x³: fold curve 27 β1² = 4 β2³, cusp at (0, 0)."""
    ds = ir_system(ode_problem(["x"], {"b1": 0.0, "b2": 1.0},
                               lambda X, P: [P["b1"] + P["b2"] * X["x"] - X["x"] ** 3]), ["b1", "b2"])
    cv = continue_equilibria(ds, np.array([1.0]), "b1", cfg(), marginal_tol=1e-9, direction=-1,
                             bounds=(-2, 2))
    f = cv.events_of("fold")[0]
    fc = continue_codim2(ds, f.point.z[:-1], "fold", ("b1", "b2"), cfg(), par={"b1": f.point.z[-1]},
                         direction=-1, bounds={"b2": (-0.5, 1.5)})
    assert np.abs(27 * fc.column("b1") ** 2 - 4 * fc.column("b2") ** 3).max() < 1e-10
    (c,) = fc.events_of("cusp")
    assert np.abs(c.point.z[-2:]).max() < 1e-9


def test_bautin():
    """ż = (β1 + i) z + β2 z|z|² − z|z|⁴: Hopf curve β1 = 0 with l1 = 2β2, Bautin at β2 = 0."""
    def rhs(X, P):
        x, y = X["x"], X["y"]
        r2 = x * x + y * y
        return [P["b1"] * x - y + P["b2"] * x * r2 - x * r2 * r2,
                x + P["b1"] * y + P["b2"] * y * r2 - y * r2 * r2]
    ds = ir_system(ode_problem(["x", "y"], {"b1": 0.0, "b2": -0.5}, rhs, "Bautin"), ["b1", "b2"])
    hc = continue_codim2(ds, np.zeros(2), "hopf", ("b1", "b2"), cfg(), bounds={"b2": (-1, 1)})
    (gh,) = hc.events_of("gh")
    assert abs(gh.point.z[-1]) < 1e-11 and abs(gh.point.z[-2]) < 1e-12
    l1 = np.array([p.data["l1"] for p in hc.points])
    assert np.abs(l1 - 2 * hc.column("b2")).max() < 1e-11


def test_fold_hopf_and_hopf_hopf():
    """Fold–Hopf (Kuznetsov §8.5 truncated form, θ = 1): ẋ = β1 + x² + |z|², ż = (β2 + i + x) z + x² z:
    fold curve β1 = 0 with ZH at (0, 0). Hopf–Hopf: two oscillators ż_k = (β_k + iω_k) z_k − z_k|z_k|²,
    ω = (1, √2): Hopf curve β1 = 0 crosses β2 = 0 (HH)."""
    def zh(X, P):
        x, u, v = X["x"], X["u"], X["v"]
        return [P["b1"] + x * x + u * u + v * v,
                (P["b2"] + x + x * x) * u - v, u + (P["b2"] + x + x * x) * v]
    ds = ir_system(ode_problem(["x", "u", "v"], {"b1": 0.0, "b2": -0.5}, zh, "fold-Hopf"), ["b1", "b2"])
    fc = continue_codim2(ds, np.zeros(3), "fold", ("b1", "b2"), cfg(), bounds={"b2": (-1, 1)})
    (z,) = fc.events_of("zh")
    assert np.abs(z.point.z[-2:]).max() < 1e-11 and np.abs(fc.column("b1")).max() < 1e-12
    w2 = np.sqrt(2.0)

    def hh(X, P):
        x1, y1, x2, y2 = X["x1"], X["y1"], X["x2"], X["y2"]
        r1, r2 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2
        return [P["b1"] * x1 - y1 - x1 * r1, x1 + P["b1"] * y1 - y1 * r1,
                P["b2"] * x2 - w2 * y2 - x2 * r2, w2 * x2 + P["b2"] * y2 - y2 * r2]
    ds2 = ir_system(ode_problem(["x1", "y1", "x2", "y2"], {"b1": 0.0, "b2": -0.5}, hh, "Hopf-Hopf"),
                    ["b1", "b2"])
    hc = continue_codim2(ds2, np.zeros(4), "hopf", ("b1", "b2"), cfg(), omega=1.0, bounds={"b2": (-1, 1)})
    (e,) = hc.events_of("hh")
    assert abs(e.point.z[-1]) < 1e-11 and "1.414213562" in e.detail


def _pd_ns_system(w=1.0, Om=0.3):
    """Oscillator (c, s) on the unit circle, period 2π/w, with a transverse plane (u, v) rotated by
    half the phase: multipliers −e^{μT}, −e^{−T} (period doubling at μ = 0, doubled orbit a² = μ,
    period 4π/w); and a plane (p, q) with ż = (ν + iΩ) z − z|z|² (Neimark–Sacker at ν = 0,
    angle 2πΩ/w)."""
    def rhs(X, P):
        c, s, u, v, p, q = (X[k] for k in "csuvpq")
        rr, uu, pp = 1 - c * c - s * s, u * u + v * v, p * p + q * q
        mu, nu = P["mu"], P["nu"]
        return [-w * s + c * rr, w * c + s * rr,
                -0.5 * w * v + 0.5 * (mu - 1) * u + 0.5 * (mu + 1) * (c * u + s * v) - uu * u,
                0.5 * w * u + 0.5 * (mu - 1) * v + 0.5 * (mu + 1) * (s * u - c * v) - uu * v,
                nu * p - Om * q - pp * p, Om * p + nu * q - pp * q]
    return ode_problem(list("csuvpq"), {"mu": -0.3, "nu": -0.2}, rhs, "PD/NS construction")


def _base_orbit(pb: PeriodicOrbitProblem, w=1.0):
    tau = pb.mesh.times()
    U = np.zeros((tau.size, 6))
    U[:, 0], U[:, 1] = np.cos(2 * np.pi * tau), np.sin(2 * np.pi * tau)
    return U


def test_period_doubling_and_switch():
    ds = ir_system(_pd_ns_system(), ["mu", "nu"])
    pb = PeriodicOrbitProblem(ds, "mu", CollocationMesh.uniform(24, 4))
    z0 = np.concatenate([_base_orbit(pb).ravel(), [2 * np.pi, -0.3]])
    oc = continue_orbits(pb, z0, cfg(ds=0.1, max_points=12, tolerance=1e-10), bounds={"mu": (-1, 0.5)})
    (pd,) = oc.events_of("pd")
    assert abs(pd.point.z[-1]) < 1e-9          # μ = 0 up to the collocation error of the multipliers
    new = branch_switch(oc, pd, cfg(ds=0.02, max_points=8, tolerance=1e-10), amplitude=0.05,
                        bounds={"mu": (-0.1, 0.3)})
    pb2 = new.context
    assert len(new.points) > 2                  # the loop below must check something
    for pt in new.points[1:]:
        U, T, mu = pb2.split(pt.z)
        a2 = np.max(U[:, 2] ** 2 + U[:, 3] ** 2)
        assert abs(T - 4 * np.pi) < 1e-8 and abs(a2 - mu) < 1e-8
    assert "period doubling" in new.explain()


def test_neimark_sacker():
    ds = ir_system(_pd_ns_system(), ["mu", "nu"])
    pb = PeriodicOrbitProblem(ds, "nu", CollocationMesh.uniform(24, 4))
    z0 = np.concatenate([_base_orbit(pb).ravel(), [2 * np.pi, -0.2]])
    oc = continue_orbits(pb, z0, cfg(ds=0.1, max_points=8, tolerance=1e-10), bounds={"nu": (-1, 0.3)})
    (ns,) = oc.events_of("ns")
    assert abs(ns.point.z[-1]) < 1e-9
    assert abs(ns.data["angle"] - 2 * np.pi * 0.3) < 1e-8


def test_unresolved_multipliers_do_not_trigger_events():
    """With a strongly unstable direction (multiplier e^{5T} ~ 4e13), the small multipliers of
    the collocation monodromy carry round-off ~ eps |μ|max and flipped the pd / ns tests at random
    (Lorenz near its homoclinic orbit, run). They are now counted as
    unresolved and left out of the tests (declared)."""
    def rhs(X, P):
        c, s, w, p = X["c"], X["s"], X["w"], X["p"]
        rr = 1 - c * c - s * s
        return [-s + c * rr, c + s * rr, 5.0 * w, P["nu"] * p]
    ds = ir_system(ode_problem(["c", "s", "w", "p"], {"nu": -0.3}, rhs, "unresolved"), ["nu"])
    pb = PeriodicOrbitProblem(ds, "nu", CollocationMesh.uniform(24, 4))
    tau = pb.mesh.times()
    U = np.zeros((tau.size, 4))
    U[:, 0], U[:, 1] = np.cos(2 * np.pi * tau), np.sin(2 * np.pi * tau)
    oc = continue_orbits(pb, np.concatenate([U.ravel(), [2 * np.pi, -0.3]]),
                         cfg(ds=0.1, max_points=6, tolerance=1e-10), bounds={"nu": (-1, 0.2)})
    assert oc.points and all(p.data["unresolved"] >= 1 for p in oc.points)
    assert not oc.events_of("pd") and not oc.events_of("ns")
    assert "unresolved" in oc.explain()


def test_branch_point_location_prints_no_lapack_warning():
    """Locating the Lorenz pitchfork (r = 1) factors an exactly singular bordered matrix (expected
    at a branch point): the driver takes the chord direction and prints nothing (tutorial 10
    showed a scipy LinAlgWarning, with a local path, in its output)."""
    import warnings

    from scipy.linalg import LinAlgWarning
    ds = ir_system(lorenz(), ["r"])
    c = cfg(ds=0.1, ds_max=2.0, fast_iterations=4, max_points=60)
    with warnings.catch_warnings(record=True) as caught:   # recorded, not raised: an exception
        warnings.simplefilter("always")                     # would change the path under test
        cv = continue_equilibria(ds, np.zeros(3), "r", c, marginal_tol=1e-9, bounds=(0, 3))
    assert not [w for w in caught if issubclass(w.category, LinAlgWarning)]
    (bp,) = cv.events_of("branch point")
    assert abs(bp.point.z[-1] - 1.0) < 1e-10
    assert "chord" in bp.point.status          # the substituted tangent is declared


@pytest.mark.parametrize("sparse", [False, True])
def test_branch_point_on_a_singular_bordered_matrix_dense_and_sparse(sparse):
    """Final review: at an exact branch point the bordered matrix is singular. Dense LU
    warned (chord tangent taken, not recorded); sparse LU (splu) raised 'Factor is exactly
    singular', reported as a corrector failure (event not converged). Both now take the chord
    of the bracket and say so in the point status."""
    import scipy.sparse as sps

    from conatus.bifurcation.driver import ArclengthDriver, CurvePoint

    def Hz(z):
        A = np.array([[z[1] - 3 * z[0] ** 2, z[0]]])
        return sps.csr_matrix(A) if sparse else A
    d = ArclengthDriver(lambda z: np.array([z[1] * z[0] - z[0] ** 3]), Hz, cfg(ds=0.1),
                        tests={"bp": lambda pt: pt.z[1]})
    a = CurvePoint(0.0, np.array([0.0, -0.5]), np.array([0.0, 1.0]), 0.0, 0, tests={"bp": -0.5})
    b = CurvePoint(1.0, np.array([0.0, 0.5]), np.array([0.0, 1.0]), 0.0, 0, tests={"bp": 0.5})
    ev = d.locate("bp", a, b, (0, 1))
    assert ev.converged and abs(ev.point.z[1]) < 1e-14
    assert "chord" in ev.point.status
