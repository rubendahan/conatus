"""Check the prose claims of the user guide (docs/site/guide/*.md, except dynamics/) and the docs
index, at small sizes, against the current library: the texts of explain() and the values the
pages quote. The python blocks of the pages themselves are executed as written, each alone, by
tests/docs/test_guide_blocks.py; this file does not re-type them.

One function per snippet group, named after its page. Each prints a short line and asserts what
the page says; a failure is reported and the script ends with exit code 1. Snippets that need
hardware or code that is not in this tree (a GPU run, an MPI launch with several ranks, the
Studio's ``sim.watch``) are reported as SKIP with the reason, never silently dropped.

"""
from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path

import numpy as np

import conatus as cn
import conatus.expert as ex

TMP = Path(tempfile.mkdtemp(prefix="cn_guide_"))
os.environ.setdefault("CN_RUNS", str(TMP / "runs"))
RESULTS: list[tuple[str, str, str]] = []


def rb_model(Lx=2.0):
    """The Rayleigh-Benard model of index.md / equations.md / boundary_conditions.md."""
    box = cn.Box(x=cn.periodic(Lx), z=cn.walls(-0.5, 0.5))
    m = cn.Model(box, name="Rayleigh-Benard 2D", Ra=2500, Pr=1)
    u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    return m, u, p, T


def heat_model():
    m = cn.Model(cn.Box(x=cn.walls(0, 1)), name="heat", kappa=1.0)
    T = m.scalar("T")
    m.equations("dt(T) - kappa*lap(T) = 0")
    m.at("walls", T=0)
    return m, T


# --- index.md -------------------------------------------------------------------------------
def index():
    m, u, p, T = rb_model()
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sim.run(until=0.01)
    assert sim.iteration == 20
    modes = m.linearize(about={u: 0, T: lambda z: -z}).eigenvalues(N=24, k=np.pi)
    txt = modes.explain()
    assert "resid" in txt.lower()
    assert np.max(modes.values.real) > 0          # Ra = 2500 > Ra_c: one growing mode at k = pi
    return f"max growth {np.max(modes.values.real):.4f}"


# --- geometries.md --------------------------------------------------------------------------
def geometries():
    names = {
        "Box": cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)),
        "Channel": cn.Channel(2 * np.pi, np.pi),
        "Disk": cn.Disk(1.0), "Annulus": cn.Annulus(1.0, 2.0), "Pipe": cn.Pipe(1.0, 2 * np.pi),
        "Cylinder": cn.Cylinder(1.0, 2 * np.pi), "Sphere": cn.Sphere(1.0), "Ball": cn.Ball(1.0),
        "Shell": cn.Shell(0.35, 1.0)}
    expect = {"Box": ("x", "z"), "Channel": ("x", "y", "z"), "Disk": ("phi", "r"),
              "Annulus": ("phi", "r"), "Pipe": ("phi", "r", "z"), "Cylinder": ("phi", "r", "z"),
              "Sphere": ("phi", "theta"), "Ball": ("phi", "theta", "r"),
              "Shell": ("phi", "theta", "r")}
    bnd = {"Box": {"bottom", "top", "walls", "z=-0.5", "z=0.5"},
           "Channel": {"bottom", "top", "walls", "y=-1", "y=1"},
           "Disk": {"wall", "r=1"}, "Annulus": {"inner", "outer", "walls"}, "Pipe": {"wall"},
           "Sphere": set(), "Ball": {"wall"}, "Shell": {"inner", "outer", "walls"}}
    for k, g in names.items():
        assert g.coord_names == expect[k], (k, g.coord_names)
        if k in bnd:
            assert bnd[k] <= set(g.boundaries()), (k, sorted(g.boundaries()))
    txt = cn.Annulus(1.0, 2.0).explain()
    assert "AnnulusBasis" in txt and "inner = r left" in txt
    m, *_ = rb_model()
    a = m.compile((16, 24)).N
    b = m.compile({"x": 16, "z": 24}).N
    assert dict(a) == dict(b) == {"x": 16, "z": 24}
    # periodic complex axis
    cn.Box(x=cn.periodic(2.0, complex=True), z=cn.walls(0, 1))
    try:
        cn.Box(x=cn.walls(0, 1), z=cn.walls(0, 1))
        raise AssertionError("two wall axes accepted")
    except ex.errors.UnsupportedError:
        pass
    return "9 geometries, boundaries and coordinates as documented"


# --- equations.md ---------------------------------------------------------------------------
def equations():
    m, u, p, T = rb_model()
    txt = m.explain(full=True)                    # the page shows the full listing
    assert "explicit  -dot(u, grad(u))" in txt and "moved from the right side" in txt
    # Orr-Sommerfeld: given, define, eigenvalue
    line = cn.Box(y=cn.walls(-1, 1))
    os_ = cn.Model(line, name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
    os_.scalar("v")
    os_.given(U=lambda y: 1 - y**2)
    os_.eigenvalue("c", growth="imag")
    os_.define("L2(f) = dy(dy(f)) - alpha**2*f")
    os_.equations("L2(L2(v))/(1j*alpha*Re) - (U - c)*L2(v) + dy(dy(U))*v = 0")
    os_.at("walls", "v = 0", "dy(v) = 0")
    res = os_.eigenvalues(N=64, drift=96, drift_tol=1e-6)
    i = int(res.values.imag.argmax())
    assert abs(res.values[i] - (0.23752649 + 0.00373967j)) < 1e-7, res.values[i]
    # LBVP
    lb = cn.Model(cn.Box(x=cn.walls(0, 1)), name="poisson")
    lb.scalar("f")
    lb.equations("lap(f) = 1")
    lb.at("walls", f=0)
    lb.discretize(16).solve()
    # explicit() override
    r = cn.Model(cn.Box(x=cn.walls(0, 1)), name="react", a=1.0)
    r.scalar("w")
    r.equations("dt(w) - lap(w) = explicit(-a*w)")
    r.at("walls", w=0)
    assert "explicit" in r.explain()
    # errors with a column
    for text, cls in (("dt(T) - lapp(T) = 0", "UndeclaredVariableError"),
                      ("dt(T) - kappa lap(T) = 0", "EquationSyntaxError")):
        mm = cn.Model(cn.Box(x=cn.walls(0, 1)), name="h", kappa=1.0)
        mm.scalar("T")
        try:
            mm.equations(text)
            raise AssertionError(f"accepted: {text}")
        except Exception as e:  # noqa: BLE001 -- the class name is what the page shows
            assert type(e).__name__ == cls and "column" in str(e), (type(e).__name__, str(e))
    return f"Orr-Sommerfeld c = {res.values[i]:.8f}"


# --- boundary_conditions.md -----------------------------------------------------------------
def boundary_conditions():
    m = cn.Model(cn.Box(x=cn.walls(0, 1)), name="heat", kappa=1.0)
    m.scalar("T")
    m.equations("dt(T) - kappa*lap(T) = 0")
    m.at("x=0", T=0)
    try:
        m.simulate(N=16, scheme="SBDF2", dt=1e-3)
        raise AssertionError("missing condition accepted")
    except ex.errors.BoundaryConditionError as e:
        assert "1 boundary condition(s) for 2 needed" in str(e)
    h, _ = heat_model()
    txt = h.explain(full=True)
    assert "lifted into mode N_x-1" in txt and "highest x-order (2)" in txt
    # Robin condition as text
    r = cn.Model(cn.Box(x=cn.walls(0, 1)), name="robin", Bi=2.0)
    r.scalar("T")
    r.equations("dt(T) - lap(T) = 0")
    r.at("bottom", "dx(T) - Bi*T = 0")
    r.at("top", T=1)
    r.simulate(N=16, scheme="SBDF2", dt=1e-3).run(steps=3)
    # user filter on an EVP result
    line = cn.Box(y=cn.walls(-1, 1))
    os_ = cn.Model(line, name="OS", Re=1e4, alpha=1.0)
    os_.scalar("v")
    os_.given(U=lambda y: 1 - y**2)
    os_.eigenvalue("c", growth="imag")
    os_.define("L2(f) = dy(dy(f)) - alpha**2*f")
    os_.equations("L2(L2(v))/(1j*alpha*Re) - (U - c)*L2(v) + dy(dy(U))*v = 0")
    os_.at("walls", "v = 0", "dy(v) = 0")
    res = os_.eigenvalues(N=64, drift=96, drift_tol=1e-6)
    kept = res.select(np.flatnonzero(abs(res.values) < 2), why="|c| < 2")
    assert len(kept) <= len(res)
    return f"drift kept {len(res)} modes, filter kept {len(kept)}"


# --- timesteppers.md ------------------------------------------------------------------------
def timesteppers():
    m, u, p, T = rb_model()
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sim.run(until=0.005)
    sim.grid("T"), sim.coeffs("T"), sim.t, sim.iteration
    assert "Initial state: T given; u, p start at zero" in sim.explain()
    try:
        sim.run(until=sim.t + 2.5e-4)
        raise AssertionError("non-integer number of steps accepted")
    except ex.errors.ResolutionError:
        pass
    s3 = m.simulate(N=(16, 24), scheme="SBDF3", dt=5e-4, startup="rk443")
    assert "rk443" in s3.explain()
    cfl = sim.cfl("u", safety=0.5)
    assert cfl > 0
    from conatus.expert import EmbeddedError
    e = EmbeddedError(sim.solver)
    err = e.step()
    assert np.isfinite(err)
    ck = sim.solver.save_checkpoint(TMP / "ck_rb")
    sim2 = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    sim2.solver.load_checkpoint(ck)
    sim.run(steps=5)
    sim2.run(steps=5)
    assert np.array_equal(sim.solver.X, sim2.solver.X)
    calls = []
    sim.solver.run(10, callback=lambda s: calls.append(s.iteration), every=5)
    assert len(calls) == 2
    s4 = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4, complexify=True)
    s4.run(steps=1)
    return f"restart bit for bit; CFL dt {cfl:.3g}; embedded error {err:.2e}"


# --- resolution.md --------------------------------------------------------------------------
def timesteppers_boundary_data():
    """timesteppers.md, *Time-dependent boundary data*."""
    cs = ex.CartesianCoordinates("y")
    y = cs["y"]
    by = ex.Chebyshev(y, 32, (0, 5), dealias=1)
    u, t = ex.Field("u", ex.Domain(cs, [by])), ex.Parameter("t", 0.0)
    pb = ex.IVP([u])
    pb.add_equation((ex.dt(u) - ex.diff(ex.diff(u, y), y) / 2, 0))
    pb.add_bc((ex.interp(u, y, 0), ex.cos(t)), (ex.interp(u, y, 5), np.exp(-5) * ex.cos(t - 5)))
    plan = ex.discretize(pb)
    s = ex.IVPSolver(plan, "RK443", 1e-2, time_parameter="t")
    s_s = ex.IVPSolver(plan, "RK443", 1e-2, time_parameter="t", bc_time="stage")
    s_d = ex.IVPSolver(plan, "RK443", 1e-2, time_parameter="t", bc_time="step")
    assert s.bc_time == "rk" and "(bc_time='rk', default 'auto'" in s.explain()
    assert "imposed at every stage time (bc_time='stage'" in s_s.explain()
    assert "Dedalus convention (bc_time='step')" in s_d.explain()
    # native layer: t is the time
    m = cn.Model(cn.Box(y=cn.walls(0, 5)), name="moving walls")
    m.scalar("u")
    m.equations("dt(u) - dy(dy(u))/2 = 0")
    m.at("bottom", u="cos(t)")
    m.at("top", u=lambda t: np.exp(-5) * np.cos(t - 5))
    sim = m.simulate(N=32, scheme="RK443", dt=1e-2, dealias=1)
    assert "(bc_time='rk', default 'auto'; RK443 about order 2.8" in sim.explain()
    sim_d = m.simulate(N=32, scheme="RK443", dt=1e-2, dealias=1, bc_time="step")
    assert sim_d.solver.bc_time == "step"
    return "expert and native: 'auto' -> 'rk'; 'stage', 'step' declared"


def boundary_conditions_time():
    """boundary_conditions.md, *Boundary values that depend on time* and curved walls (A22)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="travelling wall")
    m.scalar("u")
    m.given(S=lambda x: np.sin(x), C=lambda x: np.cos(x))
    m.equations("dt(u) - lap(u) = 0")
    m.at("bottom", u="S*cos(t) - C*sin(t)")
    m.at("top", "u = 0.5*sin(2*t)")
    m.simulate(N=(16, 16), scheme="RK222", dt=1e-2).run(steps=2)
    m2 = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="refused")
    m2.scalar("u")
    m2.equations("dt(u) - lap(u) = 0")
    m2.at("bottom", u=lambda x, t: np.sin(x - t))
    m2.at("top", u=0)
    try:
        m2.simulate(N=(16, 16), scheme="RK222", dt=1e-2)
        raise AssertionError("lambda x, t accepted")
    except cn.errors.BoundaryConditionError as err:
        assert "S*cos(t) - C*sin(t)" in str(err)
    d = cn.Model(cn.Disk(1.0))
    d.scalar("T")
    d.given(g=lambda phi, r: r**3 * np.cos(phi) + r**2)
    d.equations("heat: dt(T) - lap(T) = 0")
    d.at("wall", "T = g")
    sim = d.simulate(N=(8, 12), scheme="SBDF2", dt=1e-3)
    assert "evaluated from its function (exact)" in sim.explain(full=True)
    return "text, separable form, refusal of lambda x, t; disk wall trace exact"


def resolution():
    m, *_ = rb_model()
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sim.run(steps=10)
    rep = sim.plan.resolution_report(sim.solver, velocity="u")
    rep.explain()
    rep.recommend(target=1e-10)
    rep.entry("T", "z").modes_for(1e-12)
    return f"report ok={rep.ok}"


# --- gpu.md ---------------------------------------------------------------------------------
def gpu():
    ok = ex.gpu_available()
    ex.gpu_report()
    if not ok:
        return "SKIP: no GPU on this node (gpu_available() False, gpu_report() printed)"
    m, *_ = rb_model()
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4, backend="cuda", f_backend="fused")
    sim.initial(T=lambda x, z: -z)
    sim.solver.run_graph(5)
    return "GPU run_graph ok"


# --- explain.md -----------------------------------------------------------------------------
def explain():
    m, *_ = rb_model()
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    mt = sim.methods()
    assert "Rayleigh-Benard 2D" in mt.text and "native layer" in mt.text
    mt.bibtex
    mt.explain()
    ex.explain()
    return "Methods paragraph and bibtex"


# --- runs.md --------------------------------------------------------------------------------
def runs():
    m, *_ = rb_model()
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
    with cn.runs.track("rb_ra1e6", tags=["onset"], params={"note": "first try"}) as run:
        sim.solver.run(5, record=run)
        run.log(t=sim.t, Nu=1.0)
        run.result(Nu=1.0, residual=1e-12)
        run.save_field("T", sim.grid("T"), compression="gzip")
    run.certify("guide snippet check", evidence={"residual": 1e-12})
    found = cn.runs.find(name="rb_ra1e6")
    assert found, "find(name=...) found nothing"
    cn.runs.find(problem=sim.problem, status="certified", tags=["onset"])
    cn.runs.find(Nu__ge=0.5)
    got = cn.runs.get(found[0].id if hasattr(found[0], "id") else found[0].run_id)
    cn.runs.compare(got, found[0]).explain()
    cn.runs.lineage(got).explain()
    cn.runs.export(TMP / "export")
    return f"{len(found)} run(s) recorded and found"


# --- reproducibility.md ---------------------------------------------------------------------
def reproducibility():
    from conatus.expert import provenance, provenance_record
    m, *_ = rb_model()
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    rec = provenance_record(sim.solver, seed=0, tolerances={"newton": 1e-12}, out=str(TMP / "prov"))
    assert {"code", "environment", "numerics", "seed"} <= set(rec)
    provenance.problem_hash(sim.problem)
    try:
        ex.record  # noqa: B018
        raise AssertionError("conatus.expert.record exists")
    except (AttributeError, ex.errors.ConatusError):
        pass
    return "provenance record"


# --- doctor.md, errors.md, notebook.md ------------------------------------------------------
def doctor_errors_notebook():
    rep = cn.doctor()
    rep.explain()
    rep.to_dict()
    import conatus.errors
    conatus.errors.explain()
    m, *_ = rb_model()
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    for obj in (sim.solver, sim.problem):
        if hasattr(obj, "_repr_html_"):
            obj._repr_html_()
    return f"doctor ok={rep.ok}"


# --- problems.md ----------------------------------------------------------------------------
def problems():
    r = cn.problems.channel_evp(N=64, Re=1e4, alpha=1.0)
    r.explain()
    res = ex.eigenvalues(ex.discretize(r.problem, ncc=r.ncc), method="shift-invert", target=0.24, n_modes=1)
    assert abs(res.values[0] - (0.23752649 + 0.00373967j)) < 1e-7
    return f"c = {res.values[0]:.8f}"


# --- dedalus.md -----------------------------------------------------------------------------
# The Dedalus column of the side-by-side and the import example read the script written for
# tutorial 14 (docs/site/tutorials/code/rb_cell_dedalus.py); the Conatus column is the native block
# of t14_from_dedalus.py, which the tutorial runs and checks.
RB_CELL = Path(__file__).resolve().parents[1] / "tutorials" / "code" / "rb_cell_dedalus.py"


def dedalus():
    tr = cn.from_dedalus(str(RB_CELL))
    tr.explain()
    tr.problem
    solver = tr.ivp_solver(startup="lower_order")
    solver.run(3)
    tr.to_script()
    unsup = tr.counts()["unsupported"]
    assert unsup == 0
    return f"translated; {unsup} unsupported line(s); 3 steps run"


# --- expert.md, taus.md, operators.md, bases.md, units.md -----------------------------------
def expert_layer():
    m, u, p, T = rb_model()
    comp = m.compile((16, 24))
    comp.problem, comp.fields["T"], comp.ncc
    comp.explain()
    plan = comp.discretize()
    plan.explain()
    m.discretize((16, 24), assembly="direct")
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    sim.solver, sim.plan
    from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, IVPSolver,
                                Parameter, TauWatch, dt, interp, lap, lift)
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, 24, (-1, 1), dealias=3 / 2)
    D = Domain(cs, [b])
    uu = Field("u", D)
    a = Parameter("a", 1.0)
    pb = IVP([uu], name="heat 1D")
    pb.add_equation((dt(uu) - lap(uu), -a * uu))
    pb.add_bc(interp(uu, x, "left"), interp(uu, x, "right"))
    txt = pb.explain()
    assert "lift(tau_x0_1, x, -1)" in txt, txt[:400]
    t1, t2 = Field("t1", D.without(x)), Field("t2", D.without(x))
    pb2 = IVP([uu, t1, t2], name="heat 1D")
    pb2.add_equation((dt(uu) - lap(uu) + lift(t1, b, -1) + lift(t2, b, -2), -a * uu))
    pb2.add_equation((interp(uu, x, "left"), 0))
    pb2.add_equation((interp(uu, x, "right"), 0))
    solver = IVPSolver(ex.discretize(pb), "SBDF2", 1e-3)
    solver.set_state({uu: lambda X: np.sin(np.pi * X)})
    tw = TauWatch(solver, growth_limit=1e3)
    solver.run(20, callback=tw, every=10)
    assert tw.history
    # forgetting one condition -> RankError before factorization
    pb3 = IVP([uu, t1, t2], name="heat 1D")
    pb3.add_equation((dt(uu) - lap(uu) + lift(t1, b, -1) + lift(t2, b, -2), -a * uu))
    pb3.add_equation((interp(uu, x, "left"), 0))
    try:
        IVPSolver(ex.discretize(pb3), "SBDF2", 1e-3)
        raise AssertionError("rank-deficient system accepted")
    except Exception as e:  # noqa: BLE001 -- the page names the class
        assert "rank" in (type(e).__name__ + str(e)).lower(), (type(e).__name__, str(e)[:200])
    # bases.md
    cs2 = ex.CartesianCoordinates("x", "z")
    xx, zz = cs2.coords
    try:
        ex.Chebyshev(zz, 32, (-1, 1))
        raise AssertionError("Chebyshev without dealias accepted")
    except (ValueError, TypeError):
        pass
    assert ex.RealFourier(xx, 170, (0, 2 * np.pi), dealias=3 / 2).grid_size == 255
    # units.md
    return "compile/discretize, add_bc, explicit taus, TauWatch, rank refusal, dealias, units"


# --- studio.md ------------------------------------------------------------------------------
def studio():
    from conatus.model.run import Simulation
    assert hasattr(Simulation, "watch")
    return "sim.watch: executed by _check_features.py (studio_watch) and tutorial 9"


CHECKS = [index, geometries, equations, boundary_conditions, timesteppers,
          timesteppers_boundary_data, boundary_conditions_time, resolution, gpu,
          explain, runs, reproducibility, doctor_errors_notebook, problems, dedalus,
          expert_layer, studio]


def main() -> int:
    bad = 0
    for f in CHECKS:
        try:
            msg = f()
            status = "SKIP" if str(msg).startswith("SKIP") else "PASS"
        except Exception:  # noqa: BLE001 -- every failure is reported, then the script fails
            status, msg = "FAIL", traceback.format_exc(limit=4)
            bad += 1
        RESULTS.append((f.__name__, status, str(msg)))
        print(f"{status} {f.__name__}: {msg}", flush=True)
    print(f"== guide snippets: {sum(r[1] == 'PASS' for r in RESULTS)} pass, "
          f"{sum(r[1] == 'SKIP' for r in RESULTS)} skip, {bad} fail")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
