"""explain() is a short summary by default, the full text with full=True.

The default text of the key objects (Model, Simulation, compiled model, EVP result,
Linearization, IVPSolver) is a few lines in the words of the model: the problem, the
discretisation, the scheme, the checks and every warning; tau names and modes, the term-by-term
split, solver buckets and bands move to ``explain(full=True)``, which is the former text."""
from __future__ import annotations

import numpy as np
import pytest

import conatus as cn


def _rb():
    box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
    m = cn.Model(box, name="Rayleigh-Benard 2D", Ra=2500, Pr=1)
    u, _p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    return m, u, T


MACHINERY = ("tau_", "lifted into", "bucket", "kl=", "border rows", "ordering=", "Term placement")


def _short(text: str, max_lines: int) -> None:
    lines = text.splitlines()
    assert len(lines) <= max_lines, text
    for word in MACHINERY:
        assert word not in text, (word, text)
    assert "full=True" in lines[-1] or "full=True" in text


def test_model_and_simulation_short_by_default_full_on_request():
    m, _u, _T = _rb()
    s = m.explain()
    _short(s, 15)
    for want in ("Equations as written", "momentum: dt(u) + grad(p)", "nonlinear terms explicit "
                 "(momentum, heat)", "bottom: u = 0, T = 0.5", "integ(p) = 0", "Dealiasing: x: 3/2"):
        assert want in s, want
    f = m.explain(full=True)
    assert "tau_u1 lifted into" in f and "Term placement" in f and len(f) > 2 * len(s)

    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    s = sim.explain()
    _short(s, 15)
    for want in ("N = x: 16, z: 24: RK443, dt = 0.0005, backend cpu", "scheme: RK443",
                 "finite-state check: every 100 steps"):
        assert want in s, want
    f = sim.explain(full=True)
    assert "bucket" in f and "lifted into mode 23" in f
    assert sim.compiled.explain().count("\n") < 10 and "IR problem" in sim.compiled.explain(full=True)
    assert "GlobalSystem" in sim.solver.explain(full=True) and "GlobalSystem" not in sim.solver.explain()


def test_warnings_and_per_call_parameters_stay_in_the_short_text():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="nonpoly", nu=0.1)
    m.scalar("u")
    m.equations("dt(u) - nu*lap(u) = -sin(u)")
    s = m.explain()
    assert "warning: not polynomial" in s and "sin(u)" in s
    sim = m.simulate(N=16, scheme="RK222", dt=1e-3, nu=0.2)
    assert "Parameters for this call" in sim.explain() and "nu = 0.2 (model: 0.1)" in sim.explain()


def test_evp_result_short_keeps_filters_and_modes():
    os_ = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
    os_.scalar("v")
    os_.given(U=lambda y: 1 - y**2)
    os_.eigenvalue("c")
    os_.define("L2(f) = dy(dy(f)) - alpha**2*f")
    os_.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    os_.at("walls", "v = 0", "dy(v) = 0")
    res = os_.eigenvalues(N=48, drift=64, drift_tol=1e-6)
    s, f = res.explain(), res.explain(full=True)
    assert "kept " in s and "growth = Im c" in s and "residual" in s
    assert "pencil L + sM" not in s and "pencil L + sM" in f and "full=True" in s
    assert len(s.splitlines()) < len(f.splitlines())


def test_linearization_short_and_full():
    m, u, T = _rb()
    lin = m.linearize(about={u: 0, T: lambda z: -z})
    s = lin.explain((16, 24))
    assert "T = T0(z), a non-constant coefficient" in s and "tau_" not in s and "full=True" in s
    f = lin.explain((16, 24), full=True)
    assert "tau_u1 lifted into" in f and "eigenproblem (expert layer)" in f


def test_agent_report_and_run_registry_keep_the_full_text():
    import inspect

    from conatus import agent
    from conatus._explain import full_explain
    from conatus.runs import integrations, run
    m, _u, _T = _rb()
    # the agent report uses the one helper (its former private wrapper was a duplicate)
    assert agent.full_explain is full_explain
    assert "lifted into" in full_explain(m) and full_explain(m) == m.explain(full=True)
    sim = m.simulate(N=(8, 12), scheme="RK443", dt=1e-3)
    assert "GlobalSystem" in full_explain(sim.solver)
    # the registry records what full_explain gives (Run.attach, cn.runs.record)
    assert "full_explain(o)" in inspect.getsource(run.Run.attach)
    assert "full_explain(obj)" in inspect.getsource(integrations)

    class Plain:
        def explain(self):
            return "plain"
    assert full_explain(Plain()) == "plain"


@pytest.mark.parametrize("obj", ["model", "sim", "solver"])
def test_full_is_keyword_only(obj):
    m, _u, _T = _rb()
    sim = m.simulate(N=(8, 12), scheme="RK443", dt=1e-3)
    target = {"model": m, "sim": sim, "solver": sim.solver}[obj]
    with pytest.raises(TypeError):
        target.explain(None, None, True) if obj == "model" else target.explain(True)


def test_solver_warnings_stay_in_the_short_text():
    """Review the short solver text dropped the order note of time-dependent boundary
    data (bc_time='step': about first order) and the fallback/accuracy notes of F."""
    from conatus.discretize import discretize
    from conatus.ir import IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, cos, diff
    from conatus.ir import dt as d_t
    from conatus.ir import interp
    from conatus.timestep import IVPSolver
    cs = CartesianCoordinates("y")
    y = cs["y"]
    u, t = Field("u", Domain(cs, [Chebyshev(y, 16, (0, 1), dealias=1)])), Parameter("t", 0.0)
    pb = IVP([u])
    pb.add_equation((d_t(u) - diff(diff(u, y), y), 0))
    pb.add_bc((interp(u, y, 0), cos(t)), (interp(u, y, 1), 0.0))
    plan = discretize(pb)
    for mode, want in (("step", "about first order in time"), ("stage", "except RK443: 2")):
        s = IVPSolver(plan, "RK443", 1e-3, time_parameter="t", bc_time=mode)
        assert want in s.explain(full=True) and want in s.explain(), (mode, s.explain())
    s = IVPSolver(plan, "RK443", 1e-3)
    s._fused_note = "F: StageTimeFused with coefficient_time='step_start' (first order in time)"
    s._fc2d_note = "2D pipeline not used: probe"
    assert s._fused_note in s.explain() and s._fc2d_note in s.explain()


def test_evp_short_keeps_skipped_checks():
    """Review 'numerical test skipped' (pencils larger than NUMERICAL_MAX) is a warning."""
    from conatus.solvers.evp import EVPResult
    r = EVPResult(np.array([1.0 + 0j]), None, np.array([1e-14]), "dense QZ", "g", 0,
                  log=["pencil L + sM: structural rank 3000/3000; numerical test skipped (size 3000)",
                       "0 infinite eigenvalues removed"])
    s = r.explain()
    assert "numerical test skipped" in s and "infinite eigenvalues removed" not in s
