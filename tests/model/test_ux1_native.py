"""Parameters as attributes, per-run overrides, the
finite-state check, eigenfunctions on the grid, the growth convention of critical() and the
degree rule of the default dealiasing factor."""
import numpy as np
import pytest

import conatus as cn
from conatus.errors import InputTypeError, NumericalError, OptionError


def _ks(**params):
    m = cn.Model(cn.Box(x=cn.periodic(22.0)), name="KS", **params)
    m.scalar("u")
    m.equations("dt(u) + nu*lap(u) + lap(lap(u)) = -u*dx(u)")
    return m


def _os(alpha=1.0):
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=alpha)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    return m


# --- C1: parameters -------------------------------------------------------------------------
def test_attribute_sets_a_declared_parameter_with_validation():
    m = _ks(nu=1.0)
    assert m.nu == 1.0
    m.nu = 2                                   # used to create an unrelated attribute
    assert m.params["nu"] == 2.0 and "nu" not in vars(m)
    with pytest.raises(InputTypeError, match="not a number"):
        m.nu = "abc"
    with pytest.raises(OptionError, match=r"(?s)'nuu' is neither.*Did you mean 'nu'"):
        m.nuu = 3.0
    m.name = "renamed"                         # existing attributes keep working
    assert m.name == "renamed"


def test_params_mapping_is_validated():
    m = _ks(nu=1.0)
    m.params["nu"] = 1.5
    assert m.nu == 1.5
    with pytest.raises(OptionError, match="not a declared parameter"):
        m.params["mu"] = 1.0
    with pytest.raises(InputTypeError, match="not a number"):
        m.params["nu"] = "x"
    with pytest.raises(OptionError, match="cannot be removed"):
        del m.params["nu"]
    m.params.update(nu=3)
    assert m.params["nu"] == 3.0 and isinstance(m.params, dict)


def test_parameter_named_like_a_model_attribute_is_refused():
    with pytest.raises(InputTypeError, match="attribute of cn.Model"):
        cn.Model(cn.Box(x=cn.periodic(1.0)), solve=1.0)


def test_attribute_change_reaches_the_next_compilation():
    m = _ks(nu=1.0)
    c1 = m.compile(16)
    m.nu = 0.5
    assert c1.params["nu"].value == 1.0 and m.compile(16).params["nu"].value == 0.5


def test_simulate_parameter_override_is_declared_and_recorded():
    m = _ks(nu=1.0)
    sim = m.simulate(N=16, scheme="RK443", dt=0.01, nu=1.5)
    assert m.params["nu"] == 1.0                     # the model keeps its value
    assert sim.compiled.params["nu"].value == 1.5
    assert "nu = 1.5 (model: 1)" in sim.explain()
    assert "nu = 1.5" in sim.methods().text
    from conatus import provenance
    assert provenance.parameter_values(sim.problem)["nu"] == 1.5      # what a run record holds
    with pytest.raises(InputTypeError, match="not a number"):
        m.simulate(N=16, scheme="RK443", dt=0.01, nu="fast")


def test_simulate_unknown_keyword_lists_the_options():
    m = _ks(nu=1.0)
    with pytest.raises(OptionError, match=r"(?s)'timestepper' is neither.*Did you mean 'scheme'"
                                          r".*check_finite"):
        m.simulate(N=16, scheme="RK443", dt=0.01, timestepper="RK443")


# --- C2: finite-state check -----------------------------------------------------------------
def _blowup(**kw):
    m = _ks(nu=1.0)
    sim = m.simulate(N=16, scheme="RK111", dt=50.0, **kw)    # explicit Euler far beyond stability
    sim.initial(u=lambda x: 10 * np.cos(2 * np.pi * x / 22.0))
    return sim


def test_blown_up_run_stops_with_step_time_and_field():
    sim = _blowup()
    with np.errstate(all="ignore"):
        with pytest.raises(NumericalError, match=r"(?s)non-finite state.*iteration \d+, t = "
                                                 r".*unknown 'u'.*reduce dt.*check_finite=False"):
            sim.run(steps=1000)
    assert sim.iteration <= 1000 and sim.iteration % 100 == 0 or sim.iteration == 1000
    assert "finite-state check: every 100 steps (default)" in sim.explain()


def test_check_at_the_end_of_a_short_run_and_opt_out():
    sim = _blowup(check_finite=1000)
    with np.errstate(all="ignore"):
        with pytest.raises(NumericalError, match="end of the call"):
            sim.run(steps=40)                      # shorter than the cadence: checked at the end
    off = _blowup(check_finite=False)
    with np.errstate(all="ignore"):
        off.run(steps=40)
    assert not np.isfinite(off.grid("u")).all()
    assert "finite-state check: off" in off.explain()
    with pytest.raises(OptionError, match="check_finite"):
        _blowup(check_finite=0)


def test_no_nan_checkpoint(tmp_path):
    sim = _blowup(check_finite=False)
    with np.errstate(all="ignore"):
        sim.run(steps=40)
    sim.solver._finite_every = 100                # turned back on: the write is refused
    with pytest.raises(NumericalError, match="save_checkpoint"):
        sim.save(tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def test_finite_check_does_not_change_the_numbers():
    a = _ks(nu=1.0).simulate(N=16, scheme="RK443", dt=0.01)
    b = _ks(nu=1.0).simulate(N=16, scheme="RK443", dt=0.01, check_finite=False)
    for s in (a, b):
        s.initial(u=lambda x: np.cos(2 * np.pi * x / 22.0))
        s.run(steps=250)
    assert np.array_equal(a.solver.X, b.solver.X)


# --- C3: eigenfunctions ---------------------------------------------------------------------
def test_os_mode_on_the_grid_is_normalised():
    res = _os().eigenvalues(N=48, drift=64, drift_tol=1e-6)
    i = int(res.values.imag.argmax())
    md = res.mode(i)
    assert md.names == ["v"] and md.value == res.values[i]
    v, y = md["v"], md.coordinate("y")
    assert v.shape == y.shape == (48,)
    j = int(np.argmax(abs(v)))
    assert abs(v[j] - 1) < 1e-14                  # declared normalisation: max |v| = 1, real
    assert abs(y[j]) < 0.1                        # the least stable OS mode peaks at the centre
    assert "normalisation" in md.explain()
    with pytest.raises(InputTypeError, match="not an unknown"):
        md["u"]
    with pytest.raises(OptionError, match="mode"):
        res.mode(10_000)


def test_linearized_mode_uses_the_unknown_names():
    m = cn.problems.native.rayleigh_benard(Lx=4, Ra=2000, Pr=1)
    u, p, T = m.unknowns
    lin = m.linearize(about={u: 0, T: lambda z: -z})
    r = lin.eigenvalues(N={"x": 8, "z": 16}, k=np.pi / 2)
    md = r.mode(int(r.values.real.argmax()))
    assert md.names == ["u", "p", "T"] and md["u"].shape == (2, 8, 16)
    assert abs(abs(md["u"]).max() - 1) < 1e-14


# --- C4: critical() -------------------------------------------------------------------------
def test_growth_convention_is_declared_or_inferred():
    from conatus.solvers.parametric import growth_convention
    assert growth_convention("c")[0] == "imag" and growth_convention("s")[0] == "real"
    assert growth_convention("omega")[0] is None
    assert growth_convention("omega", "imag")[0] == "imag"
    m = _os()
    assert "growth = Im c" in m.explain()
    with pytest.raises(OptionError):
        m.eigenvalue("c", growth="Im")


def test_critical_bracket_without_sign_change_is_explained():
    from conatus.expert import critical
    P = _os(alpha=1.02055).parametric(N=48, over=("Re", "alpha"), drift=64, drift_tol=1e-6)
    with pytest.raises(cn.errors.ConvergenceError,
                       match=r"(?s)same sign at both ends.*decaying.*decaying.*try bracket="):
        critical(P, "Re", bracket=(3000, 4000), xtol=1e-6)   # alpha: the model's value


def test_critical_undeclared_growth_is_refused():
    from conatus.expert import critical
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OSw", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("w")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + w*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    P = m.parametric(N=32, over=("Re",))
    with pytest.raises(OptionError, match=r"(?s)growth of the eigenvalue 'w' is not declared"):
        critical(P, "Re", bracket=(5000, 7000), xtol=1e-6)


# --- C11: dealiasing follows the degree -----------------------------------------------------
def test_default_dealias_follows_the_polynomial_degree():
    from conatus.model.compiler import dealias_factor
    assert [dealias_factor(p) for p in (1, 2, 3, 4)] == [1.5, 1.5, 2.0, 2.5]
    sh = cn.Model(cn.Box(x=cn.periodic(10.0)), name="SH", r=0.3)
    sh.scalar("u")
    sh.equations("dt(u) - r*u + u + 2*lap(u) + lap(lap(u)) = -u*u*u")
    c = sh.compile(32)
    assert c.dealias == {"x": 2.0} and c.degree == 3
    assert "highest polynomial degree p = 3" in "\n".join(c.decisions())
    assert sh.compile(32, dealias=1.5).dealias == {"x": 1.5}      # an explicit value wins
    assert _ks(nu=1.0).compile(16).dealias == {"x": 1.5}         # quadratic: unchanged


# --- ----------------------------------------------------------------------------
def test_model_copies_and_pickles_keep_validated_parameters():
    """copy.deepcopy / pickle of a Model used to fail: the validated m.params refused every name
    while being rebuilt on an empty instance."""
    import copy
    import pickle
    m = _ks(nu=1.0)
    m2, m3 = copy.deepcopy(m), pickle.loads(pickle.dumps(m))
    m2.nu = 2.0
    assert m.nu == 1.0 and m2.nu == 2.0 and m3.nu == 1.0
    with pytest.raises(OptionError, match="not a declared parameter"):
        m3.params["nuu"] = 1.0


def test_embedded_error_loop_stops_on_a_blown_up_state():
    """A loop of EmbeddedError.step() steps the solver itself: the finite-state check runs there
    too (the NaN estimate only skipped the adaptation)."""
    from conatus.expert import EmbeddedError
    m = _ks(nu=1.0)
    sim = m.simulate(N=16, scheme="RK222", dt=50.0)
    sim.initial(u=lambda x: 10 * np.cos(2 * np.pi * x / 22.0))
    ee = EmbeddedError(sim.solver)
    with np.errstate(all="ignore"):
        with pytest.raises(NumericalError, match="EmbeddedError.step"):
            for _ in range(1000):
                ee.step()


def test_mode_leaves_out_expert_taus_whatever_their_names():
    """Expert taus are identified by their lift, not by a name prefix: the OS
    problem of tests/evp/cases.py names them tau1..tau4 (prefix without underscore)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evp"))
    from cases import orr_sommerfeld
    import conatus.expert as X
    plan = orr_sommerfeld(48)
    res = X.eigenvalues(plan)
    md = res.mode(int(np.argmax(np.where(np.abs(res.values) < 2, res.values.imag, -np.inf))))
    assert md.names == ["v"]
    assert abs(md.value - 0.23752649 - 0.00373967j) < 1e-6
