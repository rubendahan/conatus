"""External review of 0.1.1, lot B "false
assurances": a check that reports success must have measured what it names. Each test rebuilds
the reviewer's case and compares with an oracle computed here (NumPy, SciPy, the analytic value,
or the plain execution of the script), never with the library's own output.
"""
import math
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize as so

import conatus as cn
from conatus.errors import InputTypeError, OptionError
from conatus.expert import critical
from conatus.verify import verify

from tests.evp.test_parametric_functions import sh_pencil


def _backward_error(L, M, s, x):
    """Independent oracle: ||(L + s M) x|| / ((||L||_1 + |s| ||M||_1) ||x||)."""
    return np.linalg.norm((L + s * M) @ x) / (
        (np.linalg.norm(L, 1) + abs(s) * np.linalg.norm(M, 1)) * np.linalg.norm(x))


# ------------------------------------------------------------------ F4: verify() measures
def _os_mode():
    """The reviewer's case: the Orr-Sommerfeld mode near 0.2375 + 0.0037i at Re = 1e4."""
    m = cn.problems.native.orr_sommerfeld(Re=1e4, alpha=1.0, flow="poiseuille")
    P = m.parametric(32, over=("Re", "alpha"))
    r = P.solve(Re=1e4, alpha=1.0)
    i = int(np.argmin(abs(r.values - (0.2375 + 0.0037j))))
    return P, P.track(r.values[i], r.vectors[:, i], Re=1e4, alpha=1.0)


def test_f4_verify_measures_the_current_vector():
    P, mode = _os_mode()
    L, M = P.matrices(**mode.values)
    right = next(c for c in verify(P, mode).checks if c.name == "right residual")
    assert right.status == "PASS" and right.value < 1e-12
    # the vector is replaced by a coordinate vector that is not an eigenvector
    x = np.zeros_like(mode.x)
    x[0] = 1.0
    oracle = _backward_error(L, M, mode.sigma, x)
    assert oracle > 1e-3                       # the replaced vector is far from an eigenvector
    mode.x[:] = x
    rep = verify(P, mode)
    right = next(c for c in rep.checks if c.name == "right residual")
    assert right.status == "FAIL" and math.isclose(right.value, oracle, rel_tol=1e-10)
    assert rep.manual_steps >= 1


def test_f4_zero_and_misshapen_vectors():
    P, mode = _os_mode()
    mode.x[:] = 0.0
    rep = verify(P, mode)
    right = next(c for c in rep.checks if c.name == "right residual")
    tail = next(c for c in rep.checks if c.name == "spectral tail")
    assert np.isnan(right.value) and right.status == "FAIL"
    assert tail.status == "FAIL"               # a zero vector has no tail: not a PASS
    taylor = [c for c in rep.checks if "Taylor" in c.name]
    assert taylor and all(c.status == "FAIL" for c in taylor)
    _, mode = _os_mode()
    mode.x = mode.x[:-1]
    with pytest.raises(InputTypeError, match="shape"):
        verify(P, mode)


# ------------------------------------------------------------------ F5: dynamic reads
def test_f5_dynamic_attribute_read_is_never_a_stale_hit(tmp_path):
    cfg = types.ModuleType("review_cfg_f5")
    cfg.factor = 1.0
    sys.modules[cfg.__name__] = cfg
    try:
        @cn.runs.track("dynamic_attr", cache=True, root=tmp_path)
        def f(x):
            return getattr(cfg, "factor") * x              # noqa: B009 -- the reviewed read

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert f(2.0) == 2.0
            cfg.factor = 3.0
            assert f(2.0) == 6.0                           # oracle: 3 x 2
        msgs = [str(m.message) for m in w]
        assert any("getattr" in m for m in msgs), msgs
        assert not any("cache hit" in m for m in msgs), msgs
    finally:
        sys.modules.pop(cfg.__name__, None)


def test_f5_environment_read_is_never_a_stale_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("CN_REVIEW_F5", "1")

    @cn.runs.track("env_read", cache=True, root=tmp_path)
    def g(x):
        return float(os.environ["CN_REVIEW_F5"]) * x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert g(2.0) == 2.0
        monkeypatch.setenv("CN_REVIEW_F5", "5")
        assert g(2.0) == 10.0                              # oracle: 5 x 2


# ------------------------------------------------------------------ F6: critical(over=)
def _f6_curve(q):
    return (q + 0.6) ** 2 * (q - 0.5) ** 2 + 0.01 * (q + 0.6)


def test_f6_critical_over_finds_the_lower_of_two_interior_minima():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), r=0.1, q=0.1)
    m.scalar("u")
    m.eigenvalue("s", growth="real")
    m.equations("s*u=(r-((q+0.6)**2*(q-0.5)**2+0.01*(q+0.6)))*u")
    P = m.parametric(8, over=("r", "q"))
    c = critical(P, "r", bracket=(-0.5, 2), over={"q": (-1, 1)}, xtol=1e-9)
    # oracle: the neutral curve r_c(q) = f(q) minimized by SciPy on a dense grid then refined
    qs = np.linspace(-1, 1, 20001)
    q0 = qs[np.argmin(_f6_curve(qs))]
    ref = so.minimize_scalar(_f6_curve, bounds=(q0 - 1e-3, q0 + 1e-3), method="bounded",
                             options={"xatol": 1e-12})
    assert ref.fun < 0 and ref.x < 0                       # the lower branch, near q = -0.604
    assert abs(c.value - ref.fun) < 1e-8 and abs(c.over["q"] - ref.x) < 1e-4
    text = c.explain()
    assert "local minima of the scan refined (2)" in text and "OVER_SCAN_POINTS" in text


def test_f6_over_points_is_checked():
    with pytest.raises(OptionError, match="over_points"):
        critical(sh_pencil(), "r", bracket=(-0.5, 1.0), over={"q": (0.5, 1.5)}, xtol=1e-9,
                 over_points=2)


# ------------------------------------------------------------------ F10, F17: resolution
def _heat_sim():
    m = cn.Model(cn.Box(x=cn.periodic(6.0)))
    m.scalar("u")
    m.equations("dt(u)-lap(u)=0")
    return m.simulate(16, scheme="RK443", dt=0.01)


def test_f10_empty_state_is_not_resolved():
    r = _heat_sim().plan.resolution_report({}, layout="coeff")
    assert not r.ok and not r.complete and r.unmeasured == ["u"]
    text = r.explain()
    assert "NOT MEASURED" in text and "all resolved" not in text
    assert "every measured tail" not in text and "nothing was measured" in text
    assert r.to_dict()["ok"] is False


def test_f10_partial_state_is_incomplete():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.scalar("v")
    m.equations("dt(u)-lap(u)=0", "dt(v)-lap(v)=0")
    plan = m.simulate(16, scheme="RK443", dt=0.01).plan
    c = np.zeros(16)
    c[0] = 1.0                                             # the mean only: resolved
    r = plan.resolution_report({"u": c}, layout="coeff")
    assert r.entries[0].verdict == "resolved"
    assert not r.ok and r.unmeasured == ["v"] and "INCOMPLETE: not measured v" in r.explain()
    full = plan.resolution_report({"u": c, "v": c}, layout="coeff")
    assert full.ok and full.complete


@pytest.mark.parametrize("layout", ["coeff", "grid"])
def test_f17_wrong_shape_is_a_typed_refusal(layout):
    with pytest.raises(InputTypeError, match=r"shape \(2,\).*\(16,\)"):
        _heat_sim().plan.resolution_report({"u": np.zeros(2)}, layout=layout)


# ------------------------------------------------------------------ F13: agent check
def _script(tmp_path, steps):
    p = Path(tmp_path) / "run.py"
    p.write_text("import conatus as cn\nm=cn.Model(cn.Box(x=cn.periodic(6.)))\n"
                 "m.scalar('u'); m.equations('dt(u)-lap(u)=0')\n"
                 "s=m.simulate(16,scheme='RK443',dt=.01)\n"
                 f"s.run(steps={steps})\n")
    return p


def test_f13_agent_check_validates_run_arguments(tmp_path):
    from conatus.agent import check
    p = _script(tmp_path, -1)
    with pytest.raises(OptionError):                       # oracle: the script run for real
        exec(compile(p.read_text(), str(p), "exec"), {})
    r = check(p)
    assert not r["ok"] and r["refusals"][0]["type"] == "OptionError"
    assert r["refusals"][0]["line"] == 5
    assert r["validated"] == [{"call": "Simulation.run", "values": "checked",
                               "checked": "Simulation.run argument checks (_run_arguments)"}]
    good = check(_script(tmp_path, 3))
    assert good["ok"] and good["validated"][0]["values"] == "checked"


# ------------------------------------------------------------------ F16: tuple resolutions
def test_f16_tuple_resolution_report_prints():
    r = cn.uncertainty(lambda N, dt: 1 + np.exp(-N[0]) + dt ** 2,
                       N=[(8, 8), (12, 12), (16, 16)], dt=[0.04, 0.02, 0.01], scheme_order=2)
    assert math.isclose(r.value, 1 + math.exp(-16) + 1e-4, rel_tol=1e-15)   # finest N, dt
    text = r.explain()
    assert "N = (8, 8), (12, 12), (16, 16)" in text and "dt = 0.04, 0.02, 0.01" in text


@pytest.mark.parametrize("refused", ["ParameterError", "ConvergenceError"])
def test_scan_over_survives_a_refused_point_and_a_refused_refinement(refused):
    """Integration review of 0.1.2 (lots A and B): a scan point where the pencil is refused
    (lot A's ParameterError on discarded entries) aborted all of critical(over=), and a refusal
    inside the Brent refinement of a minimum next to it was not caught. Oracle: the analytic
    minimum of (q + 0.4)^2 - 1 on [-1, 1], at q = -0.4 with value -1; q > 0.7 is refused."""
    import conatus.errors as errs
    from conatus.solvers.parametric import _minimize_over
    exc = getattr(errs, refused)

    def neutral(q, fresh=False):
        if q > 0.7:
            raise exc(f"refused at q = {q}")
        return ((q + 0.4) ** 2 - 1.0, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, log = _minimize_over(neutral, "q", "r", -1.0, 1.0, 9, None, 1e-10)
    assert abs(x + 0.4) < 1e-6, (x, log)

    def neutral_edge(q, fresh=False):                 # minimum next to the refused points
        if q > 0.7:
            raise exc(f"refused at q = {q}")
        return (-q, None)                             # decreasing: the minimum is the last good point

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, log = _minimize_over(neutral_edge, "q", "r", -1.0, 1.0, 9, None, 1e-10)
    assert abs(x - 0.5) < 1e-12 and any("refinement near" in s for s in log), (x, log)


def test_verify_reports_a_refused_pencil_as_a_failed_check(monkeypatch):
    """Release 0.1.2: a pencil that refuses the mode's point (ParameterError, entries discarded at
    rounding level could matter there) made verify() raise; the report now lists the check as
    not measured (FAIL). Oracle: the refusal is forced, so no residual can have been measured."""
    from conatus.errors import ParameterError
    P, mode = _os_mode()

    def refuse(**values):
        raise ParameterError("refused here")

    monkeypatch.setattr(P, "matrices", refuse)
    rep = verify(P, mode)
    names = [c.name for c in rep.checks]
    assert "pencil at the mode's values" in names and "right residual" not in names
    assert rep.manual_steps >= 1 and "not measured" in rep.explain()


def test_verify_reports_a_refused_taylor_step_as_a_failed_check(monkeypatch):
    """The Taylor test steps the parameter by +- h; a refusal there is a failed check."""
    from conatus.errors import ParameterError
    P, mode = _os_mode()

    def refuse(*args, **kwargs):
        raise ParameterError("refused at the step")

    monkeypatch.setattr(P, "track", refuse)
    rep = verify(P, mode)
    taylor = [c for c in rep.checks if "Taylor" in c.name]
    assert taylor and all(c.status == "FAIL" and "not measured" in c.what for c in taylor)
