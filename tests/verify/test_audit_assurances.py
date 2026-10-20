"""Audit "assurances" (docs/audits/assurances.md): a positive assurance printed or
returned by the library must be backed by the check it names. Each test reproduces a case where
the check passed while the result was wrong (or nothing was measured), and asserts the corrected
statement.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import conatus as cn
import conatus.resolution as R
from conatus.errors import CorrectnessWarning
from conatus.expert import critical
from conatus.ir import CartesianCoordinates, Chebyshev
from conatus.reference.evaluate import FieldData
from conatus.verify import verify

from tests.evp.test_parametric_functions import sh_pencil
from tests.resolution.test_resolution import _heat_plan


# ------------------------------------------------------------------ resolution report
def test_nan_coefficients_are_never_zero_or_resolved():
    """Python's max(0.0, nan) is 0.0: an all-NaN field was "zero" and the report said "all
    resolved"; a NaN in the top coefficient was dropped from the tail."""
    b = Chebyshev(CartesianCoordinates("x")["x"], 16, (-1, 1), dealias=1.5)
    assert R.estimate_modes(np.full(16, np.nan), b).direction.verdict == "non-finite"
    c = 0.5 ** np.arange(16)
    c[-1] = np.nan
    assert R.estimate_modes(c, b).direction.verdict == "non-finite"
    plan, u = _heat_plan(24)
    rep = plan.resolution_report({u: FieldData(u, np.full(24, np.nan), layout="coeff")})
    assert not rep.ok and "NOT all resolved" in rep.explain()
    assert rep.recommend() and "non-finite" in rep.recommend()[0]
    assert "every tail is under the target" not in rep.explain()


def test_cfl_of_a_nan_velocity_is_nan_not_zero(monkeypatch):
    """Every non-finite CFL step used to print CFL 0 (a fluid at rest); only dt_max = inf is."""
    from conatus.timestep import control
    plan, u = _heat_plan(24)
    monkeypatch.setattr(control.CFL, "compute", lambda self: float("nan"))
    rep = plan.resolution_report({u: lambda X: np.sin(np.pi * X)}, velocity="u", dt=0.1)
    assert np.isnan(rep.cfl["cfl"])
    monkeypatch.setattr(control.CFL, "compute", lambda self: float("inf"))
    rep = plan.resolution_report({u: lambda X: np.sin(np.pi * X)}, velocity="u", dt=0.1)
    assert rep.cfl["cfl"] == 0.0


# ------------------------------------------------------------------ verify()
def _pevp(cols, n):
    return SimpleNamespace(recon_error=0.0, rank_check="regular pencil",
                           gm0=SimpleNamespace(cols=cols), params=[],
                           matrices=lambda **v: (np.eye(n), np.eye(n)))


def test_verify_spectral_tail_per_component():
    """The columns of a vector unknown hold its components one after the other: the last 4
    entries were the top of the LAST component only, so an unresolved first one passed."""
    n = 8
    cols = [("u", (0,), "var")] * n + [("u", (1,), "var")] * n
    x = np.concatenate([0.7 ** np.arange(n), 1e-3 * 0.01 ** np.arange(n)])
    mode = SimpleNamespace(residual=0.0, y=None, x=x, values={}, sigma=-1.0)
    tail = next(c for c in verify(_pevp(cols, 2 * n), mode).checks if c.name == "spectral tail")
    assert tail.status == "FAIL" and abs(tail.value - 0.7 ** 4) < 1e-12
    x[:n] = 0.01 ** np.arange(n)                                   # both resolved: passes
    tail = next(c for c in verify(_pevp(cols, 2 * n), mode).checks if c.name == "spectral tail")
    assert tail.status == "PASS"


def test_verify_spectral_tail_not_measured_is_no_pass():
    mode = SimpleNamespace(residual=0.0, y=None, x=np.ones(2), values={}, sigma=-1.0)
    rep = verify(_pevp([], 2), mode)
    tail = next(c for c in rep.checks if c.name == "spectral tail")
    assert np.isnan(tail.value) and tail.status == "FAIL" and rep.manual_steps == 1


# ------------------------------------------------------------------ critical(over=)
def test_critical_minimum_at_the_edge_of_over_is_said():
    """Swift-Hohenberg r_c(q) = (1 - q^2)^2, minimum r_c = 0 at q = 1: over q in (1.2, 1.5) the
    bounded minimizer stops at q = 1.2 (r = 0.1936), returned as the critical point, verified."""
    with pytest.warns(CorrectnessWarning, match="end q = 1.2"):
        cp = critical(sh_pencil(), "r", bracket=(-0.5, 1.0), over={"q": (1.2, 1.5)}, xtol=1e-10)
    assert abs(cp.over["q"] - 1.2) < 1e-6 and cp.edge_note
    assert "MINIMUM AT THE EDGE" in repr(cp) and "WARNING: the minimum over q" in cp.explain()


def test_critical_interior_minimum_has_no_edge_note(recwarn):
    cp = critical(sh_pencil(), "r", bracket=(-0.5, 1.0), over={"q": (0.5, 1.5)}, xtol=1e-10)
    assert abs(cp.over["q"] - 1) < 1e-4 and cp.edge_note == ""
    assert not [w for w in recwarn if issubclass(w.category, CorrectnessWarning)]


# ------------------------------------------------------------------ time_average
def test_time_average_batch_means_check_is_applied():
    """White noise plus a slow sine: Sokal's window closes at W = 7 (tau = 1.35) while the batch
    means see the sine (3.1x larger error); the mean was 2.7 standard errors from the truth and
    "trusted"."""
    rng = np.random.default_rng(0)
    k = np.arange(4000)
    r = cn.time_average(rng.standard_normal(4000) + 0.5 * np.sin(2 * np.pi * k / 3000), discard=0)
    assert r.batch_stderr > 2 * r.stderr and not r.trusted
    assert any("batch-means check" in f for f in r.flags) and "flagged" in r.explain()
    white = cn.time_average(np.random.default_rng(1).standard_normal(4000), discard=0)
    assert white.trusted


# ------------------------------------------------------------------ hinf_norm




# ------------------------------------------------------------------ kreiss_constant


# ------------------------------------------------------------------ implicit rotation
def test_implicit_rotation_at_unit_step_is_not_called_resolved():
    """|W| dt = 1 with SBDF1: the inertial wave (omega dt = 0.71) keeps 0.82 of its amplitude
    per step (exact: 1), 15 % after 10 steps, while explain() said "its waves are resolved in
    time"."""
    from tests.ivp.test_implicit_coriolis_cartesian import A0, _amplitude, _model, _u0
    sim = _model(0.5).simulate(N=(8, 8, 8), scheme="SBDF1", dt=1.0)          # |W| dt = 1
    sim.initial(u=lambda x, y, z: np.array(list(_u0(x, y, z).values())))
    sim.run(steps=10)
    ratio = np.linalg.norm(_amplitude(sim)) / np.linalg.norm(A0)
    assert ratio < 0.2, ratio                                   # the wave is damped, not resolved
    line = next(ln for ln in sim.explain().splitlines() if "implicit rotation" in ln)
    assert "resolved in time" not in line and "own error" in line


# ------------------------------------------------------------------ NaN-hiding reductions
def test_lbvp_max_residual_keeps_a_nan_group():
    """max_residual and explain() used Python's max: a NaN residual after a finite group was
    hidden (max(1e-16, nan) is 1e-16), the solution printed "max relative residual 1e-16"."""
    from conatus.discretize.plan import LBVPSolution
    plan, _u = _heat_plan(16)
    sol = LBVPSolution({}, [("k=0", 1e-16), ("k=1", float("nan"))], plan, False)
    assert np.isnan(sol.max_residual)
    assert "max relative residual nan [k=1]" in sol.explain()
    assert LBVPSolution({}, [("a", 1e-16), ("b", 3e-16)], plan, False).max_residual == 3e-16


# ------------------------------------------------------------------ review of the audit (critic)
@pytest.mark.parametrize("xtol", [1e-6, 1e-4])
def test_critical_edge_is_said_at_a_loose_xtol(xtol):
    """The bounded Brent stops within its own tolerance of the end, never on it: with xtol = 1e-6
    it stopped 5.9e-7 from q = 1.2 (above 1e-6 x the width 0.3) and the edge was not said."""
    with pytest.warns(CorrectnessWarning, match="end q = 1.2"):
        cp = critical(sh_pencil(), "r", bracket=(-0.5, 1.0), over={"q": (1.2, 1.5)}, xtol=xtol)
    assert cp.edge_note and "MINIMUM AT THE EDGE" in repr(cp)






def test_cfl_of_an_infinite_velocity_is_inf(monkeypatch):
    """dt_max = 0 (an infinite velocity) raised ZeroDivisionError."""
    from conatus.timestep import control
    plan, u = _heat_plan(24)
    monkeypatch.setattr(control.CFL, "compute", lambda self: 0.0)
    rep = plan.resolution_report({u: lambda X: np.sin(np.pi * X)}, velocity="u", dt=0.1)
    assert rep.cfl["cfl"] == np.inf


def test_parametric_reconstruction_nan_is_no_pass(monkeypatch):
    """max(1e-17, nan) is 1e-17 and nan > RECON_TOL is False: a NaN in M(p) at the check point
    passed the 'exact reconstruction' check."""
    from conatus.errors import ProblemError
    from conatus.solvers.parametric import ParametricEVP
    orig = ParametricEVP.matrices

    def nan_mass(self, **values):
        L, M = orig(self, **values)
        M = np.array(M, dtype=complex)
        M[0, 0] = np.nan
        return L, M

    monkeypatch.setattr(ParametricEVP, "matrices", nan_mass)
    with pytest.raises(ProblemError, match="reconstruction error nan"):
        sh_pencil()
