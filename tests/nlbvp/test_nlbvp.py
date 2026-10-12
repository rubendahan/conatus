"""NLBVP: Newton with the IR's exact Fréchet Jacobian (Lane-Emden, u'' = 2u^3), refusals."""
import numpy as np
import pytest

from conatus.discretize.plan import RankError
from conatus.errors import OptionError
from conatus.ir import ProblemError, dt
from conatus.reference import transforms as T
from conatus.solvers.nlbvp import NewtonNotConverged, NewtonSolver

from .nlbvp_cases import XI1_N3, cubic, lane_emden, lane_emden_n3, newton


def test_lane_emden_n1_exact():
    pb, o = lane_emden(1, 48)
    sol = newton(pb, o["rr"]).solve({o["f"]: lambda r: np.cos(np.pi * r / 2),
                                     o["R"]: lambda r: 3.0 + 0 * r})
    assert sol.converged and sol.residual < 1e-13
    assert abs(sol.value("R") - np.pi) < 1e-12
    r = T.grid(o["basis"], dealias=False)
    assert np.max(abs(sol.grid("f") - np.sinc(r))) < 1e-12        # sin(pi r)/(pi r)
    q = sol.observed_orders()
    assert q and min(q[1:] or q) > 1.7, sol.explain()             # quadratic once in the basin


@pytest.mark.slow                   # 2.4 s in the fast suite (qual_durations0)
def test_lane_emden_n3_first_zero_converged_in_N():
    vals = []
    for N in (32, 64):
        sol, _ = lane_emden_n3(N)
        assert sol.converged and sol.residual < 1e-13
        vals.append(sol.value("R"))
    assert abs(vals[0] - vals[1]) < 1e-11
    assert abs(vals[1] - XI1_N3) < 1e-11
    assert round(vals[1], 8) == 6.89684862
    q = sol.observed_orders()
    assert max(q) > 1.9, sol.explain()


def test_cubic_exact_solution():
    pb, o = cubic(32)
    s = NewtonSolver(pb, tolerance=1e-13, damping=1.0, max_iterations=20)
    sol = s.solve({o["u"]: lambda x: 1 - x / 2})
    x = T.grid(o["basis"], dealias=False)
    assert np.max(abs(sol.grid("u") - 1 / (1 + x))) < 1e-12
    assert len(sol.iterations) <= 8


def test_damping_is_honoured_and_reported():
    pb, o = cubic(24)
    sol = NewtonSolver(pb, tolerance=1e-10, damping=0.5, max_iterations=60).solve(
        {o["u"]: lambda x: 1 - x / 2})
    r = [it.rel_residual for it in sol.iterations]
    ratios = [b / a for a, b in zip(r[3:-1], r[4:])]
    assert all(0.4 < q < 0.6 for q in ratios), ratios            # linear rate 1 - damping
    assert "damping=0.5" in sol.explain()


def test_settings_are_required():
    pb, o = cubic(16)
    with pytest.raises(OptionError, match="tol must be given"):
        NewtonSolver(pb)          # no silent accuracy (damping, max_iterations: declared defaults)
    with pytest.raises(ValueError):
        NewtonSolver(pb, tolerance=1e-12, damping=1.5, max_iterations=5)


def test_missing_guess_refused():
    pb, o = lane_emden(1, 16)
    with pytest.raises(ProblemError, match="initial guess"):
        newton(pb, o["rr"]).solve({o["f"]: lambda r: 1 - r ** 2})


def test_singular_jacobian_refused():
    pb, o = lane_emden(3, 16)                                     # f = 0: the R column vanishes
    with pytest.raises(RankError):
        newton(pb, o["rr"]).solve({o["f"]: lambda r: 0 * r, o["R"]: lambda r: 3.0 + 0 * r})


def test_not_converged_is_an_error():
    pb, o = cubic(16)
    s = NewtonSolver(pb, tolerance=1e-14, damping=1.0, max_iterations=1)
    with pytest.raises(NewtonNotConverged):
        s.solve({o["u"]: lambda x: 1 - x / 2})
    sol = s.solve({o["u"]: lambda x: 1 - x / 2}, require_convergence=False)
    assert not sol.converged and "NOT converged" in sol.explain()


def test_dt_refused_and_explain():
    pb, o = cubic(16)
    with pytest.raises(ProblemError):
        pb.add_equation((dt(o["u"]), 0))
    text = NewtonSolver(pb, tolerance=1e-12, damping=1.0, max_iterations=5).explain()
    assert "Fréchet" in text and "damping = 1" in text and "NLBVP" in pb.explain()


def test_bratu_lower_branch_exact():
    """u'' + lam e^u = 0 (exp node of the IR): closed-form solution, quadratic Newton."""
    from .nlbvp_cases import bratu, bratu_exact
    lam = 0.5
    pb, o = bratu(32, lam)
    sol = NewtonSolver(pb, tolerance=1e-13, damping=1.0, max_iterations=20,
                       params={"lam": lam}).solve({o["u"]: lambda x: 0 * x})
    assert sol.converged and sol.residual < 1e-13
    x = T.grid(o["basis"], dealias=False)
    exact, _ = bratu_exact(lam, x)
    assert np.max(abs(sol.grid("u") - exact)) < 1e-11
    assert max(sol.observed_orders()) > 1.9, sol.explain()
    assert "exp" in pb.explain() and "NOT alias-free" in pb.explain()


def test_bratu_lambda_c_constant():
    """lam_c on [-1, 1] = 1/4 of the classical [0, 1] value 3.51383071912516."""
    from .nlbvp_cases import bratu_lambda_c
    assert abs(4 * bratu_lambda_c() - 3.51383071912516) < 1e-12
