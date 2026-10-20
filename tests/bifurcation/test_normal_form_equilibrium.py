"""Finding 91: a fold normal form was returned at a point that is not an
equilibrium (u' = p + u^2, p = 1, u = 0: G = 1)."""
import numpy as np
import pytest

from conatus.bifurcation import ir_system, normal_form, ode_problem
from conatus.errors import ConvergenceError


def _fold(p):
    pb = ode_problem(["u"], {"p": p}, lambda X, P: [P["p"] + X["u"] * X["u"]], "fold")
    return ir_system(pb, ["p"])


def test_off_equilibrium_refused():
    with pytest.raises(ConvergenceError, match="not an equilibrium"):
        normal_form(_fold(1.0), np.zeros(1), kind="zero", parameter="p")


def test_at_the_fold_classified():
    nf = normal_form(_fold(0.0), np.zeros(1), kind="zero", parameter="p")
    assert nf.kind == "fold"
