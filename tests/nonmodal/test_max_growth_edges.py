"""Finding 15: max_growth missed the trivial maximum G = 1 at t = 0."""
import numpy as np
import pytest

from conatus.errors import OptionError
from conatus.nonmodal.energy import max_growth
from conatus.nonmodal.statespace import StateSpace


def test_maximum_at_left_edge():
    ss = StateSpace(np.array([[-10000.]]), np.eye(1), None, 0, np.inf, 1., 1e-12)
    g, t = max_growth(ss, (0., 10.), xtol=1e-6)
    assert g == pytest.approx(1.0, abs=1e-12) and t == 0.0


def test_maximum_at_right_edge():
    ss = StateSpace(np.array([[0.5]]), np.eye(1), None, 0, np.inf, 1., 1e-12)
    g, t = max_growth(ss, (0., 2.), xtol=1e-8)
    assert t == 2.0 and g == pytest.approx(np.exp(2.0), rel=1e-12)


def test_bad_bounds_refused():
    ss = StateSpace(np.array([[-1.]]), np.eye(1), None, 0, np.inf, 1., 1e-12)
    with pytest.raises(OptionError):
        max_growth(ss, (2., 1.), xtol=1e-6)
    with pytest.raises(OptionError):
        max_growth(ss, (0., 1.), xtol=1e-6, n_scan=2)
