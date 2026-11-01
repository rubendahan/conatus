"""The steady residual of a linearization on bases with a closed-form answer. The laminar channel
U = 1 - y^2 with G = 2/Re solves the equations exactly; other forcings miss by a known amount."""
from __future__ import annotations

import numpy as np
import pytest

from conatus.errors import OptionError
from conatus.problems import native as NS

RE = 1e4


def _lin(G, U=1.0):
    m = NS.channel(Lx=2 * np.pi, Lz=2 * np.pi, Re=RE, pressure_gradient=G)
    return m.linearize(about={"u": lambda x, y, z: np.stack([U * (1 - y * y), 0 * y, 0 * y])})


@pytest.mark.parametrize("N", [16, 20, 32, 52, 64])
def test_an_exact_base_is_accepted_at_every_resolution(N):
    lin = _lin(2 / RE)
    lin.eigenvalues(N=N, k={"x": 1.0, "z": 0.0})
    assert any("a steady solution" in n for n in lin.notes)


@pytest.mark.parametrize(("G", "U"), [(-2 / RE, 1.0), (0.0, 1.0), (2 / RE, 0.5)])
@pytest.mark.parametrize("N", [16, 48])
def test_a_base_that_is_not_steady_is_refused(N, G, U):
    with pytest.raises(OptionError, match="NOT a steady solution"):
        _lin(G, U).eigenvalues(N=N, k={"x": 1.0, "z": 0.0})


def test_a_wall_condition_really_missed_is_refused():
    """U = 1 - y^2 + 1e-6: a real wall value, far above rounding of U and below the terms' scale."""
    m = NS.channel(Lx=2 * np.pi, Lz=2 * np.pi, Re=RE, pressure_gradient=2 / RE)
    lin = m.linearize(about={"u": lambda x, y, z: np.stack([1 - y * y + 1e-6, 0 * y, 0 * y])})
    with pytest.raises(OptionError, match="NOT a steady solution"):
        lin.eigenvalues(N=16, k={"x": 1.0, "z": 0.0})
