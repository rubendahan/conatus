"""The three spectral-tail measures have one definition each (``conatus.ir.tails``), checked against closed forms on geometric coefficients c_n = r^n:

* coefficient tail, last k of N: r^(N-k);
* envelope tail on M mode numbers, window n = max(2, ceil(0.1 M)): r^(M-n);
* energy tail of one wall-normal piece of n coefficients, window k = max(2, ceil(0.1 n)):
  r^(2(n-k)) (1 - r^(2k)) / (1 - r^(2n)).

Every user of a measure (NCC check, resolution report, run-time check of the time steppers,
spurious-mode flag) reads it from that module, so the numbers they print are the same function.
"""
import math
import warnings

import numpy as np
import pytest

from conatus import resolution as R
from conatus.ir import tails as T
from conatus.ir.bases import Chebyshev, RealFourier
from conatus.ir.coords import CartesianCoordinates
from conatus.solvers import spurious
from conatus.timestep import IVPSolver
from tests.resolution.test_resolution import _heat_plan


@pytest.mark.parametrize("N,k,r", [(12, 3, 0.5), (40, 3, 0.8), (17, 4, 0.3)])
def test_coefficient_tail_is_the_last_k_over_the_largest(N, k, r):
    c = r ** np.arange(N) * (-1.0) ** np.arange(N)          # signs do not matter: amplitudes
    assert math.isclose(T.coefficient_tail(c, k), r ** (N - k), rel_tol=1e-13, abs_tol=0.0)
    assert T.coefficient_tail(np.zeros(N), k) == 0.0          # nothing to measure


@pytest.mark.parametrize("N,r", [(40, 0.7), (64, 0.9), (20, 0.5)])
def test_envelope_tail_of_a_chebyshev_series(N, r):
    b = Chebyshev(CartesianCoordinates("x")["x"], N, (-1, 1), dealias=1.5)
    n = max(2, math.ceil(0.1 * N))
    est = R.estimate_modes(r ** np.arange(N), b)
    assert est.direction.n_tail == n
    assert math.isclose(est.direction.tail, r ** (N - n), rel_tol=1e-13, abs_tol=0.0)


def test_envelope_tail_of_a_real_fourier_series():
    # RealFourier: index i holds the pair n = i // 2 (cos, -sin); M = N / 2 mode numbers
    N, r = 48, 0.6
    b = RealFourier(CartesianCoordinates("x")["x"], N, (0, 2 * np.pi), dealias=1.5)
    c = r ** (np.arange(N) // 2)
    M = N // 2
    n = max(2, math.ceil(0.1 * M))
    assert math.isclose(R.estimate_modes(c, b).direction.tail, r ** (M - n), rel_tol=1e-13,
                        abs_tol=0.0)


def test_run_time_check_prints_the_envelope_tail():
    """The time steppers' run-time check uses the same envelope tail as the resolution report:
    on c_n = 0.8^n (N = 30, window 3) it reports 0.8^27 = 2.4e-3 > RUN_TAIL_TOL."""
    N, r = 30, 0.8
    plan, u = _heat_plan(N)
    s = IVPSolver(plan, "SBDF1", 1e-3)
    sizes = [int(np.prod(sh)) for sh in s.system.var_shapes]
    parts = [r ** np.arange(N) if v is u else np.zeros(sh)
             for v, sh in zip(s.system.variables, s.system.var_shapes, strict=True)]
    assert sizes[[v is u for v in s.system.variables].index(True)] == N
    s.X = s.system.gather_state(parts)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s._check_tails()
    expected = r ** (N - 3)
    assert f"tail {expected:.1e}" in s._tail_note, s._tail_note
    assert any(f"tail {expected:.1e}" in str(x.message) for x in w)


@pytest.mark.parametrize("n,r", [(24, 0.5), (40, 0.9), (10, 0.7)])
def test_energy_tail_of_one_piece(n, r):
    k = max(2, math.ceil(0.1 * n))
    exact = r ** (2 * (n - k)) * (1 - r ** (2 * k)) / (1 - r ** (2 * n))
    V = np.stack([r ** np.arange(n), 2.0 * r ** np.arange(n)], axis=1)   # scale-free: same tail
    layout = [("u", np.arange(n).reshape(1, n), [(0, n)])]
    got = T.energy_tails(layout, V)
    np.testing.assert_allclose(got, [exact, exact], rtol=1e-12, atol=0.0)
    assert spurious.spectral_tails is T.energy_tails                       # the flag reads it


def test_every_user_reads_the_one_definition():
    from conatus.discretize import ultraspherical
    from conatus.verify import report
    assert R._mode_numbers is T.mode_numbers and R._n_tail is T.tail_window
    assert R._family is T.basis_family
    assert ultraspherical.coefficient_tail is T.coefficient_tail
    assert report.coefficient_tail is T.coefficient_tail
