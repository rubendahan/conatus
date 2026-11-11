"""Argument-principle count with an even cluster of eigenvalues near an edge (hunt 5C).

A double eigenvalue 1e-3 inside (or outside) the bottom edge of [-1, 1]^2 turns the phase of
det T by 2 pi within one step; the walk tested only the principal phase of the half steps, so the
2 pi aliased to 0 at 48 of 197 positions and the count was off by 2 (or 1), with no warning.
Oracle: the eigenvalues are given (diagonal pencil, its similarity transform, a scaled copy).
"""
import numpy as np
import pytest

from conatus.contour import box, count
from conatus.errors import ConvergenceError

RE, IM = (-1.0, 1.0), (-1.0, 1.0)


def inside(ev):
    ev = np.asarray(ev)
    return int(np.sum((np.abs(ev.real) < 1) & (np.abs(ev.imag) < 1)))


@pytest.mark.parametrize("d", [1e-3, -1e-3, 1e-5])
@pytest.mark.parametrize("k", [2, 4])
def test_even_cluster_near_the_bottom_edge_is_counted_at_every_position(k, d):
    wrong = []
    for x0 in np.linspace(-0.98, 0.98, 197):
        ev = np.concatenate([x0 + 1e-7 * np.arange(k) + 1j * (-1 + d), [0.1 + 0.2j]])
        try:
            r = count((np.diag(-ev), np.eye(ev.size)), box(re=RE, im=IM))
        except ConvergenceError:
            continue                     # a refusal is allowed; a wrong integer is not
        if r.count != inside(ev):
            wrong.append((round(float(x0), 3), r.count, inside(ev)))
    assert not wrong, wrong[:6]


@pytest.mark.parametrize("c", [1e-8, 1.0, 1e8])
def test_similar_and_scaled_pencils(c):
    rng = np.random.default_rng(0)
    n = 12
    ev = np.concatenate([np.array([0.123 - 1e-7, 0.123 + 1e-7]) + 1j * (-1 + 1e-3), [0.1 + 0.2j],
                         rng.uniform(-3, 3, n - 3) + 1j * rng.uniform(-3, 3, n - 3)])
    Q = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = Q @ np.diag(ev) @ np.linalg.inv(Q)
    r = count((-c * A, c * np.eye(n)), box(re=RE, im=IM))
    assert r.count == inside(ev) == 3 + int(np.sum((np.abs(ev[3:].real) < 1) & (np.abs(ev[3:].imag) < 1)))
