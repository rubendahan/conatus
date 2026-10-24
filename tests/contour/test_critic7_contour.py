"""Critic an outside pole near the edge hid an inside zero, and count() returned 0.

Oracle: T(z) = (z - a) / (z - p) has one zero a and one pole p, so the argument principle gives
Z - P analytically from where a and p lie. The walk must return that, and warn that a zero or pole
lies close to the boundary; a function whose zeros are far from the edges gets no warning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import conatus as cn
from conatus.contour import count
from conatus.contour.count import NEAR_EDGE
from conatus.errors import CorrectnessWarning

BOX = cn.box(re=(-1, 1), im=(-1, 1))


@pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-5, 1e-8])
def test_dipole_across_the_top_edge_is_counted(eps):
    """The critic's probe: zero eps inside the top edge, pole eps outside; analytic count 1."""
    a, p = 0.123 + (1 - eps) * 1j, 0.123 + (1 + eps) * 1j
    with pytest.warns(CorrectnessWarning, match="close to the boundary"):
        c = count(lambda z: (z - a) / (z - p), BOX, poles=0)
    assert c.count == 1, c.explain()
    assert c.near_edge and "Move or shrink the box" in c.near_edge
    assert "warning:" in c.explain() and "certif" not in c.explain().lower()
    assert abs(c.smallest_at - 0.123 - 1j) < 0.05


@pytest.mark.parametrize("eps", [1e-3, 1e-6])
def test_reversed_dipole_gives_minus_one(eps):
    """Pole eps inside the right edge, zero eps outside: Z - P = 0 - 1 = -1 (P declared: Z = 0)."""
    a, p = 1 + eps - 0.31j, 1 - eps - 0.31j
    with pytest.warns(CorrectnessWarning):
        c = count(lambda z: (z - a) / (z - p), BOX, poles=1)
    assert c.winding == -1 and c.count == 0, c.explain()


def test_dipole_off_a_side_edge_with_a_smooth_factor():
    """The same pair 1e-4 across the left edge, times exp(z) (a smooth variation of det T that
    turns the phase by 2 rad along each horizontal edge): analytic count 1."""
    a, p = -1 + 1e-4 + 0.4j, -1 - 1e-4 + 0.4j
    with pytest.warns(CorrectnessWarning):
        c = count(lambda z: np.exp(z) * (z - a) / (z - p), BOX, poles=0)
    assert c.count == 1, c.explain()


def test_no_warning_when_the_zeros_are_clear_of_the_edges():
    """Zeros 0.2 and -0.3j, a pole at 0.5 + 0.5j (declared): Z = 2; nothing close to an edge."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorrectnessWarning)
        c = count(lambda z: (z - 0.2) * (z + 0.3j) / (z - 0.5 - 0.5j), BOX, poles=1)
    assert c.count == 2 and not c.near_edge
    assert c.smallest_step >= NEAR_EDGE


def test_critical_point_on_the_edge_is_not_a_zero():
    """T(z) = (z - i)^2 + 1/2 has T'(i) = 0 on the top edge (log det T bends there) but its
    zeros i(1 +- 1/sqrt 2) are 0.29 from the edge: count 1, no warning, no refusal."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorrectnessWarning)
        c = count(lambda z: (z - 1j) ** 2 + 0.5, BOX, poles=0)
    assert c.count == 1 and not c.near_edge, c.explain()


def test_pencil_walk_is_unchanged():
    """A pencil has no poles: the bend test does not run, and a zero near an edge gives no
    warning (the walk sees it by the phase or the modulus)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorrectnessWarning)
        c = count((-np.diag([0.2, 1 - 1e-6 + 0.1j]), np.eye(2)), BOX)
    assert c.count == 2 and c.poles_by_construction and not c.near_edge
