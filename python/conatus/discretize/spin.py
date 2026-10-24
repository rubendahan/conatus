"""Spin components of the curvilinear tensors (disk, annulus, sphere, ball, shell): one definition
shared by the polar and spherical evaluators, the fused spherical right-hand side, its adjoint
and the symmetry projections (one definition for all of them).

A tensor index takes the values 0, 1, 2: the spin components -1 and +1 of the two angular
directions and the radial (sphere) or axial (polar) direction, of spin 0. A component is a tuple
of such indices, one per tensor rank.
"""
from __future__ import annotations

import numpy as np

from ._xp import array_module

__all__ = ["SPIN", "spin_of", "flip", "canon", "real_component", "apply_index"]

SPIN = (-1, 1, 0)       # tensor index 0 -> spin -1, 1 -> +1, 2 (radial or axial) -> 0


def spin_of(comp: tuple) -> int:
    """Total spin of a component (the sum of the spins of its indices); on the disk the effective
    wavenumber is m + spin_of(comp), on the ball it is the total regularity shift."""
    return int(sum(SPIN[i] for i in comp))


def flip(comp: tuple) -> tuple:
    """The component with every spin reversed (index 0 <-> 1, 2 kept): the complex-conjugate
    partner of ``comp`` for a real tensor."""
    return tuple({0: 1, 1: 0, 2: 2}[i] for i in comp)


def canon(comp: tuple) -> tuple:
    """Canonical member of the spin pair {comp, flip(comp)} of a real tensor (its own if radial)."""
    f = flip(comp)
    return comp if comp <= f else f


def real_component(comp: tuple) -> bool:
    """Component real on the grid for a real tensor: every index radial (or a scalar)."""
    return all(i == 2 for i in comp)


def apply_index(M: np.ndarray, g: np.ndarray, rank: int) -> np.ndarray:
    """Apply the matrix M on each of the first ``rank`` axes of g (NumPy or CuPy g)."""
    xp = array_module(g)
    M = M if xp is np else xp.asarray(M)
    for ax in range(rank):
        g = xp.moveaxis(xp.tensordot(M, g, axes=([1], [ax])), 0, ax)
    return g
