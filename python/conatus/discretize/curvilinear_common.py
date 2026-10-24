"""Helpers shared by the polar and spherical discretizations (one implementation for both
geometries).

* :func:`spin_dot`: the dot product of two tensor fields stored in a spin basis, contracted with the
  geometry's metric ``G`` (polar: e_- . e_+ = 1/2, e_z . e_z = 1; spherical: its own ``G``). The
  geometry enters only through ``G``.
* :func:`const_value`: the value of a field-free expression built from constants, parameters,
  powers and products, or ``None`` when it is not of that form.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..ir import expr as E
from ._xp import array_module
from .compiler import DiscretizationError


def spin_dot(a: np.ndarray, ra: int, b: np.ndarray, rb: int, G: np.ndarray) -> np.ndarray:
    """Contract the last index of ``a`` (rank ``ra``) with the first of ``b`` (rank ``rb``) with the
    spin-basis metric ``G``; grid axes pointwise (broadcast)."""
    if ra == 0 or rb == 0:
        raise DiscretizationError("dot of a scalar")
    A = array_module(a).moveaxis(a, ra - 1, 0)    # (nc, a_rest..., grid)
    ar, br = ra - 1, rb - 1

    def outer(x, y):                              # tensor axes outer, grid axes pointwise
        gx, gy = x.shape[ar:], y.shape[br:]
        x = x.reshape(x.shape[:ar] + (1,) * br + gx)
        y = y.reshape((1,) * ar + y.shape[:br] + gy)
        return x * y
    out = 0
    for s in range(G.shape[0]):
        for t in range(G.shape[1]):
            if G[s, t]:
                out = out + G[s, t] * outer(A[s], b[t])
    return out


def const_value(f: E.Expr, param: Callable[[E.Parameter], Any]) -> Any:
    """Value of a field-free scalar expression (constants, parameters through ``param``, powers,
    products), or ``None`` if ``f`` is not of that form."""
    if isinstance(f, E.Constant) and f.rank == 0:
        return f.value
    if isinstance(f, E.Parameter):
        return param(f)
    if isinstance(f, E.Power) and not f.free_fields():
        b = const_value(f.args[0], param)
        return None if b is None else b ** f.exponent
    if isinstance(f, E.Mul) and not f.free_fields():
        v = 1.0
        for a in f.args:
            w = const_value(a, param)
            if w is None:
                return None
            v = v * w
        return v
    return None
