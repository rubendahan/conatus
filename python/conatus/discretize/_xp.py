"""Array-module helpers for the curvilinear transforms and evaluators (NumPy on CPU, CuPy on GPU).

The polar and spherical spaces are shared by the CPU plan and a GPU solver: the array module of a
call is the module of its *input* array (``array_module``), never a global switch, and the constant
matrices (radial stacks, harmonics, operators) are copied to the device once and cached next to
their host version (``on``). CuPy is imported only when a CuPy array is actually seen.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._arrays import array_module, is_device  # noqa: F401 (re-export: one definition)

__all__ = ["array_module", "on", "is_device"]


def on(xp, a: np.ndarray, cache: dict, key) -> Any:
    """``a`` in module ``xp``: the host array itself for NumPy, a cached device copy for CuPy
    (``key`` identifies ``a`` in ``cache``; the copy is made once)."""
    if xp is np:
        return a
    dk = ("__device__", key)
    d = cache.get(dk)
    if d is None:
        d = cache[dk] = xp.asarray(a)
    return d
