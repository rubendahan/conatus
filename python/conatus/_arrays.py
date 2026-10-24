"""Array-module helpers shared by every layer (NumPy on the CPU, CuPy on the GPU): which module an
array belongs to and its host copy. CuPy is imported only when a CuPy array is actually seen.
One definition for the package (``discretize._xp``, ``datadriven.norms``,
``diagnostics.grid``, ``diagnostics.properties`` and ``io.analysis`` had their own copies)."""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["is_device", "array_module", "host_array"]


def is_device(a: Any) -> bool:
    """True for a CuPy array (checked by type module name: no CuPy import on CPU-only nodes)."""
    return type(a).__module__.split(".")[0] == "cupy"


def array_module(*arrays: Any):
    """``cupy`` if any argument is a CuPy array, else ``numpy``."""
    for a in arrays:
        if is_device(a):
            import cupy
            return cupy
    return np


def host_array(a: Any) -> np.ndarray:
    """``a`` as a NumPy array: a CuPy array (any object with ``.get()`` that is not an ndarray) is
    copied to the host, anything else goes through ``np.asarray`` (no copy when possible)."""
    if hasattr(a, "get") and not isinstance(a, np.ndarray):     # CuPy array
        a = a.get()
    return np.asarray(a)
