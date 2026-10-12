"""Floquet–Hill toolkit (decision D9).

Two methods, always chosen explicitly (never switched silently):

- ``"chebyshev_radiation"``: Hill × Chebyshev on [0, Y] with a holomorphic radiation
  condition at y = Y;
  difference) basis, declared precision (implemented for S = 0, ``cascade.py``).
"""
from __future__ import annotations

from .hill_general import HillBlocks, exponents_dense, hill_from_plan, secant_root
from .hill_gpu import HillGPU
from .roots import CertifiedRoot, SecantRoot, fold_floquet
from .shell import shell_floquet

__all__ = ["HillBlocks",
           "hill_from_plan", "secant_root", "exponents_dense", "HillGPU", "CertifiedRoot", "SecantRoot", "fold_floquet", "shell_floquet"]

_METHODS = ("chebyshev_radiation", )


