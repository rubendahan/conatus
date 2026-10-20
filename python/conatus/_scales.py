"""Scales in the unit of the quantity they judge (critic review of the invariance audit).

Several decisions compared an eigenvalue or a parameter with ``tol * max(1, |x|)``: the ``1`` is
in the unit of ``x``, so the same problem written in another time unit (every eigenvalue times
1e4) or with an equation multiplied by a constant decided differently. The floors below are taken
from the problem itself:

* :func:`spectral_scale` -- the median modulus of the finite, nonzero eigenvalues of a computed
  spectrum (a handful of spurious modes do not move it; a spectrum whose eigenvalues are mostly
  zero falls back to its largest modulus);
* :func:`pencil_unit` -- ``||L|| / ||M||`` of a pencil ``L + s M``: the eigenvalue unit of the
  operator before any eigenvalue is known.

A quantity is "near zero" below ``SPECTRAL_FLOOR`` times such a scale, and relative tests use
``max(|x|, floor)`` instead of ``max(|x|, 1)``.
"""
from __future__ import annotations

import numpy as np

#: Fraction of the spectral scale below which an eigenvalue counts as near zero for a relative
#: test (the role the absolute 1 used to play, now in the unit of the spectrum).
SPECTRAL_FLOOR = 1e-3


def spectral_scale(values) -> float:
    """Median modulus of the finite nonzero entries of ``values``; their largest modulus when
    fewer than half are nonzero (a spectrum of mostly zero eigenvalues); 0.0 when all are zero."""
    a = np.abs(np.asarray(values, dtype=complex).ravel())
    a = a[np.isfinite(a)]
    nz = a[a > 0]
    if nz.size == 0:
        return 0.0
    if 2 * nz.size < a.size:
        return float(nz.max())
    return float(np.median(nz))


def spectral_floor(values, frac: float = SPECTRAL_FLOOR) -> float:
    """``frac`` times :func:`spectral_scale` (never below the smallest positive float)."""
    return max(frac * spectral_scale(values), np.finfo(float).tiny)


def pencil_unit(L, M, ord=1) -> float:
    """``||L|| / ||M||`` (dense or sparse, norm ``ord``): the eigenvalue unit of ``L + s M``;
    ``||L||`` when ``M`` vanishes, 1.0 when both do."""
    def nrm(A) -> float:
        if hasattr(A, "toarray"):
            A = A.toarray()
        return float(np.linalg.norm(np.asarray(A), ord)) if np.size(A) else 0.0
    nl, nm = nrm(L), nrm(M)
    if nm > 0 and nl > 0:
        return nl / nm
    return nl if nl > 0 else 1.0
