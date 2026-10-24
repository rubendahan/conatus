"""Contour methods in the complex plane.

* :func:`count` -- the number of eigenvalues of ``T(z)`` in a region by the argument principle on
  ``det T`` (unit phase and log modulus from an LU factorisation, adaptive boundary walk with a
  principal-increment bound, integer winding checked, ``N = Z - P`` stated; a test on samples
  of the boundary, whose scope is ``CountResult.scope``);
* :func:`beyn` -- every eigenvalue in the region by Beyn's contour-integral method (Beyn 2012),
  for linear and nonlinear eigenproblems ``T(z) v = 0``, each root polished and accepted by its
  residual and a local count, the roots adding up to the count;
* :func:`box` -- the region, ``cn.box(re=(a, b), im=(c, d))``, also taken by
  ``m.eigenvalues(N, ..., region=...)`` (:class:`RegionEVPResult`).

``T`` is a pair ``(L, M)`` (the pencil ``L + z M``), a :class:`conatus.floquet.HillBlocks` or a
callable ``z -> matrix`` (or scalar). Mathematics: guide:
``docs/site/guide/dynamics/contour.md``.
"""
from __future__ import annotations

from .beyn import BeynResult, beyn
from .count import CountResult, count
from .evp import RegionEVPResult
from .region import ContourBox, box

__all__ = ["box", "ContourBox", "count", "CountResult", "beyn", "BeynResult", "RegionEVPResult",
           "explain"]


def explain() -> str:
    """What the package computes and what it checks."""
    from .beyn import RESIDUAL_TOL
    from .count import PHASE_STEP_MAX
    return "\n".join([
        "conatus.contour: eigenvalues of T(z) v = 0 in a region of the complex plane",
        "  count(T, cn.box(re=, im=)): winding number of det T along the boundary = Z - P (zeros "
        "minus poles inside); det T as unit phase + log modulus from an LU (never formed), "
        f"adaptive walk with phase increments <= {PHASE_STEP_MAX:.3g} rad; a non-integer winding, "
        "an eigenvalue on the boundary or a discontinuous determinant is refused; a test on "
        "samples: for a T with poles, the walk follows where log det T bends and warns when a "
        "zero or pole is close to an edge; a zero and a pole straddling the boundary whose bend "
        "a smooth variation of det T hides are not seen (CountResult.scope)",
        "  beyn(T, cn.box(...)): Beyn's contour-integral method (block Hankel moments, rank-"
        "revealing SVD), every root polished by nonlinear inverse iteration and accepted when "
        f"its residual is <= {RESIDUAL_TOL:g} (default; scale in BeynResult.residual_definition) "
        "and a square around it counts at least one eigenvalue; the counts must add up to the "
        "count, else refused",
        "  T: (L, M) for L + z M, a HillBlocks, or a callable z -> matrix or scalar (poles= "
        "declares the poles of det T in the box)",
        "  m.eigenvalues(N, ..., region=cn.box(...)): the same on the pencil of one group "
        "(res.count, res.count_method, res.multiplicity)"])
