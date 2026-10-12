"""Reality check of real fields on the curvilinear grids (polar and spherical evaluators).

One copy of the three declared tolerances and of the check, shared by ``polar_eval`` and
``sphere_eval`` (they were duplicated identically). Each space declares the
values it uses (``real_tol``, ``state_real_tol``, ``func_real_tol``) and prints them in
``explain()``; a refusal names the attribute and its value.
"""
from __future__ import annotations

import numpy as np

from .compiler import DiscretizationError

__all__ = ["REAL_TOL", "STATE_REAL_TOL", "FUNC_REAL_TOL", "checked_real"]

REAL_TOL = 1e-12
"""Relative tolerance on the imaginary part of a real field inside the transforms of F (was a hidden 1e-9); rounding of the transforms is ~1e-15 relative. Declared on
``PolarSpace.real_tol`` / ``SphereSpace.real_tol``. Since the m = 0 projection at the exit of the
solves, ``solver.grid`` also checks a projected state at this tolerance."""

STATE_REAL_TOL = 1e-8
"""Relative tolerance on the imaginary part of a real field read from arbitrary coefficients
(``plan.to_grid``; ``solver.grid`` of a state whose m = 0 entries are not exactly real, e.g. set by
hand or read from a checkpoint written before), declared on ``space.state_real_tol``. That imaginary part is exactly the non-real m = 0 component of the coefficients (B(J c) =
conj(B c), so Re g is the exact projection onto real fields): without the projection it is born
from the rounding of the tau solves and is never reprojected by the step. Measured without the
projection: u 1.4e-12 after 100 steps at 128x63x48 (CPU and GPU), p 1e-11 after 2000 steps of a
decaying shell flow; an inconsistent state (u_+ != conj(u_-)) gives O(1)."""

FUNC_REAL_TOL = 1e-4
"""Relative tolerance on the imaginary part of the argument of a pointwise function of real fields
in the RHS evaluator (T**2, exp(T), ...), declared on ``space.func_real_tol``: the function
acts on Re g, the exact projection onto real fields. Without the m = 0 projection that
imaginary part, relative to a decaying field, grew most when the field fell to the rounding level of
the coupled state: measured up to 6.2e-8 (T) and 5.7e-7 (p) on a coupled (u, p, T) shell flow
decaying to 1e-32 (run); REAL_TOL = 1e-12 refused T**2 at amplitude
1e-20 (run). The bound is a measured margin (175x above the worst
sound case, 1e4 below an inconsistent state), not a proven one; it is kept after because the
evaluator also serves ``project_m0=False`` solvers, unprojected initial states and FlowMap seeds."""


def checked_real(g: np.ndarray, what: str, tol: float, name: str) -> np.ndarray:
    """Physical values of a real field: Re g, after checking that |Im g| <= tol * max|g| (``name``:
    the declared attribute of the space that holds ``tol``, quoted in the refusal)."""
    scale = max(float(np.max(np.abs(g))) if g.size else 0.0, 1e-300)
    im = float(np.max(np.abs(g.imag))) if g.size else 0.0
    if im > tol * scale:
        raise DiscretizationError(
            f"{what}: a real field has an imaginary part {im:.1e} (relative {im / scale:.1e} > "
            f"plan.space.{name} = {tol:g}); the m = 0 spin components are probably inconsistent "
            "(u_+ != conj(u_-)), or the field has decayed to the rounding level of the state")
    return g.real
