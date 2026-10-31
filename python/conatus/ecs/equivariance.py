"""Preconditions of the invariant-solution finders: autonomy and equivariance, checked numerically.

U(x - c t) solves the equation only when the vector field commutes with translations, and
a steady or periodic state of F(X, 0) means something only when F does not depend on time. A small
residual of the finder's own equation establishes neither of these: u_t + u = cos x + sin x has
a zero comoving residual at U = cos x, c = 1, and no such travelling solution.

Both checks evaluate the right-hand side at a few states. The translation is one cell of the grid,
which maps the grid to itself, so a pseudospectral product commutes with it exactly at any
resolution, whatever the content of the state. It is a cell of the dealiased grid, where products are
evaluated: a cell of the coefficient grid is not one on a 3/2 grid, and the aliases of a rough seed
then break the commutation in an equivariant problem. A check that passes on these probes is a
necessary condition, not a proof."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from ..errors import OptionError, SymmetryError

#: Relative defect above which the right-hand side does not have the symmetry (or autonomy)
EQUIVARIANCE_TOL = 1e-9
#: Times at which the right-hand side is compared with its value at t = 0
_PROBE_TIMES = (0.7371, 2.3)
#: A probe state holds wavenumbers up to this fraction of the largest, so a product stays resolved
LOW_MODE_FRACTION = 0.5


def _size(*arrays) -> float:
    return max((float(np.linalg.norm(a)) for a in arrays), default=0.0)


def grid_cell(solver, coord: str) -> float:
    """One cell of the grid on which the right-hand side of ``solver`` evaluates products along ``coord``."""
    for b in solver.plan.domain.bases:
        if b is not None and b.coord.name == coord:
            return float(b.length) / int(b.grid_size)
    raise OptionError(f"no basis along {coord!r}")


def require_autonomous(solver, probes: Iterable[np.ndarray], where: str) -> None:
    """Refuse a right-hand side that depends on the time (SymmetryError)."""
    L = solver.system.L
    for X in probes:
        F0 = solver.F(X, 0.0)
        for t in _PROBE_TIMES:
            d = _size(solver.F(X, t) - F0)
            if d > EQUIVARIANCE_TOL * max(_size(F0, L @ X), np.finfo(float).tiny):
                raise SymmetryError(
                    f"{where}: the right-hand side depends on time (relative change {d / max(_size(F0, L @ X), np.finfo(float).tiny):.1e} "
                    f"at t = {t:g})",
                    why="this finder solves an equation for the vector field at t = 0; a forced "
                        "system has no such steady or relative solution",
                    fix="integrate it in time (m.simulate), or for a time-periodic system use the "
                        "Floquet tools")


def equivariance_defect(solver, act: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> float:
    """max(|r(g X) - g r(X)|, |M g X - g M X|), r = F - L X, relative to the size of the terms."""
    L, M = solver.system.L, solver.system.M
    gX = act(X)
    F, Fg = solver.F(X, 0.0), solver.F(gX, 0.0)
    r, rg = F - L @ X, Fg - L @ gX
    size = max(_size(F, Fg, L @ X, L @ gX), np.finfo(float).tiny)
    msize = max(_size(M @ X), np.finfo(float).tiny)
    return max(_size(rg - act(r)) / size, _size(M @ gX - act(M @ X)) / msize)


def require_equivariant(solver, act: Callable[[np.ndarray], np.ndarray], probes: Iterable[np.ndarray],
                        where: str, symmetry: str, fix: str) -> None:
    """Refuse a problem whose right-hand side or mass does not commute with ``act`` (SymmetryError)."""
    for X in probes:
        d = equivariance_defect(solver, act, X)
        if not d <= EQUIVARIANCE_TOL:
            raise SymmetryError(
                f"{where}: the equations do not have the {symmetry} this solution type assumes "
                f"(relative defect {d:.1e} > EQUIVARIANCE_TOL = {EQUIVARIANCE_TOL:g})",
                why=f"a coefficient or a forcing that depends on the coordinate breaks the {symmetry}, "
                    "so the finder's residual would not describe a solution of these equations",
                fix=fix)


def low_mode_probe(like: np.ndarray, wavenumber: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """A deterministic random state of size ``like`` (entries with a wavenumber above half the
    largest are 0, so a product stays resolved)."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal(np.shape(like))
    if wavenumber is not None:
        V = np.where(wavenumber <= LOW_MODE_FRACTION * np.max(wavenumber, initial=0.0), V, 0.0)
    return V * (1e-3 * max(float(np.max(np.abs(like), initial=0.0)), 1.0))
