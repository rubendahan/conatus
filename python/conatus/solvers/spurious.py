"""Spectral-tail test of eigenvectors: the cheap spurious-mode flag of every eigenvalue result.

. A tau discretization of a problem with walls has finite
eigenvalues that belong to no physical mode (Gardner, Trogdon & Douglass 1989 for Orr-Sommerfeld):
OS at N = 64 without a filter put c = 0.0174 + 1470 i first, with a backward residual of 3e-17, so
the residual (the project's proof of a mode) cannot catch them. Their eigenvectors can: the
energy of such a mode sits in the last coefficients of the non-periodic axis (relative tail
energy 0.96..1.0 measured), while a resolved mode has a tail of 1e-17..1e-7 there.

Test (declared in :mod:`conatus.tolerances`): for each unknown of the mode (tau variables left out)
that depends on the non-periodic coordinate, the energy sum |c_n|^2 of its coefficients in the
last ``SPURIOUS_TAIL_FRACTION`` of that axis (at least two coefficients; per piece of a compound
axis), divided by its total energy; the mode's tail is the largest of these ratios over the
unknowns that carry at least ``SPURIOUS_SIGNIFICANT`` of the energy of the most excited one (a
field that the mode does not excite is round-off, never judged). A tail above
``SPURIOUS_TAIL_TOL`` flags the mode: it stays in the result (``values``, ``mode(i)``), is listed
by ``explain()``, and is ranked after every mode that passes by ``order()``; ``least_stable`` and
``critical`` never select it. It is a detection, not a filter: the drift test
(``drift=N2, drift_tol=``) stays the stronger opt-in filter that removes modes. ``spurious="none"``
switches the flag off (declared in the result).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..tolerances import SPURIOUS_SIGNIFICANT, SPURIOUS_TAIL_FRACTION, SPURIOUS_TAIL_TOL

__all__ = ["SPURIOUS_MODES", "check_spurious", "tail_layout", "spectral_tails", "check_selected",
           "unresolved_sentence"]

#: Accepted values of ``spurious=`` (default ``"tail"``).
SPURIOUS_MODES = ("tail", "none")


def check_spurious(spurious: Any, where: str) -> str:
    """``spurious`` validated (``"tail"`` or ``"none"``), else :class:`OptionError`."""
    if spurious not in SPURIOUS_MODES:
        from ..errors import OptionError, did_you_mean
        raise OptionError(f"{where}(spurious={spurious!r})",
                          why="'tail' flags the modes whose eigenvector has its energy in the last "
                              "coefficients (spurious tau modes) and ranks them last; 'none' "
                              "switches the flag off",
                          fix="spurious='tail' (default) or spurious='none'"
                              + did_you_mean(str(spurious), SPURIOUS_MODES),
                          doc="docs/site/guide/dynamics/eigenvalues.md")
    return spurious


def _pieces(basis: Any, nz: int) -> list[tuple[int, int]]:
    """(start, stop) of each piece of the non-periodic axis (one piece unless compound)."""
    sizes = getattr(basis, "sizes", None)
    if sizes and sum(sizes) == nz:
        out, s = [], 0
        for n in sizes:
            out.append((s, s + int(n)))
            s += int(n)
        return out
    return [(0, nz)]


def tail_layout(plan: Any, group: Any, reduced: bool) -> tuple[list, str] | tuple[None, str]:
    """([(unknown, index array (rows, nz), pieces)], note) of the vectors of ``group`` of
    ``plan``, or (None, why) when the test does not apply (no tau, no non-periodic axis, a plan
    without the Cartesian layout). Indices are into the vector the solver returns (reduced or
    full), -1 where a coefficient is outside the symmetry subspace."""
    from .evp_modes import _tau_names, _unpack
    if not hasattr(plan, "var_layouts") or not hasattr(getattr(plan, "layout", None),
                                                         "fourier_indices"):
        return None, ("spectral-tail test not available for this plan "
                      f"({type(plan).__name__}): no spurious-mode flag")
    try:
        taus = _tau_names(plan)
    except Exception:  # noqa: BLE001 -- a plan without the IR layout has no tau to report
        taus = set()
    if not taus:
        return None, ""                      # no tau: no spurious tau mode to flag
    zc = plan.layout.zcoord
    if zc is None:
        return None, ""
    coords = plan.domain.coordsys.coords
    gm = plan.matrices(group, reduced) if reduced else None
    n_full = plan.matrices(group, False).L.shape[1]
    probe = _unpack(plan, group, np.arange(1, n_full + 1, dtype=float))
    if reduced:                              # full index -> position in the reduced vector
        pos = np.full(n_full + 1, -1, dtype=np.int64)
        pos[np.asarray(gm.col_index) + 1] = np.arange(len(gm.col_index))
    out = []
    basis = plan.domain.basis(zc)
    for v in plan.variables:
        if v.name in taus:
            continue
        arr = np.asarray(probe[v.name]).real.astype(np.int64)
        za = arr.ndim - len(coords) + coords.index(zc)
        idx = np.moveaxis(arr, za, -1).reshape(-1, arr.shape[za])
        idx = idx[np.any(idx > 0, axis=1)]
        if idx.size == 0 or idx.shape[1] < 4:
            continue                         # no dependence on the wall coordinate
        idx = (pos[idx] if reduced else idx - 1)
        out.append((v.name, idx, _pieces(basis, idx.shape[1])))
    if not out:
        return None, ""
    return out, ""


def spectral_tails(layout: list, V: np.ndarray) -> np.ndarray:
    """Relative tail energy of each column of ``V`` (see the module docstring)."""
    V = np.asarray(V)
    m = V.shape[1]
    ratios, energies = [], []
    for _name, idx, pieces in layout:
        C = np.where((idx >= 0)[..., None], V[np.maximum(idx, 0)], 0.0)   # (rows, nz, m)
        E = np.abs(C) ** 2
        tot = E.sum(axis=(0, 1))
        tail = np.zeros(m)
        for a, b in pieces:
            k = max(2, int(np.ceil(SPURIOUS_TAIL_FRACTION * (b - a))))
            tail = tail + E[:, b - k:b, :].sum(axis=(0, 1))
        energies.append(tot)
        ratios.append(np.where(tot > 0, tail / np.where(tot > 0, tot, 1.0), 0.0))
    energies = np.array(energies)
    ratios = np.array(ratios)
    top = energies.max(axis=0)
    significant = energies >= SPURIOUS_SIGNIFICANT * np.where(top > 0, top, 1.0)
    return np.where(significant, ratios, 0.0).max(axis=0)


def flag(res: Any, layout: list | None, note: str) -> None:
    """Set ``res.tails``, ``res.spurious`` and ``res.spurious_note`` (in place)."""
    if layout is None or res.vectors is None or res.values.size == 0:
        res.tails, res.spurious = None, None
        res.spurious_note = note
        return
    res.tails = spectral_tails(layout, res.vectors)
    res.spurious = res.tails > SPURIOUS_TAIL_TOL
    res.spurious_note = (
        f"spectral-tail test (default, spurious='tail'): {int(res.spurious.sum())} of "
        f"{res.values.size} modes flagged as spurious tau or unresolved modes (relative energy in the last "
        f"{SPURIOUS_TAIL_FRACTION:g} of the wall-normal coefficients above SPURIOUS_TAIL_TOL = "
        f"{SPURIOUS_TAIL_TOL:g}); they are kept and ranked last by order()")


def expected_accuracy(tail: float) -> float:
    """Relative accuracy to expect of an eigenvalue whose mode has the spectral tail ``tail``
   : the tail is an *energy* fraction, so the truncated coefficients are
    about sqrt(tail) of the mode in amplitude, and the eigenvalue error is of that order (an order
    of magnitude, not a bound: Orr-Sommerfeld at N = 32, tail 1e-8, eigenvalue right to 4e-5)."""
    return float(np.sqrt(max(tail, 0.0)))


def unresolved_sentence(what: str, tail: float) -> str:
    """The one sentence of an under-resolved selected mode (the expected accuracy)."""
    from ..tolerances import SPURIOUS_TAIL_FRACTION, UNRESOLVED_TAIL_TOL
    return (f"selected mode under-resolved: {what} has a spectral tail {tail:.1e} > "
            f"UNRESOLVED_TAIL_TOL = {UNRESOLVED_TAIL_TOL:g} (relative energy in the last "
            f"{SPURIOUS_TAIL_FRACTION:g} of its wall-normal coefficients), so its value is not "
            f"converged in N: expect a relative eigenvalue accuracy of about sqrt(tail) = "
            f"{expected_accuracy(tail):.0e} (an order of magnitude, not a bound); to measure it, use "
            "drift=N2, drift_tol=1e-6 (only modes found again at N2 are kept) or a larger N")


def check_selected(res: Any) -> None:
    """Resolution of the mode ``res.order()[0]`` of a result without a drift test: above ``UNRESOLVED_TAIL_TOL`` the sentence goes to
    ``res.unresolved_note`` (repr, explain()) and one :class:`~conatus.errors.ResolutionWarning`
    is raised. With a drift test the drift is the evidence; without a tail test (no tau, polar
    plans) there is nothing to measure (said by ``spurious_note``)."""
    from ..tolerances import UNRESOLVED_TAIL_TOL
    res.unresolved_note = ""
    if res.drift is not None or res.tails is None or res.values.size == 0:
        return
    i = int(res.order()[0])            # flagged modes come last: flagged here = all are
    t = float(res.tails[i])
    if t > UNRESOLVED_TAIL_TOL:
        res.unresolved_note = unresolved_sentence(
            f"{res.eigenvalue} = {res.values[i]:.10g} (order()[0], index {i})", t)
        from ..errors import ResolutionWarning, _warn_user
        _warn_user(res.unresolved_note, ResolutionWarning)
