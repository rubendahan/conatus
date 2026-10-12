"""Column border of the group systems: a few dense columns taken out of the band.

In the first-order spherical shell (and any tau form that lifts a tau inside a gradient), the
tau of the gradient, lifted then multiplied by r, has non-zeros in every radial row of the
momentum equation: placed among the other columns it widens the band of its group to about half
the system (kl ~ 99 for n = 198), so the banded solve costs as much as a dense one. The bordered
solver then accepts a **column border** (``col_border=True``, core/include/conatus/banded.hpp):

- the c dense columns are moved among the pinned columns (in front, with those of the boundary
  rows), and c bulk rows are moved among the border rows, so the border has rank r + c;
- every entry of a bulk row in a pinned column leaves the band and is solved exactly around the
  band factors (one sparse update before / after the band solve), the border rows by Woodbury;
- the remaining core (bulk rows x other columns) is banded with the band of the other columns.

The border is solved by its Schur complement S = A00 - A01 A11^{-1} A10, whose inverse is the
block of A^{-1} on (border columns, border rows). Which bulk rows go to the border is therefore
chosen to make that block as well conditioned as possible: a column-pivoted QR (max-volume
choice) of the rows of A^{-1} belonging to the border columns, restricted to the bulk rows after
projecting out the boundary rows already in the border, on the first group of each sparsity
bucket at a generic combination M + 0.3713 L (one dense inverse per bucket, at construction).
Measured on the B6 shell (N_r = 48, pencils b = 1e-6 .. 1): solutions equal to the dense solve
to 2e-13 relative at worst (the row border: 4e-13). The result is checked (condition of the
core, band gain) and the plain row border is kept, with the reason, otherwise; each
factorization is then checked again by a backward error (timestep/polar.py,
_check_conditioning). Nothing is approximated: the same matrix is solved, in another layout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from .._repr import compact_repr

__all__ = ["ColumnBorderPlan", "plan_column_border", "BORDER_MODES"]

BORDER_MODES = ("auto", "rows")
MAX_COLS = 8               # at most this many dense columns moved (border rank r + c <= 64 on the GPU)
MIN_GAIN = 2.0             # band-cost ratio (row border / column border) required to use it
GENERIC_B = 0.3713         # generic pencil M + b L used to choose the rows (not a time step)
CORE_COND_MAX = 1e10       # the core (bulk rows x other columns) at that pencil must stay this well conditioned


@dataclass
class ColumnBorderPlan:
    """New relative order of one bucket: ``rows``/``cols`` index the old permuted positions."""
    __repr__ = compact_repr   # one line, arrays by shape

    rows: np.ndarray
    cols: np.ndarray
    n_border: int          # r + c
    n_cols: int            # c: dense columns moved
    band_before: tuple     # (kl, ku) with the row border only
    band_after: tuple      # (kl, ku) of the core
    core_cond: float

    def explain(self) -> str:
        return (f"column border: {self.n_cols} dense column(s) and {self.n_cols} row(s) moved to the "
                f"border (rank {self.n_border}); band kl/ku {self.band_before[0]}/{self.band_before[1]} -> "
                f"{self.band_after[0]}/{self.band_after[1]} (core cond {self.core_cond:.1e} at M + "
                f"{GENERIC_B} L)")


def _extents(P: sp.csc_matrix, nb: int) -> np.ndarray:
    """Max |i - j| over the bulk rows (i >= nb) of every column j (0 for j < nb or empty)."""
    n = P.shape[0]
    ext = np.zeros(n, int)
    for j in range(nb, n):
        rows = P.indices[P.indptr[j]:P.indptr[j + 1]]
        rows = rows[rows >= nb]
        if rows.size:
            ext[j] = int(max(j - rows.min(), rows.max() - j))
    return ext


def _band(A: np.ndarray, k: int) -> tuple[int, int]:
    """(kl, ku) of the bulk rows >= k over the columns >= k."""
    ii, jj = np.nonzero(A[k:, k:])
    if ii.size == 0:
        return 0, 0
    return int(max(0, (ii - jj).max())), int(max(0, (jj - ii).max()))


def plan_column_border(M, L, rperm, cperm, nb: int) -> tuple[ColumnBorderPlan | None, str]:
    """Column border for one group (permuted by rperm/cperm, nb border rows): the plan, or None
    and the reason it is not used (no dense column, too small a gain, singular core)."""
    G = (M + GENERIC_B * L).tocsr()[rperm][:, cperm]
    n = G.shape[0]
    if nb == 0 or n < 16:
        return None, "no boundary rows" if nb == 0 else "group too small"
    P = sp.csc_matrix((abs(G) > 0).astype(np.int8))
    ext = _extents(P, nb)
    order = np.argsort(-ext[nb:], kind="stable") + nb       # widest columns first
    # cost model per row of the banded solve: ~2 max(kl, ku) for the band, ~2 (rank) for Woodbury
    e = [int(ext[order[c]]) if c < order.size else 0 for c in range(MAX_COLS + 1)]
    cost = [2 * e[c] + nb + c for c in range(MAX_COLS + 1)]
    c = int(np.argmin(cost))
    if c == 0 or cost[0] < MIN_GAIN * cost[c]:
        return None, f"no dense column worth moving (band {e[0]}, best gain {cost[0] / cost[c]:.2f}x)"
    dense = np.sort(order[:c])
    A = G.toarray()
    keep_cols = np.array([j for j in range(nb, n) if j not in set(dense.tolist())], int)
    try:
        Z = np.linalg.inv(A)[np.concatenate([np.arange(nb), dense])]     # (nb + c) x n
    except np.linalg.LinAlgError:
        return None, f"pencil M + {GENERIC_B} L singular (rows not chosen)"
    if nb:
        Q, _ = np.linalg.qr(Z[:, :nb])                     # boundary rows already in the border
        Zb = Z[:, nb:] - Q @ (Q.conj().T @ Z[:, nb:])
    else:
        Zb = Z[:, nb:]
    _, _, piv = sla.qr(Zb, pivoting=True, mode="economic")
    chosen = np.sort(piv[:c] + nb)
    rest_rows = np.array([i for i in range(nb, n) if i not in set(chosen.tolist())], int)
    rows = np.concatenate([np.arange(nb), chosen, rest_rows])
    cols = np.concatenate([np.arange(nb), dense, keep_cols])
    k = nb + c
    A2 = A[rows][:, cols]
    core_cond = float(np.linalg.cond(A2[k:, k:]))
    if not core_cond <= CORE_COND_MAX:
        return None, f"core singular without the dense columns (cond {core_cond:.1e})"
    kl0, ku0 = _band(A, nb)
    kl1, ku1 = _band(A2, k)
    if 2 * max(kl0, ku0) + nb < MIN_GAIN * (2 * max(kl1, ku1) + k):
        return None, f"band gain too small after reordering ({kl0}/{ku0} -> {kl1}/{ku1})"
    return ColumnBorderPlan(rows, cols, k, c, (kl0, ku0), (kl1, ku1), core_cond), ""
