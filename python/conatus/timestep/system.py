"""Global IVP system: all Fourier groups stacked, gather/scatter maps, batched banded solves.

The state is one vector ``X`` = concatenation of the group vectors (full space, or the
symmetry subspace if the problem declares one). Every map between ``X`` and the variable
coefficient arrays is a precomputed integer gather/scatter (no Python loop over groups
at run time). Linear solves go to the C++ bordered banded solver, batched over all groups
that share one sparsity structure (``BorderedBandedBatch``).

Ordering (point 6): the plan stores each group "variable-major" (all Chebyshev modes
of one variable, then the next), so coupled systems have a band ~N wide. ``ordering=
"interleaved"`` permutes rows and columns inside each group by Chebyshev index (all
variables at mode n, then n+1, ...), boundary/constraint rows first, with the columns they
pin (lowest modes of the variables they constrain) in front: the band becomes
~ (number of blocks) x (operator bandwidth). The permutation is public (``explain()``).

Complexified solves (opt-in ``complexify=True``): with one RealFourier direction, a group
k > 0 holds a (cos, -sin) pair of every object and d/dx couples them, so its real system is
[[A, -B], [B, A]] of size 2m. That structure is checked numerically for M and L (exactly, up
to rounding; any mismatch is refused, never silently skipped) and the group is then solved as
the complex system (A + iB) z = r_c + i r_s of size m, with x_c = Re z, x_s = Im z: half the
unknowns and roughly half the band. The real path stays the oracle (``complexify=False``).
With **two or more** RealFourier directions (e.g. the channel in x and z), a group
k_x > 0, k_z > 0 holds the 4 products cos/sin x cos/sin of every object; it is the real form of
the two complex modes exp(i(k_x x +- k_z z)). Writing each basis function as
Re((i)^p e^{i theta}) (p = 0: cos, 1: -sin), the complex amplitudes are
C_s = sum_p w_s(p) a_p, w_s(p) = prod_j (i s_j)^{p_j}, s in {+1} x {+-1}^{d-1}, and
a_p = 2^{1-d} sum_s Re(conj(w_s(p)) C_s). Each group is then solved as 2^{d-1} complex systems
of the size of one mode (the matrices are extracted and their complex linearity is checked
numerically, refused otherwise). Its transpose is the same map with A_s^H.

Decoupled blocks (``decouple=True``, opt-in): a group whose system is block-diagonal
after a permutation (e.g. poloidal and toroidal potentials of the channel) is split into its
connected components, each solved as its own (smaller, narrower-band) system. Exact: same
equations, no pivoting across blocks is possible. Declared in ``explain()``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import structural_rank

from ..discretize.compiler import _comps
from ..discretize.plan import Plan, RankError
from ..runtime import BorderedBandedBatch
from .shared_factors import SharedFactor
from ..errors import OptionError, SymmetryError
from .._repr import compact_repr

__all__ = ["GlobalSystem", "GroupSlot", "ORDERINGS"]

ORDERINGS = ("interleaved", "variable-major")


@dataclass
class GroupSlot:
    """One group inside the global vector."""
    __repr__ = compact_repr   # one line, arrays by shape

    group: tuple
    label: str
    offset: int
    size: int
    M: sp.csr_matrix
    L: sp.csr_matrix
    row_kind: list[str]
    rperm: np.ndarray          # permuted position -> group row
    cperm: np.ndarray          # permuted position -> group column
    n_border: int
    block: str = ""            # decouple=True: which connected component of the group
    col_border: int = 0        # dense columns moved to the border (column border, timestep/colborder.py)


def _meta(plan: Plan, group, layouts, shapes, comps_of) -> list[tuple[int, int, int | None]]:
    """For each entry of a full group vector: (object index, flat index in the object's
    coefficient array, Chebyshev index or None)."""
    L = plan.layout
    coords = plan.domain.coordsys.coords
    zc = L.zcoord
    fax = [coords.index(d.coord) for d in L.fdirs]
    out: list = []
    for k, (ol, shape, comps) in enumerate(zip(layouts, shapes, comps_of, strict=True)):
        fidx = L.fourier_indices(ol.fdep, group)
        rank = len(comps[0]) if comps and comps[0] else 0
        for _ci, c in enumerate(comps):
            for fi in fidx:
                idx = list(c) + [0] * len(coords)
                for a, j in zip(fax, fi, strict=True):
                    idx[rank + a] = j
                if ol.zdep:
                    for n in range(ol.nz):
                        idx[rank + coords.index(zc)] = n
                        out.append((k, int(np.ravel_multi_index(idx, shape)), n))
                else:
                    out.append((k, int(np.ravel_multi_index(idx, shape)), None))
    return out


def _fpos_multi(plan: Plan, group, layouts, comps_of) -> list[tuple]:
    """Per entry of a full group vector, the position in each Fourier direction: 0 (cos) or 1
    (-sin) where the object holds a RealFourier pair in that direction, None otherwise (object
    constant along it, zero mode or complex direction). Same order as ``_meta``."""
    L = plan.layout
    out: list = []
    for ol, comps in zip(layouts, comps_of, strict=True):
        fidx = L.fourier_indices(ol.fdep, group)
        per = [len(it.indices) == 2 and dep for dep, it in zip(ol.fdep, group, strict=True)]
        for _c in comps:
            for fi in fidx:
                pos = tuple((j % 2) if pair else None for j, pair in zip(fi, per, strict=True))
                out.extend([pos] * (ol.nz if ol.zdep else 1))
    return out


def _components(gm) -> list[tuple[np.ndarray, np.ndarray]]:
    """Connected components (rows, cols) of the bipartite graph of |L| + |M| of one group."""
    from scipy.sparse.csgraph import connected_components
    n = gm.L.shape[0]
    pat = sp.csr_matrix((abs(gm.L) + abs(gm.M)) > 0, dtype=np.int8)
    big = sp.bmat([[None, pat], [pat.T, None]], format="csr")
    nc, lab = connected_components(big, directed=False)
    out = []
    for c in range(nc):
        ri = np.nonzero(lab[:n] == c)[0]
        ci = np.nonzero(lab[n:] == c)[0]
        if len(ri) != len(ci):
            raise RankError(f"[{gm.label}] decoupled block with {len(ri)} rows and {len(ci)} "
                            "columns: not square (structurally singular)")
        out.append((ri, ci))
    out.sort(key=lambda rc: rc[1][0])
    return out


def _piece_quotas(plan, n, patc, col_n, col_kind, rm, kinds):
    """Compound Chebyshev axis: (piece of a z-index, tau columns per piece, zero counters, piece
    offsets); (None, {}, {}, None) for any other plan (polar and spherical systems share the
    ordering code and have no compound axis). A tau column belongs to the piece of the bulk rows
    it is lifted into (median z-index of its rows)."""
    fam = getattr(getattr(plan, "layout", None), "zfam", None)
    offs = getattr(fam, "offsets", None)
    if offs is None:
        return None, {}, {}, None
    offs = np.asarray(offs)

    def piece(nn: int) -> int:
        return int(np.searchsorted(offs, nn, side="right") - 1)

    quota: dict[int, int] = {}
    for j in range(n):
        if col_kind[j] == "field" and col_n[j] is not None:
            continue
        rows = [r for r in patc.indices[patc.indptr[j]:patc.indptr[j + 1]]
                if kinds[r] == "bulk" and rm[r][2] is not None]
        if rows:
            p = piece(int(np.median([rm[r][2] for r in rows])))
            quota[p] = quota.get(p, 0) + 1
    return piece, quota, {p: 0 for p in range(len(offs))}, offs


def _compound_pinning(pat, border, pinned, kinds, col_n, col_kind, piece, offs, label):
    """Pinned columns on a compound interval such that the banded part stays square on every
    piece and every decoupled block within it (cos/sin halves of a real Fourier group, ...).

    The greedy per-piece quota above is kept when it works (structural rank of the banded part);
    with two or more interfaces it can hand a piece's last interface row to the wrong piece or
    the wrong cos/sin half (Fourier x 3 pieces, "banded part is singular").
    Then the columns are those that the border rows take in a perfect matching of the whole
    pattern of minimal weight (weight 1 + mode within its piece for a field column of a border
    row, 10^6 for any other column of a border row, 1 for a bulk row): a perfect matching exists
    (structural rank checked by the caller's system), and the bulk rows are then matched inside
    the remaining columns, so the banded part is structurally nonsingular."""
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    n = pat.shape[0]
    is_border = np.zeros(n, bool)
    is_border[border] = True
    rest = np.setdiff1d(np.arange(n), pinned)
    bulk = np.nonzero(~is_border)[0]
    if structural_rank(sp.csr_matrix(pat[bulk][:, rest])) == len(bulk):
        return pinned
    P = sp.csr_matrix(pat, dtype=float, copy=True)
    P.data[:] = 1.0
    for i in border:
        lo, hi = P.indptr[i], P.indptr[i + 1]
        for t in range(lo, hi):
            j = P.indices[t]
            P.data[t] = (1.0 + col_n[j] - offs[piece(col_n[j])]
                         if col_kind[j] == "field" and col_n[j] is not None else 1e6)
    try:
        _rows, cols = min_weight_full_bipartite_matching(P)
    except ValueError as err:
        raise RankError(f"[{label}] compound interval: no ordering keeps the banded part square "
                        f"({err})") from None
    return [int(cols[i]) for i in border]


_BUILD_SELF = None          # (system, groups, reduced) of a parallel build, read by forked workers


def _build_group_worker(i: int) -> list:
    sysm, groups, reduced = _BUILD_SELF
    return sysm._build_group(groups[i], reduced)


class GlobalSystem:
    """Stacked groups of an IVP plan, with gather/scatter maps and batched solvers."""

    def __init__(self, plan: Plan, reduced: bool | None = None,
                 ordering: str = "interleaved", complexify: bool = False,
                 decouple: bool = False, build_workers: int | None = None) -> None:
        if ordering not in ORDERINGS:
            raise OptionError(f"ordering must be one of {ORDERINGS}",
                              fix="ordering='interleaved' or ordering='variable-major'")
        if reduced is None:
            reduced = plan.problem.symmetry is not None
        self.plan, self.reduced, self.ordering = plan, reduced, ordering
        self.complexify = bool(complexify)
        self.decouple = bool(decouple)
        # worker processes of the group assembly: argument, else $CONATUS_BUILD_WORKERS, else
        # serial (a performance setting only: the system is the same; declared in explain())
        if build_workers is None:
            import os
            env = os.environ.get("CONATUS_BUILD_WORKERS", "")
            build_workers = int(env) if env.strip() else None
        if build_workers is not None and int(build_workers) < 1:
            raise OptionError(f"build_workers must be >= 1, got {build_workers!r}",
                              fix="build_workers=1 (or more)")
        self.build_workers = None if build_workers is None else int(build_workers)
        pb = plan.problem
        self.variables = plan.variables
        self.var_shapes = [plan._grid_shape(v.domain, v.tensorsig) for v in self.variables]
        self.eq_shapes = [plan._grid_shape(eq.domain, eq.lhs.tensorsig) for eq in pb.equations]
        var_comps = [_comps(v.tensorsig) for v in self.variables]
        eq_comps = [_comps(eq.lhs.tensorsig) for eq in pb.equations]
        self.slots: list[GroupSlot] = []
        cols_obj, cols_flat, rows_obj, rows_flat, rows_n, rows_lam = [], [], [], [], [], []
        self._var_comps, self._eq_comps = var_comps, eq_comps
        off = 0
        for parts in self._build_groups(plan.groups, reduced):
            for slot, cpart, rpart in parts:
                slot.offset = off
                for (k, fl, _) in cpart:
                    cols_obj.append(k)
                    cols_flat.append(fl)
                for k, fl, n, lam in rpart:
                    rows_obj.append(k)
                    rows_flat.append(fl)
                    rows_n.append(n)
                    rows_lam.append(lam)
                self.slots.append(slot)
                off += slot.size
        self.n = off
        self.cols_obj = np.array(cols_obj, int)
        self.cols_flat = np.array(cols_flat, int)
        self.rows_obj = np.array(rows_obj, int)
        self.rows_flat = np.array(rows_flat, int)
        self.rows_lam = np.array(rows_lam, int)
        self.complex = plan.layout.complex or any(
            np.iscomplexobj(s.L.data) or np.iscomplexobj(s.M.data) for s in self.slots)
        self.dtype = np.complex128 if self.complex else np.float64
        self.M = sp.block_diag([s.M for s in self.slots], format="csr").astype(self.dtype)
        self.L = sp.block_diag([s.L for s in self.slots], format="csr").astype(self.dtype)
        self._check_structure()
        self._buckets, self._cbuckets = self._make_buckets()
        self._zbuckets = self._make_zbuckets()
        self._factors: dict = {}
        # complex-mode buckets equal up to signs share one factorization (exact, checked at
        # factor time; timestep/shared_factors.py). False: every bucket factors its own matrices.
        self.share_factors = True
        self._zpartners: list | None = None
        self._conv: dict = {}
        # per-object gather/scatter selectors (vectorized)
        self._col_sel = [np.nonzero(self.cols_obj == k)[0] for k in range(len(self.variables))]
        self._row_sel = {}
        for k in range(len(pb.equations)):
            idx = np.nonzero(self.rows_obj == k)[0]
            for lam in np.unique(self.rows_lam[idx]):
                self._row_sel[(k, int(lam))] = idx[self.rows_lam[idx] == lam]

    build_workers: int | None = None

    def _build_groups(self, groups, reduced):
        """Per group, the list of (slot without offset, column meta, row meta) of its blocks, in
        group order; serially or on ``build_workers`` forked processes (same results)."""
        groups = list(groups)
        w = int(self.build_workers or 1)
        if w <= 1 or len(groups) < 2 * w:
            return [self._build_group(g, reduced) for g in groups]
        import multiprocessing as mp
        global _BUILD_SELF
        _BUILD_SELF = (self, groups, reduced)
        try:
            ctx = mp.get_context("fork")
            chunk = max(1, len(groups) // (8 * w))
            with ctx.Pool(w) as pool:
                return pool.map(_build_group_worker, range(len(groups)), chunksize=chunk)
        finally:
            _BUILD_SELF = None

    def _build_group(self, g, reduced) -> list:
        plan = self.plan
        var_comps, eq_comps = self._var_comps, self._eq_comps
        out: list = []
        gm = plan.matrices(g, reduced)
        if gm.L.shape[0] == 0:
            return out
        if gm.L.shape[0] != gm.L.shape[1]:
            raise RankError(f"[{gm.label}] non-square group system {gm.L.shape}")
        cmeta = _meta(plan, g, plan.var_layouts(g), self.var_shapes, var_comps)
        rmeta = _meta(plan, g, plan.eq_layouts(g), self.eq_shapes, eq_comps)
        cm_g = [cmeta[j] for j in gm.col_index]
        rm_g = [rmeta[i] for i in gm.row_index]
        parts = (_components(gm) if self.decouple else
                 [(np.arange(gm.L.shape[0]), np.arange(gm.L.shape[1]))])
        for bi, (ri, ci) in enumerate(parts):
            block = ""
            sub = gm
            if len(parts) > 1:
                names = sorted({gm.cols[j][0] for j in ci if gm.cols[j][2] != "tau"})
                block = f"block {bi + 1}/{len(parts)}: {', '.join(names)}"
                sub = SimpleNamespace(L=gm.L[ri][:, ci].tocsr(), M=gm.M[ri][:, ci].tocsr(),
                                      label=f"{gm.label} [{block}]",
                                      rows=[gm.rows[i] for i in ri],
                                      cols=[gm.cols[j] for j in ci],
                                      lam=gm.lam, row_index=gm.row_index[ri],
                                      col_index=gm.col_index[ci])
            cm = [cm_g[j] for j in ci]
            rm = [rm_g[i] for i in ri]
            rpart = []
            for (k, fl, n), (ei, c, _) in zip(rm, sub.rows, strict=True):
                lam = sub.lam.get((ei, c))
                rpart.append((k, fl, -1 if n is None else n, -1 if lam is None else lam))
            kinds = [r[2] for r in sub.rows]
            rperm, cperm, nb = self._ordering(sub, cm, rm, kinds, retry=True)
            slot = GroupSlot(g, sub.label, 0, sub.L.shape[0], sub.M.tocsr(),
                             sub.L.tocsr(), kinds, rperm, cperm, nb, block)
            # In a reflection subspace each object keeps only cos or only -sin: the group is
            # already real and half size, so there is no pair to complexify (declared in explain()).
            if self.complexify and not self.reduced:
                cpos = _fpos_multi(plan, g, plan.var_layouts(g), var_comps)
                rpos = _fpos_multi(plan, g, plan.eq_layouts(g), eq_comps)
                cpos = [cpos[j] for j in sub.col_index]
                rpos = [rpos[i] for i in sub.row_index]
                # directions where this block holds both cos and -sin (a decoupled block may
                # hold a single product, e.g. cos x cos z only: already real and minimal)
                # (a block may also pair cos of one object with -sin of another: then it is
                # a real system of one mode's size, nothing to complexify either)
                nd = len(cpos[0]) if cpos else 0

                def paired(pos, meta, j):
                    per: dict = {}
                    for p, m in zip(pos, meta, strict=True):
                        per.setdefault(m[0], set()).add(p[j])
                    return bool(per) and all(v == {0, 1} for v in per.values())
                act = [j for j in range(nd) if paired(cpos, cm, j) and paired(rpos, rm, j)]
                cpos = [tuple(p[j] if j in act else None for j in range(nd)) for p in cpos]
                rpos = [tuple(p[j] if j in act else None for j in range(nd)) for p in rpos]
                if len(act) >= 2:
                    slot.zs = self._multi_complex_views(sub, cm, rm, kinds, cpos, rpos)
                elif len(act) == 1:
                    j = act[0]
                    slot.cz = self._complex_view(sub, cm, rm, kinds, [p[j] for p in cpos],
                                                 [p[j] for p in rpos])
            out.append((slot, cm, rpart))
        return out

    def _ordering(self, gm, cm, rm, kinds, retry: bool = False):
        n = gm.L.shape[0]
        border = [i for i, k in enumerate(kinds) if k != "bulk"]
        if not border:
            return np.arange(n), np.arange(n), 0
        pat = (abs(gm.L) + abs(gm.M)).tocsr()
        patc = pat.tocsc()
        col_n = [c[2] for c in cm]
        col_kind = [c[2] for c in gm.cols]
        pinned: list[int] = []
        taken = np.zeros(n, bool)
        piece, quota, used, offs = _piece_quotas(self.plan, n, patc, col_n, col_kind, rm, kinds)
        if piece is not None:
            # rows confined to one piece (wall conditions) first, interface rows after: each then
            # takes a column in the piece with the most pinned columns still owed (see below)
            def npieces(i):
                cols = pat.indices[pat.indptr[i]:pat.indptr[i + 1]]
                return len({piece(col_n[j]) for j in cols if col_n[j] is not None})
            border = sorted(border, key=lambda i: (npieces(i), i))

        def key(j):
            """Lowest z-mode first; on a compound interval, first the piece owing the most
            pinned columns, then the lowest mode of that piece."""
            if piece is None:
                return (col_n[j] if col_n[j] is not None else 10 ** 9, j)
            if col_n[j] is None:
                return (1, 0, 10 ** 9, j)
            p = piece(col_n[j])
            return (0, -(quota.get(p, 0) - used[p]), col_n[j] - offs[p], j)

        def pin(rows):
            """Greedy: each border row takes its lowest free z-mode column."""
            for i in rows:
                nz = pat.indices[pat.indptr[i]:pat.indptr[i + 1]]
                cand = [j for j in nz if not taken[j] and col_kind[j] == "field" and col_n[j] is not None]
                if piece is not None:
                    # compound interval: the pinned columns of each piece must match its tau columns,
                    # or the banded part splits into non-square diagonal blocks (singular)
                    inq = [j for j in cand if used[piece(col_n[j])] < quota.get(piece(col_n[j]), 0)]
                    cand = inq or cand
                if not cand:
                    cand = [j for j in nz if not taken[j]]
                if not cand:
                    raise RankError(f"[{gm.label}] boundary/constraint row {i} has no free column")
                j = min(cand, key=key)
                if piece is not None and col_n[j] is not None:
                    used[piece(col_n[j])] += 1
                pinned.append(j)
                taken[j] = True

        def gaps() -> bool:
            """Some unknown has pinned z-modes that are not its lowest ones (0, 1, ..., k-1)."""
            per: dict = {}
            for j in pinned:
                if col_kind[j] == "field" and col_n[j] is not None:
                    per.setdefault((cm[j][0], cm[j][1]), []).append(col_n[j])
            return any(sorted(v) != list(range(len(v))) for v in per.values())

        def bulk_singular() -> bool:
            rest = [j for j in range(n) if not taken[j]]
            bulk = [i for i in range(n) if kinds[i] == "bulk"]
            return structural_rank(sp.csr_matrix(pat[bulk][:, rest])) < len(bulk)

        pin(border)
        # retry: only on the Cartesian groups of this class (the polar/spherical solvers and the
        # column-border layouts call this ordering too; they keep it exactly, and the structural
        # check is not paid on their many groups)
        if retry and piece is None and gaps() and bulk_singular():
            # an integral constraint int(u) = c has zero coefficients on the odd
            # Chebyshev modes; taken after a Dirichlet row (which took T_0) it pinned T_2 and left
            # the odd modes one column short in the banded part (structurally singular: "banded
            # part is singular" from the C++ solver). Retry with the most constrained rows first
            # (fewest candidate columns). Only reached when some pinned modes are not the lowest
            # ones AND the banded part is singular: Dirichlet / Neumann / Robin rows never get
            # here, so their ordering and their numbers are unchanged.
            def ncand(i):
                nz = pat.indices[pat.indptr[i]:pat.indptr[i + 1]]
                return sum(1 for j in nz if col_kind[j] == "field" and col_n[j] is not None)
            pinned.clear()
            taken[:] = False
            border = sorted(border, key=lambda i: (ncand(i), i))
            pin(border)
            if bulk_singular():                         # not fixed by the retry: say why
                raise RankError(
                    f"[{gm.label}] the boundary/constraint rows cannot be paired with z-modes so "
                    "that the remaining (banded) system is invertible",
                    fix="check the conditions with plan.rank_report(group); an integral or "
                        "nonlocal constraint must leave a condition on each mode the operator "
                        "annihilates")
        if piece is not None:
            pinned = _compound_pinning(pat, border, pinned, kinds, col_n, col_kind, piece, offs,
                                       gm.label)
            taken[:] = False
            taken[pinned] = True
        if self.ordering == "variable-major":      # plan order, border rows + pinned columns first
            bulk = [i for i in range(n) if kinds[i] == "bulk"]
            rest = [j for j in range(n) if not taken[j]]
            return np.array(border + bulk, int), np.array(pinned + rest, int), len(border)
        row_key = np.array([rm[i][2] if rm[i][2] is not None else -1 for i in range(n)], float)
        bulk = [i for i in range(n) if kinds[i] == "bulk"]
        bulk.sort(key=lambda i: (row_key[i], i))
        is_bulk = np.zeros(n, bool)
        is_bulk[bulk] = True
        ckey = np.empty(n)
        for j in range(n):
            if col_n[j] is not None and col_kind[j] == "field":
                ckey[j] = col_n[j]
            else:
                rows = patc.indices[patc.indptr[j]:patc.indptr[j + 1]]
                rows = rows[is_bulk[rows]]
                # a lifted tau spreads over a few rows (T_{N+n} converted to C^(lam)): centring it
                # on their median key halves the band compared to either end
                ckey[j] = float(np.median(row_key[rows])) if rows.size else np.inf
        rest = [j for j in range(n) if not taken[j]]
        rest.sort(key=lambda j: (ckey[j], j))
        return np.array(border + bulk, int), np.array(pinned + rest, int), len(border)

    # ------------------------------------------------------------------ checks
    def _check_structure(self) -> None:
        for s in self.slots:
            pat = sp.csr_matrix((abs(s.M) + abs(s.L)) > 0, dtype=float)
            r = structural_rank(pat)
            if r != s.size:
                raise RankError(f"[{s.label}] structurally singular IVP pencil M + dt L "
                                f"(rank {r} / {s.size}): refused before factorization")

    def _make_buckets(self):
        """Real buckets (idx, rows, cols) and complexified buckets
        (idx, rows_c, rows_s, cols_c, cols_s), grouping slots with one sparsity pattern."""
        buckets: dict = {}
        cbuckets: dict = {}
        for i, s in enumerate(self.slots):
            if getattr(s, "zs", None) is not None:
                continue                      # d >= 2 directions: _make_zbuckets
            cz = getattr(s, "cz", None)
            if cz is not None:
                pat = sp.csr_matrix((abs(cz.M) + abs(cz.L)) > 0, dtype=np.int8)[cz.rperm][:, cz.cperm]
                key = (cz.size, cz.n_border, pat.indptr.tobytes(), pat.indices.tobytes())
                cbuckets.setdefault(key, []).append(i)
                continue
            pat = sp.csr_matrix((abs(s.M) + abs(s.L)) > 0, dtype=np.int8)[s.rperm][:, s.cperm]
            key = (s.size, s.n_border, pat.indptr.tobytes(), pat.indices.tobytes())
            buckets.setdefault(key, []).append(i)
        out = []
        for idx in buckets.values():
            slots = [self.slots[i] for i in idx]
            rows = np.stack([s.offset + s.rperm for s in slots])
            cols = np.stack([s.offset + s.cperm for s in slots])
            out.append((idx, rows, cols))
        cout = []
        for idx in cbuckets.values():
            slots = [self.slots[i] for i in idx]
            g = [(s.offset, s.cz) for s in slots]
            cout.append((idx,
                         np.stack([o + cz.rc[cz.rperm] for o, cz in g]),
                         np.stack([o + cz.rs[cz.rperm] for o, cz in g]),
                         np.stack([o + cz.cc[cz.cperm] for o, cz in g]),
                         np.stack([o + cz.cs[cz.cperm] for o, cz in g])))
        return out, cout

    def _make_zbuckets(self) -> list:
        """Buckets of complex views of groups with pairs in >= 2 directions, keyed by (sparsity
        pattern, sign vector s): (keys [(slot, view)], rows (P, batch, m), cols (P, batch, m),
        wr (P,), wc (P,)), global indices. One s per bucket, so the scatter-add of the 2^{d-1}
        views of a group into x never has duplicate indices within one statement."""
        buckets: dict = {}
        for i, sl in enumerate(self.slots):
            for vi, v in enumerate(getattr(sl, "zs", None) or []):
                pat = sp.csr_matrix((abs(v.M) + abs(v.L)) > 0, dtype=np.int8)[v.rperm][:, v.cperm]
                key = (v.size, v.n_border, v.s, pat.indptr.tobytes(), pat.indices.tobytes())
                buckets.setdefault(key, []).append((i, vi))
        out = []
        for keys in buckets.values():
            vs = [(self.slots[i], self.slots[i].zs[vi]) for i, vi in keys]
            rows = np.stack([np.stack([sl.offset + v.rows[p][v.rperm] for sl, v in vs])
                             for p in range(len(vs[0][1].rows))])
            cols = np.stack([np.stack([sl.offset + v.cols[p][v.cperm] for sl, v in vs])
                             for p in range(len(vs[0][1].cols))])
            out.append((keys, rows, cols, vs[0][1].wr, vs[0][1].wc))
        return out

    # ------------------------------------------------------------------ complexification
    def _complex_view(self, gm, cm, rm, kinds, cpos, rpos):
        """Complex half-size view of a RealFourier (cos, -sin) group, or None for groups without
        a pair (k = 0). Raises if the pair exists but [[A, -B], [B, A]] does not hold."""
        if not any(p == 1 for p in cpos):
            return None

        # partner key = (object, Chebyshev index, rank among same-(object, n, position) entries):
        # _meta lists cos entries then sin entries per component, so ranks line up.
        def keyed(meta, pos):
            count: dict = {}
            keys = []
            for m, p in zip(meta, pos, strict=True):
                k = (m[0], m[2], p)
                c = count.get(k, 0)
                count[k] = c + 1
                keys.append((m[0], m[2], c))
            return keys

        def pairs2(meta, pos, what):
            keys = keyed(meta, pos)
            first = {keys[i]: i for i, p in enumerate(pos) if p == 0}
            c_idx, s_idx = [], []
            for i, p in enumerate(pos):
                if p == 1:
                    j = first.pop(keys[i], None)
                    if j is None:
                        raise RankError(f"[{gm.label}] complexify: {what} {i} (sin) has no cos partner")
                    c_idx.append(j)
                    s_idx.append(i)
            if first or any(p is None for p in pos):
                raise RankError(f"[{gm.label}] complexify: {what}s are not all in (cos, -sin) pairs")
            return np.array(c_idx, int), np.array(s_idx, int)

        cc, cs = pairs2(cm, cpos, "column")
        rc, rs = pairs2(rm, rpos, "row")
        mats = {}
        for name, X in (("M", gm.M.tocsr()), ("L", gm.L.tocsr())):
            P, Q = X[rc][:, cc], X[rc][:, cs]
            S, T = X[rs][:, cc], X[rs][:, cs]
            scale = max(1.0, abs(X).max())
            e1 = abs(P - T).max() if (P - T).nnz else 0.0
            e2 = abs(S + Q).max() if (S + Q).nnz else 0.0
            if max(e1, e2) > 1e-13 * scale:
                raise RankError(
                    f"[{gm.label}] complexify refused: {name} is not [[A, -B], [B, A]] in the "
                    f"(cos, -sin) pair (mismatch {max(e1, e2):.1e}); use complexify=False")
            mats[name] = (P + 1j * S).tocsr()
        ckinds = [kinds[i] for i in rc]
        fake = SimpleNamespace(L=mats["L"], M=mats["M"], label=gm.label,
                               cols=[gm.cols[j] for j in cc])
        rperm, cperm, nb = self._ordering(fake, [cm[j] for j in cc], [rm[i] for i in rc], ckinds)
        return SimpleNamespace(M=mats["M"], L=mats["L"], rc=rc, rs=rs, cc=cc, cs=cs,
                               rperm=rperm, cperm=cperm, n_border=nb, size=len(rc))

    def _multi_complex_views(self, gm, cm, rm, kinds, cpos, rpos) -> list:
        """Complex systems of a group with pairs in d >= 2 RealFourier directions (module
        docstring): one per sign vector s = (+1, s_2, .., s_d). Each view carries its matrices,
        ordering, and for each position p in {0,1}^d the group-local row/column indices and
        weights: r_s = sum_p wr[p] r[rows[p]], x[cols[p]] += Re(wc[p] z_s)."""
        from itertools import product as _prod
        d = len(next(p for p in cpos if p is not None and any(v is not None for v in p)))
        act = sorted({j for p in cpos for j, v in enumerate(p) if v is not None})
        for pos, what in ((cpos, "column"), (rpos, "row")):
            if any(tuple(j for j, v in enumerate(p) if v is not None) != tuple(act) for p in pos):
                raise RankError(f"[{gm.label}] complexify: {what}s do not all hold pairs in the "
                                "same Fourier directions")
        d = len(act)
        P = list(_prod((0, 1), repeat=d))

        def split(meta, pos, what):
            """{p: indices}, entries matched by (object, Chebyshev index, rank) across p."""
            count: dict = {}
            table: dict = {}
            for i, (m, q) in enumerate(zip(meta, pos, strict=True)):
                pq = tuple(q[j] for j in act)
                key = (m[0], m[2], pq)
                c = count.get(key, 0)
                count[key] = c + 1
                table.setdefault(pq, {})[(m[0], m[2], c)] = i
            keys = list(table.get(P[0], {}))
            if any(set(table.get(q, {})) != set(keys) for q in P):
                raise RankError(f"[{gm.label}] complexify: {what}s are not complete 2^{d} "
                                "cos/sin sets")
            return {q: np.array([table[q][k] for k in keys], int) for q in P}

        ci, ri = split(cm, cpos, "column"), split(rm, rpos, "row")
        views = []
        for s in _prod((1,), *[(1, -1)] * (d - 1)):
            w = {q: complex(np.prod([(1j * sj) ** qj for sj, qj in zip(s, q, strict=True)])) for q in P}
            mats = {}
            for name, X in (("M", gm.M.tocsr()), ("L", gm.L.tocsr())):
                A = B = None
                for q in P:                        # rows
                    Xq = X[ri[q]]
                    for q2 in P:                   # columns: z = 1 and z = i
                        blk = Xq[:, ci[q2]]
                        if not blk.nnz:
                            continue
                        f1 = w[q] * np.real(np.conj(w[q2])) / 2 ** (d - 1)
                        f2 = w[q] * np.real(1j * np.conj(w[q2])) / 2 ** (d - 1)
                        if f1:
                            A = blk * f1 if A is None else A + blk * f1
                        if f2:
                            B = blk * f2 if B is None else B + blk * f2
                m = len(ri[P[0]])
                A = sp.csr_matrix((m, m), dtype=complex) if A is None else sp.csr_matrix(A, dtype=complex)
                B = sp.csr_matrix((m, m), dtype=complex) if B is None else sp.csr_matrix(B, dtype=complex)
                scale = max(1.0, abs(X).max() if X.nnz else 0.0)
                err = abs(B - 1j * A).max() if (B - 1j * A).nnz else 0.0
                if err > 1e-13 * scale:
                    raise RankError(
                        f"[{gm.label}] complexify refused: {name} is not complex-linear in the "
                        f"cos/sin products of {d} directions (mismatch {err:.1e}); "
                        "use complexify=False")
                A.eliminate_zeros()
                mats[name] = A.tocsr()
            ckinds = [kinds[i] for i in ri[P[0]]]
            fake = SimpleNamespace(L=mats["L"], M=mats["M"], label=gm.label,
                                   cols=[gm.cols[j] for j in ci[P[0]]])
            rperm, cperm, nb = self._ordering(fake, [cm[j] for j in ci[P[0]]],
                                              [rm[i] for i in ri[P[0]]], ckinds)
            views.append(SimpleNamespace(
                s=s, M=mats["M"], L=mats["L"], rperm=rperm, cperm=cperm, n_border=nb,
                size=len(ri[P[0]]), rows=[ri[q] for q in P], cols=[ci[q] for q in P],
                wr=np.array([w[q] for q in P]),
                wc=np.array([np.conj(w[q]) / 2 ** (d - 1) for q in P])))
        return views

    # ------------------------------------------------------------------ solves
    def factor(self, a: float, b: float):
        """Factorizations of a M + b L for every group (cached by (a, b)): real buckets first,
        then complexified buckets (complex128)."""
        key = (float(a), float(b))
        f = self._factors.get(key)
        if f is None:
            f = []
            for idx, _, _ in self._buckets:
                mats = []
                for i in idx:
                    s = self.slots[i]
                    A = (a * s.M + b * s.L).tocsr()[s.rperm][:, s.cperm]
                    mats.append(A.astype(self.dtype))
                f.append(BorderedBandedBatch(mats, list(range(self.slots[idx[0]].n_border)),
                                             dtype=self.dtype))
            for idx, *_ in self._cbuckets:
                czs = [self.slots[i].cz for i in idx]
                mats = [(a * cz.M + b * cz.L).tocsr()[cz.rperm][:, cz.cperm].astype(np.complex128)
                        for cz in czs]
                f.append(BorderedBandedBatch(mats, list(range(czs[0].n_border)), dtype=np.complex128))
            base = len(f)
            for mats, nb, shared in self._zbucket_matrices(a, b):
                f.append(shared if shared is not None else
                         BorderedBandedBatch(mats, list(range(nb)), dtype=np.complex128))
            self._share_log(key, f[base:])
            self._factors[key] = f
        return f

    def _zbucket_matrices(self, a: float, b: float):
        """For each complex-mode bucket, in order: (matrices a M + b L in permuted order, border
        rows, SharedFactor or None). A SharedFactor (index in the full factor list) replaces the
        matrices (None then) when the bucket equals an earlier one up to signs and
        ``share_factors`` is on (timestep/shared_factors.py). Used by the CPU and GPU systems."""
        from .shared_factors import SharedFactor, compact, sign_similarity, zbucket_partners
        if self._zpartners is None:
            self._zpartners = zbucket_partners(self) if self._zbuckets else []
        base = len(self._buckets) + len(self._cbuckets)
        pending: dict = {}                 # primary bucket -> its matrices, until its partner
        users = {p for p in self._zpartners if p is not None}
        for j, (keys, *_rest) in enumerate(self._zbuckets):
            vs = [self.slots[i].zs[vi] for i, vi in keys]
            mats = [(a * v.M + b * v.L).tocsr()[v.rperm][:, v.cperm].astype(np.complex128)
                    for v in vs]
            for A in mats:                 # canonical CSR (the factorization sorts in place)
                A.sort_indices()
            p = self._zpartners[j]
            if self.share_factors and p is not None and p in pending:
                rel = sign_similarity(pending.pop(p), mats, vs[0].n_border)
                if rel is not None:
                    yield None, vs[0].n_border, SharedFactor(base + p, *compact(*rel))
                    continue
            if self.share_factors and j in users:
                pending[j] = mats
            yield mats, vs[0].n_border, None

    def _share_log(self, key, entries) -> None:
        """Record which complex-mode buckets share factors (explain())."""
        from .shared_factors import SharedFactor
        n_sh = sum(isinstance(e, SharedFactor) for e in entries)
        n_cand = sum(p is not None for p in (self._zpartners or []))
        self._share_stats = getattr(self, "_share_stats", {})
        self._share_stats[key] = (n_sh, n_cand)

    def solve(self, a: float, b: float, rhs: np.ndarray) -> np.ndarray:
        """X such that (a M + b L) X = rhs (all groups)."""
        f = self.factor(a, b)
        x = (np.zeros if self._zbuckets else np.empty)(self.n, dtype=self.dtype)
        nr, nc = len(self._buckets), len(self._cbuckets)
        for (_idx, rows, cols), batch in zip(self._buckets, f[:nr], strict=True):
            x[cols] = batch.solve(rhs[rows])
        for (_idx, rc, rs, cc, cs), batch in zip(self._cbuckets, f[nr:nr + nc], strict=True):
            z = batch.solve(rhs[rc] + 1j * rhs[rs])
            x[cc] = z.real
            x[cs] = z.imag
        for (_keys, rows, cols, wr, wc), batch in zip(self._zbuckets, f[nr + nc:], strict=True):
            zr = sum(w * rhs[r] for w, r in zip(wr, rows, strict=True))
            if isinstance(batch, SharedFactor):          # B = D_r A D_c: x = D_c A^{-1} D_r b
                z = f[batch.primary].solve(zr * batch.dr) * batch.dc
            else:
                z = batch.solve(zr)
            for w, c in zip(wc, cols, strict=True):
                x[c] += (w * z).real
        return x

    def solve_transpose(self, a: float, b: float, rhs: np.ndarray) -> np.ndarray:
        """X such that (a M + b L)^T X = rhs (plain transpose, no conjugation), with the same
        factors as :meth:`solve`. Per group A = G[rows][:, cols], so G^T x = r reads
        A^T x[rows] = r[cols]. A complexified bucket stores C = A + iB for the real block
        [[A, -B], [B, A]]; its real transpose is the complex C^H, solved as conj(C^{-T} conj(w))."""
        f = self.factor(a, b)
        x = (np.zeros if self._zbuckets else np.empty)(self.n, dtype=self.dtype)
        nr, nc = len(self._buckets), len(self._cbuckets)
        for (_idx, rows, cols), batch in zip(self._buckets, f[:nr], strict=True):
            x[rows] = batch.solve(rhs[cols], transpose=True)
        for (_idx, rc, rs, cc, cs), batch in zip(self._cbuckets, f[nr:nr + nc], strict=True):
            z = np.conj(batch.solve(np.conj(rhs[cc] + 1j * rhs[cs]), transpose=True))
            x[rc] = z.real
            x[rs] = z.imag
        # d >= 2 directions: R^T = E A^H T (module docstring), rows and columns swap roles
        for (_keys, rows, cols, wr, wc), batch in zip(self._zbuckets, f[nr + nc:], strict=True):
            r = sum(w * rhs[c] for w, c in zip(wr, cols, strict=True))
            if isinstance(batch, SharedFactor):          # B^T = D_c A^T D_r: x = D_r A^{-T} D_c b
                z = np.conj(f[batch.primary].solve(np.conj(r) * batch.dc, transpose=True) * batch.dr)
            else:
                z = np.conj(batch.solve(np.conj(r), transpose=True))
            for w, rr in zip(wc, rows, strict=True):
                x[rr] += (w * z).real
        return x

    @property
    def n_factorizations(self) -> int:
        return len(self._factors)

    def bands(self) -> list[tuple[str, int, int, int, int]]:
        """(label, n, border, kl, ku) of the first group of each bucket (for the first factor);
        complexified buckets are labelled '(complex)'.

        The bandwidths are read off the sparsity pattern of |M| + |L| (rows and columns in the
        solver order, the border rows first), as the C++ factor measures the band of its bulk
        rows, without factorizing anything (``M + L`` was factored
        with dt = 1 only to read kl and ku, and a valid run whose step matrix is singular at
        dt = 1, e.g. Allen--Cahn ``dt(T) - eps*lap(T) = T - T**3`` with ``-T`` implicit, made
        ``explain()`` raise "banded part is singular"). |M| + |L| has the union of the two
        patterns, so no cancellation can narrow the band reported for a step a M + b L."""
        out = []
        for idx, _, _ in self._buckets:
            s = self.slots[idx[0]]
            kl, ku = _band_of(s.M, s.L, s.rperm, s.cperm, s.n_border)
            out.append((s.label, s.size, s.n_border, kl, ku))
        for idx, *_ in self._cbuckets:
            s = self.slots[idx[0]]
            cz = s.cz
            kl, ku = _band_of(cz.M, cz.L, cz.rperm, cz.cperm, cz.n_border)
            out.append((s.label + " (complex)", cz.size, cz.n_border, kl, ku))
        for keys, *_ in self._zbuckets:
            s = self.slots[keys[0][0]]
            v = s.zs[keys[0][1]]
            kl, ku = _band_of(v.M, v.L, v.rperm, v.cperm, v.n_border)
            sgn = "".join("+" if t > 0 else "-" for t in v.s)
            out.append((f"{s.label} (complex, s={sgn})", v.size, v.n_border, kl, ku))
        return out

    # ------------------------------------------------------------------ gather / scatter
    def scatter(self, X: np.ndarray, only: frozenset | None = None) -> list:
        """Global vector -> coefficient arrays of the variables (zeros outside the subspace).
        ``only``: names of the variables actually needed; the others come back as ``None``."""
        out = []
        for k, shape in enumerate(self.var_shapes):
            if only is not None and self.variables[k].name not in only:
                out.append(None)
                continue
            a = np.zeros(int(np.prod(shape)), dtype=self.dtype)
            sel = self._col_sel[k]
            a[self.cols_flat[sel]] = X[sel]
            a = a.reshape(shape)
            out.append(a if self.complex else a.real)
        return out

    def gather_state(self, coeffs: Sequence[np.ndarray], check_subspace: bool = True) -> np.ndarray:
        """Coefficient arrays -> global vector; refuses data outside the symmetry subspace."""
        X = np.zeros(self.n, dtype=self.dtype)
        for k, a in enumerate(coeffs):
            a = np.asarray(a).reshape(-1)
            sel = self._col_sel[k]
            X[sel] = a[self.cols_flat[sel]]
            if check_subspace and self.reduced:
                kept = np.zeros(a.size, bool)
                kept[self.cols_flat[sel]] = True
                lost = np.abs(a[~kept])
                if lost.size and lost.max() > 1e-12 * max(1.0, np.abs(a).max()):
                    raise SymmetryError(f"initial data for {self.variables[k].name} has components outside the "
                                        f"symmetry subspace (max {lost.max():.2e})",
                                        fix="project the data on the subspace (solver.symmetrize_state(X)) "
                                            "or drop the symmetry")
        return X

    def gather_rhs(self, eq_coeffs: Sequence[np.ndarray | None]) -> np.ndarray:
        """Equation RHS coefficients (Chebyshev T along z) -> global row vector, each row
        converted to its equation's C^(lam) space."""
        b = np.zeros(self.n, dtype=self.dtype)
        zc = self.plan.layout.zcoord
        coords = self.plan.domain.coordsys.coords
        for (k, lam), sel in self._row_sel.items():
            c = eq_coeffs[k]
            if c is None:
                continue
            if lam > 0:
                eq = self.plan.problem.equations[k]
                ax = eq.lhs.rank + coords.index(zc)
                if c.shape[ax] > 1:
                    S = self._conv.get(lam)
                    if S is None:
                        S = self._conv[lam] = self.plan.layout.zconv(lam).tocsr()
                    cm = np.moveaxis(c, ax, 0)
                    c = np.moveaxis((S @ cm.reshape(cm.shape[0], -1)).reshape(cm.shape), 0, ax)
            b[sel] = np.asarray(c).reshape(-1)[self.rows_flat[sel]]
        return b

    def explain(self) -> str:
        lines = [f"GlobalSystem: {len(self.slots)} group(s), {self.n} unknowns "
                 f"({'symmetry subspace' if self.reduced else 'full space'}), dtype "
                 f"{np.dtype(self.dtype).name}, ordering={self.ordering}, complexify="
                 f"{self.complexify}" + ((" (symmetry subspace: groups already real with one of "
                 "cos/-sin per object, nothing to complexify)" if self.reduced else
                 " (RealFourier cos/-sin pairs solved as A + iB, structure checked)")
                 if self.complexify else ""),
                 *([f"  groups assembled on {self.build_workers} worker processes (forked; same "
                    "system as the serial assembly)"] if (self.build_workers or 1) > 1 else []),
                 f"  {len(self._buckets)} solver bucket(s) (groups sharing one sparsity pattern, "
                 "batched C++ bordered banded solves)"]
        for label, n, r, kl, ku in self.bands():
            lines.append(f"  [{label}] n={n}, border rows={r}, band kl={kl} ku={ku}")
        lines.append(f"  {len(self._cbuckets)} complexified bucket(s)")
        if self._zbuckets:
            lines.append(f"  {len(self._zbuckets)} bucket(s) of complex modes exp(i(k.x)) from "
                         "cos/sin products in >= 2 RealFourier directions (complex linearity "
                         "checked per group)")
            stats = getattr(self, "_share_stats", {})
            if not self.share_factors:
                lines.append("  shared factors off (share_factors=False): every bucket factors its own matrices")
            elif stats:
                n_sh, n_cand = max(stats.values())
                lines.append(f"  {n_sh} of {n_cand} candidate complex-mode bucket(s) solved with the factors of "
                             "their other sign vector (B = D_r A D_c, D = diag(+-1), checked entry by "
                             "entry: same pivots, bit-identical solves, half their factor memory)")
        if self.decouple:
            nb = sum(1 for s in self.slots if s.block)
            lines.append(f"  decouple=True: {nb} slot(s) are decoupled blocks of their group "
                         "(connected components of |L| + |M|, solved separately; exact)")
        return "\n".join(lines)


def _band_of(M, L, rperm, cperm, n_border: int) -> tuple[int, int]:
    """(kl, ku) of the bulk rows (all but the first ``n_border`` rows, the border) of the
    permuted pattern of |M| + |L|: the measure of ``measure_band`` in
    core/include/conatus/banded.hpp, structural (no factorization; System.bands Y5)."""
    A = (abs(sp.csr_matrix(M)) + abs(sp.csr_matrix(L))).tocsr()[rperm][:, cperm].tocoo()
    keep = (A.row >= n_border) & (A.data != 0)
    if not keep.any():
        return 0, 0
    d = A.row[keep].astype(np.int64) - A.col[keep].astype(np.int64)
    return max(int(d.max()), 0), max(int(-d.min()), 0)

