"""Factors shared between complex-mode buckets that are equal up to signs.

With two RealFourier directions, a group (k_x, k_z) is solved as two complex systems, one per sign
vector s = (+, +) and (+, -) (``system.py``, the modes exp(i(k_x x +- k_z z))). For many problems the two matrices differ only by the sign of the terms odd in k_z:

    B = D_r A D_c,   D_r, D_c diagonal with entries +1 or -1.

Then B x = b is solved with the factors of A: y = A^{-1} (D_r b), x = D_c y, and the transpose
B^T x = b as x = D_r A^{-T} (D_c b). Partial pivoting compares magnitudes, which D leaves
unchanged, so the factorization of B is entry by entry the one of A up to signs and the shared
solve is **bit-identical** to factoring B on its own (sign flips are exact in IEEE arithmetic).
Half the factor memory of those buckets is saved, nothing is approximated.

The relation is **checked**, never assumed: same CSR structure mode by mode, every entry of B
equal to + or - the entry of A (exact comparison), no explicit zeros, and the entry signs
factor as d_r[i] d_c[j] (found on a spanning forest of the bipartite graph of the pattern, then
verified on every entry). Any mismatch leaves the bucket with its own factors.

Border rows. ``BorderedBandedBatch`` replaces each border row i of the band part by s e_i
(s = max |row|, always positive) and moves the row into V^T. The band part of B is then
D_r A_band D_c only if d_r[i] d_c[i] = +1 on every border row (else the replaced diagonal has the
wrong sign, the band LU of B is a different factorization, and the shared solve, though exact
mathematically, is no longer the bits of B's own factors). Such buckets are not shared: the claim "bit-identical" holds for every shared bucket.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from .._repr import compact_repr

__all__ = ["SharedFactor", "zbucket_partners", "sign_similarity"]


@dataclass
class SharedFactor:
    """Factor-list entry of a bucket solved with the factors of bucket ``primary`` (index in
    the factor list): B = diag(dr) A diag(dc), dr/dc of shape (n,) when the same for every mode
    of the batch, else (batch, n); entries +1.0 / -1.0, permuted row/column order."""
    __repr__ = compact_repr   # one line, arrays by shape
    primary: int
    dr: np.ndarray
    dc: np.ndarray

    @property
    def uniform(self) -> bool:
        return self.dr.ndim == 1


def zbucket_partners(host) -> list:
    """For each complex-mode bucket j, the index i < j of a bucket with the same structure (size,
    border, sparsity pattern) over the same groups in the same order (the other sign vector of
    those groups), or None. Structural only: the numerical relation is checked at factor time."""
    out: list = []
    seen: dict = {}
    for j, (keys, *_rest) in enumerate(host._zbuckets):
        v = host.slots[keys[0][0]].zs[keys[0][1]]
        pat = sp.csr_matrix((abs(v.M) + abs(v.L)) > 0, dtype=np.int8)[v.rperm][:, v.cperm]
        skey = (v.size, v.n_border, tuple(i for i, _ in keys), pat.indptr.tobytes(),
                pat.indices.tobytes())
        i = seen.get(skey)
        if i is None:
            seen[skey] = j
            out.append(None)
        else:
            out.append(i)
            del seen[skey]                 # one partner per primary (the two sign vectors)
    return out


def _forest(indptr: np.ndarray, indices: np.ndarray, n: int):
    """Spanning forest of the bipartite graph (rows 0..n-1, columns n..2n-1) of one CSR pattern:
    (node, parent, entry) triples in breadth-first order (roots have parent -1)."""
    from scipy.sparse.csgraph import breadth_first_order
    nnz = indices.size
    rows = np.repeat(np.arange(n), np.diff(indptr))
    ent = sp.coo_matrix((np.arange(1, nnz + 1), (rows, indices)), shape=(n, n)).tocsr()
    big = sp.bmat([[None, ent], [ent.T, None]], format="csr")
    seen = np.zeros(2 * n, bool)
    order = []
    for root in range(2 * n):
        if seen[root]:
            continue
        nodes, pred = breadth_first_order(big, root, directed=False, return_predecessors=True)
        seen[nodes] = True
        for v in nodes:
            p = int(pred[v])
            if p < 0:
                order.append((int(v), -1, -1))
                continue
            order.append((int(v), p, int(big[p, v]) - 1))
    return order


def sign_similarity(prim: list, mats: list, n_border: int = 0):
    """(dr, dc) with mats[m] = diag(dr[m]) prim[m] diag(dc[m]) exactly (entries +-1), as (batch, n)
    float arrays, or None if the relation does not hold for every mode, or if a border row i
    (i < ``n_border``, permuted order) has d_r[i] d_c[i] = -1 (see module docstring)."""
    if len(prim) != len(mats) or not mats:
        return None
    A0 = prim[0]
    n = A0.shape[0]
    ip, ix = A0.indptr, A0.indices
    for A, B in zip(prim, mats, strict=True):
        if A.shape != B.shape or not (np.array_equal(A.indptr, ip) and np.array_equal(A.indices, ix)
                                      and np.array_equal(B.indptr, ip)
                                      and np.array_equal(B.indices, ix)):
            return None                    # not one common pattern: no sharing (declared)
    Ad = np.stack([A.data for A in prim])
    Bd = np.stack([B.data for B in mats])
    if not np.all(Ad != 0):
        return None                        # explicit zeros carry no sign: not shared
    pos, neg = Bd == Ad, Bd == -Ad
    if not np.all(pos | neg):
        return None                        # some entry is not +- the other (exact comparison)
    s = np.where(pos, 1.0, -1.0)           # (batch, nnz)
    d = np.ones((len(mats), 2 * n))
    for v, p, e in _forest(ip, ix, n):
        if p >= 0:
            d[:, v] = s[:, e] * d[:, p]
    dr, dc = d[:, :n], d[:, n:]
    rows = np.repeat(np.arange(n), np.diff(ip))
    if not np.array_equal(dr[:, rows] * dc[:, ix], s):
        return None                        # entry signs do not factor as d_r[i] d_c[j]
    nb = int(n_border)
    if nb and not np.all(dr[:, :nb] * dc[:, :nb] == 1.0):
        return None                        # a border row's s e_i would change sign: own factors
    return dr, dc


def compact(dr: np.ndarray, dc: np.ndarray):
    """(n,) vectors when every mode of the batch has the same signs, else unchanged."""
    if np.all(dr == dr[:1]) and np.all(dc == dc[:1]):
        return dr[0].copy(), dc[0].copy()
    return dr, dc
