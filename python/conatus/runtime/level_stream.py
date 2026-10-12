"""Size of the level-scheduled factor streams of ``cuda.BandedBatch``, computed on the host.

The device keeps the LU factors of each mode as a stream of words (``core/src/cuda/banded.cu``,
``LvlBuilder``): the rows of L' and U without their zeros, grouped by dependency level, consecutive
levels merged into warp steps of at most 32 rows, in chunks of a fixed layout. Its size depends on the
fill of the factorization, so the memory planner could not count it (measured it at +12 % of
the factors on the real path). This module replays, for one mode, the device factorization (the
``k_factor`` algorithm: banded LU with partial pivoting on the modulus, inside the band), then the
builder (levels, merging, chunking) and counts the words, forward and transposed. Nothing here runs
on the device; the planner samples a few modes per batch and reports the result as an *estimate*
(pivots and exact cancellations may differ between modes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

__all__ = ["bordered_band", "device_lu", "stream_words", "batch_stream_bytes", "host_bucket_streams"]

LVL_SEG, LVL_NSEG, WARP = 256, 4, 32
LVL_RING = LVL_SEG * LVL_NSEG
_CAP = (LVL_NSEG - 1) * LVL_SEG - 1          # merge_levels: a merged step fits the ring


def _chunk_words(rows: int, K: int, TW: int, backward: bool) -> int:
    q = (rows + 3) // 4
    return 1 + q + K * q + K * rows * TW + (rows * TW if backward else 0)


@dataclass
class _Row:
    i: int
    e: list = field(default_factory=list)    # [(column, value)]
    d: complex | float = 0.0


def bordered_band(A, n_border: int, kl: int | None = None, ku: int | None = None):
    """Dense band part B of ``make_bordered_layout`` (border rows first, replaced by s e_i), its
    band (kl, ku: the common band of the batch if given), and the number of non-zeros of V^T."""
    A = sp.csr_matrix(A)
    r = int(n_border)
    D = A.toarray()
    vt_nnz = int(np.count_nonzero(D[:r])) if r else 0
    for i in range(r):
        s = float(np.max(np.abs(D[i]))) if np.any(D[i]) else 1.0
        D[i] = 0.0
        D[i, i] = s
        # V^T row i: the row of A minus s e_i (banded.hpp); its non-zeros as counted above,
        # the diagonal entry possibly cancelled: not tracked (one entry)
    rows, cols = np.nonzero(D[r:])
    rows = rows + r
    kl0 = int(max(0, (rows - cols).max())) if rows.size else 0
    ku0 = int(max(0, (cols - rows).max())) if rows.size else 0
    return D, max(kl0, kl or 0), max(ku0, ku or 0), vt_nnz


def device_lu(D: np.ndarray, kl: int, ku: int):
    """In-place banded LU of the dense band matrix D as ``k_factor`` does it (pivot = largest
    modulus in the column below the diagonal, row interchange on columns j..ju, ju tracked).
    Returns (D with L below and U on/above the diagonal, ip) with ip[j] = j + pivot offset."""
    D = np.array(D, dtype=np.result_type(D.dtype, float), copy=True)
    n = D.shape[0]
    ip = np.arange(n)
    ju = 0
    for j in range(n):
        km = min(kl, n - 1 - j)
        col = np.abs(D[j:j + km + 1, j])
        jp = int(np.argmax(col)) if km > 0 else 0
        if col.size and col[jp] == col[0]:
            jp = 0                                  # strict '>' in the kernel: first maximum
        ip[j] = j + jp
        if D[j + jp, j] == 0:
            raise ValueError(f"device_lu: singular band at column {j}")
        ju = max(ju, min(j + ku + jp, n - 1))
        if jp:
            D[[j, j + jp], j:ju + 1] = D[[j + jp, j], j:ju + 1]
        if km > 0:
            D[j + 1:j + km + 1, j] *= 1.0 / D[j, j]
            D[j + 1:j + km + 1, j + 1:ju + 1] -= np.outer(D[j + 1:j + km + 1, j], D[j, j + 1:ju + 1])
    return D, ip


def _merge(levels: list, backward: bool, merge: bool, TW: int) -> list:
    """LvlBuilder::merge_levels: [(rows, merged)] steps."""
    steps: list = []
    cur: list = []
    where: dict = {}
    nlev = 0
    is_open = False

    def kmax(rows):
        return max((len(r.e) for r in rows), default=0)

    def fits(rows):
        return len(rows) <= WARP and _chunk_words(len(rows), kmax(rows), TW, backward) <= _CAP

    def close():
        nonlocal cur, nlev
        if cur:
            steps.append((cur, nlev > 1))
        where.clear()
        cur, nlev = [], 0

    def start(L):
        nonlocal cur, nlev, is_open
        cur = list(L)
        nlev = 1
        for s_, r in enumerate(cur):
            where[r.i] = s_
        is_open = merge and fits(cur)

    for L in levels:
        if not L:
            continue
        if not is_open:
            close()
            start(L)
            continue
        ex = []
        for R in L:
            acc: dict = {}
            for j, v in R.e:
                if j not in where:
                    acc[j] = acc.get(j, 0) + v
                    continue
                Rj = cur[where[j]]
                f = v * Rj.d if backward else v
                acc[j] = acc.get(j, 0) + f
                for k, vk in Rj.e:
                    acc[k] = acc.get(k, 0) - f * vk
            keys = sorted(acc, reverse=backward)
            ex.append(_Row(R.i, [(c, acc[c]) for c in keys if acc[c] != 0], R.d))
        trial = cur + ex
        if fits(trial):
            for E in ex:
                where[E.i] = len(cur)
                cur.append(E)
            nlev += 1
        else:
            close()
            start(L)
    close()
    return steps


def _emit_words(steps: list, backward: bool, TW: int) -> int | None:
    """LvlBuilder::emit word count (None: a chunk does not fit the ring: band kernel)."""
    words, anyc = 0, False
    lim = (LVL_NSEG - 1) * LVL_SEG
    for rows, _merged in steps:
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: -len(r.e))
        pos = 0
        while pos < len(rows):
            nr = min(WARP, len(rows) - pos)
            K = len(rows[pos].e)
            while nr > 1 and _chunk_words(nr, K, TW, backward) + 1 > lim:
                nr //= 2
            if _chunk_words(nr, K, TW, backward) + 1 > lim:
                return None
            words += _chunk_words(nr, K, TW, backward)
            pos += nr
            anyc = True
    return words if anyc else 1


def _levels(rows_in_order, deps_of) -> list:
    lev: dict = {}
    levels: list = [[]]
    for R in rows_in_order:
        l_ = max((lev.get(c, 0) for c, _ in deps_of(R)), default=0)
        lev[R.i] = l_ + 1
        while len(levels) <= l_ + 1:
            levels.append([])
        levels[l_ + 1].append(R)
    return levels


def stream_words(LU: np.ndarray, ip: np.ndarray, kl: int, ku: int, merge: bool = True,
                 transpose: bool = False) -> int | None:
    """Words of one mode's stream (forward: L' then U; transpose: U^T then L'^T), as built by
    ``BandedBatch`` (``build_level_schedule`` / ``build_transpose_schedule``), before the
    even-length padding; None if the level schedule does not apply (band kernel)."""
    n = LU.shape[0]
    TW = 2 if np.iscomplexobj(LU) else 1
    kv = kl + ku
    F = list(range(n))
    lent: list = []                                   # (row of L' in final positions, k, value)
    for k in range(n - 1, -1, -1):
        for i in range(1, min(kl, n - 1 - k) + 1):
            v = LU[k + i, k]
            if v != 0:
                lent.append((F[k + i], k, v))
        F[k], F[ip[k]] = F[ip[k]], F[k]
    if not transpose:
        lrow: dict = {}
        for r, k, v in lent:
            lrow.setdefault(r, []).append((k, v))
        L1 = [_Row(i, sorted(lrow[i])) for i in range(n) if i in lrow]
        U = []
        for i in range(n - 1, -1, -1):
            e = [(j, LU[i, j]) for j in range(min(n - 1, i + kv), i, -1) if LU[i, j] != 0]
            U.append(_Row(i, e, 1.0 / LU[i, i]))
        a = _emit_words(_merge(_levels(L1, lambda R: R.e), False, merge, TW), False, TW)
        b = _emit_words(_merge(_levels(U, lambda R: R.e), True, merge, TW), True, TW)
    else:
        UT = []
        for j in range(n):
            e = [(i, LU[i, j]) for i in range(max(0, j - kv), j) if LU[i, j] != 0]
            UT.append(_Row(j, e, 1.0 / LU[j, j]))
        lcol: dict = {}
        for r, k, v in lent:
            lcol.setdefault(k, []).append((r, v))
        LT = [_Row(k, sorted(lcol[k])) for k in range(n - 1, -1, -1) if k in lcol]
        a = _emit_words(_merge(_levels(UT, lambda R: R.e), True, merge, TW), True, TW)
        b = _emit_words(_merge(_levels(LT, lambda R: R.e), False, merge, TW), False, TW)
    if a is None or b is None:
        return None
    return a + b


def batch_stream_bytes(sample_mats: list, n_border: int, kl: int, ku: int, batch: int,
                       merge: bool = True) -> dict:
    """Estimated device bytes of the level streams of a batch of ``batch`` modes (forward with its
    gather permutation, stream offsets and sparse V^T; transposed with its positions), from the
    per-mode words of ``sample_mats`` (a few modes of the batch): a declared estimate."""
    fw, tw, vt = [], [], []
    TW, n = 1, 0
    for A in sample_mats:
        D, kl_m, ku_m, vnz = bordered_band(A, n_border, kl, ku)
        n = D.shape[0]
        TW = 2 if np.iscomplexobj(D) else 1
        LU, ip = device_lu(D, kl_m, ku_m)
        f = stream_words(LU, ip, kl_m, ku_m, merge, transpose=False)
        t = stream_words(LU, ip, kl_m, ku_m, merge, transpose=True)
        fw.append(None if f is None else f + (f & 1))         # 16-byte aligned mode streams
        tw.append(None if t is None else t + (t & 1))
        vt.append(vnz)
    if not fw or any(v is None for v in fw):
        return dict(applies=False)
    B, r = int(batch), int(n_border)
    fwd = (np.mean(fw) * B + LVL_RING) * 8 + (B + 1) * 8 + B * n * 4
    if r:
        fwd += np.mean(vt) * B * (8 * TW + 4) + B * (r + 1) * 8
    tr = (None if any(v is None for v in tw) else
          (np.mean(tw) * B + LVL_RING) * 8 + (B + 1) * 8 + B * n * 4)
    return dict(applies=True, forward_bytes=float(fwd),
                transposed_bytes=None if tr is None else float(tr),
                forward_words_per_mode=[int(v) for v in fw],
                transposed_words_per_mode=[None if v is None else int(v) for v in tw])


def host_bucket_streams(host, a: float, b: float, samples: int = 3, merge: bool = True) -> list:
    """For every bucket of a ``GlobalSystem`` (real, complexified, complex-mode: the factor-list
    order), :func:`batch_stream_bytes` at the factor key (a, b) from ``samples`` modes spread over
    the bucket (their matrices only are assembled)."""
    out = []
    nr, nc = len(host._buckets), len(host._cbuckets)
    allb = list(host._buckets) + list(host._cbuckets) + list(host._zbuckets)
    for bi, ((_lab, _n, r, kl, ku), bk) in enumerate(zip(host.bands(), allb, strict=True)):
        idx = bk[0]
        B = len(idx)
        pick = sorted({int(round(x)) for x in np.linspace(0, B - 1, min(samples, B))})
        mats = []
        for m in pick:
            if bi < nr:
                s_ = host.slots[idx[m]]
                mats.append((a * s_.M + b * s_.L).tocsr()[s_.rperm][:, s_.cperm].astype(host.dtype))
            elif bi < nr + nc:
                cz = host.slots[idx[m]].cz
                mats.append((a * cz.M + b * cz.L).tocsr()[cz.rperm][:, cz.cperm].astype(np.complex128))
            else:
                i, vi = idx[m]
                v = host.slots[i].zs[vi]
                mats.append((a * v.M + b * v.L).tocsr()[v.rperm][:, v.cperm].astype(np.complex128))
        res = batch_stream_bytes(mats, r, kl, ku, B, merge=merge)
        res.update(batch=B, sampled=pick)
        out.append(res)
    return out
