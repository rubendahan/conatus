"""Principled bounds for dot-product (adjoint) tests: the rule in one place.

A dot-product test compares a = <W, J V> with b = <J^T W, V>. For exact adjoints the difference
is the rounding of the two computations, a - b = <W, e1> - <e2, V> (e1, e2: the errors of J V and
J^T W), plus the rounding of the two pairings. A fixed literal close to one realisation of that
rounding fails on the next unrelated change of the last bits (6.1e-12 -> 1.3e-11 against a
bound of 1e-11). Here the bound is derived from the computation itself:

* the size of e1 (e2) is read off the **linearity defect** |J(V + V2) - J V - J V2| (the sum of
  three independent roundings of the same map: about sqrt(3) |e1|, taken whole, so the estimate
  is conservative); a wrong or non-linear map does not have a rounding-sized defect, so the
  defect itself is capped at its a priori level ``cap`` (eps x the condition of the step solves,
  given by the caller with its reason);
* Cauchy-Schwarz: |a - b| <= |W| e1 + |V| e2 (+ the pairings' rounding, n eps |W| |J V| at most
  each, Higham's gamma_n). This is the bound when the directions are structured (projected,
  admissible: ``dims`` left out);
* for directions drawn independently of the rounding from an isotropic distribution on
  ``dim`` coordinates, the projection <W, e1> is |W| |e1| / sqrt(dim) on average (exactly: a
  centred Gaussian of that standard deviation), so ``dims=(dim_w, dim_v)`` gives
  ``k (|W| e1 / sqrt(dim_w) + |V| e2 / sqrt(dim_v))`` with k = 4 (four standard deviations).

Every check also states its **sensitivity**: an adjoint wrong by a relative ``sensitivity``
(1e-6 by default), J^T W (1 + d), must fail the same bound on the same data; the check asserts it,
so a bound that has become too loose to catch such an error fails the test instead of passing.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EPS = float(np.finfo(float).eps)


def real_pair(a, b) -> float:
    """Re <a, b> (the real pairing of the FlowMap tests), for NumPy or CuPy arrays."""
    return float(np.real((a.conj() * b).sum()))


@dataclass
class DotCheck:
    a: float              # <W, J V>
    b: float              # <J^T W, V>
    bound: float
    e_fwd: float          # linearity defect of J (absolute)
    e_adj: float          # linearity defect of J^T (absolute)
    scale: float          # max(|W| |J V|, |V| |J^T W|)
    JV: object = None     # the computed J V and J^T W (for further checks by the caller)
    JtW: object = None

    @property
    def defect(self) -> float:
        return abs(self.a - self.b)

    def __str__(self) -> str:
        return (f"|a - b| = {self.defect:.3e} vs bound {self.bound:.3e} (relative to the scale: "
                f"{self.defect / self.scale:.2e} / {self.bound / self.scale:.2e}; linearity "
                f"defects {self.e_fwd:.2e}, {self.e_adj:.2e})")


def adjoint_dot_check(fwd: Callable, adj: Callable, V, W, V2, W2, *, cap: float,
                      pair: Callable = real_pair, dims: tuple[int, int] | None = None,
                      k: float = 4.0, sensitivity: float = 1e-6) -> DotCheck:
    """Assert <W, fwd(V)> = <adj(W), V> to the rounding of this very computation (module doc).

    ``V2``, ``W2``: second directions of the same kind as ``V``, ``W`` (the linearity probes).
    ``cap``: a priori relative level of the linearity defect (eps x condition, with the reason at
    the call). ``dims``: (dim_w, dim_v) for isotropic random directions, None for structured ones
    (Cauchy-Schwarz). Returns the measured :class:`DotCheck`."""
    norm = (lambda x: abs(pair(x, x)) ** 0.5)
    JV, JtW = fwd(V), adj(W)
    a, b = pair(W, JV), pair(JtW, V)
    e1 = norm(fwd(V + V2) - JV - fwd(V2))
    e2 = norm(adj(W + W2) - JtW - adj(W2))
    nW, nV, nJV, nJtW = norm(W), norm(V), norm(JV), norm(JtW)
    assert e1 <= cap * nJV and e2 <= cap * nJtW, \
        f"linearity defect above its a priori level {cap:g}: {e1 / nJV:.2e}, {e2 / nJtW:.2e}"
    n = int(np.size(V))
    dw, dv = dims if dims is not None else (1, 1)
    kk = k if dims is not None else 1.0
    bound = kk * (nW * e1 / np.sqrt(dw) + nV * e2 / np.sqrt(dv)) + n * EPS * (nW * nJV + nV * nJtW)
    chk = DotCheck(a, b, float(bound), e1, e2, max(nW * nJV, nV * nJtW), JV, JtW)
    assert chk.defect <= chk.bound, str(chk)
    # sensitivity: the same data with an adjoint wrong by the relative `sensitivity` fail it
    assert abs(a - (1 + sensitivity) * b) > chk.bound, \
        f"bound too loose to catch a relative adjoint error {sensitivity:g}: {chk}"
    return chk



def product_identity_bound(F, B) -> np.ndarray:
    """Entrywise bound on |F B - I| for a forward/backward transform pair (F B = I exactly).

    Rounding model (Higham, Accuracy and Stability, section 3.5): the product of the stored
    matrices errs by at most gamma_M |F| |B| (M: the inner dimension, the quadrature points), and
    the stored entries of F and B, built by recurrences over at most M terms, carry at most as
    much again: 2 M eps (|F| |B|)_ij. For an orthogonal pair (|F| |B|)_ii = 1 and the off-diagonal
    entries are below 1 (Cauchy-Schwarz), so the bound is about 2 M eps; it is a worst case, not a
    realisation of the rounding."""
    M = np.shape(F)[1]
    return 2 * M * EPS * (np.abs(F) @ np.abs(B))


def chain_rounding_bound(mats, f, k: float = 4.0) -> np.ndarray:
    """Entrywise bound on the rounding of ``A_m ... A_1 f`` computed as m matrix-vector products
    (a transform round trip: ``[F, B]``), for exactly stored matrices.

    Rounding model: a variance model under the hypothesis of the probabilistic analysis of
    Higham and Mary (SIAM J. Sci. Comput. 41, 2019: independent mean-zero roundings), not their
    theorem itself (which bounds the error by lambda sqrt(n) u times the l1 norm of the terms,
    sum |a_i v_i|): here the error of a dot product of length n is modelled as having standard
    deviation sqrt(n + 1) eps times the l2 norm of its terms (eps = 2u, a factor 2 of slack; a
    heuristic, tighter than the theorem, held up by the measured margins below). Each
    product adds the variance (n + 1) eps^2 (A^2 v^2) (n terms and the representation of the
    entries) and propagates the earlier ones through A^2; ``k`` standard deviations (4) are
    allowed. It describes the products only: a basis whose stored entries carry more than their
    representation error (inaccurate nodes, weights or function values) exceeds it, which is what
    a test built on it detects (Gauss-Jacobi on scipy's float64 nodes
    fails it 1.9 to 1900 times; the long-double nodes pass with a margin of 15 to 21)."""
    v = np.asarray(f, dtype=float)
    var = np.zeros_like(v)
    for A in mats:
        A = np.asarray(A.toarray() if hasattr(A, "toarray") else A, dtype=float)
        A2 = A ** 2
        var = A2 @ var + (A.shape[1] + 1) * EPS ** 2 * (A2 @ (v ** 2))
        v = A @ v
    return k * np.sqrt(var)
