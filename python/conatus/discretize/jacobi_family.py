"""Legendre, Jacobi and Ultraspherical (Gegenbauer) trial bases on a bounded interval.

Same construction as Olver & Townsend (2013) for Chebyshev T, with Jacobi polynomials: variables
are expanded in the declared basis (level 0), an equation of order q is written at level q, i.e.
in P_n^(a+q, b+q)(t), and its q taus are lifted into the last q modes (generalized tau). All
operators are sparse and exact on span{n < N}, built from three classical identities
(checked numerically in tests/discretize/test_jacobi_family.py):

* derivative     d/dt P_n^(a,b) = (n + a + b + 1)/2 P_{n-1}^(a+1,b+1)              (one diagonal)
* conversion     (2n+a+b+1) P_n^(a,b) = (n+a+b+1) P_n^(a+1,b) - (n+b) P_{n-1}^(a+1,b)
                 (2n+a+b+1) P_n^(a,b) = (n+a+b+1) P_n^(a,b+1) + (n+a) P_{n-1}^(a,b+1)   (bidiagonal)
* multiplication t P_n = A_n P_{n+1} + B_n P_n + C_n P_{n-1}                         (tridiagonal)

Coefficients of the variables (level 0) are those of the basis as declared by ``coeff_space()``:
P_n^(a,b) for ``Jacobi``, P_n for ``Legendre``, C_n^(lambda) for ``Ultraspherical`` (and its
Dedalus names ``ChebyshevU`` = C^(1), ``ChebyshevV`` = C^(2)): the Gegenbauer scaling
s_n = C_n^(lambda) / P_n^(lambda-1/2, lambda-1/2) is a diagonal similarity applied where an
operator reads or writes level 0. Higher levels are internal (equation rows) and use P^(a+q,b+q).

Grid: Gauss–Jacobi nodes of the weight (1 - t)^a (1 + t)^b (M = dealias * N points), transforms
by that quadrature (exact for polynomials of degree < 2M - N + 1); nodes, weights and polynomial
values are computed in long double (orthonormal recurrence, Newton-refined nodes, Christoffel
weights) and cast once, so the coefficient tails sit at the float64 round-off. A non-constant coefficient is
expanded in P^(a,b) on the N-point grid (its tail reported as for Chebyshev, never truncated
silently) and multiplied as sum_k a_k P_k(J), J the Jacobi operator of the row level.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.special import eval_jacobi, gammaln, roots_jacobi

from ..ir.bases import Basis, Chebyshev, Jacobi, Legendre, Ultraspherical
from . import ultraspherical as us
from ..errors import ResolutionError

__all__ = ["JacobiFamily", "is_jacobi_trial"]


def is_jacobi_trial(b: Any) -> bool:
    """A Jacobi-family basis handled here (Chebyshev T keeps its own code path)."""
    return isinstance(b, Jacobi) and not isinstance(b, Chebyshev)


def _frozen(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    for a in arrays:
        a.setflags(write=False)
    return arrays


def _frozen_sparse(M: sp.csr_matrix) -> sp.csr_matrix:
    """A sparse matrix returned by an ``lru_cache`` is shared by every later caller: canonical
    (sorted, no duplicates, so no SciPy call needs to reorder it in place) and read-only, so an
    in-place ``M *= c`` or ``M.data[...] = v`` raises instead of changing every later operator
    (review of the state audit, F1)."""
    M.sum_duplicates()
    M.sort_indices()
    for a in (M.data, M.indices, M.indptr):
        a.setflags(write=False)
    return M


def _abc(N: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-term coefficients t P_n = A_n P_{n+1} + B_n P_n + C_n P_{n-1}, n < N."""
    A, B, C = np.zeros(N), np.zeros(N), np.zeros(N)
    A[0], B[0] = 2.0 / (a + b + 2), (b - a) / (a + b + 2)
    n = np.arange(1, N, dtype=float)
    m = 2 * n + a + b
    A[1:] = 2 * (n + 1) * (n + a + b + 1) / ((m + 1) * (m + 2))
    B[1:] = (b * b - a * a) / (m * (m + 2))
    C[1:] = 2 * (n + a) * (n + b) / (m * (m + 1))
    return A, B, C


@lru_cache(maxsize=128)
def _J(N: int, a: float, b: float) -> sp.csr_matrix:
    """Multiplication by t on P^(a,b) coefficients (N x N truncation of the tridiagonal)."""
    A, B, C = _abc(N, a, b)
    return _frozen_sparse(sp.diags([A[:-1], B, C[1:]], [-1, 0, 1], shape=(N, N), format="csr"))


@lru_cache(maxsize=128)
def _S(N: int, a: float, b: float) -> sp.csr_matrix:
    """Conversion P^(a,b) -> P^(a+1,b+1) (upper triangular, bandwidth 2), exact."""
    n = np.arange(N, dtype=float)

    def step(a_: float, b_: float, sign: float, which: str) -> sp.csr_matrix:
        den = 2 * n + a_ + b_ + 1
        diag = np.where(n == 0, 1.0, (n + a_ + b_ + 1) / np.where(den == 0, 1.0, den))
        par = b_ if which == "a" else a_                # (n+b) for a -> a+1, (n+a) for b -> b+1
        sup = sign * (n[1:] + par) / den[1:]
        return sp.diags([diag, sup], [0, 1], shape=(N, N), format="csr")
    Sa = step(a, b, -1.0, "a")                           # (a, b) -> (a+1, b)
    Sb = step(a + 1, b, +1.0, "b")                       # (a+1, b) -> (a+1, b+1)
    return _frozen_sparse((Sb @ Sa).tocsr())


@lru_cache(maxsize=128)
def _D(N: int, a: float, b: float) -> sp.csr_matrix:
    """d/dt: P^(a,b) -> P^(a+1,b+1), single superdiagonal."""
    n = np.arange(1, N, dtype=float)
    return _frozen_sparse(sp.diags([(n + a + b + 1) / 2], [1], shape=(N, N), format="csr"))


_LD = np.longdouble


def _orthonormal(n_max: int, a: float, b: float, t: np.ndarray) -> np.ndarray:
    """Orthonormal Jacobi polynomials p_0..p_{n_max-1} (weight (1-t)^a (1+t)^b on [-1, 1]) at
    ``t``, by their three-term recurrence in long double (the
    float64 quadrature of scipy left coefficient tails ~500x above Dedalus' at N = 128)."""
    a_, b_, t = _LD(a), _LD(b), np.asarray(t, dtype=_LD)
    P = np.zeros((n_max, t.size), dtype=_LD)
    h0 = np.exp(_LD((a + b + 1) * np.log(2.0) + gammaln(a + 1) + gammaln(b + 1) - gammaln(a + b + 2)))
    P[0] = 1 / np.sqrt(h0)
    if n_max == 1:
        return P

    def alpha(n: int) -> Any:
        if n == 0:
            return (b_ - a_) / (a_ + b_ + 2)
        m = 2 * n + a_ + b_
        return (b_ * b_ - a_ * a_) / (m * (m + 2))

    def beta(n: int) -> Any:                             # n >= 1; n = 1 apart (0/0 when a + b = -1)
        if n == 1:
            return np.sqrt(4 * (1 + a_) * (1 + b_) / ((a_ + b_ + 2) ** 2 * (a_ + b_ + 3)))
        m = 2 * n + a_ + b_
        return np.sqrt(4 * n * (n + a_) * (n + b_) * (n + a_ + b_) / (m * m * (m + 1) * (m - 1)))

    P[1] = (t - alpha(0)) * P[0] / beta(1)
    for n in range(1, n_max - 1):                        # t p_n = b_{n+1} p_{n+1} + a_n p_n + b_n p_{n-1}
        P[n + 1] = ((t - alpha(n)) * P[n] - beta(n) * P[n - 1]) / beta(n + 1)
    return P


@lru_cache(maxsize=64)
def _gauss_ld(M: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Jacobi nodes and weights in long double: scipy's nodes refined by Newton on p_M
    (d/dt p_M = sqrt(M (M + a + b + 1)) p_{M-1}^(a+1,b+1)), Christoffel weights 1 / sum_n p_n^2."""
    t = np.asarray(roots_jacobi(M, a, b)[0], dtype=_LD)
    for _ in range(3):
        pM = _orthonormal(M + 1, a, b, t)[M]
        dpM = np.sqrt(_LD(M) * (M + a + b + 1)) * _orthonormal(M, a + 1, b + 1, t)[M - 1]
        t = t - pM / dpM
    w = 1 / np.sum(_orthonormal(M, a, b, t) ** 2, axis=0)
    return t, w


@lru_cache(maxsize=64)
def _nodes(M: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    t, w = _gauss_ld(M, a, b)
    return _frozen(np.asarray(t, dtype=float), np.asarray(w, dtype=float))


def gauss_legendre(M: int) -> tuple[np.ndarray, np.ndarray]:
    """M-point Gauss-Legendre nodes (ascending) and weights, Newton-refined in long double and
    cached read-only: numpy's ``leggauss`` weights carry a relative error growing
    to 3.9e-12 at M = 97 (8.5e-14 at M = 25), which left |F B - I| = 1.6e-13 in the spherical
    harmonic transform at L = 64 and 1.1e-13 in the disk transform at N = 40."""
    return _nodes(int(M), 0.0, 0.0)


class JacobiFamily:
    """Family object of a Legendre / Jacobi / Ultraspherical basis (protocol of
    ``discretize.families``)."""

    has_constant = True

    def __init__(self, basis: Basis) -> None:
        self.basis = basis
        self.N = basis.size
        self.a, self.b = float(basis.a), float(basis.b)
        self.x0, self.L = float(basis.bounds[0]), float(basis.length)
        if isinstance(basis, Ultraspherical):
            lam = float(basis.lam)
            if lam <= 0:
                from ..errors import OptionError
                raise OptionError(f"{basis!r}: Ultraspherical trial basis needs lambda > 0",
                                  fix="lambda = 0 is Chebyshev T: use Chebyshev", doc="docs/site/guide/bases.md")
            n = np.arange(self.N, dtype=float)
            # s_n = C_n^(lam)(1) / P_n^(lam-1/2, lam-1/2)(1)
            self.s = np.exp(gammaln(n + 2 * lam) - gammaln(2 * lam)
                            - gammaln(n + lam + 0.5) + gammaln(lam + 0.5))
            self.kind = f"Ultraspherical C^({lam:g})"
            self.coeffs_name = f"C_n^({lam:g})"
        else:
            self.s = np.ones(self.N)
            self.kind = "Legendre" if isinstance(basis, Legendre) else f"Jacobi P^({self.a:g},{self.b:g})"
            self.coeffs_name = "P_n" if isinstance(basis, Legendre) else f"P_n^({self.a:g},{self.b:g})"
        self._Sd = sp.diags(self.s, format="csr")
        self._Sinv = sp.diags(1.0 / self.s, format="csr")

    # ---- levels -------------------------------------------------------------------------------
    def _ab(self, lam: int) -> tuple[float, float]:
        return self.a + lam, self.b + lam

    def _in0(self, M: sp.csr_matrix, lam: int) -> sp.csr_matrix:
        """Right-multiply by the level-0 scaling when the operator reads level 0."""
        return (M @ self._Sd).tocsr() if lam == 0 else M

    # ---- grids and transforms -----------------------------------------------------------------
    def grid_size(self, dealias: bool) -> int:
        return self.basis.grid_size if dealias else self.basis.size

    def grid(self, dealias: bool = True) -> np.ndarray:
        t, _ = _nodes(self.grid_size(dealias), self.a, self.b)
        return self.x0 + (t + 1) * self.L / 2

    def matrices(self, dealias: bool) -> tuple[np.ndarray, np.ndarray]:
        """(B, F): grid values = B @ c (M x N), c = F @ grid values (N x M)."""
        return _mats(self.N, self.grid_size(dealias), self.a, self.b, tuple(self.s))

    def forward(self, f: np.ndarray, axis: int = 0, dealias: bool = True) -> np.ndarray:
        from .families import _apply_rows
        M = self.grid_size(dealias)
        if np.shape(f)[axis] != M:
            raise ValueError(f"expected {M} grid points along axis {axis}, got {np.shape(f)[axis]}")
        return _apply_rows(self.matrices(dealias)[1], f, axis)

    def backward(self, c: np.ndarray, axis: int = 0, dealias: bool = True) -> np.ndarray:
        from .families import _apply_rows
        if np.shape(c)[axis] != self.N:
            raise ValueError(f"expected {self.N} coefficients along axis {axis}, got {np.shape(c)[axis]}")
        return _apply_rows(self.matrices(dealias)[0], c, axis)

    # ---- sparse operators (compiler) ----------------------------------------------------------
    def conversion_chain(self, lam_from: int, lam_to: int) -> sp.csr_matrix:
        out = sp.identity(self.N, format="csr")
        for lam in range(lam_from, lam_to):
            out = (_S(self.N, *self._ab(lam)) @ out).tocsr()
        return self._in0(out, lam_from) if lam_to > lam_from else out

    def differentiation(self, lam: int, order: int = 1) -> sp.csr_matrix:
        out = sp.identity(self.N, format="csr")
        for k in range(order):
            out = (_D(self.N, *self._ab(lam + k)) @ out).tocsr() * (2.0 / self.L)
        return self._in0(out, lam)

    def _values(self, lam: int, t: float) -> np.ndarray:
        a, b = self._ab(lam)
        return eval_jacobi(np.arange(self.N), a, b, t)

    def evaluation_row(self, lam: int, pos: float, deriv: int = 0, side: str | None = None) -> np.ndarray:
        t = 2 * (pos - self.x0) / self.L - 1
        row = self._values(lam + deriv, t)
        if deriv:
            D = sp.identity(self.N, format="csr")
            for k in range(deriv):
                D = (_D(self.N, *self._ab(lam + k)) @ D).tocsr() * (2.0 / self.L)
            row = np.asarray(row @ D).ravel()
        return row * self.s if lam == 0 else row

    def integration_row(self, lam: int) -> np.ndarray:
        x, w = gauss_legendre(self.N // 2 + 2)          # exact for degree < N + 3 (refined)
        a, b = self._ab(lam)
        row = eval_jacobi(np.arange(self.N)[:, None], a, b, x[None, :]) @ w * (self.L / 2)
        return row * self.s if lam == 0 else row

    def constant_column(self) -> sp.csr_matrix:
        col = np.zeros((self.N, 1))
        col[0, 0] = 1.0 / self.s[0]                     # P_0 = 1 (C_0 = 1 too: s_0 = 1)
        return sp.csr_matrix(col)

    def lift_index(self, n: int, piece: int | None) -> int:
        return self.N + n

    # ---- coefficient-space operators at level 0 (explicit RHS) --------------------------------
    def diff_coeffs(self, c: np.ndarray) -> np.ndarray:
        """d/dx along axis 0, level 0 -> level 0 (exact: back-substitution in the upper
        triangular conversion level 0 -> 1, since the derivative has degree < N - 1)."""
        from scipy.linalg import solve_triangular
        c = np.asarray(c)
        sh = c.shape
        p1 = self.differentiation(0, 1) @ c.reshape(sh[0], -1)       # level-1 coefficients
        S = self.conversion_chain(0, 1).toarray()                     # level 0 -> level 1
        return solve_triangular(S, p1, lower=False).reshape(sh)

    def integ_weights(self) -> np.ndarray:
        return self.integration_row(0)

    def interp_row(self, pos: float, side: str | None = None) -> np.ndarray:
        return self.evaluation_row(0, pos, 0, side)

    # ---- non-constant coefficients ------------------------------------------------------------
    def ncc_forward(self, g: np.ndarray, axis: int) -> np.ndarray:
        """P^(a,b) coefficients (unscaled) of a coefficient sampled on the N-point grid."""
        from .families import _apply_rows
        F = _mats(self.N, self.N, self.a, self.b, tuple(np.ones(self.N)))[1]
        return _apply_rows(F, g, axis)

    def multiplication(self, a_P: np.ndarray, lam: int, *, name: str = "a", cutoff=None,
                       tail_tol=None, exact: bool = False,
                       floor: float = 0.0) -> tuple[sp.csr_matrix, Any]:
        """Multiplication by a(t) = sum_k a_k P_k^(a,b)(t) acting on level-``lam`` coefficients:
        sum_k a_k P_k(J_lam), built on N + deg + 1 modes then truncated (exact on span{n < N})."""
        coef, rep = us._ncc_coeffs(np.asarray(a_P), name, cutoff, tail_tol=tail_tol, exact=exact,
                                   floor=floor)
        rep = _JacobiNCCReport(rep, self.kind)
        m = coef.size
        Ne = self.N + m + 1
        X = _J(Ne, *self._ab(lam))
        A, B, C = _abc(max(m, 1), self.a, self.b)        # recurrence of the level-0 family
        I = sp.identity(Ne, format="csr")
        Pm1, P = None, I
        out = coef[0] * I
        for k in range(m - 1):                           # P_{k+1} = ((X - B_k) P_k - C_k P_{k-1}) / A_k
            Pn = (X @ P - B[k] * P - (C[k] * Pm1 if Pm1 is not None else 0 * I)) / A[k]
            Pm1, P = P, sp.csr_matrix(Pn)
            out = out + coef[k + 1] * P
        out = sp.csr_matrix(out)[:self.N, :self.N]
        out.eliminate_zeros()
        if lam == 0:
            out = (self._Sinv @ out @ self._Sd).tocsr()
        return out.tocsr(), rep

    def explain(self) -> str:
        return (f"{self.basis.coord.name}: {self.kind} basis, coefficients {self.coeffs_name}(t), "
                f"N={self.N}, t = 2(x - {self.x0:g})/{self.L:g} - 1; equations of order q written in "
                f"P^({self.a:g}+q,{self.b:g}+q) (sparse Jacobi derivative/conversion, Olver–Townsend "
                f"construction), taus in their last q modes; Gauss–Jacobi grid of "
                f"{self.basis.grid_size} points")


class _JacobiNCCReport:
    """NCC report (us.NCCReport) worded for a Jacobi-family axis."""

    def __init__(self, rep: Any, kind: str) -> None:
        self._rep, self.kind = rep, kind

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rep, name)

    def explain(self) -> str:
        return self._rep.explain().replace("Chebyshev coefficients", f"{self.kind} coefficients")


@lru_cache(maxsize=64)
def _mats(N: int, M: int, a: float, b: float, s: tuple) -> tuple[np.ndarray, np.ndarray]:
    if M < N:
        raise ResolutionError(f"Jacobi grid of {M} points for {N} modes: the grid needs M >= N",
                              fix="a grid of at least N points (dealias >= 1)")
    t, w = _gauss_ld(M, a, b)                      # long double throughout, cast once at the end
    p = _orthonormal(N, a, b, t)                                         # (N, M), orthonormal
    # classical P_n = r_n p_n, r_n = sqrt(h_n) = sum_j w_j P_n(t_j) p_n(t_j) (exact quadrature)
    A_, B_, C_ = (np.asarray(x, dtype=_LD) for x in _abc(N, a, b))
    P = np.zeros_like(p)
    P[0] = 1
    if N > 1:
        P[1] = (t - B_[0]) / A_[0]
    for n in range(1, N - 1):
        P[n + 1] = ((t - B_[n]) * P[n] - C_[n] * P[n - 1]) / A_[n]
    r = (P * p) @ w
    sv = np.asarray(s, dtype=_LD)
    B = (r[:, None] * p * sv[:, None]).T                                 # values = B @ c
    F = (p * w[None, :]) / (r * sv)[:, None]                             # c = F @ values
    return _frozen(np.asarray(B, dtype=float).copy(), np.asarray(F, dtype=float).copy())
