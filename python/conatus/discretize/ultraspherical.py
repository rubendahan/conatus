"""Sparse Olver–Townsend operators on [-1, 1] (spec §10, §14–15).

Coefficient spaces: lam = 0 is Chebyshev T; lam >= 1 (integer) is the
ultraspherical/Gegenbauer basis C^(lam). Derivatives raise lam by one, conversions
S_lam : C^(lam) -> C^(lam+1) are upper-bidiagonal, multiplication by a polynomial
coefficient is banded. All matrices are square N x N (the top modes that
leave the space are dropped exactly as in Olver & Townsend, SIAM Rev. 2013), and
scipy.sparse only: no dense differentiation matrix is ever built here.

Interval scaling (x in [a, b], t = 2(x - a)/L - 1) is applied by the callers:
d/dx = (2/L) d/dt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import eval_gegenbauer

from ..tolerances import NCC_TAIL_TOL

__all__ = [
    "check_lam", "conversion", "conversion_chain", "differentiation", "multiplication_by_t",
    "multiplication", "NCCReport", "NCCResolutionWarning", "suggested_size", "ncc_policy", "evaluation_row", "integration_row", "basis_values",
]


def check_lam(lam: float) -> int:
    """The level ``lam`` as an int (T: 0, C^(k): k); a non-integer or negative level is refused."""
    if float(lam) != int(lam) or lam < 0:
        raise NotImplementedError(
            f"ultraspherical index lam={lam}: only integer lam >= 0 (Chebyshev T and C^(k)) on a "
            "Chebyshev T axis; Legendre, Jacobi and Ultraspherical trial bases have their own "
            "operators (discretize/jacobi_family.py)")
    return int(lam)


def conversion(N: int, lam: int) -> sp.csr_matrix:
    """S_lam : C^(lam) -> C^(lam+1) (lam = 0: T -> C^(1))."""
    lam = check_lam(lam)
    n = np.arange(N, dtype=float)
    if lam == 0:
        diag = np.full(N, 0.5)
        diag[0] = 1.0
        upper = -0.5 * np.ones(max(N - 2, 0))
    else:
        diag = lam / (n + lam)
        upper = -lam / (n[2:] + lam)
    return sp.diags([diag, upper], [0, 2], shape=(N, N), format="csr")


def conversion_chain(N: int, lam_from: int, lam_to: int) -> sp.csr_matrix:
    """S_{to-1} ... S_{from}; identity if lam_from == lam_to."""
    if lam_to < lam_from:
        raise ValueError(f"cannot convert C^({lam_from}) down to C^({lam_to})")
    out = sp.identity(N, format="csr")
    for lam in range(lam_from, lam_to):
        out = conversion(N, lam) @ out
    return out.tocsr()


def differentiation(N: int, lam: int, order: int = 1) -> sp.csr_matrix:
    """d^order/dt^order : C^(lam) -> C^(lam+order) (lam = 0 means T)."""
    lam = check_lam(lam)
    out = sp.identity(N, format="csr")
    for k in range(lam, lam + order):
        n = np.arange(1, N, dtype=float)
        vals = n if k == 0 else np.full(N - 1, 2.0 * k)
        out = sp.diags([vals], [1], shape=(N, N), format="csr") @ out
    return out.tocsr()


def multiplication_by_t(N: int, lam: int) -> sp.csr_matrix:
    """Multiplication by t in C^(lam) (tridiagonal, the Jacobi operator)."""
    lam = check_lam(lam)
    n = np.arange(N, dtype=float)
    if lam == 0:
        sub = 0.5 * np.ones(N - 1)
        sub[0] = 1.0                         # t T_0 = T_1
        sup = 0.5 * np.ones(N - 1)           # t T_n = (T_{n+1} + T_{n-1}) / 2
    else:
        sub = (n[:-1] + 1) / (2 * (n[:-1] + lam))          # coefficient of C_{n+1}
        sup = (n[1:] + 2 * lam - 1) / (2 * (n[1:] + lam))  # coefficient of C_{n-1}
    return sp.diags([sub, sup], [-1, 1], shape=(N, N), format="csr")


@dataclass(frozen=True)
class NCCReport:
    """What was done to a non-constant coefficient (never silent, spec §4.2)."""

    name: str
    n_coeffs: int            # Chebyshev coefficients of a kept (= bandwidth)
    n_given: int             # coefficients available
    tail: float              # max |a_k| over the last 3 kept coefficients / max |a_k|
    dropped_max: float       # max |a_k| over dropped coefficients (0 if none)
    cutoff: float | None     # user cutoff (None: nothing dropped)
    exact: bool = False      # exact polynomial coefficients (no sampling, so no tail to measure)
    tail_tol: float = NCC_TAIL_TOL   # refusal threshold of the sampled tail (declared)
    rounding: float = 0.0    # > 0: the tail is at this rounding level of its computation (not judged)

    @property
    def accepted_tail(self) -> float:
        """Largest sampled tail accepted: ncc_tail_tol, or the declared ncc_cutoff if larger."""
        return max(self.tail_tol, self.cutoff or 0.0)

    @property
    def resolved(self) -> bool:
        return self.exact or self.rounding > 0 or self.tail <= self.accepted_tail

    def explain(self) -> str:
        s = (f"NCC {self.name}: {self.n_coeffs} of {self.n_given} Chebyshev coefficients used "
             f"(bandwidth {self.n_coeffs}), ")
        s += "exact polynomial coefficients" if self.exact else f"relative tail {self.tail:.2e}"
        if self.cutoff is None:
            s += ", no truncation"
        else:
            s += (f", declared ncc_cutoff={self.cutoff:g} dropped terms up to "
                  f"{self.dropped_max:.2e}")
        if self.rounding > 0:
            s += (f", tail at the rounding level of its computation (<= {self.rounding:.1e} "
                  "absolute: numerically zero or exact up to rounding), not judged")
        elif not self.exact:
            s += (f", sampled tail accepted up to {self.accepted_tail:g} (ncc_tail_tol="
                  f"{self.tail_tol:g}" + (f", ncc_cutoff={self.cutoff:g}" if self.cutoff is not None
                                          else "") + ")")
        return s


class NCCResolutionWarning(UserWarning):
    """Former warning on an under-resolved non-constant coefficient (before v0.1).

    No longer emitted since v0.1: an NCC whose sampled Chebyshev tail
    exceeds ``ncc_tail_tol`` (default :data:`conatus.tolerances.NCC_TAIL_TOL`) is refused with a
    ``DiscretizationError`` naming the resolution to use or the cutoff to declare. The class stays
    importable for one cycle so that warning filters in user code keep working."""


def check_tail_tol(tail_tol: float) -> float:
    """Validate a declared ``ncc_tail_tol`` (a finite number > 0; None was removed in v0.1)."""
    if tail_tol is None:
        from ..errors import OptionError
        raise OptionError(
            "ncc_tail_tol=None is no longer accepted (v0.1)",
            why="None used to mean 'use an under-resolved coefficient as is, with a warning'; an "
                "under-resolved non-constant coefficient is now refused, never used silently",
            fix=f"drop ncc_tail_tol (default {NCC_TAIL_TOL:g}), raise the resolution, or declare "
                "the accepted error: ncc_cutoff=1e-8 (truncation) or ncc_tail_tol=1e-8")
    if isinstance(tail_tol, bool) or not (np.isfinite(tail_tol) and tail_tol > 0):
        from ..errors import OptionError
        raise OptionError(f"ncc_tail_tol={tail_tol!r} must be a finite number > 0",
                          fix=f"drop ncc_tail_tol (default {NCC_TAIL_TOL:g}) or e.g. ncc_tail_tol=1e-8")
    return float(tail_tol)


def ncc_policy(tail_tol: float, cutoff: float | None) -> str:
    """One line for ``Plan.explain()``: how non-constant coefficients are accepted."""
    s = (f"sampled Chebyshev tail of every non-constant coefficient <= ncc_tail_tol = {tail_tol:g}"
         + (" (declared default)" if tail_tol == NCC_TAIL_TOL else " (declared)"))
    if cutoff is not None:
        s += f", or <= the declared ncc_cutoff = {cutoff:g} (truncation)"
    return s + "; above: refused, never truncated silently"


def suggested_size(a: np.ndarray, tol: float) -> int | None:
    """Coefficient count predicted to bring the relative tail of ``a`` under ``tol`` (geometric
    decay fitted on the envelope of the second half of the coefficients); None if they do not
    decay. A declared estimate, never applied."""
    a = np.abs(np.asarray(a))
    n = a.size
    amax = float(a.max()) if n else 0.0
    if n < 6 or amax == 0:
        return None
    env = np.maximum.accumulate(a[::-1])[::-1] / amax       # monotone envelope (parity zeros)
    k = np.arange(n // 2, n)
    y = np.log10(np.maximum(env[k], 1e-300))
    slope = float(np.polyfit(k, y, 1)[0])
    if not slope < -1e-3:
        return None
    tail = float(a[-3:].max()) / amax
    extra = (np.log10(tol) - np.log10(max(tail, 1e-300))) / slope
    m = int(np.ceil(n + max(extra, 1.0) + 3))
    return m + (m % 2)


def _round_up(x: float) -> float:
    """Smallest 1eK, 2eK or 5eK >= x (a readable cutoff to suggest)."""
    e = 10.0 ** np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if m * e >= x * (1 - 1e-12):
            return float(f"{m * e:.0e}")
    return float(x)


def _ncc_coeffs(a_T: np.ndarray, name: str, cutoff: float | None, *,
                tail_tol: float | None = None, exact: bool = False,
                axis: str | None = None, floor: float = 0.0) -> tuple[np.ndarray, NCCReport]:
    """Chebyshev coefficients of an NCC, truncated at the declared ``cutoff`` (None: all kept),
    with their report.

    Resolution: the relative tail of the coefficients *as sampled on the
    grid* (the last 3 of ``a_T``, before any cutoff) must not exceed ``tail_tol`` (None here:
    :data:`NCC_TAIL_TOL`), or the declared ``cutoff`` if it is larger (a cutoff declares that the
    coefficient is accepted to that relative accuracy). Above, DiscretizationError: it names the
    coefficient, its tail, the resolution predicted to resolve it and the cutoff to declare
    instead; nothing is truncated or used silently. ``exact=True``: the coefficients are an exact
    polynomial (not a grid sample), there is no tail to measure (e.g. the polar metric r, r^2).
    ``axis``: name of the sampled direction (for the message). ``floor``: absolute rounding level
    of the computation of the coefficient (``Plan``: ``NCC_ROUNDING`` x its data scale x the
    amplification of its derivatives); a tail at or below it is rounding, not under-resolution
    (e.g. U'' of a linear U, numerically zero): not judged, and the report says so."""
    tol = NCC_TAIL_TOL if tail_tol is None else tail_tol
    a = np.asarray(a_T)
    n_given = a.size
    amax = float(np.max(np.abs(a))) if a.size else 0.0
    accepted = max(tol, cutoff or 0.0)
    rounding = not exact and amax > 0 and float(np.max(np.abs(a[-3:]))) <= floor
    if not exact and amax > 0 and not rounding:
        given_tail = float(np.max(np.abs(a[-3:]))) / amax
        if not given_tail <= accepted:
            raise _unresolved(a, name, given_tail, accepted, tol, axis)
    keep = n_given
    dropped = 0.0
    if cutoff is not None and amax > 0:
        big = np.nonzero(np.abs(a) > cutoff * amax)[0]
        keep = int(big[-1]) + 1 if big.size else 1
        dropped = float(np.max(np.abs(a[keep:]))) if keep < n_given else 0.0
    a = a[:keep]
    tail = float(np.max(np.abs(a[-3:]))) / amax if amax > 0 and not exact else 0.0
    rep = NCCReport(name, keep, n_given, tail, dropped, cutoff, exact, tol, floor if rounding else 0.0)
    return a, rep


def _unresolved(a: np.ndarray, name: str, given_tail: float, accepted: float, tol: float,
                axis: str | None):
    """The DiscretizationError of an under-resolved NCC: what to write instead."""
    from ..errors import DiscretizationError
    n_given = a.size
    along = f" along {axis}" if axis else ""
    n_new = suggested_size(a, tol)
    grow = (f"raise the resolution{along} to about N = {n_new} (predicted from the decay of its "
            f"coefficients; N = {n_given} now)" if n_new is not None else
            f"raise the resolution{along} (its coefficients do not decay at N = {n_given}: is the "
            "coefficient smooth on this interval?)")
    cut = _round_up(given_tail)
    return DiscretizationError(
        f"NCC {name}: not resolved on this grid (relative Chebyshev tail {given_tail:.2e} of its "
        f"{n_given} grid coefficients > {accepted:g})",
        where=f"non-constant coefficient {name}{along}",
        why="on N grid points the coefficient is replaced by its N-term interpolant; with this "
            f"tail the equations solved are not the ones written (error ~{given_tail:.0e} in the "
            f"coefficient). Conatus never truncates it silently (refused above ncc_tail_tol = "
            f"{tol:g}" + (", the declared default conatus.tolerances.NCC_TAIL_TOL)"
                          if tol == NCC_TAIL_TOL else ", as declared)"),
        fix=f"{grow}; or accept this truncation explicitly: ncc_cutoff={cut:g} (drops the "
            f"coefficients below {cut:g} x max, the coefficient is then represented to ~{cut:g}), "
            f"or ncc_tail_tol={cut:g} (keeps all {n_given} coefficients, accepts this tail); "
            "plan.explain() reports either",
        doc="docs/site/guide/errors.md#discretizationerror")


def multiplication(a_T: np.ndarray, N: int, lam: int, *, name: str = "a",
                   cutoff: float | None = None, tail_tol: float | None = None,
                   exact: bool = False, axis: str | None = None,
                   floor: float = 0.0) -> tuple[sp.csr_matrix, NCCReport]:
    """M_lam[a]: multiplication by a(t) = sum_k a_k T_k(t) acting on C^(lam) coefficients.

    Built as sum_k a^(lam)_k C^(lam)_k(X_lam) (Olver–Townsend) on an extended space of size
    N + deg(a) + 1 so that the N x N truncation is exact. No coefficient of ``a`` is dropped
    unless the user passes ``cutoff``; either way the NCCReport says what happened.
    ``tail_tol`` (declared ncc_tail_tol) and ``exact``: see ``_ncc_coeffs``.
    """
    lam = check_lam(lam)
    a, rep = _ncc_coeffs(a_T, name, cutoff, tail_tol=tail_tol, exact=exact, axis=axis,
                         floor=floor)
    m = a.size
    Ne = N + m + 1
    a_ext = np.zeros(Ne, dtype=np.result_type(a, float))
    a_ext[:m] = a
    a_lam = conversion_chain(Ne, 0, lam) @ a_ext
    X = multiplication_by_t(Ne, lam)
    I = sp.identity(Ne, format="csr")
    Pm1 = I
    out = a_lam[0] * I
    if m > 1:
        P = X if lam == 0 else 2 * lam * X
        out = out + a_lam[1] * P
        for k in range(1, m - 1):
            if lam == 0:
                Pn = 2 * (X @ P) - Pm1
            else:
                Pn = ((2 * (k + lam)) * (X @ P) - (k + 2 * lam - 1) * Pm1) / (k + 1)
            Pm1, P = P, Pn.tocsr()
            out = out + a_lam[k + 1] * P
    out = out.tocsr()[:N, :N]
    out.eliminate_zeros()
    return out.tocsr(), rep


def basis_values(N: int, lam: int, t: float, deriv: int = 0) -> np.ndarray:
    """Row [d^deriv/dt^deriv P_n(t)]_{n<N} for P = T (lam = 0) or C^(lam)."""
    lam = check_lam(lam)
    if deriv:
        # derivative of C^(lam)_n (or T_n) = factor * C^(lam+deriv)_{n-deriv}
        D = differentiation(N, lam, deriv)
        return basis_values(N, lam + deriv, t) @ D
    n = np.arange(N)
    if lam == 0:
        if t == 1.0:
            return np.ones(N)
        if t == -1.0:
            return (-1.0) ** n
        return np.cos(n * np.arccos(np.clip(t, -1, 1)))
    return eval_gegenbauer(n, lam, t)


def evaluation_row(N: int, lam: int, t: float, deriv: int = 0) -> np.ndarray:
    """Trace row at t in [-1, 1] (the caller has checked the position): ``basis_values``."""
    return basis_values(N, lam, t, deriv)


def integration_row(N: int, lam: int) -> np.ndarray:
    """[int_{-1}^{1} P_n(t) dt]_{n<N}, exact (Gauss–Legendre on enough nodes)."""
    lam = check_lam(lam)
    if lam == 0:
        n = np.arange(N)
        out = np.zeros(N)
        even = n % 2 == 0
        out[even] = 2.0 / (1.0 - n[even] ** 2)
        return out
    from .jacobi_family import gauss_legendre    # refined weights (Q8.8)
    x, w = gauss_legendre(N // 2 + 2)
    n = np.arange(N)
    return eval_gegenbauer(n[:, None], lam, x[None, :]) @ w
