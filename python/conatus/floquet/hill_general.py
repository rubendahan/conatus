"""General Floquet–Hill operator: any linear problem, any number of base-flow harmonics.

A T-periodic linear system  M dx/dt + L(t) x = 0,  L(t) = Σ_{|m|<=M} L_m e^{i m ω t},
has Floquet solutions x = e^{μ t} Σ_n x_n e^{i n ω t}. Truncating to n = -Nh..Nh gives the Hill
system, block-banded of half-width M:

    (L_0 + (μ + i n ω) M) x_n + Σ_{m≠0} L_m x_{n-m} = 0          (block row n)

``HillBlocks`` holds these blocks and factors the Hill matrix by block-Thomas on super-blocks
of M consecutive harmonics (a block-banded matrix of half-width M is block-tridiagonal in
super-blocks), so the cost stays linear in Nh for every M. ``slogdet`` accumulates log|pivots|
and never forms det M(μ) (no FP64 overflow/underflow whatever the size).

Two builders:
- ``hill_from_plan``: the blocks come from the plan of any IR problem (EVP convention
  (L + σ M) x = 0). The base flow enters through NCC fields; L_m is obtained as
  L(U := U_m) - L(U := 0), after an explicit linearity check in U (refused otherwise).
  μ-dependent radiation rows, tests).

Top/boundary conditions are whatever the plan's BC rows say (Dirichlet, Neumann, Robin...); Mathematics and
validation: section "Generalisation".
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import scipy.linalg as sla

from ..errors import NumericalError, OptionError
from .roots import (SECANT_MAX_ITERATIONS, SECANT_STEP, SecantRoot, check_allow_unconverged,
                    check_count, determinant_secant, unconverged)
from .._deprecation import renames
from ..errors import EquationError

__all__ = ["HillBlocks", "hill_from_plan", "hill_from_blocks", "harmonic_matrices", "secant_root",
           "exponents_dense"]


class HillBlocks:
    """Block-banded Hill matrix M(μ) with half-width M = max |m| of the couplings.

    diag(μ, n) -> (m, m) block of harmonic n (includes (μ + i n ω) M); couplings[m] couples
    x_{n-m} into block row n (m ≠ 0, constant in μ). Nh >= M is required.
    """

    def __init__(self, *, diag: Callable[[complex, int], np.ndarray],
                 couplings: Mapping[int, np.ndarray], Nh: int, size: int, label: str,
                 omega: float = 1.0, notes: tuple[str, ...] = (), mass: np.ndarray | None = None):
        if any(m == 0 for m in couplings):
            raise OptionError("couplings must not contain m = 0 (it belongs to the diagonal block)",
                              fix="put the m = 0 harmonic in the diagonal block, not in couplings")
        self.Mhalf = max((abs(m) for m in couplings), default=0)
        # an integer number of harmonics (Nh=10.5 used to fail later in range()); OptionError is
        # a ValueError, as before
        if isinstance(Nh, bool) or not isinstance(Nh, (int, np.integer)) or Nh < self.Mhalf:
            raise OptionError(f"need an integer Nh >= max|m| = {self.Mhalf} (got Nh = {Nh!r})",
                              fix=f"e.g. Nh={max(self.Mhalf, 1) * 8} (check convergence in Nh)")
        for m, C in couplings.items():
            if C.shape != (size, size):
                raise OptionError(f"coupling m={m} has shape {C.shape}, expected {(size, size)}",
                                  fix="give each coupling as a square block of the diagonal block's size")
        self.diag, self.C = diag, {int(m): np.asarray(C, complex) for m, C in couplings.items()}
        self.Nh, self.size, self.label, self.omega = Nh, size, label, omega
        self.notes = tuple(notes)
        self.mass = None if mass is None else np.asarray(mass, complex)   # dM/dμ block (μ-linear case)

    # ------------------------------------------------------------------ structure
    @property
    def harmonics(self) -> list[int]:
        return list(range(-self.Nh, self.Nh + 1))

    def _super(self) -> list[list[int]]:
        """Harmonics grouped in super-blocks of Mhalf (block-tridiagonal in super-blocks)."""
        w = max(self.Mhalf, 1)
        h = self.harmonics
        return [h[i:i + w] for i in range(0, len(h), w)]

    def _block(self, mu: complex, rows: list[int], cols: list[int]) -> np.ndarray:
        """Sub-matrix of M(μ) for harmonic rows × harmonic cols (row n, col n' couples m = n - n')."""
        s = self.size
        out = np.zeros((len(rows) * s, len(cols) * s), complex)
        for i, n in enumerate(rows):
            for j, n2 in enumerate(cols):
                m = n - n2
                if m == 0:
                    out[i * s:(i + 1) * s, j * s:(j + 1) * s] = self.diag(mu, n)
                elif m in self.C:
                    out[i * s:(i + 1) * s, j * s:(j + 1) * s] = self.C[m]
        return out

    # ------------------------------------------------------------------ factorization
    def factor(self, mu: complex) -> HillFactorization | BandedHillFactorization:
        """Block-Thomas on super-blocks: S_k = D_k - L_k S_{k-1}⁻¹ U_{k-1} (partial-pivot LU
        inside each super-block, no pivoting across them).

        Block-Thomas needs every intermediate Schur complement S_k (k < last) to be well
        conditioned: a regular Hill matrix can have a singular S_k (finding 11: det -2j, cond 4.98, and slogdet / solve came back NaN). When the reciprocal
        condition of an intermediate S_k is below ``THOMAS_RCOND_MIN``, the whole matrix is
        factored instead by a banded LU with partial pivoting across super-blocks (LAPACK
        gbtrf, still linear in Nh); ``factorization.method`` says which one ran. The last S_k
        may be singular: then M(μ) is (μ is an exponent)."""
        sup = self._super()
        lus, lows, ups = [], [], []
        prev = None
        for k, rows in enumerate(sup):
            D = self._block(mu, rows, rows)
            if k > 0:
                Lk = self._block(mu, rows, sup[k - 1])
                D = D - Lk @ sla.lu_solve(prev, ups[-1], check_finite=False)
                lows.append(Lk)
            if not np.isfinite(D).all():
                return self._banded(mu)
            with warnings.catch_warnings():     # a singular block is handled just below
                warnings.simplefilter("ignore", sla.LinAlgWarning)
                prev = sla.lu_factor(D, check_finite=False)
            if k < len(sup) - 1 and _rcond(prev, D) < THOMAS_RCOND_MIN:
                return self._banded(mu)
            lus.append(prev)
            if k < len(sup) - 1:
                ups.append(self._block(mu, rows, sup[k + 1]))
        return HillFactorization(mu, sup, self.size, lus, lows, ups)

    def _banded(self, mu: complex) -> BandedHillFactorization:
        """M(μ) in LAPACK band storage, LU with partial pivoting (zgbtrf)."""
        s, h = self.size, self.harmonics
        n = len(h) * s
        kl = ku = (self.Mhalf + 1) * s - 1            # scalar half-bandwidth of the block band
        ab = np.zeros((2 * kl + ku + 1, n), complex)
        for i, row in enumerate(h):
            lo, hi = max(0, i - self.Mhalf), min(len(h), i + self.Mhalf + 1)
            blk = self._block(mu, [row], h[lo:hi])            # (s, (hi-lo) s)
            r0 = i * s
            for jj in range(blk.shape[1]):
                c = lo * s + jj
                rows = np.arange(r0, r0 + s)
                ab[kl + ku + rows - c, c] = blk[:, jj]
        lu, piv, info = sla.lapack.zgbtrf(ab, kl, ku)
        if info < 0:
            raise RuntimeError(f"zgbtrf: illegal argument {-info}")   # internal error
        return BandedHillFactorization(mu, lu, piv, kl, ku, s)

    def slogdet(self, mu: complex) -> tuple[complex, float]:
        return self.factor(mu).slogdet()

    def solve(self, mu: complex, rhs: np.ndarray) -> np.ndarray:
        """M(μ) x = rhs with its normwise backward error checked: above
        ``SOLVE_BACKWARD_ERROR_MAX`` the banded pivoted LU is tried, then NumericalError
        (never a NaN or an inaccurate solution passed on)."""
        rhs = np.asarray(rhs, complex)
        F = self.factor(mu)
        x = F.solve(rhs)
        eta = self.backward_error(mu, x, rhs)
        if not eta <= SOLVE_BACKWARD_ERROR_MAX and isinstance(F, HillFactorization):
            x = self._banded(mu).solve(rhs)
            eta = self.backward_error(mu, x, rhs)
        if not eta <= SOLVE_BACKWARD_ERROR_MAX:
            raise NumericalError(
                f"Hill solve at μ = {complex(mu):.6g}: backward error {eta:.2e} > "
                f"{SOLVE_BACKWARD_ERROR_MAX:g} (non-finite or singular to working precision)",
                fix="move μ off the exponent (a shift), or check the blocks for NaN/Inf")
        return x

    def backward_error(self, mu: complex, x: np.ndarray, rhs: np.ndarray) -> float:
        """‖M x - b‖∞ / (‖M‖∞ ‖x‖∞ + ‖b‖∞) (normwise backward error; NaN -> inf)."""
        r = self.matvec(mu, x) - rhs
        h, s = self.harmonics, self.size
        norm_m = 0.0
        for i, n in enumerate(h):
            row = np.abs(self.diag(mu, n)).sum(axis=1)
            for m, C in self.C.items():
                if 0 <= i - m < len(h):
                    row = row + np.abs(C).sum(axis=1)
            norm_m = max(norm_m, float(row.max()) if s else 0.0)
        den = norm_m * float(np.abs(x).max(initial=0.0)) + float(np.abs(rhs).max(initial=0.0))
        num = float(np.abs(r).max(initial=0.0))
        if not np.isfinite(num) or not np.isfinite(den):
            return float("inf")
        return num / den if den > 0 else (0.0 if num == 0 else float("inf"))

    def matvec(self, mu: complex, x: np.ndarray) -> np.ndarray:
        """M(μ) x for x of shape (2Nh+1, size)."""
        h, out = self.harmonics, np.zeros_like(x, dtype=complex)
        for i, n in enumerate(h):
            v = self.diag(mu, n) @ x[i]
            for m, C in self.C.items():
                j = i - m
                if 0 <= j < len(h):
                    v = v + C @ x[j]
            out[i] = v
        return out

    def dense(self, mu: complex) -> np.ndarray:
        """Assembled M(μ), small problems and oracles only."""
        n = len(self.harmonics) * self.size
        if n > 6000:
            raise MemoryError(f"dense Hill matrix of size {n} refused (use factor/slogdet)")
        return self._block(mu, self.harmonics, self.harmonics)

    def winding(self, center: complex, radius: float, n_points: int) -> float:
        """Winding number of det M on |μ - center| = radius (an integer read off the adaptive
        walk of :func:`conatus.floquet.roots.adaptive_winding`, a test on samples; ConvergenceError
        when the walk cannot be completed). ``n_points``: initial points on the circle."""
        from .roots import adaptive_winding
        return adaptive_winding(self.slogdet, center, radius, n_points)

    def explain(self) -> str:
        k = len(self.harmonics)
        return (f"Floquet–Hill ({self.label}): harmonics n = -{self.Nh}..{self.Nh} (ω = {self.omega:g}), "
                f"base-flow couplings m ∈ {sorted(self.C)} (half-width {self.Mhalf}); block size "
                f"{self.size}, Hill size {k * self.size}; block-Thomas on {len(self._super())} "
                f"super-blocks of {max(self.Mhalf, 1)} harmonics (partial-pivot LU, FP64; banded "
                f"LU pivoting across super-blocks when a Schur complement has rcond < "
                f"{THOMAS_RCOND_MIN:g}); solve checks its backward error; "
                f"slogdet accumulated as Σ log|pivot| (no overflow of det)."
                + (" " + " ".join(self.notes) if self.notes else ""))


class HillFactorization:
    """Reusable block-Thomas factors of M(μ) on super-blocks."""

    def __init__(self, mu, sup, size, lus, lows, ups):
        self.mu, self.sup, self.size = mu, sup, size
        self.lus, self.lows, self.ups = lus, lows, ups

    def _split(self, x: np.ndarray) -> list[np.ndarray]:
        out, i = [], 0
        for rows in self.sup:
            out.append(x[i:i + len(rows)].reshape(-1))
            i += len(rows)
        return out

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """M(μ) x = rhs, rhs of shape (2Nh+1, size)."""
        b = self._split(np.asarray(rhs, complex))
        g = []
        for k, lu in enumerate(self.lus):
            v = b[k] if k == 0 else b[k] - self.lows[k - 1] @ g[-1]
            g.append(sla.lu_solve(lu, v, check_finite=False))
        x = [None] * len(g)
        x[-1] = g[-1]
        for k in range(len(g) - 2, -1, -1):
            x[k] = g[k] - sla.lu_solve(self.lus[k], self.ups[k] @ x[k + 1], check_finite=False)
        return np.concatenate(x).reshape(-1, self.size)

    method = "block-Thomas"

    def slogdet(self) -> tuple[complex, float]:
        """(phase, log|det|) from the pivots of every super-block Schur complement; (0, -inf)
        for an exactly singular matrix (as numpy.linalg.slogdet), never NaN."""
        pivots, swaps = [], 0
        for lu, piv in self.lus:
            pivots.append(np.diag(lu))
            swaps += np.count_nonzero(piv != np.arange(piv.size))
        return _slogdet_from_pivots(np.concatenate(pivots), swaps)


class BandedHillFactorization:
    """LU with partial pivoting of the banded M(μ) (LAPACK zgbtrf): the fallback of
    :meth:`HillBlocks.factor` when block-Thomas meets an ill-conditioned Schur complement."""

    method = "banded LU, partial pivoting"

    def __init__(self, mu, lu, piv, kl, ku, size):
        self.mu, self.lu, self.piv, self.kl, self.ku, self.size = mu, lu, piv, kl, ku, size

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """M(μ) x = rhs, rhs of shape (2Nh+1, size)."""
        b = np.asarray(rhs, complex).reshape(-1, 1)
        x, info = sla.lapack.zgbtrs(self.lu, self.kl, self.ku, b, self.piv)
        if info != 0:
            raise RuntimeError(f"zgbtrs: illegal argument {-info}")   # internal error
        return x.reshape(-1, self.size)

    def slogdet(self) -> tuple[complex, float]:
        """(phase, log|det|) from the pivots of the banded LU; (0, -inf) if singular, never NaN."""
        d = self.lu[self.kl + self.ku]                    # diagonal of U in band storage
        swaps = np.count_nonzero(self.piv != np.arange(self.piv.size))
        return _slogdet_from_pivots(d, swaps)


# Normwise backward error accepted by HillBlocks.solve (a backward-stable LU gives ~1e-15).
SOLVE_BACKWARD_ERROR_MAX = 1e-10

# Reciprocal 1-norm condition below which an intermediate block-Thomas Schur complement is
# not trusted (review finding 11): the banded pivoted LU is used instead. 1e-10 leaves ~6 of
# the 16 digits of FP64 in the worst case of the unpivoted elimination step.
THOMAS_RCOND_MIN = 1e-10


def _rcond(lu_piv, D: np.ndarray) -> float:
    """Reciprocal 1-norm condition number of D from its LU (LAPACK zgecon); 0 if singular."""
    lu, _ = lu_piv
    if not np.all(np.diag(lu)):
        return 0.0
    anorm = float(np.abs(D).sum(axis=0).max())
    if anorm == 0.0:
        return 0.0
    rc, info = sla.lapack.zgecon(lu.astype(complex), anorm, norm="1")
    return float(rc) if info == 0 else 0.0


def _slogdet_from_pivots(d: np.ndarray, swaps: int) -> tuple[complex, float]:
    if not np.all(d):
        return 0j, float("-inf")
    phase = (-1.0) ** swaps * np.prod(d / np.abs(d))
    return complex(phase), float(np.sum(np.log(np.abs(d))))


# ---------------------------------------------------------------------- builder from the IR plan
def harmonic_matrices(build: Callable[[Mapping[Any, Any]], Any],
                      base: Mapping[Any, Mapping[int, Any]], *, group: Any = None,
                      linearity_tol: float = 1e-12):
    """(M, L(U := 0), {m: L_m}, cols) of a plan whose base-flow NCC fields carry harmonics.

    ``build(ncc)`` returns the plan (``discretize(problem, ncc=ncc)``); ``base[field][m]``
    is the harmonic U_m of each base field (callable on the grid, like any NCC; a missing
    harmonic is zero). L_m = L(U := U_m) - L(U := 0). Two checks, both refusals: the base flow
    must not enter the mass matrix (it would be dropped silently by the Hill form and by an
    IMEX propagator), and L must be linear in U (L(2 U_m) - L(0) = 2 (L(U_m) - L(0)) to
    ``linearity_tol`` relative). Shared by :func:`hill_from_plan` (Floquet) and
    :mod:`conatus.growth.periodic` (transient growth), so both see the same operator.
    """
    fields = list(base)
    ms = sorted({m for f in fields for m in base[f]})
    zero = {f: (lambda *X: 0.0 * X[0]) for f in fields}

    def mats(ncc):
        gm = build(ncc).matrices(group)
        return gm.L.toarray().astype(complex), gm.M.toarray().astype(complex), gm.cols

    L0z, Mm, cols = mats(zero)
    blocks = {}
    for m in ms:
        pick = {f: base[f].get(m, zero[f]) for f in fields}
        Lm, Mpick, _ = mats(pick)
        # The Hill form assumes a constant mass matrix: a base flow inside the σ term would be
        # dropped silently (AUDIT_J1 B1), so it is refused.
        if np.abs(Mpick - Mm).max() > linearity_tol * max(np.abs(Mm).max(), 1e-300):
            raise EquationError(f"the base flow enters the mass (σ) term (harmonic m={m}); the Hill "
                                "form here needs a mass matrix independent of the base flow",
                                fix="keep the base flow out of the eigenvalue (mass) term")
        twice = {f: (lambda *X, g=pick[f]: 2.0 * g(*X)) for f in fields}
        L2m, _, _ = mats(twice)
        d1, d2 = Lm - L0z, L2m - L0z
        scale = max(np.abs(d1).max(), 1e-300)
        if np.abs(d2 - 2 * d1).max() > linearity_tol * scale:
            raise EquationError(f"the problem is not linear in the base-flow NCC (harmonic m={m}); "
                                "Hill harmonics need L affine in the base flow",
                                fix="make the problem linear in the base-flow NCC")
        blocks[m] = d1
    return Mm, L0z, blocks, cols


def hill_from_plan(build: Callable[[Mapping[Any, Any]], Any], base: Mapping[Any, Mapping[int, Any]],
                   *, Nh: int, omega: float, label: str, group: Any = None,
                   linearity_tol: float = 1e-12) -> HillBlocks:
    """Hill operator of an IR eigenvalue problem whose base flow is T-periodic.

    ``build(ncc)`` must return the plan (``discretize(problem, ncc=ncc)``) of an EVP in the
    convention (L + σ M) x = 0, σ being the growth rate; the base-flow fields are NCC.
    ``base[field][m]`` gives the harmonic U_m of each base field (callable on the grid, like
    any NCC); a missing harmonic is zero. Harmonic m of L is L(U := U_m) - L(U := 0); linearity
    in U is checked (L(2U_m) - L(0) = 2 (L(U_m) - L(0))) and a non-linear dependence is refused
    (:func:`harmonic_matrices`). The diagonal block of harmonic n is L_0 + (μ + i n ω) M.
    An IVP plan works too: M X' + L X = 0 is the same pencil.
    """
    Mm, L0z, blocks, _ = harmonic_matrices(build, base, group=group, linearity_tol=linearity_tol)
    fields = list(base)
    return hill_from_blocks(Mm, L0z, blocks, Nh=Nh, omega=omega, label=label,
                            note=("Blocks from the discretized plan (EVP convention (L + σM)x = 0); "
                                  f"base-flow harmonics of {', '.join(str(f) for f in fields)} "
                                  "entered as NCC, linearity checked."))


def hill_from_blocks(M: np.ndarray, L0: np.ndarray, blocks: Mapping[int, np.ndarray], *, Nh: int,
                     omega: float, label: str, note: str) -> HillBlocks:
    """Hill operator of M x' + (L0 + Σ_m e^{i m ω t} L_m) x = 0 from its matrices (L_0 joins the
    diagonal block, the other harmonics are the couplings)."""
    blocks = dict(blocks)
    L0 = L0 + blocks.pop(0, 0.0)
    size = L0.shape[0]

    def diag(mu: complex, n: int) -> np.ndarray:
        return L0 + (mu + 1j * n * omega) * M

    return HillBlocks(diag=diag, couplings=blocks, Nh=Nh, size=size, label=label, omega=omega,
                      notes=(note,), mass=M)


# ---------------------------------------------------------------------- roots and exponents
@renames("conatus.floquet.hill_general:secant_root")
def secant_root(hill: HillBlocks, start: complex, *, tol: float, max_iterations: int = SECANT_MAX_ITERATIONS,
                step: complex = SECANT_STEP, allow_unconverged: bool = False) -> SecantRoot:
    """Secant on det M(μ)/|det M(start)| (log-scaled, no overflow) from an explicit start.

    Returns a :class:`~conatus.floquet.roots.SecantRoot` (``mu``, ``converged``, ``residual`` =
    the scaled determinant at ``mu``, ``iterations``, ``reasons``). Converged means the last
    secant step is below ``tol``; otherwise ConvergenceError is raised, carrying the last iterate
    (``err.result``), unless ``allow_unconverged=True`` (the v0.1
    tuple ``(μ, reasons)`` let ``secant_root(...)[0]`` be used as a root; unpacking it keeps
    working with a DeprecationWarning). The tolerance must stay above the round-off floor of
    det: it is the caller's choice. ``max_iterations``: declared default
    SECANT_MAX_ITERATIONS (a budget); ``maxit`` and ``seed`` are the deprecated former
    names of ``max_iterations`` and ``start``.
    """
    maxit = check_count(max_iterations, "max_iterations", "secant_root")
    allow = check_allow_unconverged(allow_unconverged, "secant_root")
    _, ref = hill.slogdet(start)

    def f(m):
        ph, la = hill.slogdet(m)
        return ph * np.exp(la - ref)

    run = determinant_secant(f, start, step, tol=tol, max_iterations=maxit)
    root = SecantRoot(mu=run.mu, converged=run.converged, residual=float(abs(run.f)),
                      iterations=run.iterations, last_step=run.last_step, tol=float(tol),
                      reasons=[run.reason] if run.reason else [])
    if not run.converged and not allow:
        raise unconverged(root, "secant_root", max_iterations=maxit, tol=tol)
    return root


def exponents_dense(hill: HillBlocks, *, finite_cap: float) -> np.ndarray:
    """All Floquet exponents of a μ-linear Hill operator (M(μ) = A + μ B), small sizes only.

    Generalized EVP of the assembled matrix; eigenvalues with |μ| > ``finite_cap`` (the infinite
    ones of the tau/BC rows, whose mass rows vanish) are dropped and their number returned in
    the attribute ``n_dropped`` of the result (never discarded silently).
    """
    if hill.mass is None:
        raise OptionError("exponents_dense needs a μ-linear operator built with its mass block",
                          fix="build the operator with its mass block (HillBlocks(mass=...))")
    A = hill.dense(0.0)
    K = len(hill.harmonics)
    B = np.kron(np.eye(K), hill.mass)
    w = sla.eigvals(A, -B)
    keep = np.isfinite(w) & (np.abs(w) <= finite_cap)
    out = np.asarray(w[keep])
    out = out.view(_Exponents)
    out.n_dropped = int(np.count_nonzero(~keep))
    return out


class _Exponents(np.ndarray):
    """ndarray of exponents carrying the count of dropped infinite eigenvalues."""

    n_dropped: int = 0
