"""Operators and transforms of the non-periodic bases beyond a single Chebyshev T interval.

. One *family* object per
basis gives everything the Cartesian pipeline needs along its non-periodic direction:

* sparse operators between derivative *levels* ``lam`` (the analogue of the ultraspherical index):
  conversion, differentiation, multiplication by a coefficient, evaluation and integration rows,
  the tau column of a lift and the constant function;
* grid transforms (NumPy) and the dense transform matrices used on the GPU;
* coefficient-space operators at the variables' level for the explicit right-hand side.

Families:

``HermiteFamily``   psi_n(y) = H_n(y) e^{-y^2/2} / sqrt(2^n n! sqrt(pi)), y = (x - x0)/s. One level:
                    d/dy and y are tridiagonal (psi_n' = sqrt(n/2) psi_{n-1} - sqrt((n+1)/2)
                    psi_{n+1}; y psi_n = sqrt((n+1)/2) psi_{n+1} + sqrt(n/2) psi_{n-1}), conversion
                    is the identity. The N x N truncation of a product of operators drops the
                    psi_N intermediate: it only changes the last row/column (a Galerkin truncation,
                    like the tau rows of a bounded basis).
``LaguerreFamily``  phi^k_n(y) = L^(k)_n(y) e^{-y/2}, y = (x - a)/s, level k = derivative level:
                    d/dy phi^k_n = -(phi^{k+1}_n + phi^{k+1}_{n-1})/2 and phi^k_n = phi^{k+1}_n -
                    phi^{k+1}_{n-1} (both bidiagonal and exact on span{n < N}), y phi^k_n =
                    -(n+1) phi^k_{n+1} + (2n+k+1) phi^k_n - (n+k) phi^k_{n-1}. Variables live at
                    level 0 (orthonormal); an equation of order q is written at level q and its taus
                    are lifted into its last q modes (Olver–Townsend's construction, with Laguerre
                    instead of Gegenbauer polynomials).
``CompoundFamily``  Chebyshev T per piece, block-diagonal ultraspherical operators (Olver &
                    Townsend 2013), scaled by 2/length of each piece; one-sided traces at the
                    interfaces; lifts into the last modes of a chosen piece.

Non-constant coefficients: on Hermite/Laguerre a coefficient must be a polynomial in x (its
multiplication operator is then banded and exact); its degree is read off its orthogonal-polynomial
coefficients on the Gauss grid, anything else is refused (DiscretizationError, with what to do instead).
On a compound interval each piece expands the coefficient in its own Chebyshev series (reported
per piece like a Chebyshev NCC).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.special import roots_hermite, roots_laguerre

from ..ir.bases import Basis, CompoundChebyshev, Hermite, Laguerre, Legendre, Ultraspherical
from . import ultraspherical as us
from ..errors import OptionError, ResolutionError

__all__ = ["family_of", "has_family", "HermiteFamily", "LaguerreFamily", "CompoundFamily", "NEW_FAMILIES"]

# Bases with a family object. Plain Jacobi(a, b) is handled too (
# discretize/jacobi_family.py) but cannot be listed here: Chebyshev T is a Jacobi subclass with its
# own code path, so isinstance(b, Jacobi) would capture it; use has_family(b) for a membership test.
NEW_FAMILIES = (Hermite, Laguerre, CompoundChebyshev, Legendre, Ultraspherical)


def has_family(basis: Basis | None) -> bool:
    """True when ``family_of(basis)`` is not None (without building the family)."""
    from .jacobi_family import is_jacobi_trial
    return isinstance(basis, (Hermite, Laguerre, CompoundChebyshev)) or is_jacobi_trial(basis)

# A non-constant coefficient on an unbounded axis is a polynomial of degree d when its orthonormal-
# polynomial coefficients above d are below this relative size (declared, fixed; see _poly_check).
POLY_NCC_RTOL = 1e-10
# exp(-y/2) and exp(-y^2/2) underflow (to subnormals, then 0) beyond these grid extents: refuse
# a grid that would silently lose its largest nodes.
_LAGUERRE_YMAX = 1400.0
_HERMITE_YMAX = 37.0


def family_of(basis: Basis | None):
    """The family object of a basis handled by this module (or by jacobi_family), or None
    (Fourier, Chebyshev T, curvilinear bases keep their own code paths)."""
    if isinstance(basis, Hermite):
        return HermiteFamily(basis)
    if isinstance(basis, Laguerre):
        return LaguerreFamily(basis)
    if isinstance(basis, CompoundChebyshev):
        return CompoundFamily(basis)
    from .jacobi_family import JacobiFamily, is_jacobi_trial
    if is_jacobi_trial(basis):                       # Legendre, Jacobi, Ultraspherical (not T)
        return JacobiFamily(basis)
    return None


def _moved(f: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(np.asarray(f), axis, 0)


def _apply_rows(A: np.ndarray, f: np.ndarray, axis: int) -> np.ndarray:
    """A (P x Q) applied along ``axis`` of f (length Q there)."""
    fm = _moved(f, axis)
    out = np.tensordot(A, fm, axes=(1, 0))
    return np.moveaxis(out, 0, axis)


# ------------------------------------------------------------------------------------------------
# function values (stable three-term recurrences, envelope included)
# ------------------------------------------------------------------------------------------------

# Products on Hermite / Laguerre axes (tests/transforms/test_dealias_exactness.py):
# the envelope makes a product leave the span, and the 3/2 (or 2) padding of the polynomial bases does
# not remove aliasing; measured at N = 16 with unit random coefficients: 1.6e-1 / 3.0e-2 (Hermite)
# and 3.7e-1 / 1.2e-1 (Laguerre) at 3/2 / 2, 1e-7 / 1e-10 at 3; far smaller for resolved fields.
_UNBOUNDED_ALIAS = ("the 3/2 rule of the polynomial bases does not remove aliasing here; the error "
                    "falls with the padding and with the decay of the spectra, check it with a "
                    "larger dealias")


def hermite_functions(N: int, y: np.ndarray) -> np.ndarray:
    """psi_n(y) for n < N, shape (N, len(y)); orthonormal on the line."""
    y = np.asarray(y, dtype=float)
    out = np.zeros((max(N, 1), y.size))
    out[0] = np.pi ** -0.25 * np.exp(-y ** 2 / 2)
    if N > 1:
        out[1] = math.sqrt(2.0) * y * out[0]
    for n in range(1, N - 1):
        out[n + 1] = math.sqrt(2.0 / (n + 1)) * y * out[n] - math.sqrt(n / (n + 1)) * out[n - 1]
    return out[:N]


def laguerre_functions(N: int, y: np.ndarray, k: int = 0) -> np.ndarray:
    """L^(k)_n(y) e^{-y/2} for n < N, shape (N, len(y)); orthonormal on [0, inf) for k = 0."""
    y = np.asarray(y, dtype=float)
    out = np.zeros((max(N, 1), y.size))
    out[0] = np.exp(-y / 2)
    if N > 1:
        out[1] = (1 + k - y) * out[0]
    for n in range(1, N - 1):
        out[n + 1] = ((2 * n + 1 + k - y) * out[n] - (n + k) * out[n - 1]) / (n + 1)
    return out[:N]


def _frozen(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Arrays returned by an ``lru_cache`` are shared by every later caller: read-only, so an
    in-place update by one caller cannot change the transforms of all the others (audit
    cache family of)."""
    for a in arrays:
        a.setflags(write=False)
    return arrays


@lru_cache(maxsize=64)
def _hermite_nodes(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Hermite nodes y_j and *function* weights W_j = w_j e^{y_j^2} = 1/sum_m psi_m(y_j)^2
    (Christoffel form: no overflow of e^{y^2}, no underflow of w_j at large M)."""
    y, _ = roots_hermite(M)
    if np.max(np.abs(y)) > _HERMITE_YMAX:
        raise ResolutionError(f"Hermite grid of {M} points reaches |y| = {np.max(np.abs(y)):.1f}: "
                              "exp(-y^2/2) underflows there; use fewer modes or a larger scale",
                              fix="fewer Hermite modes, or a larger scale of the basis")
    W = 1.0 / np.sum(hermite_functions(M, y) ** 2, axis=0)
    return _frozen(y, W)


@lru_cache(maxsize=64)
def _laguerre_nodes(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Laguerre nodes y_j and function weights W_j = w_j e^{y_j} = 1/sum_m phi_m(y_j)^2."""
    y, _ = roots_laguerre(M)
    if np.max(y) > _LAGUERRE_YMAX:
        raise ResolutionError(f"Laguerre grid of {M} points reaches y = {np.max(y):.0f}: exp(-y/2) "
                              "underflows there; use fewer modes or a larger scale",
                              fix="fewer Laguerre modes, or a larger scale of the basis")
    W = 1.0 / np.sum(laguerre_functions(M, y) ** 2, axis=0)
    return _frozen(y, W)


# ------------------------------------------------------------------------------------------------
# common protocol
# ------------------------------------------------------------------------------------------------

class _Family:
    """Common part: grid, matrix transforms, explain()."""

    kind = "abstract"
    has_constant = False         # is the constant function representable?

    def __init__(self, basis: Basis) -> None:
        self.basis = basis
        self.N = basis.size

    # ---- grids and transforms (NumPy) --------------------------------------------------------
    def grid_size(self, dealias: bool) -> int:
        return self.basis.grid_size if dealias else self.basis.size

    def grid(self, dealias: bool = True) -> np.ndarray:
        raise NotImplementedError

    def matrices(self, dealias: bool) -> tuple[np.ndarray, np.ndarray]:
        """(B, F): grid values = B @ coefficients (M x N), coefficients = F @ grid values (N x M)."""
        raise NotImplementedError

    def forward(self, f: np.ndarray, axis: int = 0, dealias: bool = True) -> np.ndarray:
        M = self.grid_size(dealias)
        if np.shape(f)[axis] != M:
            raise ValueError(f"expected {M} grid points along axis {axis}, got {np.shape(f)[axis]}")
        return _apply_rows(self.matrices(dealias)[1], f, axis)

    def backward(self, c: np.ndarray, axis: int = 0, dealias: bool = True) -> np.ndarray:
        if np.shape(c)[axis] != self.N:
            raise ValueError(f"expected {self.N} coefficients along axis {axis}, got {np.shape(c)[axis]}")
        return _apply_rows(self.matrices(dealias)[0], c, axis)

    # ---- sparse operators (compiler) ----------------------------------------------------------
    def constant_column(self) -> sp.csr_matrix:
        raise _no_constant(self.basis)

    def lift_index(self, n: int, piece: int | None) -> int:
        return self.N + n

    # ---- coefficient-space operators at the variables' level (explicit RHS) -------------------
    def diff_coeffs(self, c: np.ndarray) -> np.ndarray:
        """d/dx along axis 0, variables' level -> variables' level."""
        raise NotImplementedError

    def integ_weights(self) -> np.ndarray:
        return self.integration_row(0)

    def interp_row(self, pos: float, side: str | None = None) -> np.ndarray:
        return self.evaluation_row(0, pos, 0, side)

    # ---- data of a non-constant coefficient (plan._ncc_coeffs) -----------------------------
    def ncc_forward(self, g: np.ndarray, axis: int) -> np.ndarray:
        """Coefficients of a coefficient sampled on the variables' grid, in the form
        ``multiplication`` takes (polynomial coefficients on Hermite/Laguerre, Chebyshev per piece
        on a compound interval)."""
        raise NotImplementedError


def _no_constant(b: Basis):
    from ..errors import DiscretizationError
    return DiscretizationError(
        f"a constant along {b.coord.name} is not representable in {b!r}: its functions decay at "
        "infinity (a constant term, a constant tau or a gauge cannot be broadcast along an "
        "unbounded axis)",
        fix="write the problem for a decaying quantity (e.g. the deviation from a far-field "
            "state), or use a bounded or mapped Chebyshev axis")


def _poly_check(name: str, a: np.ndarray, axis_name: str) -> int:
    """Degree d of a coefficient given by its orthonormal-polynomial coefficients ``a`` on the
    N-point Gauss grid: the last coefficient above POLY_NCC_RTOL relative. Accepted only when at
    least half of the sampled coefficients are negligible (d < N/2: the sample resolves a
    polynomial); otherwise the coefficient is not a polynomial and is refused."""
    a = np.asarray(a)
    amax = float(np.max(np.abs(a))) if a.size else 0.0
    if amax == 0:
        return 0
    big = np.nonzero(np.abs(a) > POLY_NCC_RTOL * amax)[0]
    d = int(big[-1]) if big.size else 0
    if 2 * d >= a.size:
        from ..errors import DiscretizationError
        tail = float(np.max(np.abs(a[a.size // 2:]))) / amax
        raise DiscretizationError(
            f"non-constant coefficient '{name}' along the unbounded axis {axis_name} is not a "
            f"polynomial (its orthogonal-polynomial coefficients do not vanish: relative tail "
            f"{tail:.1e} over the upper half of the {a.size} sampled coefficients)",
            why="on Hermite/Laguerre axes only a polynomial coefficient has a banded, exact "
                "multiplication operator; a bounded non-polynomial coefficient would fill the "
                "matrix or be truncated silently",
            fix="move the term to the right-hand side (explicit(...) in the native layer), or use "
                "a mapped Chebyshev axis for a bounded coefficient (e.g. tanh)")
    return d


class _PolyNCCReport:
    """NCC report of a polynomial coefficient on an unbounded axis (explain())."""

    def __init__(self, name: str, degree: int, n: int, kind: str) -> None:
        self.name, self.degree, self.n, self.kind = name, degree, n, kind
        self.resolved = True

    def explain(self) -> str:
        return (f"NCC {self.name}: polynomial of degree {self.degree} in x on the {self.kind} axis "
                f"(detected from its {self.n} grid coefficients: those above degree "
                f"{self.degree} are below {POLY_NCC_RTOL:g} relative and are not used); banded "
                f"multiplication of half-bandwidth {self.degree}, exact")


# ------------------------------------------------------------------------------------------------
# Hermite
# ------------------------------------------------------------------------------------------------

class HermiteFamily(_Family):
    kind = "Hermite (whole line)"

    def __init__(self, basis: Hermite) -> None:
        super().__init__(basis)
        self.x0, self.s = basis.center, basis.scale

    def grid(self, dealias: bool = True) -> np.ndarray:
        y, _ = _hermite_nodes(self.grid_size(dealias))
        return self.x0 + self.s * y

    def matrices(self, dealias: bool) -> tuple[np.ndarray, np.ndarray]:
        M = self.grid_size(dealias)
        return _hermite_mats(self.N, M)

    @staticmethod
    def _D(N: int) -> sp.csr_matrix:
        """d/dy on psi coefficients, N x N (column n: psi_n')."""
        n = np.arange(N, dtype=float)
        up = np.sqrt(n[1:] / 2)                        # psi_n -> psi_{n-1}: row n-1, col n
        lo = -np.sqrt((n[:-1] + 1) / 2)                # psi_n -> psi_{n+1}: row n+1, col n
        return sp.diags([lo, up], [-1, 1], shape=(N, N), format="csr")

    @staticmethod
    def _J(N: int) -> sp.csr_matrix:
        """Multiplication by y on psi coefficients."""
        n = np.arange(N, dtype=float)
        return sp.diags([np.sqrt((n[:-1] + 1) / 2), np.sqrt(n[1:] / 2)], [-1, 1], shape=(N, N),
                        format="csr")

    def conversion_chain(self, lam_from: int, lam_to: int) -> sp.csr_matrix:
        return sp.identity(self.N, format="csr")

    def differentiation(self, lam: int, order: int = 1) -> sp.csr_matrix:
        D = self._D(self.N) / self.s
        out = sp.identity(self.N, format="csr")
        for _ in range(order):
            out = (D @ out).tocsr()
        return out

    def evaluation_row(self, lam: int, pos: float, deriv: int = 0, side: str | None = None) -> np.ndarray:
        y = (pos - self.x0) / self.s
        Ne = self.N + deriv                       # exact: the derivative reaches psi_{N-1+deriv}
        row = hermite_functions(Ne, np.array([y]))[:, 0]
        D = self._D(Ne) / self.s
        for _ in range(deriv):
            row = D.T @ row
        return np.asarray(row)[:self.N]

    def integration_row(self, lam: int) -> np.ndarray:
        I = np.zeros(self.N)
        I[0] = math.sqrt(2.0) * np.pi ** 0.25
        for n in range(0, self.N - 2, 2):
            I[n + 2] = math.sqrt((n + 1) / (n + 2)) * I[n]
        return I * self.s

    def diff_coeffs(self, c: np.ndarray) -> np.ndarray:
        D = self.differentiation(0, 1)
        sh = c.shape
        return np.asarray(D @ c.reshape(sh[0], -1)).reshape(sh)

    def ncc_forward(self, g: np.ndarray, axis: int) -> np.ndarray:
        """Orthonormal Hermite-polynomial coefficients a_k of g on the N-point grid (Gauss)."""
        y, W = _hermite_nodes(self.N)
        A = (hermite_functions(self.N, y) * (W * np.exp(-y ** 2 / 2))[None, :])  # w_j h_k(y_j)
        return _apply_rows(A, g, axis)

    def multiplication(self, a: np.ndarray, lam: int, *, name: str = "a", cutoff=None,
                       tail_tol=None, exact: bool = False,
                       floor: float = 0.0) -> tuple[sp.csr_matrix, Any]:
        a = np.asarray(a)
        d = _poly_check(name, a, self.basis.coord.name)
        Ne = self.N + d + 1
        J = self._J(Ne)
        h0 = np.pi ** -0.25
        I = sp.identity(Ne, format="csr")
        Pm1, P = None, h0 * I
        out = a[0] * P
        for k in range(d):
            Pn = (math.sqrt(2.0 / (k + 1)) * (J @ P)
                  - (math.sqrt(k / (k + 1)) * Pm1 if Pm1 is not None else 0 * I))
            Pm1, P = P, Pn.tocsr()
            out = out + a[k + 1] * P
        out = sp.csr_matrix(out)[:self.N, :self.N]
        out.eliminate_zeros()
        return out.tocsr(), _PolyNCCReport(name, d, a.size, "Hermite")

    def explain(self) -> str:
        return (f"{self.basis.coord.name}: Hermite functions psi_n((x - {self.x0:g})/{self.s:g}), "
                f"N={self.N} (whole line, no boundary condition); d/dx and x tridiagonal, "
                f"Gauss-Hermite grid of {self.basis.grid_size} points (products: quadrature "
                f"projection, not exact: {_UNBOUNDED_ALIAS})")


@lru_cache(maxsize=64)
def _hermite_mats(N: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    y, W = _hermite_nodes(M)
    P = hermite_functions(N, y)                     # (N, M)
    return _frozen(P.T.copy(), P * W[None, :])


# ------------------------------------------------------------------------------------------------
# Laguerre
# ------------------------------------------------------------------------------------------------

class LaguerreFamily(_Family):
    kind = "Laguerre (half line)"

    def __init__(self, basis: Laguerre) -> None:
        super().__init__(basis)
        self.a, self.s = basis.start, basis.scale

    def grid(self, dealias: bool = True) -> np.ndarray:
        y, _ = _laguerre_nodes(self.grid_size(dealias))
        return self.a + self.s * y

    def matrices(self, dealias: bool) -> tuple[np.ndarray, np.ndarray]:
        return _laguerre_mats(self.N, self.grid_size(dealias))

    @staticmethod
    def _S(N: int) -> sp.csr_matrix:
        """Level k -> k+1: phi^k_n = phi^{k+1}_n - phi^{k+1}_{n-1}."""
        return sp.diags([np.ones(N), -np.ones(N - 1)], [0, 1], shape=(N, N), format="csr")

    @staticmethod
    def _Dy(N: int) -> sp.csr_matrix:
        """d/dy, level k -> k+1: -(phi^{k+1}_n + phi^{k+1}_{n-1}) / 2."""
        return sp.diags([-0.5 * np.ones(N), -0.5 * np.ones(N - 1)], [0, 1], shape=(N, N), format="csr")

    @staticmethod
    def _J(N: int, k: int) -> sp.csr_matrix:
        """Multiplication by y at level k."""
        n = np.arange(N, dtype=float)
        return sp.diags([-(n[:-1] + 1), 2 * n + k + 1, -(n[1:] + k)], [-1, 0, 1], shape=(N, N),
                        format="csr")

    def conversion_chain(self, lam_from: int, lam_to: int) -> sp.csr_matrix:
        out = sp.identity(self.N, format="csr")
        for _ in range(lam_from, lam_to):
            out = (self._S(self.N) @ out).tocsr()
        return out

    def differentiation(self, lam: int, order: int = 1) -> sp.csr_matrix:
        out = sp.identity(self.N, format="csr")
        for _ in range(order):
            out = (self._Dy(self.N) @ out).tocsr() / self.s
        return out.tocsr()

    def evaluation_row(self, lam: int, pos: float, deriv: int = 0, side: str | None = None) -> np.ndarray:
        y = (pos - self.a) / self.s
        row = laguerre_functions(self.N, np.array([y]), lam + deriv)[:, 0]
        if deriv:
            row = row @ self.differentiation(lam, deriv)
        return np.asarray(row).ravel()

    def integration_row(self, lam: int) -> np.ndarray:
        n = np.arange(self.N)
        I = 2.0 * (-1.0) ** n                       # int_0^inf L_n(y) e^{-y/2} dy
        for _ in range(lam):
            I = np.cumsum(I)                        # L^(k)_n = sum_{j<=n} L^(k-1)_j
        return I * self.s

    def diff_coeffs(self, c: np.ndarray) -> np.ndarray:
        """d/dy at level 0 -> level 0 exactly: phi_n' = -phi_n/2 - sum_{k<n} phi_k."""
        rev = np.cumsum(c[::-1], axis=0)[::-1]       # sum_{n>=k} c_n
        out = -0.5 * c - (rev - c)
        return out / self.s

    def ncc_forward(self, g: np.ndarray, axis: int) -> np.ndarray:
        y, W = _laguerre_nodes(self.N)
        A = laguerre_functions(self.N, y) * (W * np.exp(-y / 2))[None, :]   # w_j L_k(y_j)
        return _apply_rows(A, g, axis)

    def multiplication(self, a: np.ndarray, lam: int, *, name: str = "a", cutoff=None,
                       tail_tol=None, exact: bool = False,
                       floor: float = 0.0) -> tuple[sp.csr_matrix, Any]:
        a = np.asarray(a)
        d = _poly_check(name, a, self.basis.coord.name)
        Ne = self.N + d + 1
        J = self._J(Ne, lam)
        I = sp.identity(Ne, format="csr")
        Pm1, P = None, I
        out = a[0] * P
        for k in range(d):                           # L_{k+1} = ((2k+1-y) L_k - k L_{k-1})/(k+1)
            Pn = ((2 * k + 1) * P - J @ P - (k * Pm1 if Pm1 is not None else 0 * I)) / (k + 1)
            Pm1, P = P, sp.csr_matrix(Pn)
            out = out + a[k + 1] * P
        out = sp.csr_matrix(out)[:self.N, :self.N]
        out.eliminate_zeros()
        return out.tocsr(), _PolyNCCReport(name, d, a.size, "Laguerre")

    def explain(self) -> str:
        return (f"{self.basis.coord.name}: Laguerre functions L_n((x - {self.a:g})/{self.s:g}) "
                f"exp(-(x - {self.a:g})/(2*{self.s:g})), N={self.N} (half line, wall at x = {self.a:g}); "
                "equations of order q written in the derivative bases L^(q)_n e^{-y/2} (bidiagonal "
                f"d/dy and conversions), taus in their last q modes; Gauss-Laguerre grid of "
                f"{self.basis.grid_size} points (products: quadrature projection, not exact: "
                f"{_UNBOUNDED_ALIAS})")


@lru_cache(maxsize=64)
def _laguerre_mats(N: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    y, W = _laguerre_nodes(M)
    P = laguerre_functions(N, y)
    return _frozen(P.T.copy(), P * W[None, :])


# ------------------------------------------------------------------------------------------------
# Compound Chebyshev
# ------------------------------------------------------------------------------------------------

class CompoundFamily(_Family):
    kind = "compound Chebyshev (piecewise)"
    has_constant = True

    def __init__(self, basis: CompoundChebyshev) -> None:
        super().__init__(basis)
        self.sizes, self.offsets, self.pieces = basis.sizes, basis.offsets, basis.pieces

    def _blocks(self, fn) -> sp.csr_matrix:
        return sp.block_diag([fn(j, n) for j, n in enumerate(self.sizes)], format="csr")

    def _scale(self, j: int) -> float:
        a, b = self.pieces[j]
        return 2.0 / (b - a)

    def _grid_sizes(self, dealias: bool) -> list[int]:
        return self.basis.grid_sizes() if dealias else list(self.sizes)

    def grid(self, dealias: bool = True) -> np.ndarray:
        out = []
        for (a, b), M in zip(self.pieces, self._grid_sizes(dealias), strict=True):
            t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
            out.append(a + (t + 1) * (b - a) / 2)
        return np.concatenate(out)

    def matrices(self, dealias: bool) -> tuple[np.ndarray, np.ndarray]:
        return _compound_mats(self.sizes, tuple(self._grid_sizes(dealias)))

    def forward(self, f: np.ndarray, axis: int = 0, dealias: bool = True) -> np.ndarray:
        import scipy.fft as sfft
        Ms = self._grid_sizes(dealias)
        fm = _moved(f, axis)
        if fm.shape[0] != sum(Ms):
            raise ValueError(f"expected {sum(Ms)} grid points along axis {axis}, got {fm.shape[0]}")
        parts, o = [], 0
        for M, N in zip(Ms, self.sizes, strict=True):
            g = fm[o:o + M]
            if np.iscomplexobj(g):
                y = (sfft.dct(g.real, type=2, axis=0) + 1j * sfft.dct(g.imag, type=2, axis=0)) / M
            else:
                y = sfft.dct(g, type=2, axis=0) / M
            y[0] /= 2
            sign = (-1.0) ** np.arange(M)
            parts.append((y * sign.reshape((M,) + (1,) * (g.ndim - 1)))[:N])
            o += M
        return np.moveaxis(np.concatenate(parts, axis=0), 0, axis)

    def backward(self, c: np.ndarray, axis: int = 0, dealias: bool = True) -> np.ndarray:
        import scipy.fft as sfft
        Ms = self._grid_sizes(dealias)
        cm = _moved(c, axis)
        if cm.shape[0] != self.N:
            raise ValueError(f"expected {self.N} coefficients along axis {axis}, got {cm.shape[0]}")
        parts = []
        for M, N, o in zip(Ms, self.sizes, self.offsets, strict=True):
            y = np.zeros((M,) + cm.shape[1:], dtype=cm.dtype)
            sign = (-1.0) ** np.arange(N)
            y[:N] = cm[o:o + N] * sign.reshape((N,) + (1,) * (cm.ndim - 1))
            y[0] *= 2
            if np.iscomplexobj(y):
                parts.append(sfft.idct(y.real * M, type=2, axis=0) + 1j * sfft.idct(y.imag * M, type=2, axis=0))
            else:
                parts.append(sfft.idct(y * M, type=2, axis=0))
        return np.moveaxis(np.concatenate(parts, axis=0), 0, axis)

    def conversion_chain(self, lam_from: int, lam_to: int) -> sp.csr_matrix:
        return self._blocks(lambda j, n: us.conversion_chain(n, lam_from, lam_to))

    def differentiation(self, lam: int, order: int = 1) -> sp.csr_matrix:
        return self._blocks(lambda j, n: us.differentiation(n, lam, order) * self._scale(j) ** order)

    def evaluation_row(self, lam: int, pos: float, deriv: int = 0, side: str | None = None) -> np.ndarray:
        j = self.basis.piece_of(pos, side)
        a, b = self.pieces[j]
        t = 2 * (pos - a) / (b - a) - 1
        row = np.zeros(self.N)
        o, n = self.offsets[j], self.sizes[j]
        row[o:o + n] = us.evaluation_row(n, lam, t, deriv) * self._scale(j) ** deriv
        return row

    def integration_row(self, lam: int) -> np.ndarray:
        return np.concatenate([us.integration_row(n, lam) / self._scale(j)
                               for j, n in enumerate(self.sizes)])

    def constant_column(self) -> sp.csr_matrix:
        col = np.zeros((self.N, 1))
        col[self.offsets, 0] = 1.0                   # T_0 of every piece
        return sp.csr_matrix(col)

    def lift_index(self, n: int, piece: int | None) -> int:
        if piece is None:
            raise OptionError(f"lift on {self.basis!r} needs piece=",
                              fix="lift(..., piece=k) on a piecewise basis (k = index of the piece)")
        return self.offsets[piece] + self.sizes[piece] + n

    def diff_coeffs(self, c: np.ndarray) -> np.ndarray:
        out = np.zeros_like(c)
        for j, (N, o) in enumerate(zip(self.sizes, self.offsets, strict=True)):
            cj, dj = c[o:o + N], np.zeros_like(c[o:o + N])
            if N > 1:
                dj[N - 2] = 2 * (N - 1) * cj[N - 1]
                for n in range(N - 3, -1, -1):
                    dj[n] = dj[n + 2] + 2 * (n + 1) * cj[n + 1]
                dj[0] = dj[0] / 2
            out[o:o + N] = dj * self._scale(j)
        return out

    def ncc_forward(self, g: np.ndarray, axis: int) -> np.ndarray:
        return self.forward(g, axis, dealias=False)

    def multiplication(self, a: np.ndarray, lam: int, *, name: str = "a", cutoff=None,
                       tail_tol=None, exact: bool = False,
                       floor: float = 0.0) -> tuple[sp.csr_matrix, Any]:
        mats, reps = [], []
        for j, (N, o) in enumerate(zip(self.sizes, self.offsets, strict=True)):
            M, rep = us.multiplication(np.asarray(a)[o:o + N], N, lam, name=f"{name} (piece {j})",
                                       cutoff=cutoff, tail_tol=tail_tol, exact=exact,
                                       floor=floor)
            mats.append(M)
            reps.append(rep)
        return sp.block_diag(mats, format="csr"), _CompoundNCCReport(reps)

    def explain(self) -> str:
        parts = ", ".join(f"[{a:g}, {b:g}] N={n}" for (a, b), n in zip(self.pieces, self.sizes, strict=True))
        return (f"{self.basis.coord.name}: compound Chebyshev T, pieces {parts} (total N={self.N}); "
                "block-diagonal Olver–Townsend operators, continuity at the interfaces by tau "
                "conditions; Gauss-Chebyshev grid per piece")


class _CompoundNCCReport:
    def __init__(self, reps: list) -> None:
        self.reps = reps
        self.resolved = all(getattr(r, "resolved", True) for r in reps)

    def explain(self) -> str:
        return "; ".join(r.explain() for r in self.reps)


@lru_cache(maxsize=64)
def _compound_mats(sizes: tuple, Ms: tuple) -> tuple[np.ndarray, np.ndarray]:
    Bs, Fs = [], []
    for N, M in zip(sizes, Ms, strict=True):
        t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
        B = np.cos(np.arange(N)[None, :] * np.arccos(t)[:, None])           # (M, N)
        F = (2.0 / M) * B.T                                                   # discrete orthogonality
        F[0] /= 2
        Bs.append(B)
        Fs.append(F)
    from scipy.linalg import block_diag
    return _frozen(block_diag(*Bs), block_diag(*Fs))
