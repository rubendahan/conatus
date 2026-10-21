"""The operators ``T(z)`` the contour methods accept, behind one small interface.

A contour method needs, at a point ``z`` of the complex plane: the determinant of ``T(z)`` as a
unit phase and a log modulus (the argument principle), solves with ``T(z)`` (Beyn's moments and
the inverse-iteration polish), the products ``T(z) X`` and ``T'(z) X``, and a scale of ``T(z)``
for the relative backward residual. Three kinds of ``T`` are wrapped:

* :class:`Pencil` -- ``T(z) = L + z M`` (the eigenvalue convention of :mod:`conatus.solvers.evp`),
  dense or sparse; ``det T`` is a polynomial in ``z``: no pole (``P = 0``);
* :class:`HillBlocksOperator` -- a :class:`conatus.floquet.HillBlocks` (block-banded Hill matrix,
  factored by block-Thomas, never assembled); ``P = 0`` when it is linear in ``z`` (a mass block);
* :class:`MatrixFunction` -- any callable ``T(z)`` returning a matrix (dense or sparse) or a
  scalar (an Evans function, a shooting determinant): its poles in the region are the caller's
  declaration (``poles=``), never assumed.

The determinant is read off an LU factorisation with partial pivoting: ``det T = sign(row
permutation) sign(column permutation) prod(diag U)``. The permutation signs are part of the
value, so the determinant does not depend on the pivot order the factorisation chooses at each
``z`` (a pivot-labelled product without them flips sign wherever the pivot order changes). The
phase is the sum of the arguments of the pivots and the modulus the sum of their logarithms:
``det T`` itself is never formed, so it cannot overflow or underflow whatever the size.
"""
from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..errors import InputTypeError, OptionError

__all__ = ["Operator", "Pencil", "HillBlocksOperator", "MatrixFunction", "as_operator",
           "permutation_sign", "SingularPoint"]

_DOC = "docs/site/guide/dynamics/contour.md"
#: Relative step of the central difference for T'(z) when a callable T comes without its
#: derivative: only the polish uses T' (its convergence rate; the acceptance test is the residual
#: of T itself); about eps^(1/3) for a central difference.
DERIVATIVE_STEP = 6e-6


class SingularPoint(ArithmeticError):
    """T(z) is exactly singular at ``z`` (a zero pivot): ``z`` is an eigenvalue to working precision."""

    def __init__(self, z: complex) -> None:
        super().__init__(f"T(z) is exactly singular at z = {z!r}")
        self.z = z


def permutation_sign(perm: np.ndarray) -> int:
    """Sign (+1/-1) of a permutation given as an index array (cycle decomposition, O(n))."""
    perm = np.asarray(perm)
    seen = np.zeros(perm.size, dtype=bool)
    transpositions = 0
    for i in range(perm.size):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = int(perm[j])
            length += 1
        transpositions += length - 1
    return -1 if transpositions % 2 else 1


def _phase_logabs(pivots: np.ndarray, sign: int) -> tuple[complex, float]:
    """(unit phase, log modulus) of ``sign * prod(pivots)``; SingularPoint on a zero pivot."""
    a = np.abs(pivots)
    if not np.all(a > 0) or not np.all(np.isfinite(pivots)):
        raise SingularPoint(complex("nan"))
    ang = float(np.sum(np.angle(pivots))) + (math.pi if sign < 0 else 0.0)
    return complex(math.cos(ang), math.sin(ang)), float(np.sum(np.log(a)))


class _DenseLU:
    """Partial-pivot LU of a dense matrix: solve and (phase, log|det|)."""

    def __init__(self, A: np.ndarray, z: complex) -> None:
        with warnings.catch_warnings():          # an exactly zero pivot: SingularPoint in slogdet
            warnings.simplefilter("ignore")
            self.lu, self.piv = sla.lu_factor(A, check_finite=False)
        self.z = z

    def slogdet(self) -> tuple[complex, float]:
        swaps = int(np.count_nonzero(self.piv != np.arange(self.piv.size)))
        try:
            return _phase_logabs(np.diag(self.lu), -1 if swaps % 2 else 1)
        except SingularPoint:
            raise SingularPoint(self.z) from None

    def solve(self, B: np.ndarray) -> np.ndarray:
        return sla.lu_solve((self.lu, self.piv), B, check_finite=False)

    def solve_h(self, B: np.ndarray) -> np.ndarray:
        """``A^H X = B`` (the left eigenvector of the two-sided polish)."""
        return sla.lu_solve((self.lu, self.piv), B, trans=2, check_finite=False)


class _SparseLU:
    """SuperLU of a sparse matrix: ``Pr A Pc = L U`` with unit-diagonal ``L``."""

    def __init__(self, A, z: complex) -> None:
        self.z = z
        try:
            self.lu = spla.splu(sp.csc_matrix(A))
        except RuntimeError as exc:          # "Factor is exactly singular"
            raise SingularPoint(z) from exc

    def slogdet(self) -> tuple[complex, float]:
        sign = permutation_sign(self.lu.perm_r) * permutation_sign(self.lu.perm_c)
        try:
            return _phase_logabs(self.lu.U.diagonal(), sign)
        except SingularPoint:
            raise SingularPoint(self.z) from None

    def solve(self, B: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(B, dtype=complex))

    def solve_h(self, B: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(B, dtype=complex), trans="H")


def _factor(A, z: complex):
    if sp.issparse(A):
        return _SparseLU(A, z)
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    if not np.all(np.isfinite(A)):
        raise SingularPoint(z)
    return _DenseLU(A, z)


def _norm1(A) -> float:
    if sp.issparse(A):
        return float(abs(A).sum(axis=0).max()) if A.nnz else 0.0
    return float(np.linalg.norm(np.atleast_2d(A), 1))


class Operator:
    """Interface of an operator ``T(z)`` for :mod:`conatus.contour` (see the module docstring).

    ``poles``: the number of poles of ``det T`` inside a region, when it is known by construction
    (0 for a pencil); ``None`` when it is not known (a callable: the caller declares it)."""

    n: int
    poles: int | None = None
    kind: str = "operator"
    determinant: str = ""        # how det T is evaluated, in words
    #: floor of :meth:`scale` for an operator without a structural norm (a callable T: a 1 x 1
    #: T(z) = f(z) has ||T(z)|| = |f(z)|, which vanishes at its roots and would make every residual
    #: 1); set by :func:`conatus.contour.beyn` to the median ||T|| on the boundary of the region
    reference: float = 0.0
    needs_reference: bool = False
    #: the scale s(z) of the residual ||T(z) v|| / (s(z) ||v||), in words (an operator with
    #: needs_reference gets its sentence from conatus.contour.beyn, with the reference value)
    scale_definition: str = "a norm of T(z)"
    #: True when ``poles`` holds by construction (a pencil, a linear Hill operator), not by the
    #: caller's declaration: only then may a box be cut and its halves counted on their own (a
    #: wrong declaration would otherwise hide a root behind a half with winding 0)
    structural_poles: bool = False

    def factor(self, z: complex):
        """An object with ``slogdet() -> (phase, log|det|)`` and ``solve(B)``."""
        raise NotImplementedError

    def slogdet(self, z: complex) -> tuple[complex, float]:
        """(unit phase, log modulus) of det T(z), from :meth:`factor`."""
        return self.factor(z).slogdet()

    def apply(self, z: complex, X: np.ndarray) -> np.ndarray:
        """``T(z) X``."""
        raise NotImplementedError

    def dapply(self, z: complex, X: np.ndarray) -> np.ndarray:
        """``T'(z) X``."""
        raise NotImplementedError

    def scale(self, z: complex) -> float:
        """A norm of ``T(z)`` (1-norm or an upper bound of it): the residual is relative to it."""
        raise NotImplementedError

    def explain(self) -> str:
        """What the operator is and how its determinant is evaluated."""
        return f"{self.kind} of size {self.n}; det T: {self.determinant}"


class Pencil(Operator):
    """``T(z) = L + z M`` (dense or sparse): the eigenvalue convention of conatus.solvers.evp."""

    def __init__(self, L, M, *, label: str = "") -> None:
        if L.shape != M.shape or L.shape[0] != L.shape[1]:
            raise OptionError(f"pencil: L {L.shape} and M {M.shape} must be square and of one size",
                              fix="give the two matrices of (L + s M) x = 0", doc=_DOC)
        self.sparse = sp.issparse(L) or sp.issparse(M)
        if self.sparse:
            self.L, self.M = sp.csc_matrix(L, dtype=complex), sp.csc_matrix(M, dtype=complex)
        else:
            self.L, self.M = np.asarray(L, complex), np.asarray(M, complex)
        self.n = L.shape[0]
        self.poles = 0
        self.structural_poles = True
        self.kind = f"pencil L + z M{' [' + label + ']' if label else ''}"
        self.determinant = ("a polynomial in z (no pole: P = 0), read off the "
                            f"{'sparse (SuperLU)' if self.sparse else 'dense'} LU as unit phase "
                            "and log modulus with both permutation signs")
        self._nL, self._nM = _norm1(self.L), _norm1(self.M)
        self.scale_definition = ("||L||_1 + |z| ||M||_1 (a backward error relative to the "
                                 "pencil's coefficients)")

    def matrix(self, z: complex):
        """The matrix ``L + z M``."""
        return self.L + z * self.M

    def factor(self, z: complex):
        """LU of ``L + z M`` (dense or SuperLU)."""
        return _factor(self.matrix(z), z)

    def apply(self, z, X):
        """``(L + z M) X``."""
        return self.L @ X + z * (self.M @ X)

    def dapply(self, z, X):
        """``T'(z) X = M X``."""
        return self.M @ X

    def dapply_h(self, z, Y):
        """``T'(z)^H Y``."""
        return self.M.conj().T @ Y

    def scale(self, z):
        """``||L||_1 + |z| ||M||_1``, the scale of the EVP residuals (conatus.solvers.evp); ``||M||_1``
        where that vanishes (``L = 0`` at ``z = 0``: the residual was 0 / 0)."""
        s = self._nL + abs(z) * self._nM
        return s if s > 0 else self._nM


class MatrixFunction(Operator):
    """Any callable ``T(z)`` returning a square matrix (dense or sparse) or a scalar.

    ``derivative``: ``T'(z)`` (same type), optional; without it the polish uses a central
    difference of relative step :data:`DERIVATIVE_STEP` (only its convergence rate depends on it).
    ``poles``: the number of poles of ``det T`` inside the region, declared by the caller (0 for a
    function holomorphic there); ``None`` = not declared."""

    def __init__(self, T: Callable[[complex], Any], *, derivative: Callable | None = None,
                 poles: int | None = None, label: str = "") -> None:
        self.T, self.dT = T, derivative
        raw = T(complex(0.3719, 0.6153))                 # a generic point: the shape only
        self.scalar = not sp.issparse(raw) and np.ndim(raw) == 0
        self.n = self._eval(lambda _z: raw, 0j).shape[0]
        self.poles = poles
        self.kind = ("scalar function f(z)" if self.scalar else "matrix function T(z)") + \
            (f" [{label}]" if label else "")
        self.determinant = (("f(z) itself" if self.scalar else "LU of T(z) (unit phase and log "
                             "modulus with the permutation signs)")
                            + ("; derivative given" if derivative is not None else
                               f"; T' by central difference (relative step {DERIVATIVE_STEP:g}, "
                               "polish only)"))

    @staticmethod
    def _eval(F, z):
        A = F(z)
        if sp.issparse(A):
            if A.shape[0] != A.shape[1]:
                raise InputTypeError(f"T(z) has shape {A.shape}: a square matrix is needed", doc=_DOC)
            return A
        A = np.asarray(A, dtype=complex)
        if A.ndim == 0:
            return A.reshape(1, 1)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise InputTypeError(f"T(z) has shape {A.shape}: a scalar or a square matrix is needed",
                                 doc=_DOC)
        return A

    def matrix(self, z):
        """``T(z)`` as a 2-D array or sparse matrix (a scalar as 1 x 1)."""
        return self._eval(self.T, z)

    def factor(self, z):
        """LU of ``T(z)``."""
        return _factor(self.matrix(z), z)

    def apply(self, z, X):
        """``T(z) X``."""
        return self.matrix(z) @ X

    def _derivative(self, z):
        if self.dT is not None:
            return self._eval(self.dT, z)
        h = DERIVATIVE_STEP * max(1.0, abs(z))
        return (self.matrix(z + h) - self.matrix(z - h)) / (2 * h)

    def dapply(self, z, X):
        """``T'(z) X`` (given derivative, else central difference)."""
        return self._derivative(z) @ X

    def dapply_h(self, z, Y):
        """``T'(z)^H Y`` (two-sided polish)."""
        return self._derivative(z).conj().T @ Y

    needs_reference = True

    def scale(self, z):
        """``max(||T(z)||_1, reference)``: never the vanishing norm of a scalar at its root."""
        return max(_norm1(self.matrix(z)), self.reference)


class HillBlocksOperator(Operator):
    """A :class:`conatus.floquet.HillBlocks` seen as ``T(mu)``: factored by block-Thomas on its
    super-blocks (never assembled), ``det`` from the pivots of every Schur complement."""

    def __init__(self, hill: Any, *, poles: int | None = None) -> None:
        self.H = hill
        self.K = len(hill.harmonics)
        self.n = self.K * hill.size
        linear = getattr(hill, "mass", None) is not None
        self.poles = 0 if linear else poles
        self.needs_reference = not linear
        self.structural_poles = linear
        self.scale_definition = ("an upper bound of ||T(z)||_1: the largest diagonal block plus "
                                 "the couplings")
        self.kind = f"Hill operator [{hill.label}] (harmonics -{hill.Nh}..{hill.Nh}, block {hill.size})"
        self.determinant = ("block-Thomas LU on super-blocks, phase and log modulus of the pivots "
                            "with their row interchanges"
                            + ("; linear in mu (mass block): a polynomial, P = 0" if linear else
                               "; mu-dependent blocks: poles declared by the caller"))

    def factor(self, z):
        """Block-Thomas factors of the Hill matrix at ``z`` (phase, log modulus, solves)."""
        H = self.H
        with warnings.catch_warnings():          # an exactly zero pivot: SingularPoint below
            warnings.simplefilter("ignore")
            f = H.factor(z)
            ph, la = f.slogdet()
        if not (np.isfinite(la) and np.isfinite(ph) and abs(ph) > 0):
            # block-Thomas pivots within each super-block only: a singular Schur complement of a
            # regular Hill matrix breaks it down (det -2j read as NaN, then reported as "an
            # eigenvalue on the contour"; critic review of L10, review finding 11). The assembled
            # sparse matrix, LU with row pivoting across the blocks, says whether z is singular.
            return _factor(self.assembled(z), z)

        class _F:
            def slogdet(_s):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ph, la = f.slogdet()
                if not (np.isfinite(la) and abs(ph) > 0):
                    raise SingularPoint(z)
                return complex(ph), float(la)

            def solve(_s, B):
                B = np.asarray(B, complex).reshape(self.n, -1)
                cols = [f.solve(B[:, j].reshape(self.K, H.size)).reshape(-1) for j in range(B.shape[1])]
                return np.stack(cols, axis=1)
        return _F()

    def assembled(self, z: complex):
        """The Hill matrix ``M(z)`` as a sparse matrix (harmonic blocks; the fallback factor)."""
        H, K = self.H, self.K
        blocks = [[None] * K for _ in range(K)]
        for i, n in enumerate(H.harmonics):
            blocks[i][i] = sp.csr_matrix(H.diag(z, n))
            for m, C in H.C.items():
                j = i - m
                if 0 <= j < K:
                    blocks[i][j] = sp.csr_matrix(C)
        return sp.csc_matrix(sp.bmat(blocks, format="csc"), dtype=complex)

    def apply(self, z, X):
        """``M(z) X`` column by column (never assembled)."""
        X = np.asarray(X, complex).reshape(self.n, -1)
        cols = [self.H.matvec(z, X[:, j].reshape(self.K, self.H.size)).reshape(-1)
                for j in range(X.shape[1])]
        return np.stack(cols, axis=1)

    def dapply(self, z, X):
        """``M'(z) X``: the mass block per harmonic, else a central difference."""
        X = np.asarray(X, complex).reshape(self.n, -1)
        if self.H.mass is not None:
            Mb = self.H.mass
            return np.stack([(X[:, j].reshape(self.K, -1) @ Mb.T).reshape(-1)
                             for j in range(X.shape[1])], axis=1)
        h = DERIVATIVE_STEP * max(1.0, abs(z))
        return (self.apply(z + h, X) - self.apply(z - h, X)) / (2 * h)

    def scale(self, z):
        """An upper bound of ``||M(z)||_1`` (largest diagonal block plus couplings), floored by ``reference``."""
        cpl = sum(_norm1(C) for C in self.H.C.values())
        own = max(_norm1(self.H.diag(z, n)) for n in self.H.harmonics) + cpl
        return max(own, self.reference)


def _is_hill(T: Any) -> bool:
    return all(hasattr(T, a) for a in ("factor", "matvec", "harmonics", "diag", "C", "size"))


def as_operator(T: Any, *, poles: int | None = None, derivative: Callable | None = None) -> Operator:
    """Wrap ``T``: an :class:`Operator`, a pair ``(L, M)`` (pencil ``L + z M``), a
    :class:`conatus.floquet.HillBlocks`, or a callable ``T(z)`` (matrix or scalar)."""
    if poles is not None and (isinstance(poles, bool) or not isinstance(poles, numbers.Integral)
                              or poles < 0):
        raise OptionError(f"poles={poles!r}: the number of poles inside the region, an integer >= 0",
                          fix="poles=0 for a T(z) holomorphic in the region", doc=_DOC)
    if isinstance(T, Operator):
        if poles is not None and T.poles is not None and poles != T.poles:
            raise OptionError(f"poles={poles} contradicts the operator ({T.kind}: {T.poles} by "
                              "construction)", doc=_DOC)
        return T
    if isinstance(T, tuple) and len(T) == 2:
        if poles not in (None, 0):
            raise OptionError(f"poles={poles}: the determinant of a pencil L + z M is a "
                              "polynomial, it has no pole", fix="drop poles=", doc=_DOC)
        return Pencil(*T)
    if _is_hill(T):
        if poles is not None and getattr(T, "mass", None) is not None and poles != 0:
            raise OptionError(f"poles={poles}: a Hill operator linear in mu has no pole",
                              fix="drop poles=", doc=_DOC)
        return HillBlocksOperator(T, poles=poles)
    if callable(T):
        return MatrixFunction(T, derivative=derivative, poles=poles)
    raise InputTypeError(f"contour methods need T(z): got {type(T).__name__}",
                         fix="a pair (L, M) for L + z M, a HillBlocks, or a callable z -> matrix "
                             "(or scalar)", doc=_DOC)
