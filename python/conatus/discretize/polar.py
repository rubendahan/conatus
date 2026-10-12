"""Regular spectral operators in polar coordinates: disk and annulus.

Each azimuthal Fourier mode m is treated separately (the azimuth is exactly separable).

Disk of radius R (axis included). Regularity at r = 0 is built into the representation, never
imposed by 1/(r+eps) (spec §13; Vasil, Burns, Lecoanet, Olver, Brown & Oishi, J. Comput. Phys. 325
(2016) 53–73): with rho = r/R and z = 2 rho^2 - 1 in [-1, 1],
    scalar  f     = rho^|m| w(z)                          e^{i m phi}
    vector  u_+   = u_r + i u_phi = rho^|m+1| a(z)        e^{i m phi}
            u_-   = u_r - i u_phi = rho^|m-1| b(z)        e^{i m phi}
with w, a, b polynomials expanded in Chebyshev T_n(z). Every operator then acts on w, a, b through
polynomial coefficients in z only (derivation in polar.md), so the Olver–Townsend machinery of
``ultraspherical`` gives banded matrices. Implemented for m >= 0; a real field has f_{-m} = conj(f_m).

Annulus r in [r_i, r_o] (no axis): u_r, u_phi, f expanded in T_n(t), r = c + h t. Operators are
written in r-multiplied form (r^2 lap, r grad, r div, r curl), whose coefficients are polynomials in t,
hence banded; the multiplier is declared and applied to the right-hand side as well.

Boundary conditions replace the last rows of each system (tau method in its row-replacement form,
Olver & Townsend, SIAM Rev. 55, 2013); ``rank_report`` counts unknowns, conditions and equations
and refuses anything that is not square.

The IR plans (``conatus.discretize.polar_plan``) use the generic spin toolkit below
(``Disk.G``/``Disk.L``/``Annulus.G``/``Annulus.L`` and the trace rows): every operator acts on one
spin component of effective azimuthal wavenumber k = m + (sum of its spins), takes its input in
C^(lam) and returns (matrix, lam_out, p_out), p being the declared power of r that multiplies the
held quantity (always 0 on the disk; r-multiplied forms on the annulus).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import eval_gegenbauer

from .jacobi_family import gauss_legendre
from .ultraspherical import (basis_values, conversion, conversion_chain, differentiation, integration_row,
                             multiplication)
from ..errors import BoundaryConditionError, CoordinateError, OptionError, ResolutionError
from .._repr import compact_repr

__all__ = ["Disk", "Annulus", "PolarSystem", "cheb_coeffs", "evaluate"]


# ----------------------------------------------------------------------------------------------
# small helpers on Chebyshev / ultraspherical coefficient vectors
# ----------------------------------------------------------------------------------------------

def _mult(poly_T: list[float], N: int, lam: int) -> sp.csr_matrix:
    """Multiplication by a (low-degree, exact) polynomial given by its Chebyshev coefficients."""
    # exact polynomial (r, r^2 on the annulus): no sampled tail to measure (declared exact)
    return multiplication(np.asarray(poly_T, float), N, lam, name="polar metric", exact=True)[0]


def cheb_coeffs(values_on_grid: np.ndarray) -> np.ndarray:
    """Chebyshev T coefficients from values at the N Gauss–Chebyshev points t_j = cos(pi(j+1/2)/N)."""
    N = values_on_grid.shape[-1]
    j = np.arange(N)
    t = np.cos(np.pi * (j + 0.5) / N)
    V = np.cos(np.outer(np.arccos(t), np.arange(N)))          # V[j, n] = T_n(t_j)
    return np.linalg.solve(V, values_on_grid)


def evaluate(coeffs: np.ndarray, lam: int, t) -> np.ndarray:
    """Evaluate sum_n c_n P_n(t) with P = T (lam = 0) or C^(lam)."""
    t = np.atleast_1d(np.asarray(t, float))
    n = np.arange(len(coeffs))
    if lam == 0:
        P = np.cos(np.outer(np.arccos(np.clip(t, -1, 1)), n))
    else:
        P = eval_gegenbauer(n[None, :], lam, t[:, None])
    return P @ coeffs


def to_T(coeffs: np.ndarray, lam: int) -> np.ndarray:
    """Exact change of basis C^(lam) -> T (inverse of the triangular conversion chain)."""
    if lam == 0:
        return np.asarray(coeffs)
    S = conversion_chain(len(coeffs), 0, lam).tocsr()
    return spla.spsolve_triangular(S, np.asarray(coeffs, dtype=np.result_type(coeffs, float)),
                                   lower=False)


# ----------------------------------------------------------------------------------------------
# square system with boundary rows (row-replacement tau) and a rank report
# ----------------------------------------------------------------------------------------------

@dataclass
class PolarSystem:
    """A square system for one azimuthal mode: operator rows + boundary rows, and what was done."""
    __repr__ = compact_repr   # one line, arrays by shape

    L: sp.csr_matrix                 # operator, output in C^(lam_out)
    B: np.ndarray                    # boundary rows (n_bc x N), acting on the unknown coefficients
    lam_out: int
    label: str
    notes: list[str] = field(default_factory=list)

    @property
    def N(self) -> int:
        return self.L.shape[1]

    def rank_report(self) -> str:
        n_bc = self.B.shape[0]
        n_eq = self.N - n_bc
        ok = n_eq + n_bc == self.N
        return (f"{self.label}: unknowns {self.N}, boundary conditions {n_bc}, equations kept {n_eq} "
                f"(last {n_bc} operator rows replaced by the conditions), square: {'PASS' if ok else 'FAIL'}")

    def matrix(self) -> sp.csr_matrix:
        n_bc = self.B.shape[0]
        return sp.vstack([sp.csr_matrix(self.B), self.L[: self.N - n_bc]]).tocsr()

    def solve(self, rhs_out: np.ndarray, bc_values) -> tuple[np.ndarray, float]:
        """Solve; returns coefficients and the relative residual ||A x - b|| / ||b|| (always reported)."""
        n_bc = self.B.shape[0]
        A = self.matrix()
        b = np.concatenate([np.atleast_1d(np.asarray(bc_values, dtype=np.result_type(rhs_out, float))),
                            rhs_out[: self.N - n_bc]])
        x = spla.spsolve(A.tocsc(), b)
        res = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
        return x, float(res)

    def explain(self) -> str:
        return "\n".join([self.rank_report(), *[f"  {n}" for n in self.notes]])


# ----------------------------------------------------------------------------------------------
# Disk
# ----------------------------------------------------------------------------------------------

class Disk:
    """Disk of radius R, azimuthal modes m >= 0, N Chebyshev coefficients in z = 2 (r/R)^2 - 1."""

    def __init__(self, N: int, R: float = 1.0) -> None:
        if N < 4:
            raise ResolutionError("Disk needs N >= 4 radial coefficients", fix='N >= 4 radial coefficients')
        self.N, self.R = int(N), float(R)

    def _m(self, m: int) -> int:
        if m < 0:
            raise NotImplementedError("negative m: use f_{-m} = conj(f_m) for real fields "
                                      "(operators are implemented for m >= 0)")
        return int(m)

    # --- scalar operators (input: w in T; output in C^(k)) ---------------------------------
    def laplacian(self, m: int) -> sp.csr_matrix:
        """lap(rho^m w e^{im phi}) = rho^m [8(1+z) w'' + 8(m+1) w'] / R^2 e^{im phi}; T -> C^(2)."""
        m, N = self._m(m), self.N
        L = 8.0 * _mult([1, 1], N, 2) @ differentiation(N, 0, 2) \
            + 8.0 * (m + 1) * conversion(N, 1) @ differentiation(N, 0, 1)
        return (L / self.R ** 2).tocsr()

    def identity_to(self, lam: int) -> sp.csr_matrix:
        """Conversion T -> C^(lam) (the 'mass' operator of a system written in C^(lam))."""
        return conversion_chain(self.N, 0, lam)

    def gradient(self, m: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """grad f in spin components: (a, b) with u_+ = rho^{m+1} a, u_- = rho^{|m-1|} b; T -> C^(1)."""
        m, N = self._m(m), self.N
        D1 = differentiation(N, 0, 1)
        A = 4.0 * D1                                            # a = 2 w_s = 4 w_z
        if m == 0:
            Bm = 4.0 * D1                                       # b = 2 w_s (rho^1)
        else:                                                   # b = 2m w + 2 s w_s = 2m w + 2(1+z) w_z
            Bm = 2.0 * m * conversion(N, 0) + 2.0 * _mult([1, 1], N, 1) @ D1
        return (A / self.R).tocsr(), (Bm / self.R).tocsr()

    def divergence(self, m: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """div u -> rho^m d(z): returns (Da, Db) with d = Da a + Db b; a, b in T -> C^(1)."""
        m, N = self._m(m), self.N
        D1, S0, Z = differentiation(N, 0, 1), conversion(N, 0), _mult([1, 1], N, 1)
        if m == 0:                                              # (a + s a_s) + (b + s b_s)
            Da = S0 + Z @ D1
            Db = S0 + Z @ D1
        else:                                                   # (m+1) a + s a_s + b_s
            Da = (m + 1) * S0 + Z @ D1
            Db = 2.0 * D1
        return (Da / self.R).tocsr(), (Db / self.R).tocsr()

    def curl(self, m: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """2D vorticity -> rho^m c(z): c = -i [ ... ] (see polar.md); a, b in T -> C^(1)."""
        m, N = self._m(m), self.N
        D1, S0, Z = differentiation(N, 0, 1), conversion(N, 0), _mult([1, 1], N, 1)
        if m == 0:
            Ca, Cb = S0 + Z @ D1, -(S0 + Z @ D1)
        else:
            Ca, Cb = (m + 1) * S0 + Z @ D1, -2.0 * D1
        return (-1j * Ca / self.R).tocsr(), (-1j * Cb / self.R).tocsr()

    # --- boundary rows at r = R (z = 1) ------------------------------------------------------
    def dirichlet_row(self) -> np.ndarray:
        return np.ones(self.N)                                  # rho^m w at rho = 1: w(1) = sum c_n

    def neumann_row(self, m: int) -> np.ndarray:
        """d/dr (rho^m w) at r = R: (m w(1) + 4 w'(1)) / R."""
        m, n = self._m(m), np.arange(self.N)
        return (m * np.ones(self.N) + 4.0 * n ** 2) / self.R    # T_n'(1) = n^2

    # --- generic spin toolkit for the IR plans (any integer k, input in C^(lam)) -------------
    # A held quantity at effective wavenumber k is rho^|k| X(z) with X in C^(lam); p = 0 always.
    def G(self, sigma: int, k: int, lam: int, p: int = 0) -> tuple[sp.csr_matrix, int, int]:
        """Spin derivative G_sigma = d_r - sigma k / r (sigma = +-1) on rho^|k| X: the result is
        rho^|k+sigma| Y with Y = 4 X' (k = 0 or sign(k) = sigma) or 2|k| X + 2(1+z) X' (else)."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        N, kap = self.N, abs(int(k))
        D = differentiation(N, lam, 1)
        if k == 0 or (k > 0) == (sigma > 0):
            M = 4.0 * D
        else:
            M = 2.0 * kap * conversion(N, lam) + 2.0 * _mult([1, 1], N, lam + 1) @ D
        return (M / self.R).tocsr(), lam + 1, 0

    def L(self, k: int, lam: int, p: int = 0) -> tuple[sp.csr_matrix, int, int]:
        """Laplacian of rho^|k| X (a scalar, or one spin component of a vector or tensor)."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        N, kap = self.N, abs(int(k))
        M = 8.0 * _mult([1, 1], N, lam + 2) @ differentiation(N, lam, 2) \
            + 8.0 * (kap + 1) * conversion(N, lam + 1) @ differentiation(N, lam, 1)
        return (M / self.R ** 2).tocsr(), lam + 2, 0

    def trace_row(self, k: int, lam: int, p: int, side: str) -> np.ndarray:
        """Value at r = R of rho^|k| X (rho = 1): X(z = 1)."""
        if p:                           # was ignored (audit C8); the other disk rows refuse it
            raise ValueError("disk: no r-multiplier (p must be 0)")
        if side != "outer":
            raise BoundaryConditionError("the disk has a single boundary, r = R",
                                         fix="use the boundary 'outer' (r = R): the disk has no inner wall")
        return basis_values(self.N, lam, 1.0)

    def dr_trace_row(self, k: int, lam: int, p: int, side: str) -> np.ndarray:
        """d/dr of rho^|k| X at r = R: (|k| X(1) + 4 X'(1)) / R."""
        if p:                           # refused like the other disk rows (it was ignored here)
            raise ValueError("disk: no r-multiplier (p must be 0)")
        if side != "outer":
            raise BoundaryConditionError("the disk has a single boundary, r = R",
                                         fix="use the boundary 'outer' (r = R): the disk has no inner wall")
        return (abs(k) * basis_values(self.N, lam, 1.0)
                + 4.0 * basis_values(self.N, lam, 1.0, deriv=1)) / self.R

    def integral_row(self, lam: int, p: int = 0) -> np.ndarray:
        """Area integral of an m = 0 scalar X(z): 2 pi int_0^R X r dr = (pi R^2 / 2) int X dz."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        return np.pi * self.R ** 2 / 2 * integration_row(self.N, lam)

    def multiplier(self, lam: int, j: int) -> sp.csr_matrix:
        """Disk: no r-multiplier exists (identity only)."""
        if j:
            raise ValueError("disk: no r-multiplier")
        return sp.identity(self.N, format="csr")

    # --- problems ----------------------------------------------------------------------------
    def helmholtz(self, m: int, k2: float = 0.0, bc: str = "dirichlet") -> PolarSystem:
        """(lap - k2) f = g for mode m, one boundary condition at r = R (the axis needs none)."""
        L = self.laplacian(m) - k2 * self.identity_to(2)
        B = (self.dirichlet_row() if bc == "dirichlet" else self.neumann_row(m))[None, :]
        return PolarSystem(L.tocsr(), B, 2, f"disk Helmholtz m={m}, k2={k2}, {bc}", notes=[
            "regular basis f = rho^|m| w(2 rho^2 - 1), w in Chebyshev T (no axis condition needed)",
            "operator T -> C^(2); the last row (degree N-1 of the image) is replaced by the condition"])

    def rhs(self, g_T: np.ndarray) -> np.ndarray:
        """Right-hand side g = rho^m q(z) given by q in T, projected on C^(2)."""
        return self.identity_to(2) @ g_T

    def bessel_modes(self, m: int, n_modes: int) -> np.ndarray:
        """Eigenvalues lambda of -lap f = lambda f, f(R) = 0 (dense QZ; infinite eigenvalues discarded)."""
        from scipy.linalg import eig
        sys_ = self.helmholtz(m, 0.0)
        A = sys_.matrix().toarray()
        Bm = np.vstack([np.zeros((1, self.N)), -self.identity_to(2).toarray()[: self.N - 1]])
        lam = eig(A, Bm, right=False)
        lam = lam[np.isfinite(lam)]
        lam = np.sort(np.real(lam[np.abs(np.imag(lam)) < 1e-8 * np.maximum(1, np.abs(lam))]))
        lam = lam[lam > 0]
        if lam.size < n_modes:
            # fewer real positive eigenvalues than asked: used to return a short array silently
            # (audit C8); the discarded ones are complex or non-positive (unresolved)
            raise ResolutionError(f"bessel_modes: only {lam.size} real positive eigenvalues at N = {self.N}, "
                                  f"n_modes = {n_modes} requested (increase N)",
                                  fix="a larger N, or fewer requested modes")
        return lam[:n_modes]

    def explain(self) -> str:
        return (f"Disk(R={self.R}, N={self.N}): modes m >= 0, f = rho^|m| w(z), z = 2 rho^2 - 1, "
                "w in Chebyshev T; spin components u_+- = rho^|m+-1| (a, b)(z); regular at the axis "
                "by construction (Vasil et al. 2016); tau by row replacement at r = R")


# ----------------------------------------------------------------------------------------------
# Annulus
# ----------------------------------------------------------------------------------------------

class Annulus:
    """Annulus r in [ri, ro], N Chebyshev coefficients in t, r = c + h t; operators in r-multiplied form."""

    def __init__(self, N: int, ri: float, ro: float) -> None:
        if not 0 < ri < ro:
            raise CoordinateError("Annulus needs 0 < ri < ro (use Disk when the axis is included)",
                                  fix="Annulus(ri, ro) with 0 < ri < ro, or Disk(R) when the axis is in the domain")
        self.N, self.ri, self.ro = int(N), float(ri), float(ro)
        self.c, self.h = (ro + ri) / 2, (ro - ri) / 2

    @property
    def r_T(self) -> list[float]:
        return [self.c, self.h]

    @property
    def r2_T(self) -> list[float]:
        c, h = self.c, self.h
        return [c * c + h * h / 2, 2 * c * h, h * h / 2]        # t^2 = (T_0 + T_2)/2

    def r2_laplacian(self, m: int) -> sp.csr_matrix:
        """r^2 lap f = r^2 f_rr + r f_r - m^2 f (declared multiplier r^2); T -> C^(2)."""
        N, h = self.N, self.h
        return (_mult(self.r2_T, N, 2) @ differentiation(N, 0, 2) / h ** 2
                + conversion(N, 1) @ _mult(self.r_T, N, 1) @ differentiation(N, 0, 1) / h
                - m * m * conversion_chain(N, 0, 2)).tocsr()

    def r_gradient(self, m: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """r grad f = (r f_r, i m f) in (r, phi) components; T -> C^(1)."""
        N = self.N
        return ((_mult(self.r_T, N, 1) @ differentiation(N, 0, 1) / self.h).tocsr(),
                (1j * m * conversion(N, 0)).tocsr())

    def r_divergence(self, m: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """r div u = r u_r' + u_r + i m u_phi; returns (on u_r, on u_phi); T -> C^(1)."""
        N = self.N
        return ((_mult(self.r_T, N, 1) @ differentiation(N, 0, 1) / self.h + conversion(N, 0)).tocsr(),
                (1j * m * conversion(N, 0)).tocsr())

    def r_curl(self, m: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """r curl u = r u_phi' + u_phi - i m u_r; returns (on u_r, on u_phi); T -> C^(1)."""
        N = self.N
        return ((-1j * m * conversion(N, 0)).tocsr(),
                (_mult(self.r_T, N, 1) @ differentiation(N, 0, 1) / self.h + conversion(N, 0)).tocsr())

    # --- generic spin toolkit for the IR plans (r-multiplied forms, declared power p) ---------
    # A held quantity is X = r^p q, X in C^(lam) of t; every operator returns r^(p + order) times
    # the exact result, whose coefficients are polynomials in t (banded).
    def r_power_T(self, j: int) -> np.ndarray:
        """Chebyshev T coefficients of r^j = (c + h t)^j (exact)."""
        out = np.array([1.0])
        for _ in range(int(j)):
            nxt = np.zeros(out.size + 1)
            nxt[: out.size] += self.c * out                     # c T_n
            for n, a in enumerate(out):                          # t T_n = (T_{n+1} + T_{|n-1|}) / 2
                if n == 0:
                    nxt[1] += self.h * a
                else:
                    nxt[n + 1] += self.h * a / 2
                    nxt[n - 1] += self.h * a / 2
            out = nxt
        return out

    def multiplier(self, lam: int, j: int) -> sp.csr_matrix:
        """Multiplication by r^j in C^(lam) (N x N truncation of an exact banded operator)."""
        if j == 0:
            return sp.identity(self.N, format="csr")
        return _mult(list(self.r_power_T(j)), self.N, lam)

    def G(self, sigma: int, k: int, lam: int, p: int = 0) -> tuple[sp.csr_matrix, int, int]:
        """r^(p+1) G_sigma q = (r d_r - sigma k - p) X for X = r^p q (G_sigma = d_r - sigma k / r)."""
        N = self.N
        M = _mult(self.r_T, N, lam + 1) @ differentiation(N, lam, 1) / self.h \
            - (sigma * k + p) * conversion(N, lam)
        return M.tocsr(), lam + 1, p + 1

    def L(self, k: int, lam: int, p: int = 0) -> tuple[sp.csr_matrix, int, int]:
        """r^(p+2) lap_k q = r^2 X'' + (1 - 2p) r X' + (p^2 - k^2) X for X = r^p q."""
        N, h = self.N, self.h
        M = (_mult(self.r2_T, N, lam + 2) @ differentiation(N, lam, 2) / h ** 2
             + (1 - 2 * p) * conversion(N, lam + 1) @ _mult(self.r_T, N, lam + 1)
             @ differentiation(N, lam, 1) / h
             + (p * p - k * k) * conversion_chain(N, lam, lam + 2))
        return M.tocsr(), lam + 2, p + 2

    def _side(self, side: str) -> tuple[float, float]:
        if side == "inner":
            return -1.0, self.ri
        if side == "outer":
            return 1.0, self.ro
        raise BoundaryConditionError("annulus boundaries: 'inner' (r = ri) or 'outer' (r = ro)",
                                     fix="'inner' (r = ri) or 'outer' (r = ro)")

    def trace_row(self, k: int, lam: int, p: int, side: str) -> np.ndarray:
        """q(r0) = r0^-p X(t0)."""
        t0, r0 = self._side(side)
        return basis_values(self.N, lam, t0) / r0 ** p

    def dr_trace_row(self, k: int, lam: int, p: int, side: str) -> np.ndarray:
        """q'(r0) = r0^-p (X'(t0)/h - p X(t0)/r0)."""
        t0, r0 = self._side(side)
        return (basis_values(self.N, lam, t0, deriv=1) / self.h
                - p * basis_values(self.N, lam, t0) / r0) / r0 ** p

    def integral_row(self, lam: int, p: int = 0) -> np.ndarray:
        """Area integral of an m = 0 scalar: 2 pi int q r dr = 2 pi h int X(t) (c + h t)^(1-p) dt."""
        if p not in (0, 1):
            raise OptionError("annulus integral of an r^p-multiplied quantity: only p = 0, 1",
                              fix="integrate the quantity times r^0 or r^1")
        x, w = gauss_legendre(self.N + 4)
        vals = np.array([basis_values(self.N, lam, t) for t in x])            # (nodes, N)
        weight = w * (self.c + self.h * x) ** (1 - p)
        return 2 * np.pi * self.h * (weight @ vals)

    def helmholtz(self, m: int, k2: float = 0.0) -> PolarSystem:
        """r^2 (lap - k2) f = r^2 g with Dirichlet conditions at r = ri and r = ro."""
        L = self.r2_laplacian(m) - k2 * _mult(self.r2_T, self.N, 2) @ conversion_chain(self.N, 0, 2)
        n = np.arange(self.N)
        B = np.vstack([(-1.0) ** n, np.ones(self.N)])           # T_n(-1), T_n(1)
        return PolarSystem(L.tocsr(), B, 2, f"annulus Helmholtz m={m}, k2={k2}, Dirichlet", notes=[
            "r-multiplied form r^2 (lap - k2) f = r^2 g (declared multiplier; r >= ri > 0)",
            "operator T -> C^(2); the last two rows are replaced by the wall conditions"])

    def rhs(self, g_T: np.ndarray) -> np.ndarray:
        """r^2 g projected on C^(2), g given by its T coefficients."""
        return _mult(self.r2_T, self.N, 2) @ conversion_chain(self.N, 0, 2) @ g_T

    def explain(self) -> str:
        return (f"Annulus(ri={self.ri}, ro={self.ro}, N={self.N}): f, u_r, u_phi in Chebyshev T of "
                "t = (r - c)/h; operators in r-multiplied form (r^2 lap, r grad, r div, r curl), "
                "banded; Dirichlet walls by row replacement")


# ----------------------------------------------------------------------------------------------
# Axisymmetric Taylor–Couette stability (annulus x Fourier in z), r^2-multiplied primitive form
# ----------------------------------------------------------------------------------------------

def taylor_couette_eigs(N: int, eta: float, Re: float, k: float) -> np.ndarray:
    """Eigenvalues sigma of axisymmetric perturbations of circular Couette flow (outer cylinder at rest).

    Units: length d = ro - ri, velocity Omega_i ri, Re = Omega_i ri d / nu, eta = ri / ro.
    Unknowns (u_r, u_phi, u_z, p) ~ exp(i k z + sigma t), no-slip at both walls. Each momentum equation
    is multiplied by r^2 and continuity by r (declared multipliers, r >= ri > 0), so every coefficient is
    a polynomial in r and every block is banded (Olver–Townsend). Returns all finite eigenvalues
    (the infinite ones, from the algebraic rows, have beta = 0 in the QZ solver and are dropped;
    they are not counted here).
    """
    from scipy.linalg import eig
    ri, ro = eta / (1 - eta), 1 / (1 - eta)
    B_ = 1.0 / (1.0 / ri - ri / ro ** 2)            # V = A r + B / r, V(ri) = 1, V(ro) = 0
    A_ = -B_ / ro ** 2
    an = Annulus(N, ri, ro)
    h, rT, r2T = an.h, an.r_T, an.r2_T
    D1, D2 = differentiation(N, 0, 1) / h, differentiation(N, 0, 2) / h ** 2
    S01, S12 = conversion(N, 0), conversion(N, 1)
    S02 = S12 @ S01
    Mr1, Mr2 = _mult(rT, N, 1), _mult(r2T, N, 2)
    I2 = S02
    visc = (Mr2 @ D2 + S12 @ Mr1 @ D1 - I2 - k * k * Mr2 @ S02) / Re          # r^2 (lap - 1/r^2) for u, v
    visc_w = (Mr2 @ D2 + S12 @ Mr1 @ D1 - k * k * Mr2 @ S02) / Re               # r^2 lap for w
    Z = sp.csr_matrix((N, N))
    r2 = Mr2 @ S02
    two_V_over_r = 2.0 * _mult([A_ * r2T[0] + B_, A_ * r2T[1], A_ * r2T[2]], N, 2) @ S02  # 2 (A r^2 + B)
    # rows (operator L such that sigma M x = L x)
    Lu = sp.hstack([visc, two_V_over_r, Z, -Mr2 @ S12 @ D1])
    Lv = sp.hstack([-2.0 * A_ * r2, visc, Z, Z])
    Lw = sp.hstack([Z, Z, visc_w, -1j * k * r2])
    Lc = sp.hstack([Mr1 @ D1 + S01, Z, 1j * k * Mr1 @ S01, Z])                  # r u' + u + i k r w (C^1)
    Mu, Mv, Mw = sp.hstack([r2, Z, Z, Z]), sp.hstack([Z, r2, Z, Z]), sp.hstack([Z, Z, r2, Z])
    n = np.arange(N)
    left, right = (-1.0) ** n, np.ones(N)

    def bc(field_idx: int, row: np.ndarray) -> np.ndarray:
        out = np.zeros(4 * N, dtype=complex)
        out[field_idx * N:(field_idx + 1) * N] = row
        return out

    Lrows, Mrows = [], []
    for fi, (L_, M_) in enumerate([(Lu, Mu), (Lv, Mv), (Lw, Mw)]):
        Lrows += [bc(fi, left)[None, :], bc(fi, right)[None, :], L_.tocsr()[: N - 2].toarray()]
        Mrows += [np.zeros((2, 4 * N)), M_.tocsr()[: N - 2].toarray()]
    Lrows.append(Lc.toarray())
    Mrows.append(np.zeros((N, 4 * N)))
    L, M = np.vstack(Lrows), np.vstack(Mrows)
    w = eig(L, M, right=False)
    return w[np.isfinite(w)]
