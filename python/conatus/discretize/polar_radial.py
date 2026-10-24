"""Radial operators and transforms of the polar plans.

One azimuthal group m at a time, one *spin component* at a time. A component of a tensor field with
spin indices s_1..s_q (each +-1, spin basis e_+- = (e_r -+ i e_phi)/2, u_+- = u_r +- i u_phi) behaves
under rotations like a scalar of effective wavenumber k = m + s_1 + ... + s_q (Vasil et al. 2016).
Every operator below acts on such a component and returns ``(matrix, level_out, p_out)``.

Disk (``DiskOps``): the component is rho^|k| X(z), rho = r/R, z = 2 rho^2 - 1, and X is expanded in
Jacobi polynomials P_n^(a, |k|)(z), n < N_m = N - floor(|m|/2) for *every* component of group m
(Dedalus's DiskBasis truncation: triangular in m, so that grad, div and lap map the components of a
group onto each other square; see polar.md for why a per-spin size N - floor(|k|/2) is not).
``level`` is the Jacobi parameter a (0 for fields, +1 per derivative); every operator takes the
component size ``n`` explicitly. In these bases the spin derivatives are single diagonals
(DLMF 18.9.15-16):
    G_sigma = d_r - sigma k / r,
    same sign (k = 0 or sign k = sigma):  P_n^(a,|k|) -> 2 (n+a+|k|+1) P_{n-1}^(a+1, |k|+1) / R
    opposite sign:                        P_n^(a,|k|) -> 2 (n+|k|)     P_n^(a+1, |k|-1)     / R
    Laplacian = G_-+ G_+-:                P_n^(a,|k|) -> 4 (n+a+|k|+1)(n+|k|) P_{n-1}^(a+2,|k|) / R^2
and the conversion P^(a,b) -> P^(a+1,b) is bidiagonal (DLMF 18.9.6). The axis needs no condition.

Annulus (``AnnulusOps``): Chebyshev T in t, r = c + h t, ultraspherical levels, and a declared power
p of r multiplying the held quantity (r-multiplied forms, polynomial coefficients; polar.Annulus).
"""
from __future__ import annotations

from math import lgamma

import numpy as np
import scipy.sparse as sp

from ..errors import OptionError
from . import ultraspherical as us
from .polar import Annulus
from .jacobi_family import gauss_legendre

__all__ = ["jacobi_values", "jacobi_norm", "DiskOps", "AnnulusOps"]


# ----------------------------------------------------------------------------------------------
# Jacobi polynomials P_n^(a,b) (unnormalized, DLMF 18.3)
# ----------------------------------------------------------------------------------------------

def jacobi_values(N: int, a: float, b: float, z: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """(len(z), N) array of scale * P_n^(a,b)(z), n < N, by the three-term recurrence (DLMF 18.9.2).
    ``scale`` (e.g. rho^|k|) multiplies from the start, so tiny factors never meet huge values.
    The recurrence runs in long double and the result is rounded once to float64 (in float64 the unnormalised P_n lost 1.5 digits by L = 64, |F B - I| = 1.1e-13 in the spin
    harmonic transform of m = 3, 3e-15 with the long-double recurrence)."""
    z = np.asarray(z, dtype=np.longdouble).ravel()
    s = (np.ones_like(z) if scale is None
         else np.broadcast_to(np.asarray(scale, dtype=np.longdouble).ravel(), z.shape))
    out = np.zeros((z.size, max(N, 0)), dtype=np.longdouble)
    if N == 0:
        return out.astype(float)
    a, b = np.longdouble(a), np.longdouble(b)
    out[:, 0] = s
    if N == 1:
        return out.astype(float)
    out[:, 1] = s * ((a + 1) + (a + b + 2) * (z - 1) / 2)
    for n in range(1, N - 1):
        c = 2 * n + a + b
        c1 = 2 * (n + 1) * (n + a + b + 1) * c
        c2 = (c + 1) * (c + 2) * c
        c3 = (c + 1) * (a * a - b * b)
        c4 = 2 * (n + a) * (n + b) * (c + 2)
        out[:, n + 1] = ((c2 * z + c3) * out[:, n] - c4 * out[:, n - 1]) / c1
    return out.astype(float)


def _jacobi_z(N: int, a: float, b: float) -> sp.csr_matrix:
    """Multiplication by z on P^(a,b) coefficients (N x N truncation of the tridiagonal operator):
    z P_n = A_n P_{n+1} + B_n P_n + C_n P_{n-1}."""
    rows, cols, vals = [], [], []
    for n in range(N):
        if n == 0:
            A, B, C = 2.0 / (a + b + 2), (b - a) / (a + b + 2), 0.0
        else:
            c = 2 * n + a + b
            c1 = 2 * (n + 1) * (n + a + b + 1) * c
            c2 = (c + 1) * (c + 2) * c
            c3 = (c + 1) * (a * a - b * b)
            c4 = 2 * (n + a) * (n + b) * (c + 2)
            A, B, C = c1 / c2, -c3 / c2, c4 / c2
        # coefficient vector x: (z f)_{n+1} += A_n x_n, (z f)_n += B_n x_n, (z f)_{n-1} += C_n x_n
        for r, v in ((n + 1, A), (n, B), (n - 1, C)):
            if 0 <= r < N and v != 0.0:
                rows.append(r)
                cols.append(n)
                vals.append(v)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _conv_a(N: int, a: float, b: float) -> sp.csr_matrix:
    """P^(a,b) -> P^(a+1,b): (2n+a+b+1) P_n^(a,b) = (n+a+b+1) P_n^(a+1,b) - (n+b) P_{n-1}^(a+1,b)."""
    n = np.arange(N, dtype=float)
    d = 2 * n + a + b + 1
    diag = (n + a + b + 1) / d
    up = -(n[1:] + b) / d[1:]                      # column n contributes to row n-1
    return sp.diags([diag, up], [0, 1], shape=(N, N), format="csr")


# ----------------------------------------------------------------------------------------------
# Disk
# ----------------------------------------------------------------------------------------------

def log_jacobi_norms(a: float, b: float, size: int) -> np.ndarray:
    """log s_n (long double), n < size, s_n^2 = int_{-1}^{1} ((1-z)/2)^a ((1+z)/2)^b P_n^(a,b)(z)^2 dz
    = 2/(2n+a+b+1) Gamma(n+a+1) Gamma(n+b+1) / (Gamma(n+a+b+1) n!).

    Built without log-gamma: s_0^2 = 2/(a+b+1) p! / prod_{j<=p} (q+j) with p the
    integer one of (a, b) (the smaller when both are), then s_n^2 / s_{n-1}^2 =
    (2n+a+b-1)(n+a)(n+b) / ((2n+a+b+1)(n+a+b) n), summed in long double. The float64 log-gamma
    differences lost |lgamma| * eps: 1e-13 relative at n = 60 (|F B - I| = 1.1e-13 in the spin
    harmonic transform at L = 64, m = 3), more at large m."""
    LD = np.longdouble
    a_, b_ = LD(a), LD(b)
    ai, bi = float(a).is_integer(), float(b).is_integer()
    if not (ai or bi):
        raise ValueError(f"log_jacobi_norms: one of a, b must be an integer (a={a}, b={b})")
    p, q = (a_, b_) if ai and (not bi or a_ <= b_) else (b_, a_)
    j = np.arange(1, int(p) + 1, dtype=LD)
    ls2 = np.log(LD(2) / (a_ + b_ + 1)) + np.sum(np.log(j) - np.log(q + j))
    out = np.empty(max(int(size), 0), dtype=LD)
    if out.size == 0:
        return out
    n = np.arange(1, out.size, dtype=LD)
    c = 2 * n + a_ + b_
    lr = np.log(c - 1) - np.log(c + 1) + np.log(n + a_) + np.log(n + b_) - np.log(n + a_ + b_) - np.log(n)
    out[0] = ls2
    out[1:] = ls2 + np.cumsum(lr)
    return out / 2


def jacobi_norm(a: int, b: int, size: int) -> np.ndarray:
    """s_n = sqrt(int_{-1}^{1} ((1-z)/2)^a ((1+z)/2)^b P_n^(a,b)(z)^2 dz), n < size
    (:func:`log_jacobi_norms`, no overflow)."""
    return np.exp(log_jacobi_norms(a, b, size)).astype(float)


def _check_lift_index(n: int, size: int) -> None:
    """A tau lift counts modes from the top: n must be a negative integer (lift(tau, basis, -1)
    is the last mode); n >= 0 would index past the end of the column."""
    if int(n) != n or n >= 0:
        raise OptionError(f"lift index n = {n!r}: the tau lift takes a negative mode index "
                          f"(-1 is the last of the {size} modes)",
                          fix="lift(tau, basis, -1) for the last mode, -2 for the one before")


class DiskOps:
    """Operators of the regular disk basis for one resolution N and radius R (see module doc).

    Coefficients refer to the *normalized* functions rho^|k| Q_n^(a,|k|), Q_n = P_n / s_n
    (``jacobi_norm``; Dedalus normalizes the same way): at level 0 they are orthonormal on the disk
    (up to the constant pi R^2 / 2), so the forward transform is B^T W and coefficient magnitudes
    follow function magnitudes (unnormalized P_n^(0,|k|)(-1) = binom(n+|k|, n)). Every operator is
    the unnormalized one (formulas in the module doc) rescaled by diagonal matrices: same sparsity.
    Note: a Laplacian evaluated pointwise amplifies the round-off tail like eps (2N)^4 (the radial
    degree in rho is 2N), as eps N^4 for Chebyshev: ~1e-8 at N = 32, 2.6e-6 at N = 48 (measured)."""

    kind = "disk"

    def __init__(self, N: int, R: float) -> None:
        self.N, self.R = int(N), float(R)
        self._cache: dict = {}

    def _s(self, level: int, kap: int, size: int) -> np.ndarray:
        key = ("norm", level, kap, size)
        if key not in self._cache:
            self._cache[key] = jacobi_norm(level, kap, size)
        return self._cache[key]

    def _scaled(self, M, lev_in: int, kap_in: int, lev_out: int, kap_out: int, size: int):
        """diag(s_out) M diag(1/s_in): unnormalized operator -> normalized coefficients."""
        return (sp.diags(self._s(lev_out, kap_out, size)) @ M
                @ sp.diags(1.0 / self._s(lev_in, kap_in, size))).tocsr()

    # sizes -------------------------------------------------------------------------------------
    def size(self, m: int) -> int:
        """N_m = N - floor(|m|/2) radial coefficients of every component of group m (0 if negative)."""
        return max(self.N - abs(int(m)) // 2, 0)

    # operators -----------------------------------------------------------------------------------
    def G(self, sigma: int, k: int, level: int, p: int, size: int):
        """Spin-raising (sigma = +1) or lowering (-1) gradient component on spin-k coefficients: (matrix, output level, output r-power)."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        kap = abs(k)
        n = np.arange(size, dtype=float)
        if k == 0 or (k > 0) == (sigma > 0):          # -> P_{n-1}^(a+1, |k|+1)
            vals = 2.0 * (n[1:] + level + kap + 1) / self.R
            M = sp.csr_matrix((vals, (n[1:].astype(int) - 1, n[1:].astype(int))), shape=(size, size))
        else:                                         # -> P_n^(a+1, |k|-1), diagonal
            vals = 2.0 * (n + kap) / self.R
            M = sp.csr_matrix((vals, (n.astype(int), n.astype(int))), shape=(size, size))
        return self._scaled(M, level, kap, level + 1, abs(k + sigma), size), level + 1, 0

    def L(self, k: int, level: int, p: int, size: int):
        """Laplacian on spin-k coefficients (one superdiagonal, Zernike levels +2): (matrix, output level, output r-power)."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        kap = abs(k)
        n = np.arange(1, size, dtype=float)
        vals = 4.0 * (n + level + kap + 1) * (n + kap) / self.R ** 2
        M = sp.csr_matrix((vals, (n.astype(int) - 1, n.astype(int))), shape=(size, size))
        return self._scaled(M, level, kap, level + 2, kap, size), level + 2, 0

    def convert(self, k: int, lev_from: int, lev_to: int, size: int) -> sp.csr_matrix:
        """Conversion between Jacobi levels (identity map of the function, cached)."""
        if lev_to < lev_from:
            raise ValueError(f"cannot convert level {lev_from} down to {lev_to}")
        key = ("conv", abs(k), lev_from, lev_to, size)
        if key not in self._cache:
            out = sp.identity(size, format="csr")
            for a in range(lev_from, lev_to):
                out = _conv_a(size, a, abs(k)) @ out
            self._cache[key] = self._scaled(out, lev_from, abs(k), lev_to, abs(k), size)
        return self._cache[key]

    def multiply_r(self, k: int, level: int, j: int, size: int) -> sp.csr_matrix:
        """Multiplication by r^j: only j = 0 on the disk (no r-multiplied forms)."""
        if j:
            raise ValueError("disk: no r-multiplier")
        return sp.identity(size, format="csr")

    def ncc(self, g_leg: np.ndarray, k: int, level: int, size: int) -> sp.csr_matrix:
        """Multiplication by an axisymmetric regular coefficient g = sum_j g_j P_j(z) (Legendre
        coefficients) on the level-``level`` space of |k|: sum_j g_j P_j(Z) by the Legendre
        recurrence on the Jacobi matrix Z, built on size + deg + 1 modes then truncated."""
        deg = len(g_leg) - 1
        Ne = size + deg + 1
        Z = _jacobi_z(Ne, level, abs(k))
        I = sp.identity(Ne, format="csr")
        P0, P1 = I, Z
        acc = g_leg[0] * P0 + (g_leg[1] * P1 if deg >= 1 else 0 * I)
        for j in range(1, deg):
            P0, P1 = P1, ((2 * j + 1) * (Z @ P1) - j * P0) / (j + 1)
            acc = acc + g_leg[j + 1] * P1
        return self._scaled(sp.csr_matrix(acc)[:size, :size], level, abs(k), level, abs(k), size)

    def ncc_spin(self, c_coef: np.ndarray, k_c: int, k_in: int, level: int, size: int) -> sp.csr_matrix:
        """Multiplication of a component rho^|k_in| X (level ``level``) by an axisymmetric coefficient
        component rho^|k_c| Xc(z) (normalized level-0 coefficients ``c_coef``), giving the component
        of wavenumber k_out = k_in + k_c:
            rho^|k_in| rho^|k_c| X Xc = rho^|k_out| ((1+z)/2)^(e/2) X Xc,  e = |k_in| + |k_c| - |k_out|
        (e even >= 0). The Jacobi parameter moves from |k_in| to |k_out|: down with d = |k_in| - |k_out|
        factors (1+z)/2 (P^(a,b) -> P^(a,b-1), bidiagonal, DLMF 18.9.6), up by exact conversions
        (DLMF 18.9.5); the remaining factor ((1+z)/2)^J Xc multiplies in P^(a,|k_out|). Built on an
        extended size, then truncated (as every NCC; the coefficient's tail is reported by the plan)."""
        ki, kc, ko = abs(k_in), abs(k_c), abs(k_in + k_c)
        e = ki + kc - ko
        if e < 0 or e % 2:
            raise ValueError("internal: inconsistent spin product")
        d = ki - ko
        J = e // 2 - max(d, 0)
        # Xc as Legendre coefficients (exact: evaluate on Gauss-Legendre nodes and project)
        nq = len(c_coef) + kc + J + 4
        x, w = gauss_legendre(nq)
        Xc = jacobi_values(len(c_coef), 0, kc, x) @ (np.asarray(c_coef) / self._s(0, kc, len(c_coef)))
        h = Xc * ((1 + x) / 2) ** J
        deg = len(c_coef) + J
        Pl = jacobi_values(deg + 1, 0, 0, x)
        h_leg = ((2 * np.arange(deg + 1) + 1) / 2) * (Pl.T @ (w * h))
        Ne = size + deg + abs(d) + 2
        T = sp.identity(Ne, format="csr")
        b = ki
        for _ in range(max(d, 0)):                    # (1+z)/2 P_n^(a,b) in P^(a,b-1)
            n = np.arange(Ne, dtype=float)
            den = 2 * n + level + b + 1
            low = sp.diags([(n + b) / den, ((n + 1) / den)[:-1]], [0, -1], shape=(Ne, Ne), format="csr")
            T = low @ T
            b -= 1
        for _ in range(max(-d, 0)):                   # exact conversion P^(a,b) -> P^(a,b+1)
            n = np.arange(Ne, dtype=float)
            den = 2 * n + level + b + 1
            up = sp.diags([(n + level + b + 1) / den, ((n[1:] + level) / den[1:])], [0, 1],
                          shape=(Ne, Ne), format="csr")
            T = up @ T
            b += 1
        Z = _jacobi_z(Ne, level, ko)
        I = sp.identity(Ne, format="csr")
        P0, P1 = I, Z
        acc = h_leg[0] * P0 + (h_leg[1] * P1 if deg >= 1 else 0 * I)
        for j in range(1, deg):
            P0, P1 = P1, ((2 * j + 1) * (Z @ P1) - j * P0) / (j + 1)
            acc = acc + h_leg[j + 1] * P1
        M = sp.csr_matrix(acc @ T)[:size, :size]
        return (sp.diags(self._s(level, ko, size)) @ M @ sp.diags(1.0 / self._s(level, ki, size))).tocsr()

    def legendre_from(self, c0: np.ndarray) -> np.ndarray:
        """Legendre (unnormalized P_n^(0,0)) coefficients of an m = 0 scalar from its normalized
        level-0 coefficients (for ``ncc``)."""
        return np.asarray(c0) / self._s(0, 0, len(c0))

    # rows (functionals) --------------------------------------------------------------------------
    def trace_row(self, k: int, level: int, p: int, side: str, size: int) -> np.ndarray:
        """Value at r = R (rho = 1): sum_n c_n Q_n(1), P_n^(a,b)(1) = binom(n+a, n)."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        if side != "outer":
            raise ValueError("the disk has a single boundary, r = R ('outer')")
        n = np.arange(size)
        vals = np.exp([lgamma(i + level + 1) - lgamma(i + 1) - lgamma(level + 1) for i in n])
        return vals / self._s(level, abs(k), size)

    def point_row(self, k: int, level: int, p: int, r0: float, deriv: int, size: int) -> np.ndarray:
        """Value (deriv = 0) or d/dr (deriv = 1) of rho^|k| X(z) at r = r0 (0 <= r0 <= R)."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        kap, Nk = abs(k), size
        rho = r0 / self.R
        z0 = np.array([2 * rho * rho - 1])
        s = self._s(level, kap, size)
        if deriv == 0:
            return jacobi_values(Nk, level, kap, z0, np.array([rho ** kap]))[0] / s
        if deriv != 1:
            raise ValueError("only the first radial derivative is available at a point")
        P = jacobi_values(Nk, level, kap, z0)[0]
        dP = np.zeros(Nk)
        n = np.arange(1, Nk)
        dP[1:] = (n + level + kap + 1) / 2 * jacobi_values(Nk - 1, level + 1, kap + 1, z0)[0]
        t1 = kap * rho ** (kap - 1) * P if kap else 0.0 * P
        return (t1 + 4.0 * rho ** (kap + 1) * dP) / self.R / s

    def integral_row(self, level: int, p: int, size: int) -> np.ndarray:
        """Area integral of an m = 0 scalar X(z): (pi R^2 / 2) int_{-1}^{1} X dz (Gauss-Legendre)."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        x, w = gauss_legendre(self.N + level + 2)
        return np.pi * self.R ** 2 / 2 * (w @ jacobi_values(size, level, 0, x)) / self._s(level, 0, size)

    def lift_col(self, k: int, level: int, n: int, size: int) -> np.ndarray:
        """Tau column: the mode size + n (n < 0) of the level-``level`` basis. A group whose
        truncated size is below -n gets a zero column (no such mode): the plan's rank report
        flags the resulting singular block, it is not hidden."""
        _check_lift_index(n, size)
        col = np.zeros(size)
        if size + n >= 0:
            col[size + n] = 1.0
        return col

    # transforms -----------------------------------------------------------------------------------
    def grid(self, M: int) -> np.ndarray:
        """Radial nodes r_i = R sqrt((1 + z_i)/2), z_i Gauss-Legendre (ascending)."""
        z, _ = gauss_legendre(M)
        return self.R * np.sqrt((1 + z) / 2)

    def backward(self, k: int, M: int, level: int, p: int, size: int) -> np.ndarray:
        """(M x size): values rho_i^|k| Q_n^(level,|k|)(z_i) on the Gauss-Legendre nodes."""
        if p:
            raise ValueError("disk: no r-multiplier (p must be 0)")
        key = ("bwd", abs(k), M, level, size)
        if key not in self._cache:
            z, _ = gauss_legendre(M)
            rho = np.sqrt((1 + z) / 2)
            self._cache[key] = (jacobi_values(size, level, abs(k), z, rho ** abs(k))
                                / self._s(level, abs(k), size))
        return self._cache[key]

    def forward(self, k: int, M: int, size: int) -> np.ndarray:
        """(size x M) = B^T W: orthogonal projection on span{rho^|k| Q_n^(0,|k|)} in L2(disk), by
        Gauss-Legendre quadrature in z (the disk measure is dz/4). Exact on the space when
        2M - 1 >= |k| + 2(size - 1): always for scalars with M >= N; the spin components of odd m
        need M >= N + 1 (callable data are projected from the dealiased grid, at least N + 1)."""
        key = ("fwd", abs(k), M, size)
        if key not in self._cache:
            _, w = gauss_legendre(M)
            self._cache[key] = (self.backward(k, M, 0, 0, size) * w[:, None]).T.copy()
        return self._cache[key]

    def explain(self) -> str:
        """Radius, size, radial basis, grid and operators of the disk, one line."""
        return (f"disk R={self.R:g}, N={self.N}: component of effective wavenumber k = rho^|k| sum_n "
                "c_n Q_n^(a,|k|)(2 rho^2 - 1) (normalized Jacobi, Zernike), n < N - floor(|m|/2) in "
                "group m; Gauss-Legendre radial grid in z; spin derivatives single-diagonal "
                "(Vasil et al. 2016)")


# ----------------------------------------------------------------------------------------------
# Annulus
# ----------------------------------------------------------------------------------------------

class AnnulusOps:
    """Operators of the annulus basis (Chebyshev/ultraspherical levels, r-multiplied forms).

    Same interface as DiskOps (size, G, L, convert, multiply_r, ncc, trace_row, integral_row,
    lift_col, grid, backward, forward); ``size`` arguments are accepted for that interface and are
    always N here (no per-m truncation on the annulus)."""

    kind = "annulus"

    def __init__(self, N: int, ri: float, ro: float) -> None:
        self.an = Annulus(N, ri, ro)
        self.N, self.ri, self.ro = int(N), float(ri), float(ro)
        self._cache: dict = {}

    def size(self, k: int) -> int:
        """Number of radial coefficients (N for every component)."""
        return self.N

    def G(self, sigma: int, k: int, level: int, p: int, size: int):
        """Spin-raising/lowering radial operator (sigma) of component ``k`` at ``level`` with
        r-power ``p``."""
        return self.an.G(sigma, k, level, p)

    def L(self, k: int, level: int, p: int, size: int):
        """Radial part of the Laplacian of component ``k`` at ``level`` with r-power ``p``."""
        return self.an.L(k, level, p)

    def convert(self, k: int, lev_from: int, lev_to: int, size: int) -> sp.csr_matrix:
        """Conversion from level ``lev_from`` to ``lev_to``."""
        return us.conversion_chain(self.N, lev_from, lev_to)

    def multiply_r(self, k: int, level: int, j: int, size: int) -> sp.csr_matrix:
        """Multiplication by r^j at ``level``."""
        return self.an.multiplier(level, j)

    def ncc(self, g_T: np.ndarray, k: int, level: int, size: int) -> sp.csr_matrix:
        # coefficients already judged and reported by the plan's NCC check
        """Multiplication by the radial NCC of Chebyshev coefficients ``g_T`` at ``level``."""
        return us.multiplication(np.asarray(g_T), self.N, level, name="ncc", exact=True)[0]

    def ncc_spin(self, c_coef: np.ndarray, k_c: int, k_in: int, level: int, size: int) -> sp.csr_matrix:
        """Annulus: components are plain Chebyshev series; multiply by the coefficient's T series."""
        # coefficients already judged and reported by the plan's NCC check
        return us.multiplication(np.asarray(c_coef), self.N, level, name="ncc", exact=True)[0]

    def trace_row(self, k: int, level: int, p: int, side: str, size: int) -> np.ndarray:
        """Row of the value at wall ``side`` of component ``k`` at ``level`` with r-power ``p``."""
        return self.an.trace_row(k, level, p, side)

    def point_row(self, k: int, level: int, p: int, r0: float, deriv: int, size: int) -> np.ndarray:
        """Value or d/dr of q = r^-p X at r = r0 (ri <= r0 <= ro)."""
        t0 = (r0 - self.an.c) / self.an.h
        v = us.basis_values(self.N, level, t0)
        if deriv == 0:
            return v / r0 ** p
        if deriv != 1:
            raise ValueError("only the first radial derivative is available at a point")
        return (us.basis_values(self.N, level, t0, deriv=1) / self.an.h - p * v / r0) / r0 ** p

    def integral_row(self, level: int, p: int, size: int) -> np.ndarray:
        """Row of the radial integral at ``level`` with r-power ``p``."""
        return self.an.integral_row(level, p)

    def lift_col(self, k: int, level: int, n: int, size: int) -> np.ndarray:
        """Tau column: the mode N + n (-N <= n < 0) of the level-``level`` basis."""
        _check_lift_index(n, self.N)
        if self.N + n < 0:          # a negative index would wrap around to another mode
            raise OptionError(f"annulus: lift index n = {n} below -N = {-self.N}",
                              fix=f"use -{self.N} <= n <= -1")
        col = np.zeros(self.N)
        col[self.N + n] = 1.0
        return col

    def grid(self, M: int) -> np.ndarray:
        """Radial Gauss-Chebyshev grid of ``M`` points in [ri, ro]."""
        t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
        return self.an.c + self.an.h * t

    def backward(self, k: int, M: int, level: int, p: int, size: int) -> np.ndarray:
        """(M x N): values r_i^-p C^(level)_n(t_i) (T_n for level 0) on the Gauss-Chebyshev nodes;
        r >= ri > 0, so undoing the declared r-multiplier on the grid is exact arithmetic."""
        key = ("bwd", M, level, p)
        if key not in self._cache:
            t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
            if level == 0:
                V = np.cos(np.outer(np.arccos(t), np.arange(self.N)))
            else:
                V = np.array([us.basis_values(self.N, level, ti) for ti in t])
            self._cache[key] = V / (self.an.c + self.an.h * t)[:, None] ** p
        return self._cache[key]

    def forward(self, k: int, M: int, size: int) -> np.ndarray:
        """Chebyshev interpolation on the M Gauss-Chebyshev nodes, truncated to N modes."""
        key = ("fwd", M)
        if key not in self._cache:
            t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
            V = np.cos(np.outer(np.arccos(t), np.arange(M)))
            self._cache[key] = np.linalg.inv(V)[: self.N]
        return self._cache[key]

    def explain(self) -> str:
        """Radii, size, radial basis, row multiplication and grid of the annulus, one line."""
        return (f"annulus [{self.ri:g}, {self.ro:g}], N={self.N}: Chebyshev T in t (spin components), "
                "r-multiplied operators (declared power of r per equation), Gauss-Chebyshev radial grid")
