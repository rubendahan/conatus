"""Spherical toolkits: spin-weighted spherical harmonics, the
shell's r-multiplied Chebyshev operators and the ball's regular (3D Zernike) operators.

Conventions (Vasil, Lecoanet, Burns, Oishi & Brown, J. Comput. Phys. X 3 (2019) 100013, "Tensor
calculus in spherical coordinates using Jacobi polynomials. Part-I: mathematical analysis and
derivations"; Lecoanet et al., ibid. 100012, "Part-II: implementation and examples"):

- Spin basis of the tangent plane: e_+ = (e_theta - i e_phi)/sqrt(2), e_- = (e_theta + i e_phi)/sqrt(2);
  the components are u_+- = (u_theta +- i u_phi)/sqrt(2) (unitary: u.v = u_+ v_- + u_- v_+ + u_r v_r).
  Tensor index order (-, +, r), Dedalus's order; a component with spins s_1..s_q has total spin
  s = sum s_i (r counts 0).
- A spin-s component of azimuthal mode m is sum_l c_l Phi_l^{m,s}(cos theta) e^{i m phi},
  l = max(|m|, |s|)..Lmax, with the *normalized* functions (int_{-1}^{1} Phi^2 dx = 1)
      Phi_l^{m,s}(x) = ((1-x)/2)^{a/2} ((1+x)/2)^{b/2} P_n^{(a,b)}(x) / s_n,
      a = |m+s|, b = |m-s|, n = l - max(|m|, |s|),
  regular at both poles by construction. Phi^{-m,-s} = Phi^{m,s}: a real field stores m >= 0 and
  its mode -m is the conjugate with every spin flipped.
- Angular derivatives (unit sphere): G_+- = (d_theta -+ (m + s cos theta)/sin theta)/sqrt(2) map spin s
  to s +- 1 and are *diagonal in l* (single Jacobi identities, DLMF 18.9.15-16, derived in
  spherical.md); |G_+ Phi_l| = sqrt((l-s)(l+s+1)/2), |G_- Phi_l| = sqrt((l+s)(l-s+1)/2).
  In 3D the gradient adds the connection terms of the frame (spherical.md): nothing is divided by
  sin(theta), and 1/r only appears in the r-multiplied shell rows or inside the ball's regular
  operators D_+ = d_r - k/r (k -> k+1) and D_- = d_r + (k+1)/r (k -> k-1).
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import scipy.sparse as sp
from scipy.special import xlogy

from ..errors import OptionError
from . import ultraspherical as us
from .jacobi_family import _nodes as _gauss_jacobi
from .jacobi_family import gauss_legendre
from .polar import Annulus
from .spin import SPIN, spin_of  # noqa: F401 (re-export)
from .polar_radial import _check_lift_index, _conv_a, _jacobi_z, jacobi_values, log_jacobi_norms

__all__ = ["SphereHarmonics", "ShellOps", "BallOps", "log_jacobi_norm", "regularity_Q", "cross_table",
           "SPIN", "spin_of"]



def log_jacobi_norm(a: float, b: float, n: int) -> float:
    """log s_n, s_n^2 = int_{-1}^{1} ((1-x)/2)^a ((1+x)/2)^b P_n^(a,b)(x)^2 dx
    = 2/(2n+a+b+1) Gamma(n+a+1) Gamma(n+b+1) / (Gamma(n+a+b+1) n!) (polar_radial.log_jacobi_norms)."""
    return float(log_jacobi_norms(a, b, int(n) + 1)[int(n)])


def _norms(a: float, b: float, size: int) -> np.ndarray:
    return np.exp(log_jacobi_norms(a, b, size)).astype(float)


# ----------------------------------------------------------------------------------------------
# spin-weighted spherical harmonics
# ----------------------------------------------------------------------------------------------

class SphereHarmonics:
    """Spin-weighted spherical harmonics of degree l <= Lmax (module doc): values on colatitude
    nodes, Gauss-Legendre grid and projection, the diagonal angular derivatives G_+-, and products
    with axisymmetric (m = 0) angular coefficients (couplings l_in -> l_out, computed by exact
    Gauss quadrature and kept inside the triangle |l_out - l_in| <= l_c by structure)."""

    def __init__(self, Lmax: int) -> None:
        self.L = int(Lmax)
        self._cache: dict = {}

    def lmin(self, m: int, s: int) -> int:
        """Smallest degree of the spin-``s`` harmonics of order ``m``: max(|m|, |s|)."""
        return max(abs(int(m)), abs(int(s)))

    def valid(self, m: int, s: int) -> np.ndarray:
        """Boolean over l = 0..Lmax: is Phi_l^{m,s} part of the basis."""
        return np.arange(self.L + 1) >= self.lmin(m, s)

    # values --------------------------------------------------------------------------------
    def values(self, m: int, s: int, x: np.ndarray, L: int | None = None) -> np.ndarray:
        """(len(x), L+1) array of Phi_l^{m,s}(x), l = 0..L (zero columns for l < max(|m|,|s|));
        the envelope and the normalization are applied in log space (no overflow at large m)."""
        L = self.L if L is None else int(L)
        x = np.asarray(x, float)
        out = np.zeros((x.size, L + 1))
        l0 = self.lmin(m, s)
        N = L + 1 - l0
        if N <= 0:
            return out
        a, b = abs(m + s), abs(m - s)
        logenv = 0.5 * (xlogy(a, (1 - x) / 2) + xlogy(b, (1 + x) / 2))
        lns = log_jacobi_norms(a, b, N)                  # long double
        V = jacobi_values(N, a, b, x, np.exp(logenv - lns[0]))
        V *= np.exp(lns[0] - lns).astype(float)[None, :]
        out[:, l0:] = V
        return out

    def grid(self, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(theta, x = cos theta, w): M Gauss-Legendre nodes, theta ascending (north pole first)."""
        key = ("grid", M)
        if key not in self._cache:
            x, w = gauss_legendre(M)
            x, w = x[::-1].copy(), w[::-1].copy()
            out = (np.arccos(x), x, w)
            for a in out:              # shared by every later caller (sim.coordinate("theta"),
                a.setflags(write=False)  # data callables): read-only (audit state)
            self._cache[key] = out
        return self._cache[key]

    def backward(self, m: int, s: int, M: int) -> np.ndarray:
        """(M, L+1): synthesis matrix on the M-point grid."""
        key = ("bwd", m, s, M)
        if key not in self._cache:
            self._cache[key] = self.values(m, s, self.grid(M)[1])
        return self._cache[key]

    def forward(self, m: int, s: int, M: int) -> np.ndarray:
        """(L+1, M) = B^T W: orthogonal projection (exact for polynomial integrands of degree
        <= 2M - 1 in x, i.e. products of band-limited fields on the dealiased grid)."""
        key = ("fwd", m, s, M)
        if key not in self._cache:
            self._cache[key] = (self.backward(m, s, M) * self.grid(M)[2][:, None]).T.copy()
        return self._cache[key]

    # derivatives -----------------------------------------------------------------------------
    def G(self, sigma: int, m: int, s: int) -> np.ndarray:
        """Diagonal of G_sigma (unit sphere, factor 1/sqrt(2) included) from spin s to s + sigma, over
        l = 0..Lmax (0 where the source or the target function does not exist)."""
        key = ("G", sigma, m, s)
        if key not in self._cache:
            if sigma == -1:                      # G_-^{(m,s)} = G_+^{(-m,-s)} (Phi^{-m,-s} = Phi^{m,s})
                self._cache[key] = self.G(+1, -m, -s)
            elif sigma != 1:
                raise ValueError("sigma must be +1 or -1")
            else:
                self._cache[key] = np.array([_raise(m, s, ell) for ell in range(self.L + 1)])
        return self._cache[key]

    # axisymmetric angular coefficients ---------------------------------------------------------
    def coupling(self, m: int, s_in: int, s_c: int, l_c: int) -> sp.csr_matrix:
        """(L+1, L+1) matrix I[l_out, l_in] = int Phi_{l_out}^{m, s_in+s_c} Phi_{l_c}^{0, s_c}
        Phi_{l_in}^{m, s_in} dx: multiplication of a spin-s_in component of mode m by the m = 0,
        spin-s_c harmonic of degree l_c. Exact Gauss quadrature; entries outside the triangle
        |l_out - l_in| <= l_c (zero by the Clebsch-Gordan selection rule) are not stored."""
        key = ("cpl", m, s_in, s_c, l_c)
        if key not in self._cache:
            nq = self.L + (l_c + 1) // 2 + 3
            x, w = gauss_legendre(nq)
            Po = self.values(m, s_in + s_c, x)
            Pc = self.values(0, s_c, x, L=max(l_c, abs(s_c)))[:, l_c]
            Pi = self.values(m, s_in, x)
            D = Po.T @ ((w * Pc)[:, None] * Pi)
            lo, li = np.meshgrid(np.arange(self.L + 1), np.arange(self.L + 1), indexing="ij")
            D[np.abs(lo - li) > l_c] = 0.0
            self._cache[key] = sp.csr_matrix(D)
        return self._cache[key]

    def explain(self) -> str:
        """Degree, angular functions, grid and derivative structure of the spin-weighted
        harmonics."""
        return (f"spin-weighted spherical harmonics, Lmax = {self.L}: Phi_l^(m,s)(cos theta) = "
                "sin(theta/2)^|m+s| cos(theta/2)^|m-s| P_n^(|m+s|,|m-s|)(cos theta) / s_n, "
                "l >= max(|m|,|s|); Gauss-Legendre grid; angular derivatives diagonal in l "
                "(Vasil et al. 2019)")


def _raise(m: int, s: int, ell: int) -> float:
    """G_+ = (d_theta - (m + s x)/sin theta)/sqrt(2) on the normalized Phi_l^{m,s}: one Jacobi identity
    per sign pattern (spherical.md), the image being c Phi_l^{m,s+1}."""
    a, b = abs(m + s), abs(m - s)
    n = ell - max(abs(m), abs(s))
    a2, b2 = abs(m + s + 1), abs(m - s - 1)
    n2 = ell - max(abs(m), abs(s + 1))
    if n < 0 or n2 < 0:
        return 0.0
    if m + s >= 0 and m - s > 0:        # (1+x)P' + bP = (n+b) P_n^(a+1,b-1)
        c = -(n + b)
    elif m + s >= 0:                    # P' = (n+a+b+1)/2 P_{n-1}^(a+1,b+1)
        c = -(n + a + b + 1)
    elif m - s <= 0:                    # (1-x)P' - aP = -(n+a) P_n^(a-1,b+1)
        c = n + a
    else:                               # d/dx[(1-x)^a(1+x)^b P] = -2(n+1)(..)^(a-1,b-1) P_{n+1}^(a-1,b-1)
        c = n + 1
    return c * float(np.exp(log_jacobi_norms(a2, b2, n2 + 1)[n2] - log_jacobi_norms(a, b, n + 1)[n])) / sqrt(2.0)


def regularity_Q(H: SphereHarmonics, m: int, ell: int) -> np.ndarray:
    """(3, 3) unitary matrix Q with u_spin = Q u_reg for a vector of degree l (ball): columns are the
    vector spherical harmonics of regularity l-1 (index 0), l+1 (index 1) and l (toroidal, index 2),
    written in spin components (-, +, r):
        q_{l-1} = (g_-, g_+, l) / sqrt(l(2l+1)),   q_{l+1} = (g_-, g_+, -(l+1)) / sqrt((l+1)(2l+1)),
        q_l     = (-i g_-, i g_+, 0) / sqrt(l(l+1)),     g_+- = G_+-(l; m, s=0) (unit sphere),
    the directions of grad(r^l Y), grad(r^-(l+1) Y) and r x grad Y. At l = 0 only q_{l+1} = -e_r exists
    (the other columns are zero)."""
    gm, gp = H.G(-1, m, 0)[ell], H.G(1, m, 0)[ell]
    Q = np.zeros((3, 3), dtype=complex)
    if ell >= 1:
        Q[:, 0] = np.array([gm, gp, ell]) / np.sqrt(ell * (2 * ell + 1))
        Q[:, 2] = np.array([-1j * gm, 1j * gp, 0]) / np.sqrt(ell * (ell + 1))
    Q[:, 1] = np.array([gm, gp, -(ell + 1)]) / np.sqrt((ell + 1) * (2 * ell + 1))
    return Q


def cross_table() -> dict:
    """Cross product in the spin basis {e_-, e_+, e_r} (index 0, 1, 2), right-handed (e_r, e_theta,
    e_phi): e_+ x e_- = i e_r, e_r x e_+ = i e_+, e_r x e_- = -i e_-. Returns {(i, j): (k, c)} with
    e_i x e_j = c e_k, applied to the *coefficients*: (v x w)_k += c' v_i' w_j' (spherical.md)."""
    # components: (v x w)_r = i (v_+ w_- - v_- w_+); (v x w)_+ = i (v_r w_+ - v_+ w_r);
    #             (v x w)_- = -i (v_r w_- - v_- w_r)
    return {(1, 0): (2, 1j), (0, 1): (2, -1j), (2, 1): (1, 1j), (1, 2): (1, -1j),
            (2, 0): (0, -1j), (0, 2): (0, 1j)}


# ----------------------------------------------------------------------------------------------
# shell: Chebyshev T in t, r = c + h t, rows multiplied by a declared power of r
# ----------------------------------------------------------------------------------------------

class ShellOps:
    """Radial operators of the spherical shell. A held quantity at (level, p) is q = X / r^p with X in
    C^(level) of t (ultraspherical); every operator returns the r-multiplied form (as polar.Annulus
    and Dedalus's r^-k Jacobi(k-1/2, k-1/2) derivative bases): d_r -> (r X' - p X, level+1, p+1),
    1/r -> (X, level, p+1). A tau lifted into C^(lam) is held at p = lam (the lifted function is
    r^-lam C^(lam)_{N+n}, sphere_compiler._lift): a row of larger r-power multiplies it by
    r^(p - lam) like any other term."""

    kind = "shell"

    def __init__(self, N: int, ri: float, ro: float) -> None:
        self.an = Annulus(N, ri, ro)
        self.N, self.ri, self.ro = int(N), float(ri), float(ro)
        self._cache: dict = {}

    def size(self, ell: int) -> int:
        """Number of radial coefficients (N for every degree)."""
        return self.N

    def _memo(self, key, fn):
        """Operators are rebuilt for every group otherwise (compile time dominated by them)."""
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def dr(self, level: int, p: int, size: int):
        """r^(p+1) d_r (X / r^p) = r X' - p X."""
        return self._memo(("dr", level, p), lambda: self.an.G(1, 0, level, p))

    def over_r(self, level: int, p: int, size: int):
        """Division by r: identity on the coefficients, the row's r power raised by one (rows
        are r-multiplied)."""
        return sp.identity(self.N, format="csr"), level, p + 1

    def convert(self, lev_from: int, lev_to: int, size: int) -> sp.csr_matrix:
        """Conversion from level ``lev_from`` to ``lev_to`` (ValueError downwards; cached)."""
        if lev_to < lev_from:
            raise ValueError(f"cannot convert level {lev_from} down to {lev_to}")
        return self._memo(("conv", lev_from, lev_to),
                          lambda: us.conversion_chain(self.N, lev_from, lev_to).tocsr())

    def multiply_r(self, level: int, j: int, size: int) -> sp.csr_matrix:
        """Multiplication by r^j at ``level`` (cached)."""
        return self._memo(("mulr", level, j), lambda: self.an.multiplier(level, j).tocsr())

    def ncc(self, g_T: np.ndarray, level: int, size: int) -> sp.csr_matrix:
        """Multiplication by the radial NCC of Chebyshev coefficients ``g_T`` at ``level`` (a
        constant: scaled identity; cached)."""
        g = np.asarray(g_T)
        if g.size == 1:            # constant in r (e.g. e_z): exact, no series to resolve
            return (complex(g.ravel()[0]) if np.iscomplexobj(g) else float(g.ravel()[0])) * sp.identity(self.N, format="csr")
        return self._memo(("ncc", level, g.tobytes(), g.dtype.str),
                          # coefficients already judged and reported by the plan's NCC check
                          lambda: us.multiplication(g, self.N, level, name="ncc", exact=True)[0])

    def point_row(self, level: int, p: int, r0: float, deriv: int, size: int) -> np.ndarray:
        """Value or d/dr of q = r^-p X at r = r0 (ri <= r0 <= ro)."""
        t0 = (r0 - self.an.c) / self.an.h
        v = us.basis_values(self.N, level, t0)
        if deriv == 0:
            return v / r0 ** p
        if deriv != 1:
            raise ValueError("only the first radial derivative is available at a point")
        return (us.basis_values(self.N, level, t0, deriv=1) / self.an.h - p * v / r0) / r0 ** p

    def integral_row(self, level: int, p: int, size: int) -> np.ndarray:
        """int q r^2 dr = h int X(t) (c + h t)^(2-p) dt by Gauss-Legendre on N + 4 nodes: exact for
        p <= 2 (polynomial integrand); for p > 2 the integrand is rational and the rule converges
        geometrically (error ~ rho^(-2N), rho >= 3.9 for ri/ro >= 0.35): rounding in practice, not
        exact (SC6)."""
        key = ("int", level, p)
        if key not in self._cache:
            t, w = gauss_legendre(self.N + 4)
            V = np.array([us.basis_values(self.N, level, ti) for ti in t])
            r = self.an.c + self.an.h * t
            self._cache[key] = self.an.h * ((w * r ** (2 - p)) @ V)
        return self._cache[key]

    def lift_col(self, level: int, n: int, size: int) -> np.ndarray:
        """Tau column: the mode N + n (-N <= n < 0) of the level-``level`` basis. Checked like the
        annulus (polar_radial.AnnulusOps.lift_col): n >= 0 indexed past the end (bare
        IndexError) and n < -N wrapped around to another mode silently."""
        _check_lift_index(n, self.N)
        if self.N + n < 0:          # a negative index would wrap around to another mode
            raise OptionError(f"shell: lift index n = {n} below -N = {-self.N}",
                              fix=f"use -{self.N} <= n <= -1")
        col = np.zeros(self.N)
        col[self.N + n] = 1.0
        return col

    def const_col(self) -> np.ndarray:
        """Coefficients (level 0, p 0) of the constant function 1: T_0."""
        col = np.zeros(self.N)
        col[0] = 1.0
        return col

    # transforms ----------------------------------------------------------------------------------
    def grid(self, M: int) -> np.ndarray:
        """Radial Gauss-Chebyshev grid of ``M`` points in [ri, ro]."""
        t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
        return self.an.c + self.an.h * t

    def backward(self, M: int, what: str = "val") -> np.ndarray:
        """(M, N) synthesis of T_n on the Gauss-Chebyshev nodes: 'val' (values), 'dr' (d/dr),
        'over_r' (values / r; r >= ri > 0 on the nodes, exact arithmetic)."""
        key = ("bwd", M, what)
        if key not in self._cache:
            t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
            if what == "dr":
                V = np.array([us.basis_values(self.N, 0, ti, deriv=1) for ti in t]) / self.an.h
            else:
                V = np.cos(np.outer(np.arccos(t), np.arange(self.N)))
                if what == "over_r":
                    V = V / (self.an.c + self.an.h * t)[:, None]
            self._cache[key] = V
        return self._cache[key]

    def forward(self, M: int) -> np.ndarray:
        """(N, M): Chebyshev interpolation on the M Gauss-Chebyshev nodes, truncated to N modes."""
        key = ("fwd", M)
        if key not in self._cache:
            t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
            V = np.cos(np.outer(np.arccos(t), np.arange(M)))
            self._cache[key] = np.linalg.inv(V)[: self.N]
        return self._cache[key]

    def explain(self) -> str:
        """Radii, size, radial basis, row multiplication and grid of the shell, one line."""
        return (f"shell [{self.ri:g}, {self.ro:g}], N={self.N}: Chebyshev T in t for every spin "
                "component, rows multiplied by r^p (declared, p = largest r-power of the row's "
                "terms), Gauss-Chebyshev radial grid")


# ----------------------------------------------------------------------------------------------
# ball: rho^k Q_n^(a, k+1/2)(z), z = 2 rho^2 - 1, per regularity component (k = l + sigma)
# ----------------------------------------------------------------------------------------------

class BallOps:
    """Regular radial operators of the ball of radius R (Vasil et al. 2019, Part I §6; Lecoanet et al.
    2019). A regularity component of index k = l + sigma is rho^k X(z) with X expanded in the
    normalized Jacobi polynomials Q_n^(a, k+1/2) (a = derivative level), n < N - floor(l/2) for every
    component of degree l (Dedalus's truncation). Single-diagonal derivatives (DLMF 18.9.15-16):
        D_+ = d_r - k/r     : P_n^(a,b) rho^k -> 2(n+a+b+1) P_{n-1}^(a+1,b+1) rho^(k+1) / R
        D_- = d_r + (k+1)/r : P_n^(a,b) rho^k -> 2(n+b)     P_n^(a+1,b-1)     rho^(k-1) / R
    (b = k + 1/2), and the Laplacian D_-(k+1) D_+(k) keeps k. The centre needs no condition."""

    kind = "ball"

    def __init__(self, N: int, R: float) -> None:
        self.N, self.R = int(N), float(R)
        self._cache: dict = {}

    def size(self, ell: int) -> int:
        """Number of radial coefficients of degree ``ell``: N - floor(|ell|/2)."""
        return max(self.N - abs(int(ell)) // 2, 0)

    def _s(self, level: int, k: int, size: int) -> np.ndarray:
        key = ("norm", level, k, size)
        if key not in self._cache:
            self._cache[key] = _norms(level, k + 0.5, size)
        return self._cache[key]

    def _scaled(self, M, lev_in: int, k_in: int, lev_out: int, k_out: int, size: int):
        return (sp.diags(self._s(lev_out, k_out, size)) @ M
                @ sp.diags(1.0 / self._s(lev_in, k_in, size))).tocsr()

    def up(self, k: int, level: int, size: int):
        """D_+(k) = d_r - k/r: component k -> k+1, level -> level+1."""
        b = k + 0.5
        n = np.arange(1, size, dtype=float)
        M = sp.csr_matrix((2.0 * (n + level + b + 1) / self.R, (n.astype(int) - 1, n.astype(int))),
                          shape=(size, size))
        return self._scaled(M, level, k, level + 1, k + 1, size)

    def down(self, k: int, level: int, size: int):
        """D_-(k) = d_r + (k+1)/r: component k -> k-1 (k >= 1), level -> level+1."""
        if k < 1:
            raise ValueError("D_- needs k >= 1")
        b = k + 0.5
        n = np.arange(size, dtype=float)
        M = sp.diags(2.0 * (n + b) / self.R, 0, shape=(size, size), format="csr")
        return self._scaled(M, level, k, level + 1, k - 1, size)

    def lap(self, k: int, level: int, size: int):
        """Laplacian of a component k (scalar degree k, or a vector regularity component):
        D_-(k+1) D_+(k), k -> k, level -> level+2."""
        return (self.down(k + 1, level + 1, size) @ self.up(k, level, size)).tocsr()

    def convert(self, k: int, lev_from: int, lev_to: int, size: int) -> sp.csr_matrix:
        """Conversion of regularity component ``k`` from level ``lev_from`` to ``lev_to``
        (ValueError downwards; cached)."""
        if lev_to < lev_from:
            raise ValueError(f"cannot convert level {lev_from} down to {lev_to}")
        key = ("conv", k, lev_from, lev_to, size)
        if key not in self._cache:
            out = sp.identity(size, format="csr")
            for a in range(lev_from, lev_to):
                out = _conv_a(size, a, k + 0.5) @ out
            self._cache[key] = self._scaled(out, lev_from, k, lev_to, k, size)
        return self._cache[key]

    def ncc(self, g_leg: np.ndarray, k: int, level: int, size: int) -> sp.csr_matrix:
        """Multiplication by a regular radial scalar g(z) = sum_j g_j P_j(z) (Legendre coefficients)
        on the level-``level`` space of component k (Jacobi matrix recurrence, extended then
        truncated, as polar_radial.DiskOps.ncc)."""
        deg = len(g_leg) - 1
        Ne = size + deg + 1
        Z = _jacobi_z(Ne, level, k + 0.5)
        I = sp.identity(Ne, format="csr")
        P0, P1 = I, Z
        acc = g_leg[0] * P0 + (g_leg[1] * P1 if deg >= 1 else 0 * I)
        for j in range(1, deg):
            P0, P1 = P1, ((2 * j + 1) * (Z @ P1) - j * P0) / (j + 1)
            acc = acc + g_leg[j + 1] * P1
        return self._scaled(sp.csr_matrix(acc)[:size, :size], level, k, level, k, size)

    # rows -------------------------------------------------------------------------------------
    def point_row(self, k: int, level: int, r0: float, deriv: int, size: int) -> np.ndarray:
        """Value (deriv 0) or d/dr (deriv 1) of rho^k X(z) at r = r0."""
        rho = r0 / self.R
        z0 = np.array([2 * rho * rho - 1])
        b = k + 0.5
        s = self._s(level, k, size)
        P = jacobi_values(size, level, b, z0)[0]
        if deriv == 0:
            return rho ** k * P / s
        if deriv != 1:
            raise ValueError("only the first radial derivative is available at a point")
        dP = np.zeros(size)
        n = np.arange(1, size)
        dP[1:] = (n + level + b + 1) / 2 * jacobi_values(size - 1, level + 1, b + 1, z0)[0]
        t1 = k * rho ** (k - 1) * P if k else 0.0 * P
        return (t1 + 4.0 * rho ** (k + 1) * dP) / self.R / s

    def integral_row(self, level: int, size: int) -> np.ndarray:
        """int_0^R X(z) r^2 dr of a k = 0 component (the angular factor is applied by the caller):
        R^3 / (4 sqrt 2) int (1+z)^(1/2) X dz, Gauss-Jacobi (0, 1/2)."""
        z, w = _gauss_jacobi(self.N + level + 2, 0.0, 0.5)
        return self.R ** 3 / (4 * np.sqrt(2.0)) * (w @ jacobi_values(size, level, 0.5, z)) / self._s(level, 0, size)

    def lift_col(self, level: int, n: int, size: int) -> np.ndarray:
        """Tau column: the mode size + n (n < 0) of the level-``level`` basis of a degree-l
        component (size = N - floor(l/2)). A degree whose truncated size is below -n gets a zero
        column (no such mode), as on the disk: the plan's rank report flags the singular block."""
        _check_lift_index(n, self.N)
        col = np.zeros(size)
        if size + n >= 0:
            col[size + n] = 1.0
        return col

    def const_col(self, size: int) -> np.ndarray:
        """Normalized level-0 coefficients of the constant 1 (k = 0): Q_0 = 1 / s_0."""
        col = np.zeros(size)
        col[0] = self._s(0, 0, 1)[0]
        return col

    # transforms ----------------------------------------------------------------------------------
    def nodes(self, M: int) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Jacobi (0, 1/2) nodes z (ascending) and weights (weight (1+z)^(1/2)), Newton-refined
        in long double (scipy's roots_jacobi nodes are off by ~4e-14 relative at
        M = 32, which left |F B - I| = 6e-14 and a 60x larger round-trip error than Dedalus')."""
        key = ("nodes", M)
        if key not in self._cache:
            self._cache[key] = _gauss_jacobi(M, 0.0, 0.5)
        return self._cache[key]

    def grid(self, M: int) -> np.ndarray:
        """Radial Gauss-Jacobi grid of ``M`` points, r = R sqrt((1 + z)/2)."""
        z, _ = self.nodes(M)
        return self.R * np.sqrt((1 + z) / 2)

    def backward(self, k: int, M: int, size: int, what: str = "val") -> np.ndarray:
        """(M, size): 'val' rho^k Q_n(z), 'dr' its r-derivative, 'over_r' value / r, on the nodes."""
        key = ("bwd", k, M, size, what)
        if key not in self._cache:
            z, _ = self.nodes(M)
            rho = np.sqrt((1 + z) / 2)
            b = k + 0.5
            s = self._s(0, k, size)
            P = jacobi_values(size, 0, b, z)
            if what == "val":
                V = rho[:, None] ** k * P
            elif what == "over_r":
                V = rho[:, None] ** (k - 1) * P / self.R
            elif what == "dr":
                dP = np.zeros_like(P)
                if size > 1:
                    n = np.arange(1, size)
                    dP[:, 1:] = (n + b + 1) / 2 * jacobi_values(size - 1, 1, b + 1, z)
                t1 = k * rho[:, None] ** (k - 1) * P if k else 0.0 * P
                V = (t1 + 4.0 * rho[:, None] ** (k + 1) * dP) / self.R
            else:
                raise ValueError(what)
            self._cache[key] = V / s[None, :]
        return self._cache[key]

    def forward(self, k: int, M: int, size: int) -> np.ndarray:
        """(size, M): orthogonal projection on span{rho^k Q_n^(0, k+1/2)} in L2(r^2 dr), by
        Gauss-Jacobi quadrature (exact for the products of band-limited fields on the grid)."""
        key = ("fwd", k, M, size)
        if key not in self._cache:
            _, w = self.nodes(M)
            self._cache[key] = (self.backward(k, M, size) * (w / np.sqrt(2.0))[:, None]).T.copy()
        return self._cache[key]

    def explain(self) -> str:
        """Radius, size, radial basis, grid and operators of the ball, one line."""
        return (f"ball R={self.R:g}, N={self.N}: regularity component k = rho^k sum_n c_n "
                "Q_n^(a, k+1/2)(2 rho^2 - 1) (normalized Jacobi, 3D Zernike), n < N - floor(l/2); "
                "Gauss-Jacobi(0, 1/2) radial grid; D_+-, Laplacian single-diagonal (Vasil et al. 2019)")
