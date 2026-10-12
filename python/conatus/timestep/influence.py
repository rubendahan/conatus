"""Kleiser–Schumann influence-matrix Stokes solve for one channel Fourier mode (T4).

Solves, for one mode exp(i(kx x + kz z)) on y in (-1, 1),

    c u - (1/Re)(D^2 - k^2) u + grad p = r,   div u = 0,   u(+-1) = 0,        k^2 = kx^2 + kz^2,

(``c`` = 1/(dt a_ii) in an implicit stage) by decoupled scalar problems instead of the coupled
generalized-tau system of the plan:

  1. p solves (D^2 - k^2) p = div r with Dirichlet data p(+-1) = lambda_{+-} (unknown);
  2. v solves (c + k^2/Re - D^2/Re) v = r_y - D p, v(+-1) = 0;
  3. lambda is fixed by D v(+-1) = 0 (continuity at the walls, since u = w = 0 there), a 2x2
     *influence matrix* built once per mode from the two homogeneous problems (r = 0,
     lambda = e_1, e_2);
  4. u, w solve the same Helmholtz problem with r_x - i kx p, r_z - i kz p.

Discretization: Chebyshev T coefficients, ultraspherical operators (Olver–Townsend) with the two
boundary rows replacing the last two rows of each C^(2) equation (tau). No tau-error correction
is applied: the interior divergence is therefore *not* zero to roundoff but O(last tau
coefficients) — measured and reported, never hidden (T4).
The mean mode k = 0 (no pressure coupling) and the pressure gauge are outside this solver."""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from ..discretize import ultraspherical as us

__all__ = ["KleiserSchumannStokes"]


class KleiserSchumannStokes:
    """Per-mode influence-matrix Stokes solver (dense LU per scalar operator, CPU reference)."""

    def __init__(self, N: int, Re: float, c: float) -> None:
        self.N, self.Re, self.c = int(N), float(Re), float(c)
        N = self.N
        self.S = (us.conversion(N, 1) @ us.conversion(N, 0)).toarray()      # T -> C^(2)
        self.D1 = us.differentiation(N, 0, 1).toarray()                     # T -> C^(1)
        self.S1 = us.conversion(N, 1).toarray()                             # C^(1) -> C^(2)
        self.D2 = us.differentiation(N, 0, 2).toarray()                     # T -> C^(2)
        self.bc = np.vstack([us.evaluation_row(N, 0, -1.0), us.evaluation_row(N, 0, 1.0)])
        self.dbc = np.vstack([us.evaluation_row(N, 0, -1.0, 1), us.evaluation_row(N, 0, 1.0, 1)])
        self._modes: dict = {}

    def explain(self) -> str:
        return (f"KleiserSchumannStokes: N = {self.N}, Re = {self.Re:g}, c = {self.c:g}; per mode "
                "1 Poisson + 3 Helmholtz Chebyshev-tau solves (2 boundary rows each) and a 2x2 "
                "influence matrix for D v(+-1) = 0; no tau-error correction (interior divergence "
                "reported, not zero)")

    def _tau(self, A: np.ndarray) -> np.ndarray:
        """Replace the last two rows of a C^(2) operator by the Dirichlet rows at y = -1, +1."""
        return np.vstack([A[: self.N - 2], self.bc])

    def _rhs(self, b: np.ndarray, left: complex = 0.0, right: complex = 0.0) -> np.ndarray:
        return np.concatenate([b[: self.N - 2], [left, right]])

    def _mode(self, kx: float, kz: float):
        key = (kx, kz)
        if key not in self._modes:
            k2 = kx * kx + kz * kz
            P = sla.lu_factor(self._tau(self.D2 - k2 * self.S))
            H = sla.lu_factor(self._tau((self.c + k2 / self.Re) * self.S - self.D2 / self.Re))
            # homogeneous problems: p with p(-1), p(+1) = e_j, r = 0 -> v -> D v at the walls
            Ainf = np.zeros((2, 2), complex)
            hom = []
            for j in range(2):
                e = np.zeros(2)
                e[j] = 1.0
                p = sla.lu_solve(P, self._rhs(np.zeros(self.N), *e))
                v = sla.lu_solve(H, self._rhs(-self.S1 @ (self.D1 @ p)))
                Ainf[:, j] = self.dbc @ v
                hom.append((p, v))
            self._modes[key] = (P, H, sla.lu_factor(Ainf), hom)
        return self._modes[key]

    def solve(self, kx: float, kz: float, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """r: (3, N) complex Chebyshev-T coefficients of the mode's forcing (x, y, z components).
        Returns u (3, N) and p (N) in Chebyshev-T coefficients."""
        if kx == 0 and kz == 0:
            raise ValueError("mode (0, 0): no pressure coupling, not handled by the influence solver")
        P, H, Ainf, hom = self._mode(kx, kz)
        r = np.asarray(r, complex)
        divr = self.S @ (1j * kx * r[0] + 1j * kz * r[2]) + self.S1 @ (self.D1 @ r[1])
        p0 = sla.lu_solve(P, self._rhs(divr))
        v0 = sla.lu_solve(H, self._rhs(self.S @ r[1] - self.S1 @ (self.D1 @ p0)))
        lam = sla.lu_solve(Ainf, -(self.dbc @ v0))
        p = p0 + lam[0] * hom[0][0] + lam[1] * hom[1][0]
        v = v0 + lam[0] * hom[0][1] + lam[1] * hom[1][1]
        u = sla.lu_solve(H, self._rhs(self.S @ (r[0] - 1j * kx * p)))
        w = sla.lu_solve(H, self._rhs(self.S @ (r[2] - 1j * kz * p)))
        return np.array([u, v, w]), p

    def divergence(self, kx: float, kz: float, u: np.ndarray) -> np.ndarray:
        """Chebyshev-T coefficients (in C^(1)) of div u for the mode."""
        S0 = us.conversion(self.N, 0).toarray()                            # T -> C^(1)
        return S0 @ (1j * kx * u[0] + 1j * kz * u[2]) + self.D1 @ u[1]
