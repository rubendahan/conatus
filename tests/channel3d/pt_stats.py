"""Plane-averaged one-point statistics of the poloidal–toroidal channel, computed spectrally.

For real Fourier series in x and z (coefficients of cos(k x), -sin(k x); RealFourier layout:
index 2n = cos, 2n+1 = -sin), the tensor-product functions are orthogonal on the periodic box
with norms w(n_x) w(n_z), w(0) = 1, w(n >= 1) = 1/2. Hence the xz-average of a product of two
fluctuations is  <f g>(y) = sum_{i,k} w_i w_k f_{i k}(y) g_{i k}(y)  (Parseval), with f_{ik}(y)
the Chebyshev series evaluated at the requested y points (evaluation matrices T_n(y) and
T_n'(y), not a differentiation matrix). Fluctuations from the potentials, (kx, kz) != (0, 0):
    u' = dx dy phi + dz psi,   v' = -lap_h phi,   w' = dz dy phi - dx psi.
The mean flow is U(y) (and W(y)) exactly (phi = psi = 0 in the (0, 0) group).
Works with NumPy or CuPy arrays (``xp``). Checked against the grid evaluation of the reference
evaluator in tests/channel3d/test_pt_stats.py. Used by run_retau180_long.py."""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C


def cheb_eval_matrices(Ny: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E0[n, j] = T_n(y_j), E1[n, j] = T_n'(y_j) for y in [-1, 1] (channel half-height 1)."""
    E0 = np.empty((Ny, len(y)))
    E1 = np.empty((Ny, len(y)))
    for n in range(Ny):
        e = np.zeros(Ny)
        e[n] = 1.0
        E0[n] = C.chebval(y, e)
        E1[n] = C.chebval(y, C.chebder(e)) if n else 0.0
    return E0, E1


class PTStats:
    """Spectral plane averages of the PT channel at fixed y points.

    ``Nx, Ny, Nz``: coefficient sizes; ``Lx, Lz``: periods; ``y``: output points in [-1, 1]."""

    def __init__(self, Nx: int, Ny: int, Nz: int, Lx: float, Lz: float, y: np.ndarray, xp=np):
        self.xp, self.y = xp, np.asarray(y, dtype=float)
        E0, E1 = cheb_eval_matrices(Ny, self.y)
        self.E0, self.E1 = xp.asarray(E0), xp.asarray(E1)

        def kpair(N, L):
            n = np.arange(N) // 2
            return 2 * np.pi * n / L, np.where(n == 0, 1.0, 0.5)
        kx, wx = kpair(Nx, Lx)
        kz, wz = kpair(Nz, Lz)
        self.kx = xp.asarray(kx)[:, None, None]
        self.kz = xp.asarray(kz)[None, None, :]
        self.w = xp.asarray(wx[:, None] * wz[None, :])[:, None, :]      # (Nx, 1, Nz)
        mask = np.ones((Nx, 1, Nz))
        mask[:2, :, :2] = 0.0                                             # drop the (0, 0) group
        self.mask = xp.asarray(mask)

    def _dfourier(self, c, ax: int):
        """d/dx (ax = 0) or d/dz (ax = 2) of a RealFourier (cos, -sin) coefficient array:
        (a, b) -> k (-b, a), as timestep/fused.py _diff."""
        xp = self.xp
        cm = xp.moveaxis(c, ax, 0)
        sw = xp.empty_like(cm)
        sw[0::2] = -cm[1::2]
        sw[1::2] = cm[0::2]
        k = self.kx if ax == 0 else self.kz
        return k * xp.moveaxis(sw, 0, ax)

    def fluctuations(self, phi, psi):
        """(u', v', w') as (Nx, len(y), Nz) Fourier-coefficient arrays evaluated at y."""
        xp = self.xp
        phi0 = xp.einsum("inj,nm->imj", phi, self.E0) * self.mask
        phi1 = xp.einsum("inj,nm->imj", phi, self.E1) * self.mask
        psi0 = xp.einsum("inj,nm->imj", psi, self.E0) * self.mask
        u = self._dfourier(phi1, 0) + self._dfourier(psi0, 2)
        v = (self.kx**2 + self.kz**2) * phi0
        w = self._dfourier(phi1, 2) - self._dfourier(psi0, 0)
        return u, v, w

    def moments(self, phi, psi) -> dict:
        """xz-averaged second moments at y: uu, vv, ww, uv (each an array of len(y))."""
        u, v, w = self.fluctuations(phi, psi)
        W = self.w

        def avg(a, b):
            return (W * a * b).sum(axis=(0, 2))
        return dict(uu=avg(u, u), vv=avg(v, v), ww=avg(w, w), uv=avg(u, v))

    def mean(self, U) -> np.ndarray:
        """Mean profile U(y) at the output points from its Chebyshev coefficients."""
        return self.xp.asarray(U).reshape(-1) @ self.E0
