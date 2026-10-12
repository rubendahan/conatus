"""Plane Couette flow in the poloidal–toroidal IR formulation: boxes, the shift-reflect
symmetry group S of the Nagata equilibria, and the diagnostics of Gibson, Halcrow & Cvitanović
(2009, JFM 638, 243-266; arXiv:0808.3375, eq. 2.3 and Table 1).

Units: walls y = ±1 moving at U = ±1, Re = U h / nu (Waleffe 2003 "rigid-rigid" convention and
GHC 2009); laminar flow U = y.

Symmetries (GHC 2009 eq. 3.3; Waleffe 2003 eqs. 24, 26):
  s1 [u,v,w](x,y,z) = [u,v,-w](x + Lx/2, y, -z)           (shift-reflect)
  s2 [u,v,w](x,y,z) = [-u,-v,w](-x + Lx/2, -y, z + Lz/2)  (shift-rotate)
and s3 = s1 s2. On the potentials (v = -lap_h phi, eta = dz u - dx w = lap_h psi):
  s1: phi -> +phi, psi -> -psi, U -> +U, W -> -W;
  s2: phi -> -phi, psi -> -psi, U -> -U, W -> +W  (with the coordinate actions above).
The signs are checked by equivariance of the discrete step (test_couette_pt.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "channel3d"))
from ch3d_cases import channel_3d_pt  # noqa: E402

from conatus.ecs.subspace import AxisAction, DiagonalSymmetry, DiagonalSymmetryGroup  # noqa: E402

# boxes (Lx, Lz), wavenumbers alpha = 2 pi / Lx, gamma = 2 pi / Lz
BOX_GHC = (2 * np.pi / 1.14, 2 * np.pi / 2.5)        # GHC 2009 eq. 2.2, Re = 400 tables
BOX_W03 = (2 * np.pi / 0.577, 2 * np.pi / 1.150)     # Waleffe 2003 eq. 28 (fold Re_sn = 127.705)


def couette_pt(Nx, Ny, Nz, Re, box, **kw):
    """Plane Couette (walls at U = -1, +1, no pressure gradient, declared) in the PT form."""
    Lx, Lz = box
    return channel_3d_pt(Nx, Ny, Nz, Re, Lx, Lz, pressure_gradient=0.0,
                         wall_velocity=(-1.0, 1.0), **kw)


S1 = DiagonalSymmetry("s1 (shift-reflect)",
                      {"x": AxisAction(shift_half=True), "z": AxisAction(reflect=True)},
                      {"phi": 1, "psi": -1, "U": 1, "W": -1})
S2 = DiagonalSymmetry("s2 (shift-rotate)",
                      {"x": AxisAction(shift_half=True, reflect=True), "y": AxisAction(reflect=True),
                       "z": AxisAction(shift_half=True)},
                      {"phi": -1, "psi": -1, "U": -1, "W": 1})


def shift_reflect_group(solver) -> DiagonalSymmetryGroup:
    """S = {e, s1, s2, s3} acting on the solver's state vector."""
    return DiagonalSymmetryGroup(solver, [S1, S2])


# ------------------------------------------------------------------ GHC 2009 diagnostics
class CouetteDiagnostics:
    """Normalized as GHC 2009 eq. 2.3 (laminar: E = 1/6, D = I = 1):
      ||u||  = L2 norm (volume mean) of the deviation from laminar u - y ex,
      E      = (1/2) <|u_tot|^2>,  D = <|curl u_tot|^2>,  I = (1/(2 Lx Lz)) ∮ dU/dy over both walls,
    all by Parseval on the coefficients (Chebyshev integrals by Clenshaw–Curtis quadrature on a
    Gauss grid of 3 Ny points: exact for the degree 2 (Ny - 1) + 2 polynomials involved)."""

    def __init__(self, solver, box):
        from pt_stats import PTStats
        plan = solver.plan
        self.s = solver
        self.Lx, self.Lz = box
        dom = plan.domain
        self.Nx, self.Ny, self.Nz = (dom.basis(c).size for c in dom.coordsys.coords)
        M = 3 * self.Ny
        # Gauss–Chebyshev nodes with Fejér weights: exact for polynomials of degree < M
        th = np.pi * (np.arange(M) + 0.5) / M
        self.y = -np.cos(th)
        k = np.arange(1, M // 2 + 1)
        self.wq = np.array([2 / M * (1 - 2 * np.sum(np.cos(2 * k * t) / (4 * k**2 - 1)))
                            for t in th])
        self.st = PTStats(self.Nx, self.Ny, self.Nz, self.Lx, self.Lz, self.y, xp=np)
        names = [v.name for v in solver.system.variables]
        self.idx = {n: names.index(n) for n in ("phi", "psi", "U", "W")}

    def _fields(self, X):
        xs = self.s.system.scatter(X)
        h = (lambda a: a) if self.s.backend == "cpu" else self.s.xp.asnumpy
        return {n: np.asarray(h(xs[i])) for n, i in self.idx.items()}

    def __call__(self, X) -> dict:
        from numpy.polynomial import chebyshev as C
        f = self._fields(X)
        st, w = self.st, self.wq
        u, v, ww = st.fluctuations(f["phi"], f["psi"])
        W2 = st.w
        mean2 = lambda a, b: (W2 * a * b).sum(axis=(0, 2))            # noqa: E731
        Uc, Wc = f["U"].reshape(-1), f["W"].reshape(-1)
        Uy, Wy = C.chebval(self.y, Uc), C.chebval(self.y, Wc)
        dU, dW = C.chebval(self.y, C.chebder(Uc)), C.chebval(self.y, C.chebder(Wc))
        fl2 = mean2(u, u) + mean2(v, v) + mean2(ww, ww)
        vol = lambda g: 0.5 * float(np.sum(w * g))                     # (1/2) ∫_{-1}^{1}  # noqa: E731
        norm = np.sqrt(vol(fl2 + (Uy - self.y) ** 2 + Wy**2))
        E = 0.5 * vol(fl2 + Uy**2 + Wy**2)
        # vorticity of the fluctuations, spectrally: omega = curl u'
        dx, dz = (lambda a: st._dfourier(a, 0)), (lambda a: st._dfourier(a, 2))
        dy = self._dy_fluct(f)
        ox = dy["w"] - dz(v)
        oy = dz(u) - dx(ww)
        oz = dx(v) - dy["u"]
        D = vol(mean2(ox, ox) + mean2(oy, oy) + mean2(oz, oz) + dU**2 + dW**2)
        Ic = C.chebder(Uc)
        I = 0.5 * float(C.chebval(1.0, Ic) + C.chebval(-1.0, Ic))
        return dict(norm=float(norm), E=float(E), D=float(D), I=float(I))

    def _dy_fluct(self, f):
        """d/dy of u', w' at the quadrature points (T_n'' evaluation for the phi part)."""
        from numpy.polynomial import chebyshev as C
        st, Ny = self.st, self.Ny
        E1, E2 = np.empty((Ny, len(self.y))), np.empty((Ny, len(self.y)))
        for n in range(Ny):
            e = np.zeros(Ny)
            e[n] = 1.0
            E1[n] = C.chebval(self.y, C.chebder(e))
            E2[n] = C.chebval(self.y, C.chebder(e, 2))
        m = st.mask
        p2 = np.einsum("inj,nm->imj", f["phi"], E2) * m
        s1 = np.einsum("inj,nm->imj", f["psi"], E1) * m
        return dict(u=st._dfourier(p2, 0) + st._dfourier(s1, 2),
                    w=st._dfourier(p2, 2) - st._dfourier(s1, 0))
