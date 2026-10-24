"""Spherical domains (S2, shell, ball): grid <-> coefficient transforms and pseudospectral evaluation.

Layouts:

- **Coefficients** of a field on Domain(S2Coordinates(phi, theta), ...): complex array of shape
  (2,)*rank + (Nm, Lmax+1); on Domain(SphericalCoordinates(phi, theta, r), ...): (3,)*rank +
  (Nm, Lmax+1, Nr). Tensor indices hold spin components (0: u_- = (u_theta - i u_phi)/sqrt 2,
  1: u_+ = (u_theta + i u_phi)/sqrt 2, 2: u_r), except the *bulk* of a ball where they hold the
  regularity components (0: l-1, 1: l+1, 2: l, sphere_ops.regularity_Q; rank <= 1). The azimuth axis
  holds the stored wavenumbers (RealFourier: m = 0..Nphi/2-1, mode -m by conjugation with the spins
  flipped; ComplexFourier: fftfreq order without Nyquist), the second axis the degree l itself
  (entries l < max(|m|, |s|) are 0), the radial axis Chebyshev (shell) or Jacobi (ball) coefficients
  (the ball holds n < Nr - floor(l/2), zero-padded). A missing coordinate gives an axis of length 1
  (m = 0, l = 0 or a single radial value); without theta the entry is still the l = 0 *harmonic*
  coefficient (value * sqrt(2), Phi_0 = 1/sqrt(2)), so that every object shares one convention.
- **Grid values**: physical components in the coordinate order (phi, theta[, r]), shape
  (nc,)*rank + (Mphi, Mtheta[, Mr]); phi_j = 2 pi j / Mphi, theta ascending (Gauss-Legendre in
  cos theta), r ascending (Gauss-Chebyshev on the shell, Gauss-Jacobi(0, 1/2) in 2r^2 - 1 on the ball).
- **Mixed space** (internal): spin components as functions of (m, l, r-grid) -- angular spectral,
  radial physical. The evaluator's derivatives live there: angular derivatives are diagonal in l,
  radial derivatives and the 1/r factors of the frame are exact values on the radial nodes (never
  at r = 0: Gauss nodes are interior; never 1/(r + eps)).
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any

import numpy as np

from ..ir import expr as E
from ..ir.bases import BallBasis, ComplexFourier, RealFourier, ShellBasis
from ..ir.domain import Domain
from ._xp import array_module, on
from .compiler import DiscretizationError
from .curvilinear_common import spin_dot as _dot
# reality tolerances and check, shared with the other curvilinear evaluator (historical names)
from .real_fields import FUNC_REAL_TOL, REAL_TOL, STATE_REAL_TOL
from .real_fields import checked_real as _real
from .sphere_ops import BallOps, ShellOps, SphereHarmonics, cross_table, regularity_Q, spin_of
from .spin import apply_index as _apply_index
from .spin import flip as _flip

__all__ = ["SphereSpace", "SphereEvaluator", "comps", "rhs_rpow", "embed_coeffs", "REAL_TOL",
           "STATE_REAL_TOL", "FUNC_REAL_TOL"]

_R2 = 1.0 / np.sqrt(2.0)
SP3 = np.array([[-1j * _R2, _R2, 0.0], [1j * _R2, _R2, 0.0], [0.0, 0.0, 1.0]])   # (phi, theta, r) -> spin
PH3 = np.linalg.inv(SP3)                                                             # spin -> physical
METRIC = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])            # e_-.e_+ = 1, e_r.e_r = 1


def comps(rank: int, nc: int) -> list[tuple]:
    """Every component index tuple of a rank-``rank`` tensor with ``nc`` components per index."""
    return list(product(*[range(nc)] * rank)) if rank else [()]


class SphereSpace:
    """Wavenumbers, harmonics, radial operators and transforms of one S2 / shell / ball domain."""

    def __init__(self, domain: Domain) -> None:
        cs = domain.coordsys
        if cs.kind not in ("S2", "Spherical"):
            raise DiscretizationError(f"SphereSpace needs S2 or spherical coordinates, got {cs.name}")
        self.domain, self.coordsys = domain, cs
        self.s2 = cs.kind == "S2"
        self.nc = 2 if self.s2 else 3
        self.SP, self.PH = SP3[:self.nc, :self.nc], PH3[:self.nc, :self.nc]
        self.G = METRIC[:self.nc, :self.nc]
        self.phi, self.theta = cs.coords[0], cs.coords[1]
        self.r = None if self.s2 else cs.coords[2]
        self.bphi, self.bth = domain.basis(self.phi), domain.basis(self.theta)
        self.br = None if self.s2 else domain.basis(self.r)
        if self.bphi is None or self.bth is None or (not self.s2 and self.br is None):
            raise DiscretizationError(
                "a spherical problem needs a Fourier basis in phi, a SphereBasis in theta (and a "
                "BallBasis/ShellBasis in r) among its variables")
        self.L = self.bth.size - 1
        self.H = SphereHarmonics(self.L)
        self.R = self.bth.radius if self.s2 else None
        self.ball = isinstance(self.br, BallBasis)
        self.shell = isinstance(self.br, ShellBasis)
        self.N = 1 if self.s2 else self.br.size
        if self.ball:
            self.ops = BallOps(self.N, self.br.radius)
        elif self.shell:
            self.ops = ShellOps(self.N, *self.br.bounds)
        else:
            self.ops = None
        self.real = isinstance(self.bphi, RealFourier)
        self.real_tol = REAL_TOL          # declared tolerance on imaginary parts of real fields (R9)
        self.state_real_tol = STATE_REAL_TOL   # same, for a field read from a solver state (to_grid)
        self.func_real_tol = FUNC_REAL_TOL     # same, argument of a pointwise function in F (T**2)
        if not isinstance(self.bphi, (RealFourier, ComplexFourier)):  # pragma: no cover (Domain refuses)
            raise DiscretizationError(f"{self.bphi!r}: Fourier basis required in phi")
        Np = self.bphi.size
        if Np % 2:
            raise DiscretizationError(f"{self.bphi!r}: Fourier sizes must be even")
        if self.real:
            ms = list(range(Np // 2))
        else:
            freq = np.fft.fftfreq(Np, 1.0 / Np).astype(int)
            ms = [int(f) for i, f in enumerate(freq) if i != Np // 2]
        self.ms = np.array(ms, dtype=int)
        if np.abs(self.ms).max() > self.L:
            # a mode |m| > Lmax has no harmonic: its data would be dropped in silence
            raise DiscretizationError(
                f"azimuthal wavenumbers up to |m| = {np.abs(self.ms).max()} but Lmax = {self.L}: modes "
                f"|m| > Lmax have no spherical harmonic; use Nphi/2 - 1 <= Lmax (Nphi <= "
                f"{2 * (self.L + 1)})")
        self._cache: dict = {}

    # ---------------------------------------------------------------- shapes
    def has(self, dom: Domain | None) -> tuple[bool, bool, bool]:
        """Presence of (phi, theta, r) in an object's domain (r always False on S2)."""
        if dom is None:
            return (False, False, False)
        return (dom.has(self.phi), dom.has(self.theta), (not self.s2) and dom.has(self.r))

    def bulk(self, dom: Domain | None) -> bool:
        """Bulk objects carry the innermost coordinate (r in 3D, theta on S2)."""
        hp, ht, hr = self.has(dom)
        return hr if not self.s2 else ht

    def coeff_shape(self, dom: Domain | None, rank: int) -> tuple:
        """Coefficient array shape of a rank-``rank`` object on domain ``dom``."""
        hp, ht, hr = self.has(dom)
        sh = (self.nc,) * rank + (len(self.ms) if hp else 1, self.L + 1 if ht else 1)
        if not self.s2:
            sh += (self.N if hr else 1,)
        return sh

    def sizes(self, dealias) -> tuple:
        """(Mphi, Mtheta[, Mr]): dealias=True -> floor(N * dealias); False -> N; "data" -> the
        dealiased grid, at least N + 1 radial points on the ball (exact projection of data)."""
        bs = [self.bphi, self.bth] + ([] if self.s2 else [self.br])
        if dealias == "data":
            out = [max(b.grid_size, b.size) for b in bs]
            if self.ball:
                out[2] = max(out[2], self.br.size + 1)
            return tuple(out)
        return tuple(b.grid_size if dealias else b.size for b in bs)

    def grid_shape(self, dom: Domain | None, rank: int, dealias, fold=None) -> tuple:
        """Grid-array shape; with a ``fold`` (curvilinear_symmetry.Fold) the folded grid: the
        northern hemisphere ceil(Mtheta / 2) and / or the sector Mphi / m0."""
        hp, ht, hr = self.has(dom)
        M = list(self.sizes(dealias))
        if fold is not None:
            if fold.half is not None:
                M[1] = (M[1] + 1) // 2
            M[0] //= fold.m0
        sh = (self.nc,) * rank + (M[0] if hp else 1, M[1] if ht else 1)
        if not self.s2:
            sh += (M[2] if hr else 1,)
        return sh

    def grids(self, dealias=False) -> tuple:
        """(phi (Mphi,), theta (Mtheta,)[, r (Mr,)]) of the requested grid."""
        M = self.sizes(dealias)
        out = (2 * np.pi * np.arange(M[0]) / M[0], self.H.grid(M[1])[0])
        if not self.s2:
            out += (self.ops.grid(M[2]),)
        return out

    def mindex(self, dom: Domain | None, m_index: int) -> int:
        """Index of the azimuthal mode in the arrays of an object on ``dom`` (0 when it does not
        depend on phi)."""
        return m_index if self.has(dom)[0] else 0

    def rsize(self, ell: int) -> int:
        """Radial coefficients of the degree-l components (1 on S2)."""
        return 1 if self.s2 else self.ops.size(ell)

    def valid(self, m: int, comp: tuple, ell: int, dom: Domain | None) -> bool:
        """Does component ``comp`` of an object on ``dom`` have a coefficient block at (m, l)."""
        hp, ht, hr = self.has(dom)
        if (not hp and m != 0) or (not ht and ell != 0):
            return False
        if self.ball and len(comp) == 1:
            return self.valid_reg(m, comp, ell)
        s = spin_of(comp)
        return ell >= max(abs(m), abs(s)) and ell <= self.L

    def valid_reg(self, m: int, comp: tuple, ell: int) -> bool:
        """Regularity components of the ball (rank <= 1): l+1 (index 1) exists for every l >= |m|,
        l-1 (index 0) and l (toroidal, index 2) need l >= 1 (same count as the spin components,
        different labels: at l = 0 a vector is -U_{l+1} e_r). Ball plans also label the rows and
        unknowns of rank-1 *surface* quantities (boundary rows, taus) by regularity, so that each
        boundary row involves one component (sphere_plan)."""
        if ell < abs(m) or ell > self.L:
            return False
        return not comp or comp[0] == 1 or ell >= 1

    # ---------------------------------------------------------------- per-m stacks
    def _m_of(self, dom: Domain | None) -> np.ndarray:
        return self.ms if self.has(dom)[0] else np.array([0])

    def _ang_stack(self, which: str, ms: np.ndarray, s: int, M: int, xp=np) -> np.ndarray:
        """(nm, M, L+1) synthesis or (nm, L+1, M) projection matrices of spin s, one per m (in
        module ``xp``: a device copy is cached once for CuPy, batched cuBLAS products)."""
        key = ("ang", which, tuple(ms), s, M)
        if key not in self._cache:
            f = self.H.backward if which == "bwd" else self.H.forward
            self._cache[key] = np.stack([f(int(m), s, M) for m in ms])
        return on(xp, self._cache[key], self._cache, key)

    def _ang_fold(self, which: str, ms: np.ndarray, s: int, M: int, xp=np, sigma: int = 0):
        """Hemisphere stacks of the equatorial folding (rows / columns theta_j, j < ceil(M/2)):
        'bwd' synthesis (nm, h, L+1); 'fwdN' / 'fwdS' projection parts (nm, L+1, h): fwdS holds
        the columns of the mirror points theta_{M-1-j} (zero for the equator of an odd M);
        'fwd1' = fwdN + sigma fwdS (a component that is its own mirror, sigma = +-1)."""
        key = ("angfold", which, tuple(ms), s, M, sigma)
        if key not in self._cache:
            h = (M + 1) // 2
            if which == "bwd":
                self._cache[key] = np.ascontiguousarray(self._ang_stack("bwd", ms, s, M)[:, :h, :])
            else:
                F = self._ang_stack("fwd", ms, s, M)
                FN = F[:, :, :h]
                FS = np.zeros_like(FN)
                FS[:, :, :M // 2] = F[:, :, ::-1][:, :, :M // 2]
                out = {"fwdN": FN, "fwdS": FS, "fwd1": FN + sigma * FS}[which]
                self._cache[key] = np.ascontiguousarray(out)
        return on(xp, self._cache[key], self._cache, key)

    def mirror_signs(self, ms: np.ndarray, s: int, ntan: int, p: int, xp=np) -> np.ndarray:
        """(nm, L+1) signs of the equatorial mirror of the coefficients: c^{-s}_l =
        p (-1)^ntan (-1)^(l - max(|m|, |s|)) c^{s}_l (curvilinear_symmetry.object_action)."""
        key = ("mirror", tuple(ms), abs(s), ntan, p)
        if key not in self._cache:
            ell = np.arange(self.L + 1)[None, :]
            n = ell - np.maximum(np.abs(ms)[:, None], abs(s))
            self._cache[key] = (p * (-1) ** ntan * (-1.0) ** np.abs(n)).astype(float)
        return on(xp, self._cache[key], self._cache, key)

    def Gstack(self, sigma: int, ms: np.ndarray, s: int, xp=np) -> np.ndarray:
        """(nm, L+1) diagonal of G_sigma (unit sphere) on spin s, per stored m."""
        key = ("G", sigma, tuple(ms), s)
        if key not in self._cache:
            self._cache[key] = np.stack([self.H.G(sigma, int(m), s) for m in ms])
        return on(xp, self._cache[key], self._cache, key)

    def _rad_stack(self, which: str, what: str, sk: int, Mr: int, xp=np) -> np.ndarray:
        """Radial matrices of the ball stacked over the degree l, zero-padded so that one batched
        product replaces the loop over l: 'bwd' (L+1, N, Mr) with (coeffs (l, m, N)) @ B = values
        on the nodes ('val', 'dr', 'over_r' of rho^k Q_n, k = l + sk), 'fwd' (L+1, Mr, N) the
        Gauss-Jacobi projection. Entries with no coefficient (n <= 0 or k < 0) are zero blocks."""
        key = ("rad", which, what, sk, Mr)
        if key not in self._cache:
            L1, N = self.L + 1, self.N
            out = np.zeros((L1, N, Mr) if which == "bwd" else (L1, Mr, N))
            for ell in range(L1):
                n, k = self.ops.size(ell), ell + sk
                if n <= 0 or k < 0:
                    continue
                if which == "bwd":
                    out[ell, :n, :] = self.ops.backward(k, Mr, n, what).T
                else:
                    out[ell, :, :n] = self.ops.forward(k, Mr, n).T
            self._cache[key] = out
        return on(xp, self._cache[key], self._cache, key)

    def _shell_mat(self, which: str, what: str, Mr: int, xp=np) -> np.ndarray:
        """Transposed shell radial matrix: 'bwd' (N, Mr) synthesis, 'fwd' (Mr, N) projection."""
        key = ("shell", which, what, Mr)
        if key not in self._cache:
            A = self.ops.backward(Mr, what) if which == "bwd" else self.ops.forward(Mr)
            self._cache[key] = np.ascontiguousarray(A.T)
        return on(xp, self._cache[key], self._cache, key)

    def Qstack(self, ms: np.ndarray, xp=np) -> np.ndarray:
        """(nm, L+1, 3, 3) regularity -> spin matrices of the ball (rank 1)."""
        key = ("Q", tuple(ms))
        if key not in self._cache:
            out = np.zeros((len(ms), self.L + 1, 3, 3), dtype=complex)
            for i, m in enumerate(ms):
                for ell in range(abs(int(m)), self.L + 1):
                    out[i, ell] = regularity_Q(self.H, int(m), ell)
            self._cache[key] = out
        return on(xp, self._cache[key], self._cache, key)

    # ---------------------------------------------------------------- equatorial class (radial stage)
    def eq_class(self, ms: np.ndarray, comp: tuple, rank: int, p: int):
        """How spin component ``comp`` of a shell value of equatorial parity p is computed in the
        radial stage: ('mask', keep (nm, L+1) bool) for a component that is its own mirror (scalar,
        u_r: only the (m, l) of the class are nonzero), ('pair', mirror, None) for the first of a
        spin pair (computed whole, its mirror follows by the coefficient signs), ('skip', mirror,
        sign) for the second."""
        key = ("eqclass", tuple(ms), comp, rank, p)
        if key not in self._cache:
            ell = np.arange(self.L + 1)[None, :]
            m = np.asarray(ms)[:, None]
            mirror = _flip(comp)
            ntan = sum(1 for i in comp if i != 2)
            if mirror == comp:
                sg = p * (-1) ** ntan * (-1.0) ** np.abs(ell - np.abs(m))
                out = ("mask", sg > 0, None)
            elif comp < mirror:
                out = ("pair", mirror, None)
            else:
                out = ("skip", mirror, self.mirror_signs(np.asarray(ms), spin_of(comp), ntan, p))
            self._cache[key] = out
        return self._cache[key]

    def _radial_class(self, x, keep: np.ndarray, M, xp):
        """Shell radial product (``M`` (n_in, n_out), the same for every (m, l)) restricted to the
        kept (m, l) entries of x (nm, L+1, n_in); the other entries are exactly 0."""
        nm, L1, nin = x.shape
        out = xp.zeros((nm, L1, M.shape[-1]), dtype=complex)
        k = xp.asarray(keep) if xp is not np else keep
        out[k] = x[k] @ M
        return out

    # ---------------------------------------------------------------- radial stage
    def coeff_to_mixed(self, c: np.ndarray, dom: Domain | None, rank: int, dealias,
                       what: str = "val", fold=None) -> np.ndarray:
        """Coefficients -> mixed space (spin components, (nm, L+1, Mr)); ``what`` in 'val', 'dr',
        'over_r' (radial derivative or value / r at the nodes, 3D only). On S2 (no radial axis) the
        mixed space is the coefficient space itself ('over_r' then means / R). NumPy or CuPy
        (the module of ``c``); the ball's loop over l is one batched product (``_rad_stack``)."""
        xp = array_module(c)
        c = xp.asarray(c)
        if self.s2:
            return c / self.R if what == "over_r" else (c if what == "val" else None)
        hp, ht, hr = self.has(dom)
        if rank and self.ball and hr and rank > 1:
            raise DiscretizationError("ball: tensors of rank >= 2 in coefficient space are not "
                                      "implemented (rank <= 1)")
        Mr = self.sizes(dealias)[2] if hr else 1
        ms = self._m_of(dom)
        sel = self._class_rows(fold, hp)
        if sel is not None:                   # m0-fold class: only its wavenumbers are transformed
            full_shape = c.shape[:-1] + (Mr,)
            c = c[(slice(None),) * rank + (sel,)]
            ms = ms[sel]
        out = xp.zeros(c.shape[:-1] + (Mr,), dtype=complex)
        # equatorial class (symmetry subspace): only the (m, l) entries of the class are nonzero;
        # a spin pair is computed once, its mirror from the coefficient signs
        # (the ball keeps the plain batched product: measured, the gather of the kept m-rows per l
        # costs more than the halved product at the sizes of interest, run)
        eq = fold is not None and fold.half is not None and ht and hr and not fold.zero and not self.ball
        pending = []
        for comp in comps(rank, self.nc):
            x = c[comp] if comp else c
            y = xp.zeros(x.shape[:-1] + (Mr,), dtype=complex)
            if eq:
                kind, a, b = self.eq_class(ms, comp, rank, fold.half)
                if kind == "skip":
                    pending.append((comp, a, b))
                    continue
                if kind == "mask":                   # (shell only: the ball is not class-restricted)
                    y = self._radial_class(x, a, self._shell_mat("bwd", what, Mr, xp), xp)
                    if comp:
                        out[comp] = y
                    else:
                        out[...] = y
                    continue
            if not hr:
                if what != "val":
                    raise DiscretizationError("radial derivative of a quantity without r-dependence")
                y[...] = x
            elif self.shell:
                y = x @ self._shell_mat("bwd", what, Mr, xp)
            else:                                                # batched over l: (l, m, N) @ (l, N, Mr)
                sk = spin_of(comp) if rank else 0
                B = self._rad_stack("bwd", what, sk, Mr, xp)[:x.shape[1]]
                y = xp.matmul(xp.swapaxes(x, 0, 1), B).swapaxes(0, 1)
            if comp:
                out[comp] = y
            else:
                out[...] = y
        for comp, mirror, sg in pending:                         # mirrors of spin pairs
            out[comp] = (sg if xp is np else xp.asarray(sg))[..., None] * out[mirror]
        if self.ball and hr and rank == 1:                       # regularity -> spin, per (m, l)
            Q = self.Qstack(ms, xp)
            out = xp.einsum("mlsq,qmlr->smlr", Q, out)
        if sel is not None:
            full = xp.zeros(full_shape, dtype=complex)
            full[(slice(None),) * rank + (sel,)] = out
            out = full
        return out

    def _class_rows(self, fold, hp: bool):
        """Stored-wavenumber rows of an m0-fold class (None: all rows)."""
        if fold is None or fold.m0 <= 1 or not hp or fold.zero:
            return None
        from .curvilinear_symmetry import kept_index
        return kept_index(self.ms, fold)

    def mixed_to_coeff(self, x: np.ndarray, dom: Domain | None, rank: int, dealias,
                       fold=None) -> np.ndarray:
        """Mixed space (spin, radial nodes) -> coefficients (radial projection; ball: spin ->
        regularity by Q^H, then projection on rho^k Q_n^(0, k+1/2)); with an equatorial ``fold``
        only the (m, l) entries of the class are projected (a spin pair once)."""
        xp = array_module(x)
        if self.s2:
            return xp.asarray(x)
        hp, ht, hr = self.has(dom)
        x = xp.asarray(x, dtype=complex)
        if self.ball and hr and rank > 1:
            raise DiscretizationError("ball: projection of a tensor of rank >= 2 is not implemented "
                                      "(rank <= 1; products like u.grad(u) are contracted first)")
        ms = self._m_of(dom)
        full_out = xp.zeros(self.coeff_shape(dom, rank), dtype=complex)
        if fold is not None and fold.zero:
            return full_out
        sel = self._class_rows(fold, hp)
        if sel is not None:                   # m0-fold class: only its wavenumbers are projected
            x = x[(slice(None),) * rank + (sel,)]
            ms = ms[sel]
        if self.ball and hr and rank == 1:
            Q = self.Qstack(ms, xp)
            x = xp.einsum("mlsq,smlr->qmlr", Q.conj(), x)
        out = full_out if sel is None else xp.zeros(x.shape[:-1] + (full_out.shape[-1],), dtype=complex)
        Mr = x.shape[-1]
        eq = fold is not None and fold.half is not None and ht and hr and not self.ball
        pending = []
        for comp in comps(rank, self.nc):
            y = x[comp] if comp else x
            if eq:
                kind, a, b = self.eq_class(ms, comp, rank, fold.half)
                if kind == "skip":
                    pending.append((comp, a, b))
                    continue
                if kind == "mask":                   # (shell only: the ball is not class-restricted)
                    z = self._radial_class(y, a, self._shell_mat("fwd", "val", Mr, xp), xp)
                    if comp:
                        out[comp] = z
                    else:
                        out[...] = z
                    continue
            if not hr:
                z = y
            elif self.shell:
                z = y @ self._shell_mat("fwd", "val", Mr, xp)
            else:                                                # batched over l: (l, m, Mr) @ (l, Mr, N)
                sk = spin_of(comp) if rank else 0
                F = self._rad_stack("fwd", "val", sk, Mr, xp)[:y.shape[1]]
                z = xp.matmul(xp.swapaxes(y, 0, 1), F).swapaxes(0, 1)
            if comp:
                out[comp] = z
            else:
                out[...] = z
        for comp, mirror, sg in pending:                         # mirrors of spin pairs
            out[comp] = (sg if xp is np else xp.asarray(sg))[..., None] * out[mirror]
        if sel is not None:
            full_out[(slice(None),) * rank + (sel,)] = out
            out = full_out
        return out

    # ---------------------------------------------------------------- angular + azimuthal stages
    def mixed_to_grid(self, x: np.ndarray, dom: Domain | None, rank: int, dealias,
                      physical: bool = True, fold=None, real_tol: float | None = None) -> np.ndarray:
        """Mixed space -> grid values (physical components if ``physical``, else spin). With a
        ``fold`` (symmetry subspace, curvilinear_symmetry.Fold): values on the folded grid only
        (northern hemisphere and / or azimuthal sector)."""
        hp, ht, hr = self.has(dom)
        M = self.sizes(dealias)
        Mp, Mt = M[0], M[1]
        xp = array_module(x)                  # NumPy (CPU) or CuPy (GPU: cuFFT, batched cuBLAS)
        x = xp.asarray(x)
        ms = self._m_of(dom)
        half = fold is not None and fold.half is not None and ht
        sector = fold is not None and fold.m0 > 1 and hp
        if sector:                             # only the wavenumbers of the m0-fold class
            from .curvilinear_symmetry import kept_index
            sel = kept_index(self.ms, fold)
            ms = self.ms[sel]
        spec = {}
        for comp in comps(rank, self.nc):
            y = x[comp] if comp else x
            y = y if not self.s2 else y[..., None]                   # (nm, L+1|1, Mr|1)
            if sector:
                y = y[sel]
            if ht:
                B = (self._ang_fold("bwd", ms, spin_of(comp), Mt, xp) if half else
                     self._ang_stack("bwd", ms, spin_of(comp), Mt, xp))  # (nm, Mt | h, L+1)
                y = xp.matmul(B, y)
            else:                          # no theta: the single entry is the l = 0 coefficient
                y = y * _R2 if spin_of(comp) == 0 else 0 * y         # Phi_0^{0,0} = 1/sqrt(2)
            spec[comp] = y
        out = xp.zeros(self.grid_shape(dom, rank, dealias, fold), dtype=complex)
        for comp in comps(rank, self.nc):
            V = spec[comp]
            if sector:                         # azimuthal sector of the m0-fold class
                from .curvilinear_symmetry import sector_synth
                g = sector_synth(V, spec[_flip(comp)], ms, Mp, self.real, fold, xp)
            elif hp:
                S = xp.zeros((Mp,) + V.shape[1:], dtype=complex)
                if self.real:
                    S[self.ms] = V
                    S[(-self.ms[1:]) % Mp] = xp.conj(spec[_flip(comp)][1:])
                else:
                    S[self.ms % Mp] = V
                g = xp.fft.ifft(S, axis=0) * Mp
            else:
                g = V
            g = g[..., 0] if self.s2 else g
            if comp:
                out[comp] = g
            else:
                out[...] = g
        if physical:
            out = _apply_index(self.PH, out, rank)
            if self.real:
                out = _real(out, "backward transform", self.real_tol if real_tol is None else real_tol,
                             "real_tol" if real_tol is None else "state_real_tol")
        return out

    def grid_to_mixed(self, g: np.ndarray, dom: Domain | None, rank: int, dealias,
                      physical: bool = True, fold=None) -> np.ndarray:
        """Grid values -> mixed space. With a ``fold``: g holds the folded grid only; the missing
        part is the mirror / rotated image of the value's class (equatorial: component s of the
        south is p (-1)^(tangential indices) times component -s of the north), and the
        projection of a mirror pair is computed once (the partner by the coefficient signs)."""
        hp, ht, hr = self.has(dom)
        M = self.sizes(dealias)
        Mp, Mt = M[0], M[1]
        xp = array_module(g)
        g = xp.asarray(g, dtype=complex)
        if physical:
            g = _apply_index(self.SP, g, rank)
        ms = self._m_of(dom)
        Mr = (M[2] if hr else 1) if not self.s2 else 1
        out = xp.zeros((self.nc,) * rank + (len(ms), self.L + 1 if ht else 1, Mr), dtype=complex)
        if fold is not None and fold.zero:
            return out[..., 0] if self.s2 else out
        half = fold is not None and fold.half is not None and ht
        sector = fold is not None and fold.m0 > 1 and hp
        sel = None
        if sector:                             # only the rows of the m0-fold class are computed
            from .curvilinear_symmetry import kept_index, sector_analysis
            sel = kept_index(self.ms, fold)
            ms = self.ms[sel]
        az = {}

        def azim(comp):                        # azimuthal projection of one component (memo)
            if comp not in az:
                y = g[comp] if comp else g
                y = y if not self.s2 else y[..., None]
                if sector:
                    y = sector_analysis(y, self.ms, Mp, fold, xp, kept_only=True)
                elif hp:
                    y = (xp.fft.fft(y, axis=0) / Mp)[self.ms % Mp]
                az[comp] = y
            return az[comp]
        done: dict = {}
        for comp in comps(rank, self.nc):
            if comp in done:
                continue
            s = spin_of(comp)
            if half:
                mirror = _flip(comp)
                ntan = sum(1 for i in comp if i != 2)
                sig = fold.half * (-1) ** ntan
                if mirror == comp:             # own mirror: one half-size product
                    y = xp.matmul(self._ang_fold("fwd1", ms, s, Mt, xp, sig), azim(comp))
                else:                          # pair (s, -s): one projection, partner by signs
                    y = (xp.matmul(self._ang_fold("fwdN", ms, s, Mt, xp), azim(comp))
                         + sig * xp.matmul(self._ang_fold("fwdS", ms, s, Mt, xp), azim(mirror)))
                    sg = self.mirror_signs(ms, s, ntan, fold.half, xp)
                    done[mirror] = sg[..., None] * y
            elif ht:
                F = self._ang_stack("fwd", ms, s, Mt, xp)  # (nm, L+1, Mt)
                y = xp.matmul(F, azim(comp))
            else:
                y = azim(comp)
                y = y / _R2 if s == 0 else 0 * y
            done[comp] = y
        for comp, y in done.items():
            tgt = out[comp] if comp else out
            if sel is not None:
                tgt[sel] = y
            else:
                tgt[...] = y
        return out[..., 0] if self.s2 else out

    # ---------------------------------------------------------------- full transforms
    def backward(self, c: np.ndarray, dom: Domain | None, rank: int, dealias,
                 physical: bool = True, fold=None, real_tol: float | None = None) -> np.ndarray:
        """Coefficients -> grid values; ``real_tol`` overrides real_tol for this call (state
        outputs use state_real_tol, plan.to_grid)."""
        return self.mixed_to_grid(self.coeff_to_mixed(c, dom, rank, dealias, fold=fold), dom, rank, dealias,
                                  physical, fold, real_tol)

    def forward(self, g: np.ndarray, dom: Domain | None, rank: int, dealias,
                physical: bool = True, fold=None) -> np.ndarray:
        """Coefficients of the grid values ``g`` (grid -> mixed -> coefficients)."""
        return self.mixed_to_coeff(self.grid_to_mixed(g, dom, rank, dealias, physical, fold), dom, rank, dealias,
                                   fold=fold)

    def explain(self) -> str:
        """Bases, groups, grid sizes and radial operators of the spherical space, one line."""
        M = self.sizes(True)
        rad = "" if self.s2 else f"; {self.ops.explain()}"
        return (f"SphereSpace: {type(self.bphi).__name__}(phi, {self.bphi.size}) -> {len(self.ms)} "
                f"azimuthal wavenumbers{' (real fields, m >= 0 stored)' if self.real else ''}"
                + (f", imaginary parts refused above {self.real_tol:g} relative ({self.state_real_tol:g} "
                   "for a field read from a solver state: non-real m = 0 part of the state, born from "
                   "solve rounding)" if self.real else "") + "; "
                f"{self.H.explain()}{rad}; dealiased grid {' x '.join(str(v) for v in M)}")


# ----------------------------------------------------------------------------------------------
# pseudospectral evaluator (NumPy; the oracle of any later fast version)
# ----------------------------------------------------------------------------------------------

class _C:
    """Coefficients (level 0; spin components, or regularity components in the ball's bulk).
    ``f``: fold of the value in a folded (symmetry-subspace) evaluation, else None."""
    __slots__ = ("a", "dom", "rank", "f")

    def __init__(self, a, dom, rank, f=None):
        self.a, self.dom, self.rank, self.f = a, dom, rank, f


class _X:
    """Mixed space: spin components as (nm, L+1, Mr) arrays on the dealiased radial grid."""
    __slots__ = ("x", "dom", "rank", "f")

    def __init__(self, x, dom, rank, f=None):
        self.x, self.dom, self.rank, self.f = x, dom, rank, f


class _G:
    """Dealiased grid (the folded grid in a folded evaluation), spin components."""
    __slots__ = ("g", "dom", "rank", "f")

    def __init__(self, g, dom, rank, f=None):
        self.g, self.dom, self.rank, self.f = g, dom, rank, f


class SphereEvaluator:
    """Evaluate right-hand-side (and coefficient) expressions on S2 / shell / ball domains: angular
    derivatives diagonal in l, radial derivatives and frame terms on the radial nodes (mixed space),
    products on the dealiased grid, projection back (level 0). ``xp``: array module of the
    evaluation (NumPy, the oracle, or CuPy: same code on the GPU; the module of the data decides
    each call, ``xp`` only creates the constants)."""

    def __init__(self, space: SphereSpace, xp=None, fold=None) -> None:
        self.S = space
        self.xp = np if xp is None else xp
        # symmetry subspace (curvilinear_symmetry.FoldSpec): every value lives on the folded grid
        # (northern hemisphere, azimuthal sector) with the parity of its expression
        self.fold = fold if (fold is not None and fold.active) else None

    def _fold_of(self, x):
        return None if self.fold is None else self.fold.fold(x)

    def _tag(self, v, x):
        """Attach the fold of expression x to an array value (numbers pass through)."""
        if self.fold is not None and isinstance(v, (_C, _X, _G)):
            v.f = self.fold.fold(x)
        return v

    # ------------------------------------------------------------ public
    def evaluate(self, e: E.Expr, coeffs: Mapping[E.Field, np.ndarray],
                 params: Mapping[str, Any], rpow: int = 0) -> np.ndarray:
        """Coefficients (level 0) of ``e``; on the shell, ``rpow = k`` returns those of r^k e, projected
        from the dealiased grid (the projection Dedalus makes in its r^-k derivative basis of the
        equation, ``rhs_rpow``; r^k u.grad(u) is then a polynomial for k >= 1)."""
        v = self._ev(e, coeffs, params)
        fe = self._fold_of(e)
        if rpow:
            if not self.S.shell:
                raise DiscretizationError("rpow is for shell right-hand sides")
            g = self._grid(v)
            xp = array_module(g.g)
            dom = self.rpow_domain(e.domain)
            full = xp.broadcast_to(g.g, self.S.grid_shape(dom, e.rank, True, fe))
            rk = on(xp, self.S.ops.grid(full.shape[-1]) ** rpow, self.S._cache,
                    ("rpow", full.shape[-1], rpow))
            return self.S.forward(full * rk, dom, e.rank, True, physical=False, fold=fe)
        if isinstance(v, _C):
            return array_module(v.a).broadcast_to(v.a, self.S.coeff_shape(e.domain, e.rank)).copy()
        if isinstance(v, _X):
            return self.S.mixed_to_coeff(self._xfull(v, e.domain, e.rank), e.domain, e.rank, True, fold=fe)
        g = self._grid(v)
        full = array_module(g.g).broadcast_to(g.g, self.S.grid_shape(e.domain, e.rank, True, fe))
        return self.S.forward(full, e.domain, e.rank, True, physical=False, fold=fe)

    def rpow_domain(self, dom: Domain | None) -> Domain:
        """Domain of r^k e (k >= 1, shell rows): that of e with r added when e lacks it. r^k e
        depends on r even when e does not (a forcing f(theta) on a momentum equation): its
        coefficients carry the full radial axis (projecting on e's own domain
        dropped the factor r^k and the embedding then placed f, not r^k f, in the row)."""
        if self.S.has(dom)[2]:
            return dom
        return (dom if dom is not None else Domain(self.S.coordsys, [])).with_basis(self.S.br)

    # ------------------------------------------------------------ conversions
    def _xfull(self, v: _X, dom, rank) -> np.ndarray:
        S = self.S
        hp, ht, hr = S.has(dom)
        shape = (S.nc,) * rank + (len(S.ms) if hp else 1, S.L + 1 if ht else 1)
        if not S.s2:
            shape += (S.sizes(True)[2] if hr else 1,)
        return array_module(v.x).broadcast_to(v.x, shape)

    def _grid(self, v) -> _G:
        if isinstance(v, _G):
            return v
        if isinstance(v, _C):
            return _G(self.S.backward(v.a, v.dom, v.rank, True, physical=False, fold=v.f), v.dom, v.rank, v.f)
        if isinstance(v, _X):
            return _G(self.S.mixed_to_grid(v.x, v.dom, v.rank, True, physical=False, fold=v.f),
                      v.dom, v.rank, v.f)
        return _G(self.xp.asarray(v, dtype=complex), None, 0)

    def _coeff(self, v) -> _C:
        if isinstance(v, _C):
            return v
        if isinstance(v, _X):
            return _C(self.S.mixed_to_coeff(v.x, v.dom, v.rank, True, fold=v.f), v.dom, v.rank, v.f)
        v = self._grid(v)
        full = array_module(v.g).broadcast_to(v.g, self.S.grid_shape(v.dom, v.rank, True, v.f))
        return _C(self.S.forward(full, v.dom, v.rank, True, physical=False, fold=v.f), v.dom, v.rank, v.f)

    # ------------------------------------------------------------ derivatives (mixed space)
    def _grad(self, v: _C) -> _X:
        """Spin components of grad(T) (rank + 1) in mixed space (module doc of sphere_ops; the frame's
        connection terms in 3D: (grad u)_{+,-} += u_r / r, (grad u)_{+,r} -= u_+ / r,
        (grad u)_{-,+} += u_r / r, (grad u)_{-,r} -= u_- / r, applied index by index)."""
        S = self.S
        hp, ht, hr = S.has(v.dom)
        if not (ht if S.s2 else hr):
            raise DiscretizationError("gradient of a quantity without dependence on the innermost "
                                      "coordinate (theta on S2, r in 3D)")
        V = S.coeff_to_mixed(v.a, v.dom, v.rank, True, "val", fold=v.f)
        A = S.coeff_to_mixed(v.a, v.dom, v.rank, True, "over_r", fold=v.f)
        D = None if S.s2 else S.coeff_to_mixed(v.a, v.dom, v.rank, True, "dr", fold=v.f)
        ms = S._m_of(v.dom)
        nc = S.nc
        xp = array_module(V)
        out = xp.zeros((nc,) + V.shape, dtype=complex)
        for c in comps(v.rank, nc):
            s = spin_of(c)
            a = A[c] if c else A
            for d, sigma in ((0, -1), (1, 1)):
                G = S.Gstack(sigma, ms, s, xp)[:, :a.shape[1]]
                gv = G[..., None] * a if not S.s2 else G * a
                out[(d,) + c] += gv
            if not S.s2:
                out[(2,) + c] += D[c] if c else D
                for i, t in enumerate(c):                            # frame (connection) terms
                    rep = lambda j, c=c, i=i: c[:i] + (j,) + c[i + 1:]  # noqa: E731
                    if t == 0:
                        out[(1,) + c] += A[rep(2)]
                    elif t == 1:
                        out[(0,) + c] += A[rep(2)]
                    else:
                        out[(1,) + c] -= A[rep(1)]
                        out[(0,) + c] -= A[rep(0)]
        return _X(out, v.dom, v.rank + 1, v.f)

    def _contract(self, g: _X) -> _X:
        """Metric contraction of the first two indices (div of the tensor whose gradient is g)."""
        x = g.x
        out = 0
        for d in range(self.S.nc):
            for t in range(self.S.nc):
                if self.S.G[d, t]:
                    out = out + self.S.G[d, t] * x[d, t]
        return _X(out, g.dom, g.rank - 2, g.f)

    def _curl_of_grad(self, g: _X, rank_in: int) -> _X:
        out = self._curl_of_grad_raw(g, rank_in)
        if self.fold is not None and g.f is not None:
            out.f = self.fold.derived(g.f, det_flip=True)      # pseudo-tensor under a reflection
        return out

    def _curl_of_grad_raw(self, g: _X, rank_in: int) -> _X:
        S = self.S
        x = g.x
        if S.s2:
            if rank_in == 0:          # curl psi = grad(psi) x e_r: (i G_- psi, -i G_+ psi) / R
                return _X(array_module(x).stack([1j * x[0], -1j * x[1]]), g.dom, 1)
            if rank_in == 1:          # e_r . curl u = (i/R)(G_+ u_- - G_- u_+)
                return _X(1j * (x[1, 0] - x[0, 1]), g.dom, 0)
            raise DiscretizationError("curl of a rank >= 2 tensor on S2")
        if rank_in != 1:
            raise DiscretizationError("curl in 3D: vectors only")
        out = array_module(x).zeros(x.shape[1:], dtype=complex)
        for (d, t), (k, cst) in cross_table().items():
            out[k] += cst * x[d, t]
        return _X(out, g.dom, 1)

    def _lap(self, v: _C) -> _X | _C:
        S = self.S
        if S.s2:                      # spectral: (G_- G_+ + G_+ G_-) / R^2 per spin component
            ms = S._m_of(v.dom)
            xp = array_module(v.a)
            out = xp.zeros(v.a.shape, dtype=complex)
            for c in comps(v.rank, S.nc):
                s = spin_of(c)
                lam = (S.Gstack(-1, ms, s + 1, xp) * S.Gstack(1, ms, s, xp)
                       + S.Gstack(1, ms, s - 1, xp) * S.Gstack(-1, ms, s, xp))[:, :v.a.shape[-1]]
                val = lam * (v.a[c] if c else v.a) / S.R ** 2
                if c:
                    out[c] = val
                else:
                    out[...] = val
            return _C(out, v.dom, v.rank, v.f)
        g = self._grad(v)
        return self._contract(self._grad(self._coeff(g)))

    # ------------------------------------------------------------ recursion
    def _ev(self, x: E.Expr, coeffs, params):
        """Value of node x (tagged with its fold in a folded evaluation)."""
        return self._tag(self._ev_raw(x, coeffs, params), x)

    def _ev_raw(self, x: E.Expr, coeffs, params):
        S = self.S
        if isinstance(x, E.Field):
            if x not in coeffs:
                raise DiscretizationError(f"no data for field {x.name}")
            return _C(self.xp.asarray(coeffs[x]), x.domain, x.rank)
        if isinstance(x, E.Constant):
            if x.rank:
                raise DiscretizationError("tensor constants in spherical right-hand sides")
            return x.value
        if isinstance(x, E.Parameter):
            v = params.get(x.name, x.value)
            if v is None:
                raise DiscretizationError(f"parameter {x.name} has no value")
            return v
        if isinstance(x, (E.Gradient, E.Divergence, E.Curl)):
            v = self._coeff(self._ev(x.args[0], coeffs, params))
            g = self._grad(v)
            if isinstance(x, E.Gradient):
                return g
            if isinstance(x, E.Divergence):
                if v.rank == 0:
                    raise DiscretizationError("div of a scalar")
                return self._contract(g)
            return self._curl_of_grad(g, v.rank)
        if isinstance(x, E.Laplacian):
            return self._lap(self._coeff(self._ev(x.args[0], coeffs, params)))
        if isinstance(x, E.Differentiate):
            v = self._coeff(self._ev(x.args[0], coeffs, params))
            if x.coord != S.phi or x.args[0].rank:
                raise DiscretizationError(
                    f"'{x}': partial derivatives in a spherical RHS: only d/dphi of a scalar (use "
                    "grad/div/lap/curl, which carry the metric terms)")
            m = S._m_of(v.dom)
            mm = array_module(v.a).asarray(m.reshape((-1,) + (1,) * (v.a.ndim - 1)))
            return _C(1j * mm * v.a, v.dom, 0)
        if isinstance(x, E.Add):
            vals = [self._grid(self._ev(a, coeffs, params)) for a in x.args]
            out = vals[0].g
            for v in vals[1:]:
                out = out + v.g
            return _G(out, x.domain, x.rank)
        if isinstance(x, E.Mul):
            vals = [self._ev(a, coeffs, params) for a in x.args]
            num = [v for v in vals if not isinstance(v, (_C, _X, _G))]
            arr = [self._grid(v) for v in vals if isinstance(v, (_C, _X, _G))]
            c = 1.0
            for v in num:
                c = c * v
            if not arr:
                return c
            if sum(1 for a in arr if a.rank) > 1:
                raise DiscretizationError(f"'{x}': product of two tensors, use dot() or cross()")
            out = c
            for a in arr:
                out = out * a.g
            return _G(out, x.domain, x.rank)
        if isinstance(x, E.Dot):
            a = self._grid(self._ev(x.args[0], coeffs, params))
            b = self._grid(self._ev(x.args[1], coeffs, params))
            return _G(_dot(a.g, a.rank, b.g, b.rank, S.G), x.domain, x.rank)
        if isinstance(x, E.Outer):
            a = self._grid(self._ev(x.args[0], coeffs, params))
            b = self._grid(self._ev(x.args[1], coeffs, params))
            ga = a.g.reshape(a.g.shape[:a.rank] + (1,) * b.rank + a.g.shape[a.rank:])
            gb = b.g.reshape((1,) * a.rank + b.g.shape)
            return _G(ga * gb, x.domain, x.rank)
        if isinstance(x, E.Cross):
            a = self._grid(self._ev(x.args[0], coeffs, params))
            b = self._grid(self._ev(x.args[1], coeffs, params))
            out = array_module(a.g, b.g).zeros(np.broadcast_shapes(a.g.shape, b.g.shape), dtype=complex)
            for (i, j), (k, cst) in cross_table().items():
                out[k] = out[k] + cst * a.g[i] * b.g[j]
            return _G(out, x.domain, 1)
        if isinstance(x, E.Power):
            v = self._ev(x.args[0], coeffs, params)
            if not isinstance(v, (_C, _X, _G)):
                return v ** x.exponent
            g = self._grid(v)
            return _G(self._real_scalar(g.g, x) ** x.exponent, x.domain, 0)
        if isinstance(x, E.UnaryFunction):
            g = self._grid(self._ev(x.args[0], coeffs, params))
            if x.func not in ("exp", "log", "sin", "cos", "tanh", "sqrt", "abs", "sign"):
                # a user function (m.function / pointwise) as on the disk and in Cartesian boxes
                # (review finding 34: KeyError on the sphere)
                if not E.is_user_function(x.func):
                    raise KeyError(x.func)
                xp = array_module(g.g)
                return _G(xp.asarray(x.numpy_func(self._real_scalar(g.g, x))), x.domain, 0)
            f = getattr(array_module(g.g), x.func)     # numpy.exp or cupy.exp, ...
            return _G(f(self._real_scalar(g.g, x)), x.domain, 0)
        if isinstance(x, E.Component):
            i = x.index
            if S.s2 or i != 2:
                raise DiscretizationError(
                    f"'{x}': the tangential components (u_phi, u_theta) are not regular functions on "
                    "the sphere (e_theta, e_phi are singular at the poles); use dot products, "
                    "grad/div/curl, or the radial component (index 2) in 3D")
            g = self._grid(self._ev(x.args[0], coeffs, params))
            return _G(g.g[2], x.domain, x.rank)
        if isinstance(x, E.Conj):
            g = self._grid(self._ev(x.args[0], coeffs, params))
            xp = array_module(g.g)
            out = xp.conj(g.g)
            for ax in range(g.rank):          # conj(u_+) = u_- for a real vector: flip every spin
                out = xp.take(out, xp.asarray([1, 0, 2][:S.nc]), axis=ax)
            return _G(out, x.domain, x.rank)
        raise DiscretizationError(f"cannot evaluate node {type(x).__name__} ('{x}') in a spherical RHS")

    def _real_scalar(self, g: np.ndarray, x: E.Expr) -> np.ndarray:
        if self.S.real:
            return _real(g, f"'{x}'", self.S.func_real_tol, "func_real_tol")
        return g


def embed_coeffs(c, shape: tuple, what: str = "right-hand side"):
    """Coefficients ``c`` of an expression whose domain lacks some coordinates of the equation,
    in the equation's layout ``shape``. A missing coordinate is an axis of length 1
    holding the m = 0 / l = 0 / T_0 entry (module doc: the same convention for every object), so
    the data goes to index 0 of the full axis and the other entries are zero: the exact
    coefficients of the same function on the larger domain (a field of theta only is constant
    in phi), not a copy of the values. Any other shape mismatch is refused, naming ``what``."""
    shape = tuple(shape)
    if c.shape == shape:
        return c
    if c.ndim != len(shape) or any(a != b and a != 1 for a, b in zip(c.shape, shape)):
        raise DiscretizationError(
            f"{what}: coefficient shape {c.shape} cannot be embedded in the equation's layout "
            f"{shape} (only coordinates missing from the expression can be added)")
    xp = array_module(c)
    out = xp.zeros(shape, dtype=complex)
    out[tuple(slice(0, n) for n in c.shape)] = c
    return out


def rhs_rpow(e: E.Expr) -> int:
    """Dedalus's derivative-basis index k of an expression on the shell: fields, constants and
    coefficients 0; grad, div, curl +1; lap +2; lift(., lam) lam; interpolations and integrals 0 (no
    radial dependence left); sums and products the maximum of their terms. An equation lives in the
    basis k_eq = max(k(lhs), k(rhs)): its rows are multiplied by r^k_eq in C^(k_eq) and its
    right-hand side is projected as r^k_eq F from the dealiased grid (exactly what Dedalus does in its
    r^-k Jacobi(k-1/2, k-1/2) bases; sphere_plan)."""
    if isinstance(e, (E.Gradient, E.Divergence, E.Curl)):
        return rhs_rpow(e.args[0]) + 1
    if isinstance(e, E.Laplacian):
        return rhs_rpow(e.args[0]) + 2
    if isinstance(e, E.Lift):
        return int(getattr(e, "lam", 0))
    if isinstance(e, (E.Interpolate, E.Integrate)) or not e.args:
        return 0
    return max(rhs_rpow(a) for a in e.args)
