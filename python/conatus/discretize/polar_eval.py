"""Polar and cylindrical domains: grid <-> coefficient transforms and pseudospectral evaluation.

Layouts ("Polar plans"):

- **Coefficients** of a field on Domain(PolarCoordinates(phi, r), ...): complex array of shape
  (2,)*rank + (Nm, N); on Domain(CylindricalCoordinates(phi, r, z), ...): (3,)*rank + (Nm, N, Nz).
  Tensor indices hold *spin* components (index 0: u_- = u_r - i u_phi, index 1: u_+ = u_r + i u_phi,
  Dedalus's order; index 2: u_z, spin 0); the azimuth axis holds the stored wavenumbers
  ``PolarSpace.ms`` (RealFourier: m = 0 .. Nphi/2 - 1, the field being real, the mode (-m, -k_z)
  obtained by conjugation with all spins flipped; ComplexFourier: numpy fftfreq order without the
  Nyquist mode); the radial axis holds the N_m coefficients of the component (``polar_radial``),
  zero-padded to N; the z axis (ComplexFourier only) holds the Nz coefficients in fftfreq order,
  the Nyquist entry always 0. A domain without a coordinate has that axis of length 1 (m = 0 or
  k_z = 0 only for a missing phi or z).
- **Grid values**: physical components in the coordinate order (phi, r[, z]), shape
  (nc,)*rank + (Mphi, Mr[, Mz]), phi_j = 2 pi j / Mphi, r_i = radial nodes (ascending),
  z_j = a + Lz j / Mz. Real for RealFourier in phi.

Nonlinear terms are evaluated on the grid of M = floor(N * dealias) points per direction (the
dealias factors of the bases, never implicit): FFT truncation in phi and z (3/2 rule) and, on the
disk, a Gauss-Legendre radial projection that is exact for quadratic products when Mr >= 3N/2 thanks
to the triangular truncation (``polar_radial``).
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any

import numpy as np

from ..ir import expr as E
from ..ir.bases import AnnulusBasis, ComplexFourier, DiskBasis, RealFourier
from ..ir.domain import Domain
from ._xp import array_module, on
from .compiler import DiscretizationError
from .curvilinear_common import spin_dot as _dot
# reality tolerances and check, shared with the other curvilinear evaluator (historical names)
from .real_fields import FUNC_REAL_TOL, REAL_TOL, STATE_REAL_TOL
from .real_fields import checked_real as _real
from .polar_radial import AnnulusOps, DiskOps

__all__ = ["PolarSpace", "PolarEvaluator", "spin_k", "SPIN", "SP", "PH"]

SPIN = (-1, 1, 0)                                # tensor index 0 -> spin -1, 1 -> +1, 2 (z) -> 0
SP = np.array([[-1j, 1.0, 0.0], [1j, 1.0, 0.0], [0.0, 0.0, 1.0]])   # physical (phi, r, z) -> spin
PH = np.linalg.inv(SP)                                              # spin -> physical


def spin_k(comp: tuple) -> int:
    """Sum of the spins of a spin-component tuple (effective wavenumber = m + spin_k)."""
    return int(sum(SPIN[i] for i in comp))


def _flip(comp: tuple) -> tuple:
    return tuple({0: 1, 1: 0, 2: 2}[i] for i in comp)


def _comps(rank: int, nc: int = 2) -> list[tuple]:
    return list(product(*[range(nc)] * rank)) if rank else [()]


def _apply_index(M: np.ndarray, g: np.ndarray, rank: int) -> np.ndarray:
    """Apply the matrix M on each of the first ``rank`` axes of g (NumPy or CuPy g)."""
    xp = array_module(g)
    M = M if xp is np else xp.asarray(M)
    for ax in range(rank):
        g = xp.moveaxis(xp.tensordot(M, g, axes=([1], [ax])), 0, ax)
    return g


class PolarSpace:
    """Azimuthal (and axial) wavenumbers, radial operators and transforms of one polar/cylindrical domain."""

    def __init__(self, domain: Domain) -> None:
        cs = domain.coordsys
        if cs.kind not in ("Polar", "Cylindrical"):
            raise DiscretizationError(f"PolarSpace needs polar or cylindrical coordinates, got {cs.name}")
        self.domain = domain
        self.coordsys = cs
        self.cyl = cs.kind == "Cylindrical"
        self.nc = 3 if self.cyl else 2
        self.SP, self.PH = SP[:self.nc, :self.nc], PH[:self.nc, :self.nc]
        # metric in the spin basis: e_- . e_+ = 1/2, e_z . e_z = 1
        self.G = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 1.0]])[:self.nc, :self.nc]
        self.phi, self.r = cs.coords[0], cs.coords[1]
        self.zc = cs.coords[2] if self.cyl else None
        self.bphi, self.br = domain.basis(self.phi), domain.basis(self.r)
        self.bz = domain.basis(self.zc) if self.cyl else None
        if self.bphi is None or self.br is None or (self.cyl and self.bz is None):
            raise DiscretizationError(
                "a polar problem needs a Fourier basis in phi and a DiskBasis/AnnulusBasis in r "
                "(and a Fourier basis in z on a cylinder) among its variables")
        if self.cyl and not isinstance(self.bz, ComplexFourier):
            raise NotImplementedError(
                f"{self.bz!r}: the axial direction of a cylinder must be ComplexFourier (a real field "
                "then stores m >= 0 and every k_z; RealFourier in z is not implemented)")
        self.real = isinstance(self.bphi, RealFourier)
        self.N = self.br.size
        if isinstance(self.br, DiskBasis):
            self.ops = DiskOps(self.N, self.br.radius)
        elif isinstance(self.br, AnnulusBasis):
            self.ops = AnnulusOps(self.N, *self.br.bounds)
        else:  # pragma: no cover (Domain refuses it)
            raise DiscretizationError(f"radial basis {self.br!r} not supported")
        Np = self.bphi.size
        if Np % 2:
            raise DiscretizationError(f"{self.bphi!r}: Fourier sizes must be even")
        if self.real:
            ms = list(range(Np // 2))
        else:
            freq = np.fft.fftfreq(Np, 1.0 / Np).astype(int)
            ms = [int(f) for i, f in enumerate(freq) if i != Np // 2]
        self.ms = np.array(ms, dtype=int)
        if self.cyl:
            Nz = self.bz.size
            if Nz % 2:
                raise DiscretizationError(f"{self.bz!r}: Fourier sizes must be even")
            self.zfreq = np.fft.fftfreq(Nz, 1.0 / Nz).astype(int)          # integer n, all Nz entries
            self.kz_all = 2 * np.pi * self.zfreq / self.bz.length
            self.zkeep = np.array([j for j in range(Nz) if j != Nz // 2])   # Nyquist not represented
        self.real_tol = REAL_TOL          # declared tolerance on imaginary parts of real fields (R9)
        self.state_real_tol = STATE_REAL_TOL   # same, for a field read from a solver state (to_grid)
        self.func_real_tol = FUNC_REAL_TOL     # same, argument of a pointwise function in F (T**2)
        self._cache: dict = {}

    # ---------------------------------------------------------------- shapes
    def kind(self, dom: Domain | None) -> str:
        """Presence of (phi, r): 'full', 'edge' (phi only), 'radial' (r only), 'const'."""
        hp = dom is not None and dom.has(self.phi)
        hr = dom is not None and dom.has(self.r)
        return {(True, True): "full", (True, False): "edge", (False, True): "radial",
                (False, False): "const"}[(hp, hr)]

    def hz(self, dom: Domain | None) -> bool:
        """True if ``dom`` has the axial (z) direction of a cylinder."""
        return self.cyl and dom is not None and dom.has(self.zc)

    def mlist(self, kind: str) -> np.ndarray:
        """Azimuthal wavenumbers stored for a domain kind (only m = 0 without phi)."""
        return self.ms if kind in ("full", "edge") else np.array([0])

    def kzlist(self, dom: Domain | None) -> np.ndarray:
        """Axial wavenumbers along the stored z axis (all entries; [0] without z)."""
        return self.kz_all if self.hz(dom) else np.array([0.0])

    def coeff_shape(self, dom: Domain | None, rank: int) -> tuple:
        """Coefficient-array shape (components, m, [k_z,] radial) of a field on ``dom``."""
        k = self.kind(dom)
        sh = (self.nc,) * rank + (len(self.ms) if k in ("full", "edge") else 1,
                                  self.N if k in ("full", "radial") else 1)
        if self.cyl:
            sh += (self.bz.size if self.hz(dom) else 1,)
        return sh

    def embed(self, c, dom: Domain | None, rank: int, what: str = "right-hand side"):
        """Coefficients ``c`` of an expression whose domain lacks coordinates of the equation's
        domain ``dom``, in the equation's layout (the defect of the
        spherical solver here too: a forcing f(r) failed at the first step with a matmul
        dimension error). A missing coordinate is an axis of length 1 holding its m = 0 / k_z = 0 /
        constant radial entry (index 0 of each axis: m = 0 first in both Fourier orderings, the
        constant first in the radial bases at m = 0 and in the annulus's Chebyshev T at every m),
        so the data goes to index 0 and the rest is zero: the exact coefficients of the same
        function on the larger domain. On the disk a function of phi alone (not constant) or a
        vector of phi alone is not regular at the centre: refused, as any other mismatch."""
        shape = self.coeff_shape(dom, rank)
        if tuple(c.shape) == shape:
            return c
        if c.ndim != len(shape) or any(a != b and a != 1 for a, b in zip(c.shape, shape, strict=True)):
            raise DiscretizationError(
                f"{what}: coefficient shape {tuple(c.shape)} cannot be embedded in the equation's "
                f"layout {shape} (only coordinates missing from the expression can be added)")
        xp = array_module(c)
        if isinstance(self.br, DiskBasis) and c.shape[rank + 1] == 1 and shape[rank + 1] > 1:
            m_rest = c[(slice(None),) * rank + (slice(1, None),)]
            scale = float(xp.abs(c).max()) if c.size else 0.0
            if rank > 0 or (m_rest.size and float(xp.abs(m_rest).max()) > 1e-13 * max(scale, 1e-300)):
                raise DiscretizationError(
                    f"{what}: data independent of r but not of phi (or a vector of phi alone) "
                    "on the disk: not regular at the centre r = 0",
                    fix="declare the data on (phi, r), e.g. a function f(phi, r) that vanishes "
                        "like r^|m| at the centre")
        out = xp.zeros(shape, dtype=complex)
        out[tuple(slice(0, n) for n in c.shape)] = c
        return out

    def sizes(self, dealias) -> tuple:
        """(Mphi, Mr[, Mz]): dealias=True -> floor(N * dealias) (products); False -> N (output grid);
        "data" -> the dealiased grid with at least N + 1 radial points (projection of callable
        data, exact also for the spin components of odd m, polar_radial.DiskOps.forward)."""
        bs = [self.bphi, self.br] + ([self.bz] if self.cyl else [])
        if dealias == "data":
            out = [max(b.grid_size, b.size) for b in bs]
            out[1] = max(out[1], self.br.size + 1)
            return tuple(out)
        return tuple(b.grid_size if dealias else b.size for b in bs)

    def grid_shape(self, dom: Domain | None, rank: int, dealias, fold=None) -> tuple:
        """Grid-array shape of a field on ``dom`` (dealiased or not); with a ``fold``
        (curvilinear_symmetry.Fold, m0-fold class) the azimuthal sector Mphi / m0."""
        k = self.kind(dom)
        M = list(self.sizes(dealias))
        if fold is not None:
            M[0] //= fold.m0
            if fold.zq is not None and self.cyl:
                M[2] = M[2] // 2 + 1                   # z_j, j = 0..Mz/2 (both fixed points)
        sh = (self.nc,) * rank + (M[0] if k in ("full", "edge") else 1, M[1] if k in ("full", "radial") else 1)
        if self.cyl:
            sh += (M[2] if self.hz(dom) else 1,)
        return sh

    def grids(self, dealias=False) -> tuple:
        """(phi (Mphi,), r (Mr,)[, z (Mz,)]) of the requested grid (default: non-dealiased)."""
        M = self.sizes(dealias)
        out = (2 * np.pi * np.arange(M[0]) / M[0], self.ops.grid(M[1]))
        if self.cyl:
            out += (self.bz.bounds[0] + self.bz.length * np.arange(M[2]) / M[2],)
        return out

    def cindex(self, dom: Domain | None, group, j: int = 0) -> tuple:
        """Index (azimuth, radial[, z]) of coefficient j of object ``dom`` in group ``group``."""
        kind = self.kind(dom)
        out = (group.index if kind in ("full", "edge") else 0, j if kind in ("full", "radial") else 0)
        if self.cyl:
            out += (group.zindex if self.hz(dom) else 0,)
        return out

    # ---------------------------------------------------------------- radial stacks
    def _ks(self, kind: str, comp: tuple) -> np.ndarray:
        return self.mlist(kind) + spin_k(comp)

    def _ns(self, kind: str) -> np.ndarray:
        """Radial size of every component, per stored m (group truncation, polar_radial)."""
        return np.array([self.ops.size(int(m)) for m in self.mlist(kind)], dtype=int)

    def zfold(self, Mz: int, q: int, xp=np):
        """Axial folding (reflection z -> -z of a cylinder) for component sign q: (pos, neg, Bz, Fz)
        with pos the stored z entries of k_z >= 0 (Nyquist excluded), neg[a] the entry of -k for
        pos[a] (k > 0; -1 for k = 0), Bz (h, npos) the synthesis on z_j, j = 0..Mz/2, from the
        k >= 0 coefficients (c_{-k} = q c_k: e^{ikz} + q e^{-ikz}), Fz (npos, h) the projection
        (the full-grid FFT with the mirror half folded in: weights 1/2 at the fixed points)."""
        key = ("zfold", Mz, q)
        if key not in self._cache:
            f = self.zfreq
            Nz = len(f)
            pos = np.array([j for j in range(Nz) if f[j] >= 0 and j != Nz // 2], int)
            look = {int(f[j]): j for j in range(Nz) if j != Nz // 2}
            neg = np.array([look[-int(f[j])] if f[j] > 0 else -1 for j in pos], int)
            h = Mz // 2 + 1
            jj = np.arange(h)[:, None]
            n = f[pos][None, :]
            th = 2 * np.pi * n * jj / Mz
            Bz = np.where(n == 0, 1.0 + 0j, np.exp(1j * th) + q * np.exp(-1j * th))
            w = np.ones(h)
            w[0] = 0.5
            if Mz % 2 == 0:
                w[-1] = 0.5
            Fz = (w[:, None] * (np.exp(-1j * th) + q * np.exp(1j * th))).T / Mz
            self._cache[key] = (pos, neg, np.ascontiguousarray(Bz), np.ascontiguousarray(Fz))
        pos, neg, Bz, Fz = self._cache[key]
        return pos, neg, on(xp, Bz, self._cache, key + ("B",)), on(xp, Fz, self._cache, key + ("F",))

    def _stack_sel(self, which: str, ks: np.ndarray, ns: np.ndarray, M: int, level: int, p: int,
                   sel: np.ndarray, xp=np) -> np.ndarray:
        """``_stack`` restricted to the rows ``sel`` (the wavenumbers of an m0-fold class)."""
        key = ("sel", which, tuple(ks), tuple(ns), M, level, p, tuple(sel))
        if key not in self._cache:
            self._cache[key] = np.ascontiguousarray(self._stack(which, ks, ns, M, level, p)[sel])
        return on(xp, self._cache[key], self._cache, key)

    def _stack(self, which: str, ks: np.ndarray, ns: np.ndarray, M: int, level: int = 0,
               p: int = 0, xp=np) -> np.ndarray:
        """Stacked radial synthesis ('bwd', (nm, M, N)) or projection ('fwd', (nm, N, M)) matrices,
        in module ``xp`` (device copy cached once for CuPy: batched cuBLAS products)."""
        key = (which, tuple(ks), tuple(ns), M, level, p)
        if key not in self._cache:
            N = self.N
            if which == "bwd":
                out = np.zeros((len(ks), M, N))
                for j, (k, nk) in enumerate(zip(ks, ns, strict=True)):
                    if nk > 0:
                        out[j, :, :nk] = self.ops.backward(int(k), M, level, p, int(nk))
            else:
                out = np.zeros((len(ks), N, M))
                for j, (k, nk) in enumerate(zip(ks, ns, strict=True)):
                    if nk > 0:
                        out[j, :nk, :] = self.ops.forward(int(k), M, int(nk))
            self._cache[key] = out
        return on(xp, self._cache[key], self._cache, key)

    # 3D internal views: (azimuth, radial, z) with a z axis of length 1 on polar domains
    def _3(self, x: np.ndarray) -> np.ndarray:
        return x if self.cyl else x[..., None]

    def _2(self, x: np.ndarray) -> np.ndarray:
        return x if self.cyl else x[..., 0]

    # ---------------------------------------------------------------- transforms
    def backward(self, c: np.ndarray, dom: Domain | None, rank: int, dealias,
                 level: int = 0, p: int = 0, physical: bool = True, fold=None,
                 real_tol: float | None = None) -> np.ndarray:
        """Coefficients (spin) -> grid values (physical components if ``physical``, else spin);
        with a ``fold`` (m0-fold class of a symmetry subspace) on the azimuthal sector only."""
        kind, hz = self.kind(dom), self.hz(dom)
        M = self.sizes(dealias)
        Mp, Mr = M[0], M[1]
        xp = array_module(c)                  # NumPy (CPU) or CuPy (GPU: cuFFT, batched cuBLAS)
        c = xp.asarray(c)
        comps = _comps(rank, self.nc)
        sector = fold is not None and fold.m0 > 1 and kind in ("full", "edge")
        if sector:                                  # only the wavenumbers of the class are nonzero
            from .curvilinear_symmetry import kept_index
            sel = kept_index(self.ms, fold)
        zf = fold is not None and fold.zq is not None and hz
        zq = {comp: (fold.zq * (-1) ** sum(1 for i in comp if i == 2) if zf else None) for comp in comps}
        spec = {}
        for comp in comps:
            x = self._3(c[comp] if comp else c)                             # (nm, N|1, nz|1)
            if sector:
                x = x[sel]
            if zf:                                  # axial folding: the k_z >= 0 entries only
                x = x[..., self.zfold(M[2], zq[comp])[0]]
            if kind in ("full", "radial"):
                B = (self._stack_sel("bwd", self._ks(kind, comp), self._ns(kind), Mr, level, p, sel, xp)
                     if sector else self._stack("bwd", self._ks(kind, comp), self._ns(kind), Mr, level, p, xp))
                x = xp.matmul(B, x)                                         # (nm, Mr, nz)
            if hz and not zf:                                                # z modes -> length Mz
                Mz = M[2]
                S = xp.zeros(x.shape[:2] + (Mz,), dtype=complex)
                S[..., self.zfreq[self.zkeep] % Mz] = x[..., self.zkeep]
                x = S
            spec[comp] = x
        out = xp.zeros(self.grid_shape(dom, rank, dealias, fold), dtype=complex)
        for comp in comps:
            V = spec[comp]
            # real fields: mode (-m, k_z) is conj of (m, -k_z) with spins flipped; with the axial
            # folding (k >= 0 stored) c_{m,-k} = q c_{m,k}, so no reversal of the z axis
            Vf = spec[_flip(comp)]
            if zf:
                Vf = zq[comp] * Vf
            elif hz:
                Vf = xp.roll(Vf[..., ::-1], 1, axis=-1)                      # k_z -> -k_z
            if sector:                             # azimuthal sector of the m0-fold class
                from .curvilinear_symmetry import sector_synth
                g = sector_synth(V, Vf, self.ms[sel], Mp, self.real, fold, xp)
            elif kind in ("full", "edge"):
                S = xp.zeros((Mp,) + V.shape[1:], dtype=complex)
                if self.real:
                    S[self.ms] = V
                    S[(-self.ms[1:]) % Mp] = xp.conj(Vf[1:])                # modes (-m, -k_z)
                else:
                    S[self.ms % Mp] = V
                g = xp.fft.ifft(S, axis=0) * Mp
            else:
                g = V
            if zf:                                  # z_j, j = 0..Mz/2, from the k >= 0 entries
                g = xp.matmul(g, self.zfold(M[2], zq[comp], xp)[2].T)
            elif hz:
                g = xp.fft.ifft(g, axis=2) * g.shape[2]
            g = self._2(g)
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

    def forward(self, g: np.ndarray, dom: Domain | None, rank: int, dealias,
                physical: bool = True, fold=None) -> np.ndarray:
        """Grid values -> coefficients (level 0, spin): FFT truncation in phi (and z) and radial
        projection (``polar_radial``); with a ``fold``, g holds the azimuthal sector of the class."""
        kind, hz = self.kind(dom), self.hz(dom)
        M = self.sizes(dealias)
        Mp, Mr = M[0], M[1]
        xp = array_module(g)                  # NumPy (CPU) or CuPy (GPU: cuFFT, batched cuBLAS)
        g = xp.asarray(g, dtype=complex)
        if physical:
            g = _apply_index(self.SP, g, rank)
        out = xp.zeros(self.coeff_shape(dom, rank), dtype=complex)
        if fold is not None and fold.zero:
            return out
        sector = fold is not None and fold.m0 > 1 and kind in ("full", "edge")
        if sector:
            from .curvilinear_symmetry import kept_index, sector_analysis
            sel = kept_index(self.ms, fold)
        zf = fold is not None and fold.zq is not None and hz
        for comp in _comps(rank, self.nc):
            x = self._3(g[comp] if comp else g)
            if zf:                                 # folded z grid -> the k_z >= 0 coefficients
                q = fold.zq * (-1) ** sum(1 for i in comp if i == 2)
                zpos, zneg, _, Fz = self.zfold(M[2], q, xp)
                x = xp.matmul(x, Fz.T)
            if sector:                             # the class's rows only
                x = sector_analysis(x, self.ms, Mp, fold, xp, kept_only=True)
            elif kind in ("full", "edge"):
                x = (xp.fft.fft(x, axis=0) / Mp)[self.ms % Mp]
            if hz and not zf:
                Mz = M[2]
                xz = xp.fft.fft(x, axis=2) / Mz
                y = xp.zeros(xz.shape[:2] + (self.bz.size,), dtype=complex)
                y[..., self.zkeep] = xz[..., self.zfreq[self.zkeep] % Mz]
                x = y
            if kind in ("full", "radial"):
                F = (self._stack_sel("fwd", self._ks(kind, comp), self._ns(kind), Mr, 0, 0, sel, xp)
                     if sector else self._stack("fwd", self._ks(kind, comp), self._ns(kind), Mr, 0, 0, xp))
                x = xp.matmul(F, x)
            x = self._2(x)
            tgt = out[comp] if comp else out
            if zf:                                 # the -k_z entries: c_{-k} = q c_k
                full = xp.zeros(x.shape[:-1] + (tgt.shape[-1],), dtype=complex)
                full[..., zpos] = x
                kpos = np.nonzero(zneg >= 0)[0]
                full[..., zneg[kpos]] = q * x[..., kpos]
                x = full
            if sector:
                tgt[sel] = x
            else:
                tgt[...] = x
        return out

    def explain(self) -> str:
        M = self.sizes(True)
        z = (f"; z: ComplexFourier({self.bz.size}), k_z = 2 pi n / {self.bz.length:g}" if self.cyl else "")
        return (f"PolarSpace: {type(self.bphi).__name__}(phi, {self.bphi.size}) -> {len(self.ms)} "
                f"azimuthal wavenumbers m in [{self.ms.min()}, {self.ms.max()}]"
                f"{' (real fields, m >= 0 stored)' if self.real else ''}"
                + (f", imaginary parts refused above {self.real_tol:g} relative ({self.state_real_tol:g} "
                   "for a field read from a solver state: non-real m = 0 part of the state, born from "
                   "solve rounding)" if self.real else "")
                + f"; {self.ops.explain()}{z}; "
                f"dealiased grid {' x '.join(str(v) for v in M)}")


# ----------------------------------------------------------------------------------------------
# pseudospectral evaluator (NumPy; the oracle of any later fused/GPU version)
# ----------------------------------------------------------------------------------------------

class _C:
    """A value held in coefficient space (spin components), at a level and an r-power; ``f``:
    its fold in a folded (symmetry-subspace) evaluation, else None."""

    __slots__ = ("a", "dom", "rank", "level", "p", "f")

    def __init__(self, a, dom, rank, level=0, p=0, f=None):
        self.a, self.dom, self.rank, self.level, self.p, self.f = a, dom, rank, level, p, f


class _G:
    """A value held on the dealiased grid (the azimuthal sector in a folded evaluation), spin
    components (complex)."""

    __slots__ = ("g", "dom", "rank", "f")

    def __init__(self, g, dom, rank, f=None):
        self.g, self.dom, self.rank, self.f = g, dom, rank, f


class PolarEvaluator:
    """Evaluate right-hand-side (and coefficient) expressions on a polar or cylindrical domain:
    spectral derivatives on coefficients (spin toolkit, i k_z along z), products on the dealiased
    grid, projection back (level 0). ``xp``: array module of the evaluation (NumPy, the oracle, or
    CuPy: same code on the GPU, transforms by cuFFT and batched cuBLAS; the module of the data
    decides each call, ``xp`` only creates the constants)."""

    def __init__(self, space: PolarSpace, xp=None, fold=None) -> None:
        self.S = space
        self.xp = np if xp is None else xp
        self._ops: dict = {}
        # symmetry subspace (curvilinear_symmetry.FoldSpec, m0-fold class): values on the sector
        self.fold = fold if (fold is not None and fold.active) else None

    def _fold_of(self, x):
        return None if self.fold is None else self.fold.fold(x)

    # ------------------------------------------------------------ public
    def evaluate(self, e: E.Expr, coeffs: Mapping[E.Field, np.ndarray],
                 params: Mapping[str, Any]) -> np.ndarray:
        """Coefficients (level 0, spin, shape PolarSpace.coeff_shape) of expression ``e``."""
        v = self._ev(e, coeffs, params)
        if isinstance(v, _C) and v.level == 0 and v.p == 0:
            return array_module(v.a).broadcast_to(v.a, self.S.coeff_shape(e.domain, e.rank)).copy()
        g = self._grid(v)
        xp = array_module(g.g)
        fe = self._fold_of(e)
        return self.S.forward(xp.broadcast_to(g.g, self.S.grid_shape(e.domain, e.rank, True, fe)),
                              e.domain, e.rank, True, physical=False, fold=fe)

    # ------------------------------------------------------------ conversions
    def _grid(self, v) -> _G:
        if isinstance(v, _G):
            return v
        if not isinstance(v, _C):                     # a number
            return _G(self.xp.asarray(v, dtype=complex), None, 0)
        g = self.S.backward(v.a, v.dom, v.rank, True, v.level, v.p, physical=False, fold=v.f)
        return _G(g, v.dom, v.rank, v.f)

    def _coeff(self, v) -> _C:
        if isinstance(v, _C):
            return v
        v = self._grid(v)
        full = array_module(v.g).broadcast_to(v.g, self.S.grid_shape(v.dom, v.rank, True, v.f))
        return _C(self.S.forward(full, v.dom, v.rank, True, physical=False, fold=v.f), v.dom, v.rank, f=v.f)

    # ------------------------------------------------------------ spectral operators
    def _opstack(self, name: str, ks: np.ndarray, ns: np.ndarray, level: int, p: int,
                 sigma: int = 0, xp=np) -> np.ndarray:
        """Per-m operator matrices of one component, zero-padded to N x N, stacked (nm, N, N).
        name: 'G' (spin derivative), 'L' (Laplacian L_k), 'C' (one level up and, on the annulus,
        one power of r: the z-derivative term next to a radial derivative), 'C2' (two levels, r^2)."""
        key = (name, tuple(ks), tuple(ns), level, p, sigma)
        if key not in self._ops:
            N, ops = self.S.N, self.S.ops
            ann = ops.kind == "annulus"
            out = np.zeros((len(ks), N, N))
            for j, (k, nk) in enumerate(zip(ks, ns, strict=True)):
                k, nk = int(k), int(nk)
                if nk <= 0:
                    continue
                if name == "G":
                    M = ops.G(sigma, k, level, p, nk)[0]
                elif name == "L":
                    M = ops.L(k, level, p, nk)[0]
                else:
                    d = 1 if name == "C" else 2
                    M = ops.convert(k, level, level + d, nk)
                    if ann:
                        M = ops.multiply_r(k, level + d, d, nk) @ M
                M = M.toarray()
                out[j, :M.shape[0], :M.shape[1]] = M
            self._ops[key] = out
        return on(xp, self._ops[key], self._ops, key)      # device copy cached once (CuPy)

    def _apply(self, name: str, v: _C, comp: tuple, sigma: int = 0) -> np.ndarray:
        kind = self.S.kind(v.dom)
        if kind not in ("full", "radial"):
            raise DiscretizationError("radial derivative of a quantity without a radial basis")
        xp = array_module(v.a)
        x = self.S._3(v.a[comp] if comp else v.a)
        if v.f is not None and v.f.zq is not None and self.S.hz(v.dom) and not v.f.zero:
            # axial folding: the operator acts on the k_z >= 0 entries, the -k_z entries follow
            q = v.f.zq * (-1) ** sum(1 for i in comp if i == 2)
            zpos, zneg, _, _ = self.S.zfold(self.S.sizes(True)[2], q)
            A = self._opstack(name, self.S._ks(kind, comp), self.S._ns(kind), v.level, v.p, sigma, xp)
            y = xp.matmul(A, x[..., zpos])
            out = xp.zeros(x.shape, dtype=complex)
            out[..., zpos] = y
            kpos = np.nonzero(zneg >= 0)[0]
            out[..., zneg[kpos]] = q * y[..., kpos]
            return self.S._2(out)
        if v.f is not None and v.f.m0 > 1 and kind == "full":
            # m0-fold class: only its wavenumbers are nonzero, the others are not computed
            from .curvilinear_symmetry import kept_index
            sel = kept_index(self.S.ms, v.f)
            key = ("sel", name, tuple(self.S._ks(kind, comp)), v.level, v.p, sigma, tuple(sel))
            if key not in self._ops:
                self._ops[key] = np.ascontiguousarray(
                    self._opstack(name, self.S._ks(kind, comp), self.S._ns(kind), v.level, v.p, sigma)[sel])
            out = xp.zeros(x.shape, dtype=complex)
            out[sel] = xp.matmul(on(xp, self._ops[key], self._ops, key), x[sel])
            return self.S._2(out)
        A = self._opstack(name, self.S._ks(kind, comp), self.S._ns(kind), v.level, v.p, sigma, xp)
        return self.S._2(xp.matmul(A, x))                   # batched over m (cuBLAS on the GPU)

    def _ikz(self, v: _C, x: np.ndarray, power: int = 1) -> np.ndarray:
        """(i k_z)^power along the z axis of a component array."""
        kz = self.S.kzlist(v.dom)
        xp = array_module(x)
        return x * xp.asarray((1j * kz) ** power)

    def _shift(self, v: _C, d: int = 1) -> tuple[int, int]:
        """(level, p) after d derivatives."""
        return v.level + d, v.p + (d if self.S.ops.kind == "annulus" else 0)

    def _set(self, out: np.ndarray, comp: tuple, val: np.ndarray) -> None:
        if comp:
            out[comp] = val
        else:
            out[...] = val

    def _grad(self, v: _C) -> _C:
        nc = self.S.nc
        out = array_module(v.a).zeros((nc,) + v.a.shape, dtype=complex)
        for comp in _comps(v.rank, nc):
            for si, sigma in enumerate(SPIN[:nc]):
                if si < 2:
                    val = self._apply("G", v, comp, sigma)
                else:                                   # z: i k_z, raised to the common level
                    val = self._ikz(v, self._apply("C", v, comp))
                out[(si,) + comp] = val
        lev, p = self._shift(v)
        return _C(out, v.dom, v.rank + 1, lev, p)

    def _div(self, v: _C) -> _C:
        if v.rank == 0:
            raise DiscretizationError("div of a scalar")
        out = array_module(v.a).zeros(v.a.shape[1:], dtype=complex)
        for comp in _comps(v.rank - 1, self.S.nc):
            t = 0.5 * (self._apply("G", v, (0,) + comp, +1) + self._apply("G", v, (1,) + comp, -1))
            if self.S.cyl:
                t = t + self._ikz(v, self._apply("C", v, (2,) + comp))
            self._set(out, comp, t)
        lev, p = self._shift(v)
        return _C(out, v.dom, v.rank - 1, lev, p)

    def _lap(self, v: _C) -> _C:
        out = array_module(v.a).zeros(v.a.shape, dtype=complex)
        for comp in _comps(v.rank, self.S.nc):
            y = self._apply("L", v, comp)
            if self.S.cyl:
                y = y + self._ikz(v, self._apply("C2", v, comp), 2)
            self._set(out, comp, y)
        lev, p = self._shift(v, 2)
        return _C(out, v.dom, v.rank, lev, p)

    def _curl(self, v: _C) -> _C:
        if self.S.cyl:
            raise DiscretizationError("curl on cylindrical domains: not implemented (use grad/dot)")
        lev, p = self._shift(v)
        if v.rank == 0:          # curl(psi) = grad(psi) x e_z: spin (+i G_- psi, -i G_+ psi)
            out = array_module(v.a).zeros((2,) + v.a.shape, dtype=complex)
            out[0] = 1j * self._apply("G", v, (), -1)
            out[1] = -1j * self._apply("G", v, (), +1)
            return _C(out, v.dom, 1, lev, p)
        if v.rank == 1:          # e_z . curl(u) = (i/2) (G_+ u_- - G_- u_+)
            out = 0.5j * (self._apply("G", v, (0,), +1) - self._apply("G", v, (1,), -1))
            return _C(out, v.dom, 0, lev, p)
        raise DiscretizationError("curl of a rank >= 2 tensor in 2D")

    # ------------------------------------------------------------ recursion
    def _ev(self, x: E.Expr, coeffs, params):
        """Value of node x, tagged with its fold in a folded evaluation (a rotation keeps the
        parity through grad / div / lap / curl, so derived values carry the node's own fold)."""
        v = self._ev_raw(x, coeffs, params)
        if self.fold is not None and isinstance(v, (_C, _G)):
            v.f = self.fold.fold(x)
        return v

    def _ev_raw(self, x: E.Expr, coeffs, params):
        S = self.S
        if isinstance(x, E.Field):
            if x not in coeffs:
                raise DiscretizationError(f"no data for field {x.name}")
            return _C(self.xp.asarray(coeffs[x]), x.domain, x.rank)
        if isinstance(x, E.Constant):
            if x.rank:
                raise DiscretizationError("tensor constants in polar right-hand sides")
            return x.value
        if isinstance(x, E.Parameter):
            v = params.get(x.name, x.value)
            if v is None:
                raise DiscretizationError(f"parameter {x.name} has no value")
            return v
        if isinstance(x, E.UnitVector):             # e_z of a cylinder (spin 0 component)
            g = self.xp.zeros((S.nc,) + (1,) * (3 if S.cyl else 2), dtype=complex)
            g[x.index] = 1.0
            return _G(g, None, 1)
        if isinstance(x, (E.Gradient, E.Divergence, E.Laplacian, E.Curl)):
            v = self._coeff(self._ev(x.args[0], coeffs, params))
            return {E.Gradient: self._grad, E.Divergence: self._div, E.Laplacian: self._lap,
                    E.Curl: self._curl}[type(x)](v)
        if isinstance(x, E.Differentiate):
            v = self._coeff(self._ev(x.args[0], coeffs, params))
            if S.cyl and x.coord == S.zc:
                return _C(self._ikz(v, v.a), v.dom, v.rank, v.level, v.p)
            if x.coord != S.phi or x.args[0].rank:
                raise DiscretizationError(
                    f"'{x}': partial derivatives in a polar RHS: only d/dphi of a scalar and d/dz (use "
                    "grad/div/lap/curl, which carry the metric terms)")
            m = S.mlist(S.kind(v.dom))
            mm = array_module(v.a).asarray(m[:, None, None] if S.cyl else m[:, None])
            return _C(1j * mm * v.a, v.dom, 0, v.level, v.p)
        if isinstance(x, E.Add):
            vals = [self._grid(self._ev(a, coeffs, params)) for a in x.args]
            out = vals[0].g
            for v in vals[1:]:
                out = out + v.g
            return _G(out, x.domain, x.rank)
        if isinstance(x, E.Mul):
            vals = [self._ev(a, coeffs, params) for a in x.args]
            num = [v for v in vals if not isinstance(v, (_C, _G))]
            arr = [self._grid(v) for v in vals if isinstance(v, (_C, _G))]
            c = 1.0
            for v in num:
                c = c * v
            if not arr:
                return c
            if sum(1 for a in arr if a.rank) > 1:
                raise DiscretizationError(f"'{x}': product of two tensors, use dot()")
            out = c
            for a in arr:
                out = out * a.g
            return _G(out, x.domain, x.rank)
        if isinstance(x, E.Dot):
            a = self._grid(self._ev(x.args[0], coeffs, params))
            b = self._grid(self._ev(x.args[1], coeffs, params))
            return _G(_dot(a.g, a.rank, b.g, b.rank, S.G), x.domain, x.rank)
        if isinstance(x, E.Power):
            v = self._ev(x.args[0], coeffs, params)
            if not isinstance(v, (_C, _G)):
                return v ** x.exponent
            g = self._grid(v)
            return _G(self._real_scalar(g.g, x) ** x.exponent, x.domain, 0)
        if isinstance(x, E.UnaryFunction):
            g = self._grid(self._ev(x.args[0], coeffs, params))
            if x.func not in ("exp", "log", "sin", "cos", "tanh", "sqrt", "abs", "sign"):
                if not E.is_user_function(x.func):
                    raise KeyError(x.func)
                return _G(np.asarray(x.numpy_func(self._real_scalar(g.g, x))), x.domain, 0)
            f = getattr(array_module(g.g), x.func)     # numpy.exp or cupy.exp, ...
            return _G(f(self._real_scalar(g.g, x)), x.domain, 0)
        if isinstance(x, E.Component):
            if (x.index < 2 and S.ops.kind == "disk"
                    and S.kind(x.args[0].domain) in ("full", "radial")):
                raise DiscretizationError(
                    f"'{x}': a physical component (u_r, u_phi) is not a regular function on the disk "
                    "(e_r, e_phi are singular at the axis: u = e_x gives u_r = cos(phi)); use dot "
                    "products, grad/div/curl, or the component at a boundary")
            g = self._grid(self._ev(x.args[0], coeffs, params))
            i = x.index
            val = sum(S.PH[i, s] * g.g[s] for s in range(S.nc))
            return _G(val, x.domain, x.rank)
        if isinstance(x, E.Conj):
            g = self._grid(self._ev(x.args[0], coeffs, params))
            xp = array_module(g.g)
            out = xp.conj(g.g)
            for ax in range(g.rank):          # conj(u_+) = u_- for a real vector: flip every spin
                out = xp.take(out, xp.asarray([1, 0, 2][:S.nc]), axis=ax)
            return _G(out, x.domain, x.rank)
        raise DiscretizationError(f"cannot evaluate node {type(x).__name__} ('{x}') in a polar RHS")

    def _real_scalar(self, g: np.ndarray, x: E.Expr) -> np.ndarray:
        """Pointwise functions of real fields act on real values (imaginary part checked)."""
        if self.S.real:
            return _real(g, f"'{x}'", self.S.func_real_tol, "func_real_tol")
        return g


