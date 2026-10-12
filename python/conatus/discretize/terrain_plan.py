"""Discretization on a terrain-following mapped domain: all Fourier modes coupled, banded in xi.

Domain: RealFourier along the periodic horizontal coordinates x (, y), ``TerrainChebyshev`` along
the vertical (``conatus.ir.terrain``: z = b + (t - b)(1 + xi)/2). The equations are the physical
ones (``lap``, ``grad``, ``div``, ``diff`` in x, y, z); each physical derivative is written by the
chain rule,

    dz = J dxi,      dx|_z = dx|_xi + (A_x + B_x xi) dxi,      dV = H dxi dx,

and compiled into sparse operators on the coefficients c[k, n] (complex Fourier modes
|k_d| <= N_d/2 - 1 of real fields, Chebyshev T_n / ultraspherical C^(lam)_n along xi), exactly as
the flat Cartesian discretizer does per Fourier group (Olver–Townsend: derivatives raise lam,
conversions S, multiplication by xi tridiagonal) -- but the metric functions J, A, B, H of x are
**convolutions** in k: the Fourier modes are coupled.

Choice ( mapped): the metric terms are **implicit**, in one coupled sparse
system for the whole domain (no splitting of the metric into an explicit correction): along xi
the bandwidth is the one of the flat channel plus one (metric polynomial of degree <= 1 in xi,
exact); along k it is the number of kept Fourier coefficients of the metric (reported by
``explain()``). Rationale: no stability or convergence condition on the wall slope (a split
explicit metric converges only for small slopes and would make every LBVP an iteration), the
same IR and the same tau recipe as the flat channel, and a flat map gives back the flat
operator (block diagonal in k). Cost: one sparse LU of the coupled system (SuperLU) instead of
one banded solve per group; refused above MAX_COUPLED_UNKNOWNS / MAX_COUPLED_NNZ (measured
costs in the guide).

Non-constant coefficients (given fields) are functions of (x, xi): 2D coefficients, multiplied
as a sum over Fourier offsets of Chebyshev multiplication operators; the metric and every NCC
are checked against ``ncc_tail_tol`` and refused if under-resolved. Explicit terms of an
IVP (and data) are evaluated pointwise on the dealiased physical grid (x_j, z(x_j, xi_l)).

Supported: LBVP (``plan.solve()``), EVP (``conatus.eigenvalues(plan)``, one coupled "group"),
IVP (``TerrainSimulation``: SBDF1-4, RK111/222/443; time-dependent boundary data imposed at the
stage time). Refused with a message: traces other than at the two surfaces, integrals along x
at fixed z, curl/cross, conj, complex Fourier, GPU.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import structural_rank

from ..errors import DiscretizationError, OptionError, RankError, ResolutionError, UnsupportedError
from ..ir import expr as E
from ..ir.bases import RealFourier
from ..ir.domain import Domain
from ..ir.problem import EVP, IVP, LBVP, Problem
from ..ir.terrain import TerrainChebyshev, is_terrain
from ..tolerances import NCC_ROUNDING, NCC_TAIL_TOL
from . import ultraspherical as us
from .compiler import _comps, eigen_split
from .param_keys import check_param_keys
from .plan import GroupMatrices

__all__ = ["TerrainPlan", "TerrainSolution", "TerrainSimulation", "has_terrain"]

DATA = "data"                  # pseudo-key of a known coefficient column (data, state)
_DOC = "docs/site/guide/mapped.md"
# Size limits of the direct coupled solve (declared in conatus.tolerances.DECLARED, refusal with
# the measured size): the sparse LU of a system that couples every Fourier mode fills in far
# beyond nnz(L). Measured on one CPU core (SuperLU): 2D Stokes 128 x 32 (12 701 unknowns,
# nnz(L) = 4.2e6) ~ 90 s and 1.8 GB; 3D Poisson 24 x 24 x 16 (9 522 unknowns, nnz 9.7e6) ~ 16 s and
# 1.1 GB. Raise them (module attributes) knowingly.
MAX_COUPLED_UNKNOWNS = 100_000   # checked before assembly (the operators alone would not fit)
MAX_COUPLED_NNZ = 10_000_000     # nnz(L) + nnz(M), checked before any factorization


def has_terrain(problem: Problem) -> bool:
    """True if a variable of ``problem`` lives on a terrain-following basis."""
    for v in problem.variables:
        if v.domain is not None and any(is_terrain(b) for b in v.domain.bases):
            return True
    return False


@dataclass
class Ent:
    """One tensor component of a compiled expression: lam (None: constant along xi), f (False:
    constant in the horizontal directions, only the mean mode), and its blocks: sparse matrices
    acting on the unknowns' coefficients, or a vector under DATA (known values)."""

    lam: int | None
    f: bool
    blocks: dict

    @property
    def empty(self) -> bool:
        return not self.blocks

    @property
    def is_data(self) -> bool:
        return bool(self.blocks) and set(self.blocks) == {DATA}

    @property
    def is_scalar(self) -> bool:
        return self.is_data and self.lam is None and not self.f


def _zero() -> Ent:
    return Ent(None, False, {})


def _scalar(v: complex) -> Ent:
    return Ent(None, False, {DATA: np.array([v], dtype=complex)})


@dataclass
class NCCRecord:
    name: str
    fourier: int          # Fourier offsets used
    cheb: int             # Chebyshev coefficients used
    tail_f: float
    tail_z: float
    tol: float

    def explain(self) -> str:
        return (f"NCC {self.name}: {self.fourier} Fourier offsets x {self.cheb} Chebyshev "
                f"coefficients, sampled tails {self.tail_f:.1e} (Fourier) / {self.tail_z:.1e} "
                f"(Chebyshev), accepted up to ncc_tail_tol = {self.tol:g}")


class TerrainPlan:
    """Discretized problem on a terrain-following mapped domain (one coupled sparse system)."""

    def __init__(self, problem: Problem, ncc: Mapping[E.Field, Any] | None = None,
                 params: Mapping[str, Any] | None = None, ncc_cutoff: float | None = None,
                 assembly: str = "direct", ncc_tail_tol: float = NCC_TAIL_TOL) -> None:
        if assembly != "direct":
            raise OptionError(f"assembly={assembly!r} on a mapped domain",
                              why="the Fourier modes are coupled by the metric: there is one "
                                  "system, nothing to interpolate between wavenumbers",
                              fix="drop assembly=", doc=_DOC)
        if ncc_cutoff is not None:
            raise OptionError("ncc_cutoff= on a mapped domain",
                              why="a truncation of the 2D (Fourier x Chebyshev) coefficients is "
                                  "not implemented; nothing is truncated",
                              fix="drop ncc_cutoff= (or declare ncc_tail_tol=)", doc=_DOC)
        self.problem = problem
        self.ncc_tail_tol = us.check_tail_tol(ncc_tail_tol)
        self.variables = tuple(problem.variables)
        self.V = frozenset(self.variables)
        dom = None
        for v in self.variables:
            dom = Domain.union(dom, v.domain)
        for eq in problem.equations:
            dom = Domain.union(dom, eq.domain)
        self.domain = dom
        cs = dom.coordsys
        if cs.curvilinear:
            raise UnsupportedError("terrain-following map in curvilinear coordinates", doc=_DOC)
        self.coords = tuple(cs.coords)
        zb = [b for b in dom.bases if is_terrain(b)]
        if len(zb) != 1:
            raise DiscretizationError("a mapped domain needs exactly one TerrainChebyshev basis")
        self.zb: TerrainChebyshev = zb[0]
        self.zc = self.zb.coord
        self.terrain = self.zb.terrain
        self.fdirs = []
        for c in self.coords:
            if c == self.zc:
                continue
            b = dom.basis(c)
            if not isinstance(b, RealFourier):
                raise UnsupportedError(f"mapped domain: {c.name} must be RealFourier, got {b!r}",
                                       doc=_DOC)
            if b.size % 2 or b.size < 4:
                raise DiscretizationError(f"{b!r}: an even Fourier size >= 4 is required")
            self.fdirs.append((c, b))
        if tuple(c.name for c, _ in self.fdirs) != self.terrain.names:
            raise DiscretizationError("the periodic coordinates of the domain and of the terrain "
                                      "map differ")
        self.Nz = self.zb.size
        self.Ks = [b.size // 2 - 1 for _, b in self.fdirs]
        self.nks = [2 * k + 1 for k in self.Ks]
        self.nk = int(np.prod(self.nks))
        self.Ls = [b.length for _, b in self.fdirs]
        # wavenumber of each flattened mode, per direction
        grids = np.meshgrid(*[np.arange(-k, k + 1) for k in self.Ks], indexing="ij")
        self.kvec = [2 * np.pi * g.ravel() / L for g, L in zip(grids, self.Ls, strict=True)]
        self.k0 = int(np.ravel_multi_index(tuple(self.Ks), tuple(self.nks)))
        # parameters
        self.params: dict[str, Any] = {}
        for p in self._parameters():
            if p.value is not None:
                self.params[p.name] = p.value
        if params:
            # a key naming no Parameter of the problem (a typo) was ignored here and the value
            # declared on the Parameter used instead: refused, as by
            # the flat, polar and spherical plans (audit R1)
            check_param_keys(params, {p.name for p in self._parameters()},
                             where="discretize (mapped domain)")
            self.params.update(params)
        self.eigenvalue = problem.eigenvalue if isinstance(problem, EVP) else None
        # data of the given fields; data for a field the problem does not use is refused (same rule)
        used = frozenset(f for eq in problem.equations for side in (eq.lhs, E.as_expr(eq.rhs))
                         for f in side.free_fields())
        stray = sorted(f.name for f in (ncc or {}) if f not in used)
        if stray:
            raise DiscretizationError(
                f"ncc given for field(s) not used by the problem: {', '.join(stray)}",
                fix="remove them from ncc= (or check the field names)", doc=_DOC)
        self._ncc_src = dict(ncc or {})
        # metric ( check at this resolution)
        try:
            self.metric, self.metric_reports = self.terrain.metric_coeffs(
                tuple(b.size for _, b in self.fdirs), self.ncc_tail_tol)
        except ValueError as err:
            raise DiscretizationError(str(err), where="terrain map", doc=_DOC) from None
        self._conv_cache: dict = {}
        self._ccache: dict = {}
        self._dcache: dict = {}
        self._ncc_ops: dict = {}
        self.ncc_reports: dict[str, NCCRecord] = {}
        # relative Nyquist amplitude of each given field sampled on the N grid (the kept modes
        # |k| <= N/2 - 1 drop it: energy there is aliasing that the tail of the kept modes misses)
        self._nyquist: dict[E.Field, float] = {}
        self._state: dict | None = None
        self.taus = {n.args[0] for eq in problem.equations for n in eq.lhs.walk()
                     if isinstance(n, E.Lift) and isinstance(n.args[0], E.Field)}
        self.groups = [("coupled",)]
        self._gm: GroupMatrices | None = None
        self._layout()

    # ------------------------------------------------------------------ layout
    def _parameters(self):
        out = set()
        for eq in self.problem.equations:
            out |= eq.lhs.free_parameters() | E.as_expr(eq.rhs).free_parameters()
        return out

    def _space(self, dom: Domain | None) -> tuple[int | None, bool]:
        """(lam, f) of a field or equation living on ``dom``."""
        if dom is None:
            return None, False
        has_z = dom.has(self.zc)
        has_f = any(dom.has(c) for c, _ in self.fdirs)
        return (0 if has_z else None), (has_f or has_z)

    def rows(self, lam: int | None, f: bool) -> int:
        return (self.nk if f else 1) * (self.Nz if lam is not None else 1)

    def _layout(self) -> None:
        self.var_space, self.var_off = {}, {}
        off = 0
        for v in self.variables:
            lam, f = self._space(v.domain)
            for c in _comps(v.tensorsig):
                self.var_space[(v, c)] = (lam, f)
                self.var_off[(v, c)] = off
                off += self.rows(lam, f)
        self.ncol = off
        if off > MAX_COUPLED_UNKNOWNS:
            raise UnsupportedError(
                f"mapped domain: {off} coupled unknowns > MAX_COUPLED_UNKNOWNS = "
                f"{MAX_COUPLED_UNKNOWNS}",
                why="the metric couples every Fourier mode: one sparse LU of the whole system, "
                    "whose memory and time grow much faster than the number of unknowns",
                fix="lower the resolution (the coupled system holds (2K+1) x N_z coefficients per "
                    "field and per horizontal direction product), or raise "
                    "conatus.discretize.terrain_plan.MAX_COUPLED_UNKNOWNS knowingly", doc=_DOC)

    # ------------------------------------------------------------------ elementary operators
    def _IF(self, f: bool):
        return sp.identity(self.nk if f else 1, format="csr")

    def _IZ(self, lam):
        return sp.identity(self.Nz if lam is not None else 1, format="csr")

    @staticmethod
    def _left(e: Ent, A, lam, f) -> Ent:
        return Ent(lam, f, {k: A @ b for k, b in e.blocks.items()})

    def bcast_f(self, e: Ent) -> Ent:
        if e.f or e.empty:
            return Ent(e.lam, True, e.blocks) if e.empty else e
        col = sp.csr_matrix(([1.0], ([self.k0], [0])), shape=(self.nk, 1))
        return self._left(e, sp.kron(col, self._IZ(e.lam), format="csr"), e.lam, True)

    def bcast_z(self, e: Ent) -> Ent:
        if e.lam is not None:
            return e
        col = sp.csr_matrix(([1.0], ([0], [0])), shape=(self.Nz, 1))
        return self._left(e, sp.kron(self._IF(e.f), col, format="csr"), 0, e.f)

    def convert(self, e: Ent, lam: int) -> Ent:
        if e.lam == lam:
            return e
        S = us.conversion_chain(self.Nz, e.lam, lam)
        return self._left(e, sp.kron(self._IF(e.f), S, format="csr"), lam, e.f)

    def add(self, ents: list[Ent]) -> Ent:
        ents = [e for e in ents if not e.empty]
        if not ents:
            return _zero()
        f = any(e.f for e in ents)
        zl = [e.lam for e in ents if e.lam is not None]
        lam = max(zl) if zl else None
        out: dict = {}
        for e in ents:
            if f:
                e = self.bcast_f(e)
            if lam is not None:
                e = self.convert(self.bcast_z(e), lam)
            for k, b in e.blocks.items():
                out[k] = b if k not in out else out[k] + b
        return Ent(lam, f, out)

    @staticmethod
    def scale(e: Ent, c: complex) -> Ent:
        if c == 1:
            return e
        if c == 0:
            return _zero()
        return Ent(e.lam, e.f, {k: c * b for k, b in e.blocks.items()})

    def dxi(self, e: Ent) -> Ent:
        if e.empty or e.lam is None:
            return _zero()
        D = us.differentiation(self.Nz, e.lam, 1)
        return self._left(e, sp.kron(self._IF(e.f), D, format="csr"), e.lam + 1, e.f)

    def ximul(self, e: Ent) -> Ent:
        if e.empty:
            return e
        e = self.bcast_z(e)
        X = us.multiplication_by_t(self.Nz, e.lam)
        return self._left(e, sp.kron(self._IF(e.f), X, format="csr"), e.lam, e.f)

    def dfourier(self, e: Ent, d: int) -> Ent:
        if e.empty or not e.f:
            return _zero()
        Dk = sp.diags(1j * self.kvec[d], format="csr")
        return self._left(e, sp.kron(Dk, self._IZ(e.lam), format="csr"), e.lam, True)

    def _conv(self, a: np.ndarray) -> sp.csr_matrix:
        """Convolution (multiplication by sum_p a_p e^{i k_p x}) on the kept modes; ``a`` holds
        the offsets -P_d..P_d along each direction (shape (2 P_d + 1, ...)). Built in one
        vectorized COO pass: out[k, k - p] = a_p for every kept k and k - p."""
        P = np.array([(s - 1) // 2 for s in a.shape])
        offs = np.argwhere(a != 0)
        if offs.size == 0:
            return sp.csr_matrix((self.nk, self.nk), dtype=complex)
        vals = a[tuple(offs.T)]
        p = offs - P                                            # (np, d)
        K = np.array(self.Ks)
        m = np.stack(np.unravel_index(np.arange(self.nk), tuple(self.nks)), axis=1) - K   # (nk, d)
        src = m[:, None, :] - p[None, :, :]                     # (nk, np, d)
        ok = np.all(np.abs(src) <= K, axis=2)
        r, j = np.nonzero(ok)
        cols = np.ravel_multi_index(tuple((src[r, j] + K).T), tuple(self.nks))
        return sp.csr_matrix((vals[j], (r, cols)), shape=(self.nk, self.nk))

    def metric_conv(self, name: str) -> sp.csr_matrix:
        if name not in self._conv_cache:
            self._conv_cache[name] = self._conv(self.metric[name])
        return self._conv_cache[name]

    def fmul(self, e: Ent, name: str) -> Ent:
        """Multiplication by the metric function ``name`` of the horizontal coordinates."""
        if e.empty:
            return e
        e = self.bcast_f(e)
        C = self.metric_conv(name)
        if C.nnz == 0:
            return _zero()
        return self._left(e, sp.kron(C, self._IZ(e.lam), format="csr"), e.lam, True)

    # physical derivatives (chain rule) ----------------------------------------------------
    def pdiff(self, e: Ent, coord) -> Ent:
        if e.empty:
            return e
        if coord == self.zc:
            return self.fmul(self.dxi(e), "J")
        d = next((i for i, (c, _) in enumerate(self.fdirs) if c == coord), None)
        if d is None:
            raise DiscretizationError(f"derivative along {coord.name}: not a coordinate of the "
                                      "mapped domain")
        terms = [self.dfourier(e, d)]
        if e.lam is not None:
            g = self.dxi(e)
            n = coord.name
            terms += [self.fmul(g, f"A_{n}"), self.fmul(self.ximul(g), f"B_{n}")]
        return self.add(terms)

    def interp_side(self, e: Ent, pos: float) -> Ent:
        if e.empty or e.lam is None:
            return e
        row = sp.csr_matrix(us.evaluation_row(self.Nz, e.lam, pos).reshape(1, -1))
        return self._left(e, sp.kron(self._IF(e.f), row, format="csr"), None, e.f)

    def integ_z(self, e: Ent) -> Ent:
        if e.empty:
            return e
        e = self.bcast_z(self.fmul(e, "H"))
        row = sp.csr_matrix(us.integration_row(self.Nz, e.lam).reshape(1, -1))
        return self._left(e, sp.kron(self._IF(e.f), row, format="csr"), None, e.f)

    def integ_f(self, e: Ent) -> Ent:
        if e.empty:
            return e
        L = float(np.prod(self.Ls))
        if not e.f:
            return self.scale(e, L)
        row = sp.csr_matrix(([L], ([0], [self.k0])), shape=(1, self.nk))
        return self._left(e, sp.kron(row, self._IZ(e.lam), format="csr"), e.lam, False)

    def lift(self, e: Ent, n: int, lam: int) -> Ent:
        if e.empty:
            return e
        if e.lam is not None:
            raise DiscretizationError("lift of an operand that depends on the vertical coordinate")
        m = self.Nz + n
        col = sp.csr_matrix(([1.0], ([m], [0])), shape=(self.Nz, 1))
        return self._left(e, sp.kron(self._IF(e.f), col, format="csr"), lam, e.f)

    # ------------------------------------------------------------------ grids and transforms
    def grid_sizes(self, dealias: bool) -> tuple[list[int], int]:
        Ms = [(b.grid_size if dealias else b.size) for _, b in self.fdirs]
        Mz = self.zb.grid_size if dealias else self.Nz
        for M, n in zip(Ms, self.nks, strict=True):
            if M < n:
                raise DiscretizationError(f"dealiased Fourier grid of {M} points < {n} modes")
        return Ms, max(Mz, 1)

    def grids(self, dealias: bool) -> tuple[list[np.ndarray], np.ndarray]:
        """Horizontal grids (broadcastable, coordinate order) and xi points."""
        Ms, Mz = self.grid_sizes(dealias)
        d = len(self.fdirs)
        hg = []
        for i, ((_, b), M) in enumerate(zip(self.fdirs, Ms, strict=True)):
            sh = [1] * d
            sh[i] = M
            hg.append((b.bounds[0] + b.length * np.arange(M) / M).reshape(sh))
        xi = -np.cos(np.pi * (np.arange(Mz) + 0.5) / Mz)
        return hg, xi

    def physical_grid(self, dealias: bool = False) -> list[np.ndarray]:
        """Coordinate arrays in coordinate order, broadcast on the full grid (x_j, z(x_j, xi_l))."""
        hg, xi = self.grids(dealias)
        Z = self.terrain.z_of(xi, hg)
        full = Z.shape
        out = []
        it = iter(hg)
        for c in self.coords:
            if c == self.zc:
                out.append(Z)
            else:
                out.append(np.broadcast_to(next(it)[..., None], full))
        return out

    def _fidx(self, Ms: list[int]):
        return np.ix_(*[np.arange(-k, k + 1) % M for k, M in zip(self.Ks, Ms, strict=True)])

    def forward(self, vals: np.ndarray, lam_none: bool, f: bool) -> np.ndarray:
        """Grid values -> coefficient vector (T along xi) of space (lam 0 | None, f)."""
        d = len(self.fdirs)
        v = np.asarray(vals, dtype=complex)
        if not f:
            v = v.reshape(-1)
            if lam_none:
                return v[:1].copy()
            raise AssertionError("xi-dependent data constant in x")
        Ms = list(v.shape[:d])
        c = np.fft.fftn(v, axes=tuple(range(d))) / float(np.prod(Ms))
        c = c[self._fidx(Ms)] if lam_none else c[self._fidx(Ms) + (slice(None),)]
        if lam_none:
            return c.reshape(-1)
        Mz = v.shape[-1]
        y = sfft.dct(c.real, type=2, axis=-1) + 1j * sfft.dct(c.imag, type=2, axis=-1)
        y = y / Mz
        y[..., 0] /= 2
        y = y * (-1.0) ** np.arange(Mz)
        y = y[..., :self.Nz]
        if Mz < self.Nz:
            y = np.concatenate([y, np.zeros(y.shape[:-1] + (self.Nz - Mz,))], axis=-1)
        return y.reshape(-1)

    def backward(self, vec: np.ndarray, lam, f: bool, dealias: bool) -> np.ndarray:
        """Coefficient vector of space (lam, f) -> values on the full grid (complex)."""
        Ms, Mz = self.grid_sizes(dealias)
        d = len(self.fdirs)
        vec = np.asarray(vec, dtype=complex)
        if lam is not None and lam > 0:
            S = us.conversion_chain(self.Nz, 0, lam).tocsr()
            Y = vec.reshape(-1, self.Nz).T
            vec = np.asarray(spla.spsolve_triangular(S, Y, lower=False)).T.reshape(-1)
        nkf = self.nk if f else 1
        nz = self.Nz if lam is not None else 1
        c = vec.reshape(nkf, nz)
        if not f:
            full = np.zeros((self.nk, nz), dtype=complex)
            full[self.k0] = c[0]
            c = full
        c = c.reshape(tuple(self.nks) + (nz,))
        if lam is not None:
            n = np.arange(self.Nz)
            y = c * (-1.0) ** n
            y = y.copy()
            y[..., 1:] /= 2
            if Mz > self.Nz:
                y = np.concatenate([y, np.zeros(y.shape[:-1] + (Mz - self.Nz,))], axis=-1)
            else:
                y = y[..., :Mz]
            z = sfft.dct(y.real, type=3, axis=-1) + 1j * sfft.dct(y.imag, type=3, axis=-1)
        else:
            z = c
        arr = np.zeros(tuple(Ms) + (z.shape[-1],), dtype=complex)
        arr[self._fidx(Ms) + (slice(None),)] = z
        g = np.fft.ifftn(arr, axes=tuple(range(d))) * float(np.prod(Ms))
        if lam is None:
            g = np.broadcast_to(g, tuple(Ms) + (Mz,))
        return g

    # ------------------------------------------------------------------ data
    def _param(self, p: E.Parameter) -> complex:
        if self.eigenvalue is not None and p == self.eigenvalue:
            raise DiscretizationError(f"eigenvalue {p.name} outside the linear pencil")
        if p.name not in self.params:
            raise DiscretizationError(f"parameter {p.name} has no value",
                                      fix=f"Parameter('{p.name}', value) or params={{'{p.name}': ...}}")
        return complex(self.params[p.name])

    def _given(self, f: E.Field) -> dict:
        """Coefficients of a given field (data callable sampled on the N grid, physical z)."""
        key = ("given", f)
        if key in self._dcache:
            return self._dcache[key]
        if f not in self._ncc_src:
            raise DiscretizationError(f"no data for the field {f.name}",
                                      fix="pass ncc={field: callable of the coordinate grids}")
        d = self._ncc_src[f]
        lam, fd = self._space(f.domain)
        if not callable(d):
            raise DiscretizationError(f"data for {f.name}: a callable of the coordinate grids is "
                                      "expected on a mapped domain (FieldData is not supported)")
        hg, _ = self.grids(False)
        hshape = tuple(a.shape[i] for i, a in enumerate(hg))
        if lam is not None:
            args = self.physical_grid(False)
            shape = args[0].shape
        else:
            it = iter(hg)
            args = [next(it) if c != self.zc else np.zeros((1,) * len(hg)) for c in self.coords]
            shape = hshape
        val = np.asarray(d(*args))
        out = {}
        nyq = 0.0
        for c in _comps(f.tensorsig):
            g = np.asarray(val[c] if c else val)
            if not np.all(np.isfinite(g)):
                raise DiscretizationError(f"data of {f.name} has non-finite values on the grid")
            if lam is None and not fd:
                out[c] = _scalar(complex(g.reshape(-1)[0]))
                continue
            g = np.broadcast_to(g, shape)
            nyq = max(nyq, self._nyquist_ratio(g))
            out[c] = Ent(lam, True, {DATA: self.forward(g, lam is None, True)})
        self._nyquist[f] = nyq
        self._dcache[key] = out
        return out

    def _nyquist_ratio(self, g: np.ndarray) -> float:
        """max |coefficient at the Nyquist wavenumber of some direction| / max |coefficient| of
        grid values ``g`` (horizontal axes first, N-point grid)."""
        d = len(self.fdirs)
        c = np.abs(np.fft.fftn(np.asarray(g, dtype=complex), axes=tuple(range(d))))
        amax = float(c.max())
        if amax == 0:
            return 0.0
        worst = 0.0
        for i in range(d):
            m = c.shape[i]
            if m % 2 == 0:
                worst = max(worst, float(np.take(c, m // 2, axis=i).max()) / amax)
        return worst

    def _check_ncc_aliasing(self, x: E.Expr) -> None:
        """A coefficient multiplying an unknown (implicit term) must not have energy at the
        Nyquist wavenumber of the N grid: there its samples alias (e.g. exp(3 sin 4x) at N = 16
        puts mode 8 on Nyquist and mode 12 on -4), the tail of the kept modes can be zero and the
        operator would silently be another one."""
        floor = NCC_ROUNDING * np.finfo(float).eps
        for f in x.free_fields() - self.V:
            if f not in self._ncc_src:
                continue
            self._given(f)
            r = self._nyquist.get(f, 0.0)
            if r > max(self.ncc_tail_tol, floor):
                raise DiscretizationError(
                    f"NCC {f.name}: not resolved on this grid (relative amplitude {r:.2e} at the "
                    f"Nyquist wavenumber of the periodic direction(s) > ncc_tail_tol = "
                    f"{self.ncc_tail_tol:g})",
                    where=f"non-constant coefficient {x} on the mapped domain",
                    why="its samples on the N-point grid alias higher wavenumbers onto the kept "
                        "ones: the operator would multiply by another function (refused, never "
                        "truncated silently)",
                    fix=f"raise the horizontal resolution, or declare the accepted tail with "
                        f"ncc_tail_tol={_round_up(r):g}", doc=_DOC)

    def _grid_value(self, e: Ent, dealias: bool = True) -> np.ndarray:
        if e.empty:
            Ms, Mz = self.grid_sizes(dealias)
            return np.zeros(tuple(Ms) + (Mz,), dtype=complex)
        if e.is_scalar:
            return e.blocks[DATA][0]
        return self.backward(e.blocks[DATA], e.lam, e.f, dealias)

    def _from_grid(self, g, lam_none: bool) -> Ent:
        if np.ndim(g) == 0:
            return _scalar(complex(g))
        if lam_none:
            g = np.asarray(g)[..., 0]
        return Ent(None if lam_none else 0, True, {DATA: self.forward(g, lam_none, True)})

    def _pointwise(self, ents: list[Ent], fn: Callable) -> Ent:
        """fn(values...) evaluated on the dealiased grid, back to coefficients."""
        if all(e.is_scalar for e in ents):
            return _scalar(complex(fn(*[e.blocks[DATA][0] for e in ents])))
        lam_none = all(e.lam is None for e in ents if not e.empty)
        vals = [self._grid_value(e, True) for e in ents]
        g = fn(*vals)
        if not np.all(np.isfinite(g)):
            raise DiscretizationError("non-finite values in a pointwise evaluation on the grid")
        return self._from_grid(g, lam_none)

    def _ncc_operator(self, d: Ent, lam: int, f: bool, name: str) -> sp.csr_matrix:
        """Matrix of the multiplication by the data entry ``d`` on space (lam, f) -> (lam, True)."""
        key = (name, lam, f)        # name: the coefficient expression and its component
        if key in self._ncc_ops:
            return self._ncc_ops[key]
        vec = d.blocks[DATA]
        nkf = self.nk if d.f else 1
        if d.lam is not None and d.lam > 0:
            S = us.conversion_chain(self.Nz, 0, d.lam).tocsr()
            vec = np.asarray(spla.spsolve_triangular(S, vec.reshape(-1, self.Nz).T,
                                                     lower=False)).T.reshape(-1)
        nz = self.Nz if d.lam is not None else 1
        a = vec.reshape(nkf, nz)
        if not d.f:
            full = np.zeros((self.nk, nz), dtype=complex)
            full[self.k0] = a[0]
            a = full
        a = a.reshape(tuple(self.nks) + (nz,))
        amax = float(np.abs(a).max()) if a.size else 0.0
        tz = tf = 0.0
        if amax > 0:
            floor = NCC_ROUNDING * np.finfo(float).eps
            if nz > 3:
                tz = float(np.abs(a[..., -3:]).max()) / amax
            shell = np.zeros(a.shape[:-1], dtype=bool)
            for i, n in enumerate(self.nks):
                sel = np.zeros(n, dtype=bool)
                sel[[0, 1, n - 2, n - 1]] = True
                sh = [1] * len(self.nks)
                sh[i] = n
                shell |= np.broadcast_to(sel.reshape(sh), shell.shape)
            if d.f:
                tf = float(np.abs(a[shell]).max()) / amax
            for tail, along in ((tz, f"the vertical ({self.zc.name}, xi)"),
                                (tf, "the periodic direction(s)")):
                if tail > max(self.ncc_tail_tol, floor):
                    raise DiscretizationError(
                        f"NCC {name}: not resolved on this grid (relative tail {tail:.2e} along "
                        f"{along} > ncc_tail_tol = {self.ncc_tail_tol:g})",
                        where=f"non-constant coefficient {name} on the mapped domain",
                        why="the coefficient is replaced by its interpolant on the N-point grid; "
                            "with this tail the equations solved are not the ones written "
                            "(refused, never truncated silently)",
                        fix=f"raise the resolution along {along}, or declare the accepted tail "
                            f"with ncc_tail_tol={_round_up(tail):g}", doc=_DOC)
        P = [k for k in self.Ks]
        out = sp.csr_matrix((self.rows(lam, True), self.rows(lam, f)), dtype=complex)
        bcol = None
        if not f:
            bcol = sp.kron(sp.csr_matrix(([1.0], ([self.k0], [0])), shape=(self.nk, 1)),
                           self._IZ(lam), format="csr")
        nf = 0
        cheb = 0
        it = np.ndindex(*a.shape[:-1])
        for idx in it:
            col = a[idx]
            if not np.any(np.abs(col) > 0):
                continue
            p = [int(i) - q for i, q in zip(idx, P, strict=True)]
            S = None
            for pd, n in zip(p, self.nks, strict=True):
                D = sp.diags(np.ones(n - abs(pd)), -pd, shape=(n, n), format="csr")
                S = D if S is None else sp.kron(S, D, format="csr")
            if lam is None:
                if nz != 1:
                    raise DiscretizationError(f"NCC {name} depends on the vertical coordinate in an "
                                              "expression constant along it")
                Z = sp.identity(1, format="csr") * col[0]
            else:
                if nz == 1:
                    Z = col[0] * sp.identity(self.Nz, format="csr")
                else:
                    Z, _ = us.multiplication(col, self.Nz, lam, name=name, exact=True)
                    cheb = max(cheb, int(np.nonzero(np.abs(col) > 0)[0][-1]) + 1)
            out = out + (sp.kron(S, Z, format="csr") @ bcol if bcol is not None
                         else sp.kron(S, Z, format="csr"))
            nf += 1
        self.ncc_reports[name] = NCCRecord(name, nf, max(cheb, 1), tf, tz, self.ncc_tail_tol)
        out = out.tocsr()
        self._ncc_ops[key] = out
        return out

    def ncc_mul(self, e: Ent, d: Ent, name: str) -> Ent:
        if e.empty or d.empty:
            return _zero()
        if d.is_scalar:
            return self.scale(e, d.blocks[DATA][0])
        if d.lam is not None:
            e = self.bcast_z(e)
        A = self._ncc_operator(d, e.lam, e.f, name)
        return self._left(e, A, e.lam, True)

    # ------------------------------------------------------------------ compiler
    def _sub(self, x: E.Expr, mode: str) -> dict:
        if mode == "V":
            return self._value(x)
        if not (x.free_fields() & self.V):
            self._check_ncc_aliasing(x)   # a coefficient (NCC) operand of an implicit term
            return self._data(x)
        return self._c(x, mode)

    def _data(self, x: E.Expr) -> dict:
        """Value of an expression without unknowns (cached for the current parameter values: one
        entry per expression, recomputed when a value it uses changes, e.g. the time t)."""
        pk = tuple(sorted((p.name, self.params.get(p.name)) for p in x.free_parameters()
                          if p != self.eigenvalue))
        hit = self._dcache.get(("data", x))
        if hit is None or hit[0] != pk:
            hit = (pk, self._c(x, "V"))
            self._dcache[("data", x)] = hit
        return hit[1]

    def _value(self, x: E.Expr) -> dict:
        unk = x.free_fields() & self.V
        if not unk:
            return self._data(x)
        if x.degrees(self.V) == frozenset({1}) and not any(
                isinstance(n, E.TimeDerivative) for n in x.walk()):
            ents = self._c(x, "L")
            out = {}
            for c, e in ents.items():
                acc = None
                for k, blk in e.blocks.items():
                    v = blk @ self._state[k]
                    acc = v if acc is None else acc + v
                out[c] = Ent(e.lam, e.f, {DATA: acc} if acc is not None else {})
            return out
        return self._c(x, "V")

    def _c(self, x: E.Expr, mode: str) -> dict:
        if mode != "V":
            key = (x, mode)
            if key in self._ccache:
                return self._ccache[key]
            out = self._c_raw(x, mode)
            self._ccache[key] = out
            return out
        return self._c_raw(x, mode)

    def _c_raw(self, x: E.Expr, mode: str) -> dict:
        if mode in ("L", "M") and not (x.free_fields() & self.V):
            if isinstance(x, E.TimeDerivative):
                return {}
            raise DiscretizationError(f"term '{x}' does not depend on the unknowns: it belongs to "
                                      "the right-hand side")
        if isinstance(x, E.Field):
            if x in self.V:
                if mode == "M":
                    return {}
                if mode == "V":
                    return {c: Ent(*self.var_space[(x, c)], {DATA: self._state[(x, c)]})
                            for c in _comps(x.tensorsig)}
                out = {}
                for c in _comps(x.tensorsig):
                    lam, f = self.var_space[(x, c)]
                    out[c] = Ent(lam, f, {(x, c): sp.identity(self.rows(lam, f), format="csr")})
                return out
            return self._given(x)
        if isinstance(x, E.Parameter):
            return {(): _scalar(self._param(x))}
        if isinstance(x, E.Constant):
            v = np.asarray(x.value)
            if x.rank == 0:
                return {(): _scalar(complex(v))} if v != 0 else {}
            return {c: _scalar(complex(v[c])) for c in _comps(x.tensorsig) if v[c] != 0}
        if isinstance(x, E.UnitVector):
            return {(x.index,): _scalar(1.0)}
        if isinstance(x, E.Add):
            parts = [self._sub(t, mode) for t in x.args]
            comps = {c for p in parts for c in p}
            return {c: self.add([p[c] for p in parts if c in p]) for c in comps}
        if isinstance(x, E.Mul):
            return self._mul(x, mode)
        if isinstance(x, E.Dot):
            a, b = (self._sub(t, mode) for t in x.args)
            out: dict = {}
            for ca, ea in a.items():
                for cb, eb in b.items():
                    if ca[-1] != cb[0]:
                        continue
                    c = ca[:-1] + cb[1:]
                    out.setdefault(c, []).append(self._prod(ea, eb, mode, f"{x}{list(ca)}{list(cb)}"))
            return {c: self.add(v) for c, v in out.items()}
        if isinstance(x, E.Component):
            a = self._sub(x.args[0], mode)
            return {c[1:]: e for c, e in a.items() if c[0] == x.index}
        if isinstance(x, E.Differentiate):
            a = self._sub(x.args[0], mode)
            return {c: self.pdiff(e, x.coord) for c, e in a.items()}
        if isinstance(x, E.Gradient):
            a = self._sub(x.args[0], mode)
            return {(i,) + c: self.pdiff(e, q) for c, e in a.items()
                    for i, q in enumerate(self.coords)}
        if isinstance(x, E.Divergence):
            a = self._sub(x.args[0], mode)
            out = {}
            for c, e in a.items():
                out.setdefault(c[1:], []).append(self.pdiff(e, self.coords[c[0]]))
            return {c: self.add(v) for c, v in out.items()}
        if isinstance(x, E.Laplacian):
            a = self._sub(x.args[0], mode)
            return {c: self.add([self.pdiff(self.pdiff(e, q), q) for q in self.coords])
                    for c, e in a.items()}
        if isinstance(x, E.TimeDerivative):
            if mode == "M":
                return self._c(x.args[0], "L")
            if mode == "L":
                return {}
            raise DiscretizationError(f"'{x}': a time derivative on the right-hand side")
        if isinstance(x, E.Lift):
            if x.basis.coord != self.zc:
                raise DiscretizationError("lift along a periodic direction")
            a = self._sub(x.args[0], mode)
            return {c: self.lift(e, x.n, x.lam) for c, e in a.items()}
        if isinstance(x, E.Interpolate):
            if x.coord != self.zc or x.position not in (-1.0, 1.0) or x.side is not None:
                raise UnsupportedError(
                    f"trace '{x}' on a mapped domain",
                    why="xi = const is not a horizontal plane: only the traces at the two "
                        "surfaces (xi = -1: bottom, xi = +1: top) are defined",
                    fix="conditions at 'bottom'/'top'/'walls' only", doc=_DOC)
            a = self._sub(x.args[0], mode)
            return {c: self.interp_side(e, x.position) for c, e in a.items()}
        if isinstance(x, E.Integrate):
            a = self._sub(x.args[0], mode)
            fset = {c for c, _ in self.fdirs}
            got = set(x.coords)
            if got & fset and got & fset != fset:
                raise UnsupportedError(f"'{x}': integral along some periodic directions only",
                                       fix="integrate over all horizontal directions", doc=_DOC)
            out = {}
            for c, e in a.items():
                if self.zc in got:
                    e = self.integ_z(e)
                elif got & fset and e.lam is not None:
                    raise UnsupportedError(
                        f"'{x}': horizontal integral at fixed {self.zc.name} on a mapped domain",
                        why="the planes z = const are not coordinate surfaces; the integral is "
                            "not represented in the terrain-following coordinate",
                        fix=f"integrate over the whole domain (integ(f)) or along {self.zc.name}",
                        doc=_DOC)
                if got & fset:
                    e = self.integ_f(e)
                out[c] = e
            return out
        if mode == "V":
            if isinstance(x, E.Power):
                a = self._value(x.args[0])
                p = x.exponent
                return {(): self._pointwise([a.get((), _zero())], lambda u: u ** p)}
            if isinstance(x, E.UnaryFunction):
                a = self._value(x.args[0])
                fn = x.numpy_func
                return {(): self._pointwise([a.get((), _zero())], fn)}
        raise UnsupportedError(f"'{type(x).__name__}' ({x}) on a mapped domain"
                               + (" in an implicit term" if mode != "V" else ""),
                               why="not implemented by the terrain-following discretizer",
                               fix="write it with grad/div/lap/diff and products, or move a "
                                   "nonlinear term to the right-hand side", doc=_DOC)

    def _prod(self, a: Ent, b: Ent, mode: str, what: str) -> Ent:
        if a.empty or b.empty:
            return _zero()
        if a.is_scalar:
            return self.scale(b, a.blocks[DATA][0])
        if b.is_scalar:
            return self.scale(a, b.blocks[DATA][0])
        if a.is_data and b.is_data:
            return self._pointwise([a, b], lambda u, v: u * v)
        if a.is_data:
            return self.ncc_mul(b, a, what)
        if b.is_data:
            return self.ncc_mul(a, b, what)
        raise DiscretizationError(f"'{what}': product of two unknowns in an implicit term "
                                  "(nonlinear): move it to the right-hand side")

    def _mul(self, x: E.Mul, mode: str) -> dict:
        if mode == "V":
            acc = {(): _scalar(1.0)}
            for f in x.args:
                v = self._value(f)
                new = {}
                for ca, ea in acc.items():
                    for cb, eb in v.items():
                        new[ca + cb] = self._prod(ea, eb, "V", str(x))
                acc = new
            return acc
        lin = [f for f in x.args if f.free_fields() & self.V]
        if len(lin) != 1:
            raise DiscretizationError(f"'{x}': product of two unknowns in an implicit term "
                                      "(nonlinear): move it to the right-hand side")
        data = [f for f in x.args if f is not lin[0]]
        if data:
            self._check_ncc_aliasing(E.mul(*data))
        d = self._data(E.mul(*data)) if data else {(): _scalar(1.0)}
        e = self._c(lin[0], mode)
        out = {}
        name = str(E.mul(*data)) if data else "1"
        for cd, ed in d.items():
            for cl, el in e.items():
                out[cd + cl] = self._prod(ed, el, mode, name + (str(list(cd)) if cd else ""))
        return out

    # ------------------------------------------------------------------ assembly
    def _eq_space(self, i: int, ents: list[Ent]) -> tuple[int | None, bool]:
        eq = self.problem.equations[i]
        lam0, f = self._space(eq.domain)
        if lam0 is None:
            return None, f
        return max([e.lam for e in ents if e.lam is not None], default=0), f

    def _assemble(self) -> GroupMatrices:
        if self._gm is not None:
            return self._gm
        pb = self.problem
        parts: dict[str, list] = {"L": [], "M": []}
        rows, lams, self.eq_space, self.eq_off = [], {}, {}, {}
        r = 0
        for i, eq in enumerate(pb.equations):
            if isinstance(pb, EVP):
                sp_ = eigen_split(eq.lhs, self.eigenvalue)
                fL = self._c(sp_.get(0, E.Zero(eq.lhs.tensorsig)), "L") if 0 in sp_ else {}
                fM = self._c(sp_[1], "L") if 1 in sp_ else {}
            else:
                fL = self._c(eq.lhs, "L")
                fM = self._c(eq.lhs, "M") if isinstance(pb, IVP) else {}
            kind = ("boundary" if any(isinstance(n, E.Interpolate) for n in eq.lhs.walk())
                    else "bulk" if self._space(eq.domain)[0] is not None else "constraint")
            for c in _comps(eq.lhs.tensorsig):
                ents = [f[c] for f in (fL, fM) if c in f]
                lam, fq = self._eq_space(i, ents)
                n = self.rows(lam, fq)
                self.eq_space[(i, c)] = (lam, fq)
                self.eq_off[(i, c)] = r
                lams[(i, c)] = lam
                rows += [(i, c, kind)] * n
                for f, tgt in ((fL, "L"), (fM, "M")):
                    if c not in f or f[c].empty:
                        continue
                    e = f[c]
                    if fq:
                        e = self.bcast_f(e)
                    if lam is not None:
                        e = self.convert(self.bcast_z(e), lam)
                    elif e.lam is not None:
                        raise DiscretizationError(f"equation [{i}]: a term depends on the vertical "
                                                  "coordinate in an equation that does not")
                    for k, blk in e.blocks.items():
                        if k == DATA:
                            raise DiscretizationError(f"equation [{i}]: a data term on the left side")
                        parts[tgt].append((r, self.var_off[k], sp.coo_matrix(blk)))
                r += n
        self.nrow = r

        def build(ps):
            if not ps:
                return sp.csr_matrix((r, self.ncol), dtype=complex)
            R = np.concatenate([b.row + r0 for r0, _, b in ps])
            C = np.concatenate([b.col + c0 for _, c0, b in ps])
            D = np.concatenate([b.data.astype(complex) for _, _, b in ps])
            A = sp.coo_matrix((D, (R, C)), shape=(r, self.ncol)).tocsr()
            A.sum_duplicates()
            A.eliminate_zeros()
            return A
        L, M = build(parts["L"]), build(parts["M"])
        cols = []
        for v in self.variables:
            kind = ("tau" if v in self.taus else
                    "field" if self._space(v.domain) != (None, False) else "constant")
            for c in _comps(v.tensorsig):
                cols += [(v.name, c, kind)] * self.rows(*self.var_space[(v, c)])
        self._gm = GroupMatrices(("coupled",), "all Fourier modes coupled (mapped domain)", L, M,
                                 rows, cols, np.arange(r), np.arange(self.ncol), lams)
        return self._gm

    def matrices(self, group=None, reduced: bool | None = None) -> GroupMatrices:
        if reduced:
            raise UnsupportedError("symmetry subspaces on a mapped domain", doc=_DOC)
        return self._check_size(self._assemble())

    def _check_size(self, gm: GroupMatrices) -> GroupMatrices:
        """Refuse a coupled system too large for the direct sparse LU (before factorizing)."""
        nnz = int(gm.L.nnz + gm.M.nnz)
        if nnz > MAX_COUPLED_NNZ:
            raise UnsupportedError(
                f"mapped domain: coupled system {gm.L.shape[0]} x {gm.L.shape[1]} with nnz(L) + "
                f"nnz(M) = {nnz:.3g} > MAX_COUPLED_NNZ = {MAX_COUPLED_NNZ:.3g}",
                why="the direct sparse LU of the coupled system (all Fourier modes) would fill in "
                    "well beyond this (time and memory); no iterative solver is implemented",
                fix="lower the horizontal resolution or the wall slope (fewer metric Fourier "
                    "coefficients, printed by explain()), or raise "
                    "conatus.discretize.terrain_plan.MAX_COUPLED_NNZ knowingly", doc=_DOC)
        return gm

    def check(self, reduced: bool | None = None) -> list:
        gm = self._check_size(self._assemble())
        n, m = gm.L.shape
        if n != m:
            raise RankError(f"mapped domain: {n} equations for {m} unknowns",
                            why="boundary conditions and taus do not balance",
                            fix="check the conditions (m.explain())")
        pat = sp.csr_matrix((abs(gm.L) + abs(gm.M)) > 0, dtype=float)
        sr = int(structural_rank(pat)) if n else 0
        if sr < n:
            raise RankError(f"mapped domain: structural rank {sr} / {n}",
                            why="some unknown is reached by no equation (missing condition or "
                                "gauge)", fix="check the conditions (m.explain())")
        return []

    # ------------------------------------------------------------------ right-hand sides
    def rhs_vector(self, which: Callable[[int], bool] | None = None) -> np.ndarray:
        """Right-hand side of every equation in the row space (data and, in a simulation, the
        current state for the unknowns it contains)."""
        self._assemble()
        b = np.zeros(self.nrow, dtype=complex)
        for i, eq in enumerate(self.problem.equations):
            if which is not None and not which(i):
                continue
            rhs = E.as_expr(eq.rhs)
            if rhs.is_zero():
                continue
            vals = self._value(rhs)
            for c in _comps(eq.lhs.tensorsig):
                if c not in vals or vals[c].empty:
                    continue
                lam, fq = self.eq_space[(i, c)]
                e = vals[c]
                if fq:
                    e = self.bcast_f(e)
                if lam is not None:
                    e = self.convert(self.bcast_z(e), lam)
                elif e.lam is not None:
                    raise DiscretizationError(f"equation [{i}]: right side depends on the "
                                              "vertical coordinate, the equation does not")
                r0 = self.eq_off[(i, c)]
                v = e.blocks[DATA]
                b[r0:r0 + v.size] += v
        return b

    # ------------------------------------------------------------------ LBVP
    def solve(self, data: Mapping[E.Field, Any] | None = None,
              reduced: bool | None = None) -> TerrainSolution:
        if not isinstance(self.problem, LBVP):
            raise DiscretizationError("solve() is for LBVPs (IVP: Model.simulate; EVP: "
                                      "eigenvalues(plan))")
        if reduced:
            raise UnsupportedError("symmetry subspaces on a mapped domain", doc=_DOC)
        if data:
            self._ncc_src.update(data)
            self._dcache.clear()
        self.check()
        gm = self.matrices()
        b = self.rhs_vector()
        A = gm.L.tocsc()
        try:
            x = spla.splu(A).solve(b)
        except RuntimeError as err:
            raise RankError(f"mapped domain: the coupled system is singular ({err})",
                            why="a missing gauge or a redundant condition",
                            fix="check the conditions (m.explain())") from None
        res = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
        return TerrainSolution(self.split(x), [(gm.label, res)], self)

    def split(self, x: np.ndarray) -> dict:
        out = {}
        for v in self.variables:
            cs = _comps(v.tensorsig)
            out[v.name] = {c: x[self.var_off[(v, c)]:self.var_off[(v, c)]
                                 + self.rows(*self.var_space[(v, c)])] for c in cs}
        return out

    def to_grid(self, name: str, coeffs: dict, dealias: bool = False) -> np.ndarray:
        """Grid values (N-point grid unless ``dealias``) of variable ``name`` from its coefficients
        {component: vector}; real part of a real problem (the imaginary rounding is reported by
        ``imag_level``), complex otherwise."""
        v = next(v for v in self.variables if v.name == name)
        vals = []
        for c in _comps(v.tensorsig):
            lam, f = self.var_space[(v, c)]
            g = self.backward(coeffs[c], lam, f, dealias)
            if lam is None:
                g = g[..., 0]
            vals.append(g)
        g = np.stack(vals) if v.tensorsig else vals[0]
        if v.dtype.kind != "c" and not any(np.iscomplexobj(np.asarray(p))
                                          for p in self.params.values()):
            self.imag_level = max(getattr(self, "imag_level", 0.0),
                                  float(np.abs(g.imag).max() / max(np.abs(g.real).max(), 1e-300)))
            if self.imag_level > 1e-8:
                return g                       # never dropped silently
            return g.real.copy()
        return g

    # ------------------------------------------------------------------ explain
    def explain(self) -> str:
        gm = self._assemble()
        lines = [f"TerrainPlan of {self.problem.kind} '{self.problem.name}' on a mapped domain",
                 "  map: " + self.terrain.formula(),
                 "  chain rule: " + self.terrain.metric_text(),
                 "  metric terms implicit (one coupled sparse system, all Fourier modes; banded "
                 "along xi: metric polynomial of degree <= 1 in xi, exact)",
                 f"  Fourier modes |k| <= N/2 - 1 per direction ({self.nk} complex modes of real "
                 f"fields), {self.Nz} Chebyshev modes along xi",
                 f"  system: {gm.L.shape[0]} x {gm.L.shape[1]}, nnz(L) = {gm.L.nnz}, nnz(M) = "
                 f"{gm.M.nnz}; direct sparse LU (SuperLU), refused above "
                 f"{MAX_COUPLED_UNKNOWNS} unknowns or nnz {MAX_COUPLED_NNZ:.3g}"]
        lines += ["  " + r.explain() for r in self.metric_reports]
        lines += ["  " + r.explain() for r in self.ncc_reports.values()]
        lines.append("  " + us.ncc_policy(self.ncc_tail_tol, None))
        lines.append("  data and explicit terms: pointwise on the dealiased physical grid "
                     "(x_j, z(x_j, xi_l))")
        return "\n".join(lines)


def _round_up(x: float) -> float:
    e = 10.0 ** np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if m * e >= x * (1 - 1e-12):
            return float(f"{m * e:.0e}")
    return float(x)


class TerrainSolution:
    """Solution of an LBVP on a mapped domain: grid values on the physical grid."""

    def __init__(self, coeffs: dict, residuals: list, plan: TerrainPlan) -> None:
        self.coeffs, self.residuals, self.plan = coeffs, residuals, plan
        self.reduced = False

    @property
    def max_residual(self) -> float:
        return max((r for _, r in self.residuals), default=0.0)

    def grid(self, name: str) -> np.ndarray:
        """Values of a variable on the N-point grid (x_j, xi_l), physical position
        ``coordinate(...)``."""
        return self.plan.to_grid(name, self.coeffs[name])

    def coordinate(self, name: str) -> np.ndarray:
        """Physical coordinate ``name`` on the grid of ``grid()`` (full arrays: z depends on x)."""
        names = [c.name for c in self.plan.coords]
        if name not in names:
            raise KeyError(f"no coordinate {name!r} (coordinates: {names})")
        return self.plan.physical_grid(False)[names.index(name)]

    def explain(self) -> str:
        worst = max(self.residuals, key=lambda r: r[1]) if self.residuals else ("-", 0.0)
        return (f"LBVP solution on a mapped domain: one coupled system, relative residual "
                f"{worst[1]:.2e} (sparse LU, scipy SuperLU)")


class TerrainSimulation:
    """Time integration of a native IVP on a mapped domain (``Model.simulate``).

    M dX/dt + L X = F(X, t) with L, M the coupled sparse operators (metric implicit) and F the
    right-hand sides evaluated on the dealiased physical grid. Schemes: SBDF1-4 (startup
    ``lower_order``: orders 1..q-1 for the first steps, declared) and RK111/RK222/RK443 (IMEX
    Runge–Kutta, same tableaux as the flat solver). Rows of equations without time derivative
    (boundary conditions, constraints) are imposed at the stage time with their data
    (``bc_time="stage"`` convention). One sparse LU per distinct implicit coefficient."""

    SCHEMES = ("SBDF1", "SBDF2", "SBDF3", "SBDF4", "RK111", "RK222", "RK443")

    def __init__(self, comp: Any, *, scheme: str, dt: float, backend: str = "cpu",
                 **kw: Any) -> None:
        from ..model.model import ncc_options
        if backend != "cpu":
            raise UnsupportedError(f"backend={backend!r} on a mapped domain",
                                   why="the coupled sparse solve of the mapped discretization runs "
                                       "on the CPU only", fix='backend="cpu"', doc=_DOC)
        if scheme not in self.SCHEMES:
            raise OptionError(f"scheme={scheme!r} on a mapped domain",
                              fix="one of " + ", ".join(self.SCHEMES), doc=_DOC)
        bc_time = kw.pop("bc_time", "auto")
        if bc_time in ("rk", "step"):
            raise UnsupportedError(
                f"bc_time={bc_time!r} on a mapped domain",
                why="the mapped stepper imposes time-dependent boundary data at the stage times "
                    "only (the 'rk' variant of the flat solvers and the Dedalus 'step' convention "
                    "are not implemented here)",
                fix='bc_time="stage" (or leave the default "auto", which resolves to "stage")',
                doc=_DOC)
        if bc_time not in ("auto", "stage"):
            raise OptionError(f"bc_time={bc_time!r}", fix='"auto" or "stage"', doc=_DOC)
        # resolved mode (as solver.bc_time of the flat solvers) and the reason of an "auto" fallback
        self.bc_time = "stage"
        self._bc_note = ("" if bc_time == "stage" else
                         ' (bc_time="auto": the "rk" variant is not implemented on a mapped domain)')
        nkw = ncc_options(kw)
        from ..timestep.integrator import finite_cadence
        # finite-state check: same option and cadence as IVPSolver
        self._finite_every = finite_cadence(kw.pop("check_finite", True), "simulate()")
        extra = set(kw) - {"ncc_cutoff", "ncc_tail_tol"}
        if extra:
            raise OptionError(f"simulate(): option(s) {sorted(extra)} not supported on a mapped "
                              "domain", doc=_DOC)
        if not (isinstance(dt, numbers.Real) and dt > 0):
            raise OptionError(f"dt={dt!r}: a positive number is required", doc=_DOC)
        self.compiled = comp
        self.problem = comp.problem
        reg = getattr(comp.model, "_register", None)
        if reg is not None:                        # sim.id; named by later parameter changes
            reg(self)
        self.plan: TerrainPlan = comp.discretize(**nkw)
        self.scheme, self._dt = scheme, float(dt)
        P = self.plan
        P.check()
        gm = P.matrices()
        self.L, self.M = gm.L.tocsr(), gm.M.tocsr()
        self.X = np.zeros(P.ncol, dtype=complex)
        self._t, self._it = 0.0, 0
        self._tname = comp.time.name if getattr(comp, "time", None) is not None else None
        if self._tname is not None and any(
                self._tname in {p.name for p in eq.lhs.free_parameters()}
                for eq in self.problem.equations):
            raise UnsupportedError(
                f"a left-hand side depends on the time {self._tname} on a mapped domain",
                why="the coupled operator is factorized once per implicit coefficient; an "
                    "operator that changes in time would need a factorization per stage",
                fix=f"move the {self._tname}-dependent terms to the right-hand side (explicit)",
                doc=_DOC)
        # equations without time derivative: their rows are imposed at the stage time
        self.static = []
        for i, eq in enumerate(self.problem.equations):
            if not any(isinstance(n, E.TimeDerivative) for n in eq.lhs.walk()):
                if E.as_expr(eq.rhs).free_fields() & P.V:
                    raise UnsupportedError(f"equation [{i}] has no time derivative but its right "
                                           "side depends on the unknowns", doc=_DOC)
                self.static.append(i)
        mask = np.zeros(P.nrow, dtype=bool)
        for (i, c), r0 in P.eq_off.items():
            if i in self.static:
                mask[r0:r0 + P.rows(*P.eq_space[(i, c)])] = True
        self._static_rows = mask
        self._lu: dict = {}
        self._hist: list = []         # SBDF history of (X, F)

    # -- state ------------------------------------------------------------------------------
    @property
    def t(self) -> float:
        """Current simulation time."""
        return self._t

    @property
    def dt(self) -> float:
        """Time step (fixed at ``simulate``)."""
        return self._dt

    @property
    def iteration(self) -> int:
        """Number of steps taken."""
        return self._it

    def initial(self, **values: Any) -> TerrainSimulation:
        """Initial state by unknown name: callables of the coordinate names (physical z) or
        numbers; unknowns not given start at zero."""
        from ..model.compiler import _coord_args, _on_all_grids
        P = self.plan
        names = tuple(c.name for c in P.coords)
        grids = P.physical_grid(False)
        for k, v in values.items():
            f = self.compiled.fields.get(k)
            if f is None or f not in P.V:
                raise OptionError(f"initial({k}=...): not an unknown", doc=_DOC)
            fn = (_on_all_grids(v, _coord_args(v, names, f"initial {k}"), names) if callable(v)
                  else (lambda val0: (lambda *g: np.full(grids[0].shape, val0)))(v))
            val = np.asarray(fn(*grids))
            for c in _comps(f.tensorsig):
                g = np.broadcast_to(val[c] if c else val, grids[0].shape)
                o = P.var_off[(f, c)]
                vec = P.forward(g, False, True)
                self.X[o:o + vec.size] = vec
        self._hist = []
        return self

    def coeffs(self, name: str) -> dict:
        """Coefficients of unknown ``name``: {component: vector of the coupled layout}."""
        return self.plan.split(self.X)[name]

    def grid(self, name: str) -> np.ndarray:
        """Values of unknown ``name`` on the N-point grid (x_j, xi_l)."""
        return self.plan.to_grid(name, self.coeffs(name))

    def coordinate(self, name: str) -> np.ndarray:
        """Physical coordinate ``name`` on the grid of :meth:`grid` (z depends on x)."""
        names = [c.name for c in self.plan.coords]
        if name not in names:
            raise OptionError(f"coordinate({name!r}): not a coordinate of the mapped domain",
                              fix="coordinates: " + ", ".join(names), doc=_DOC)
        return self.plan.physical_grid(False)[names.index(name)]

    # -- stepping ---------------------------------------------------------------------------
    def _F(self, X: np.ndarray, t: float) -> np.ndarray:
        P = self.plan
        if self._tname is not None:
            P.params[self._tname] = t
        P._state = {k: X[o:o + P.rows(*P.var_space[k])] for k, o in P.var_off.items()}
        return P.rhs_vector()

    def _g(self, t: float) -> np.ndarray:
        P = self.plan
        if self._tname is not None:
            P.params[self._tname] = t
        P._state = {k: np.zeros(P.rows(*P.var_space[k]), dtype=complex) for k in P.var_off}
        return P.rhs_vector(lambda i: i in self.static)

    def _solve(self, a: float, b: float, rhs: np.ndarray) -> np.ndarray:
        key = (a, b)
        if key not in self._lu:
            A = (a * self.M + b * self.L).tocsc()
            try:
                self._lu[key] = spla.splu(A)
            except RuntimeError as err:
                raise RankError(f"mapped domain: the implicit system is singular ({err})") from None
        return self._lu[key].solve(rhs)

    def step(self) -> None:
        """One step of the declared scheme (boundary and constraint rows at the stage time)."""
        dt, t = self._dt, self._t
        X = self.X
        if self.scheme.startswith("SBDF"):
            q = min(int(self.scheme[-1]), len(self._hist) + 1)
            from ..timestep.integrator import sbdf_coefficients
            a, c = sbdf_coefficients(q)
            Fn = self._F(X, t)
            hist = [(X, Fn)] + self._hist
            rhs = np.zeros_like(X)
            for j in range(1, q + 1):
                Xj, Fj = hist[j - 1]
                rhs += dt * c[j] * Fj - a[j] * (self.M @ Xj)
            rhs[self._static_rows] = dt * self._g(t + dt)[self._static_rows]
            Xn = self._solve(a[0], dt, rhs)
            self._hist = hist[:3]
        else:
            from ..timestep.integrator import rk_tableau
            A, H, cc = rk_tableau(self.scheme)
            s = len(cc)
            Xs, Fs, LXs = [X], [], []
            MX0 = self.M @ X
            for i in range(1, s):
                Fs.append(self._F(Xs[i - 1], t + cc[i - 1] * dt))
                LXs.append(self.L @ Xs[i - 1])
                rhs = MX0.copy()
                for j in range(i):
                    if A[i][j]:
                        rhs += dt * A[i][j] * Fs[j]
                    if H[i][j] and j < i:
                        rhs -= dt * H[i][j] * LXs[j]
                hii = H[i][i]
                rhs[self._static_rows] = dt * hii * self._g(t + cc[i] * dt)[self._static_rows]
                Xs.append(self._solve(1.0, dt * hii, rhs))
            Xn = Xs[-1]
        self.X = Xn
        self._t = t + dt
        self._it += 1

    def run(self, until: float | None = None, *, steps: int | None = None,
            callback: Callable | None = None, every: int | None = None) -> TerrainSimulation:
        """Advance to ``until`` (a whole number of steps) or by ``steps``; ``callback(sim)`` every
        ``every`` steps."""
        from ..timestep.integrator import check_callback, check_steps
        if (until is None) == (steps is None):     # with a fix line
            raise OptionError("run(): give until= or steps=" + ("" if until is None else ", not both"),
                              fix="run(until=<end time, a whole number of steps of dt>) or "
                                  "run(steps=100)", doc=_DOC)
        # the checks shared by every run (steps=2.7 took 2 steps
        # without a word here, and every=0.5 failed only at the first modulo)
        check_callback(callback, every, "run()")
        if steps is not None:
            steps = check_steps(steps, "run()", "steps")
        if steps is None:
            n = (until - self._t) / self._dt
            steps = int(round(n))
            if steps < 0:                      # (the fix said run(steps=-2))
                # ResolutionError, as Simulation.run (was OptionError here)
                raise ResolutionError(f"run(until={until}): the simulation is already at t = "
                                      f"{self._t:g}, after until",
                                      why="a run goes forward in time; until is an absolute time, "
                                          "not a duration",
                                      fix=f"run(until=t_end) with t_end > {self._t:g}", doc=_DOC)
            from ..timestep.integrator import is_whole    # the one whole-steps rule
            if not is_whole(n):
                raise ResolutionError(f"run(until={until}): not a whole number of steps of dt = "
                                      f"{self._dt:g}", why="the time step is never adjusted silently",
                                      fix=f"run(steps={math.floor(n)})", doc=_DOC)
        from ..timestep.integrator import check_state_finite
        fe = self._finite_every
        for _ in range(steps):
            self.step()
            if fe and self._it % fe == 0:
                check_state_finite(self, "TerrainSimulation.run")
            # on iteration multiples, as IVPSolver.run: the same steps whatever the run chunks
            if callback is not None and self._it % every == 0:
                callback(self)
        if fe and steps:
            check_state_finite(self, "TerrainSimulation.run (end of the call)")
        return self

    def explain(self) -> str:
        return (f"TerrainSimulation: {self.scheme}, dt = {self._dt:g}, t = {self._t:g} "
                f"({self._it} steps); implicit: the coupled operator (metric terms implicit), "
                "explicit: right-hand sides on the dealiased physical grid; equations without "
                "time derivative imposed with their data at the stage time (bc_time = 'stage'"
                + self._bc_note + ")"
                + ("; SBDF startup lower_order (orders 1..q-1 for the first q-1 steps)"
                   if self.scheme.startswith("SBDF") else "")
                + (f"; finite-state check every {self._finite_every} steps and at the end of "
                   "each run call (check_finite=False turns it off)" if self._finite_every
                   else "; finite-state check off (check_finite=False)")
                + "\n" + self.plan.explain())
