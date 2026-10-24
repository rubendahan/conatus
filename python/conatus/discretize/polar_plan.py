"""Discretization plans on polar domains: one group per azimuthal wavenumber m.

``discretize(problem)`` dispatches here when the problem lives on ``PolarCoordinates``. The public
face is the Cartesian ``Plan``'s: ``groups``, ``matrices(group)`` (``GroupMatrices``: L, M, rows,
columns, boundary rows, tau columns), ``rank_report``/``check`` (refused before any
factorization), ``solve`` (LBVP, residual reported), ``explain()``, ``methods()``; EVPs go through
``conatus.solvers.eigenvalues`` and IVPs through ``IVPSolver`` exactly as in Cartesian geometry.

Inside a group, every tensor component is a *spin* component of effective wavenumber
k = m + (sum of its spins) and the IR nodes compile to the spin toolkit of ``polar_radial``:

    grad f      -> (G_- f, G_+ f)                       (spin index prepended)
    div u       -> (G_+ u_- + G_- u_+) / 2
    lap         -> spin-diagonal L_k (vector Laplacian included)
    curl psi    -> (+i G_- psi, -i G_+ psi) = grad(psi) x e_z      (2D convention of the IR)
    curl u      -> (i/2) (G_+ u_- - G_- u_+) = e_z . curl(u)
    interp(., r, r0), interp(dr(.), r, r0), physical components at a trace, integ over the domain,
    lift(tau, radial_basis, n, lam) (tau in mode N_k + n of level lam), dt, the EVP eigenvalue,
    constants, parameters and axisymmetric non-constant coefficients g(r) (m = 0).

Disk rows of an equation component live at the largest Jacobi level of its terms (lower levels are
converted, bidiagonal); annulus rows are multiplied by the largest power of r of their terms
(declared r-multiplied form, the right-hand side is multiplied the same way; lifted taus are not).
Nothing is ever divided by r: the regularity at the axis is in the basis (polar_radial).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..errors import InputTypeError, OptionError
from ..ir import expr as E
from ..ir.domain import Domain
from ..ir.problem import EVP, IVP, LBVP, Problem
from ..reference.evaluate import FieldData
from ..tolerances import NCC_TAIL_TOL
from . import ultraspherical as us
from .compiler import DiscretizationError
from .curvilinear_common import const_value
from .curvilinear_symmetry import (CurvilinearReduction, plan_embed, plan_matrices, plan_project,
                                   reduced_lbvp_solve)
from .plan import GroupMatrices, LBVPSolution, Plan, data_snapshot, rebind_for_data
from .polar_eval import PH, PolarEvaluator, PolarSpace, _comps, spin_k

__all__ = ["PolarPlan", "PolarGroup"]


@dataclass(frozen=True)
class PolarGroup:
    """One azimuthal wavenumber m (and, on a cylinder, one axial wavenumber k_z); ``index`` and
    ``zindex`` are the positions along the azimuth and z axes of the coefficient arrays."""

    m: int
    index: int
    kz: float = 0.0
    zindex: int = 0
    cyl: bool = False

    @property
    def label(self) -> str:
        """Short group label (m, and k_z on a cylinder)."""
        return f"m={self.m}" + (f", kz={self.kz:.6g}" if self.cyl else "")

    def __str__(self) -> str:
        return self.label


class _Ent:
    """Compiled form of one output component: bulk (radial dependence, level, r-power p; p None for
    a lifted tau) or point (one row: trace, integral, boundary variable); blocks per (field, comp)."""

    __slots__ = ("bulk", "k", "level", "p", "blocks")

    def __init__(self, bulk: bool, k: int, level, p, blocks: dict) -> None:
        self.bulk, self.k, self.level, self.p, self.blocks = bulk, k, level, p, blocks

    def map(self, A, **kw) -> _Ent:
        e = _Ent(self.bulk, self.k, self.level, self.p, {key: (A @ b).tocsr() for key, b in self.blocks.items()})
        for a, v in kw.items():
            setattr(e, a, v)
        return e

    def scale(self, c) -> _Ent:
        return _Ent(self.bulk, self.k, self.level, self.p, {key: (c * b).tocsr() for key, b in self.blocks.items()})


# ----------------------------------------------------------------------------------------------
# compiler (one group)
# ----------------------------------------------------------------------------------------------

class _PolarCompiler:
    def __init__(self, plan: PolarPlan, group: PolarGroup) -> None:
        self.P, self.g, self.m, self.kz = plan, group, group.m, group.kz
        self.ops = plan.space.ops
        self.nc = plan.space.nc
        self.n = plan.space.ops.size(group.m)        # radial size of every component of this group
        self.vars = frozenset(plan.variables)
        self.mode = "L"

    # ---------------------------------------------------------------- helpers
    def nrow(self, k: int) -> int:
        return self.n

    def add(self, ents: list[_Ent], k: int) -> _Ent:
        """Sum of entries of one component: common level (conversion), common r-power
        (multiplication, annulus), point entries broadcast as constants in r when regular."""
        ents = [e for e in ents if e is not None]
        if not any(e.bulk for e in ents):
            return _Ent(False, k, None, None, _sum_blocks(ents))
        level = max(e.level for e in ents if e.bulk)
        ps = [e.p for e in ents if e.bulk and e.p is not None]
        p = max(ps) if ps else 0
        return _Ent(True, k, level, p, _sum_blocks([self.normalize(e, k, level, p) for e in ents]))

    def normalize(self, e: _Ent, k: int, level: int, p: int) -> _Ent:
        if not e.bulk:
            e = self.broadcast(e, k)
        out = e.map(self.ops.convert(k, e.level, level, self.n)) if e.level != level else e
        if e.p is not None and e.p != p:
            if p < e.p:
                raise DiscretizationError("internal: r-power above the row power")
            out = out.map(self.ops.multiply_r(k, level, p - e.p, self.n))
        return _Ent(True, k, level, p, out.blocks)

    def broadcast(self, e: _Ent, k: int) -> _Ent:
        """A point quantity (no r-dependence) inside a bulk equation: the constant function of r.
        On the disk this is regular only for k = 0 (never e^{i m phi} without rho^|m|)."""
        if all(b.shape[1] == 0 for b in e.blocks.values()):     # e.g. a constant at m != 0: absent
            return _Ent(True, k, 0, 0, {key: sp.csr_matrix((self.nrow(k), 0)) for key in e.blocks})
        if self.ops.kind == "disk" and k != 0:
            raise DiscretizationError(
                f"[m={self.m}] a quantity without radial dependence enters a disk equation "
                f"component of wavenumber k={k}: it is not regular at the axis (only constants "
                "can be broadcast; use lift() for taus)")
        col = np.zeros((self.nrow(k), 1))
        # the constant function 1 in the level-0 radial basis of wavenumber 0: 1 / (value of mode 0).
        # Chebyshev T_0 = 1 on the annulus; on the disk the normalized mode 0 is 1/sqrt(2), and the
        # coefficient 1 used here before broadcast sqrt(2)^-1 times the quantity (a
        # pressure-gradient unknown G e_z came out sqrt(2) G; the gauge taus only rescaled)
        col[0, 0] = 1.0 / float(np.real(self.ops.backward(0, 1, 0, 0, self.nrow(k))[0, 0]))
        return _Ent(True, k, 0, 0, {key: (sp.csr_matrix(col) @ b).tocsr() for key, b in e.blocks.items()})

    def _param(self, p: E.Parameter):
        v = self.P.params.get(p.name, p.value)
        if v is None:
            raise DiscretizationError(f"parameter {p.name} has no value")
        return v

    def _const(self, f: E.Expr):
        return const_value(f, self._param)

    # ---------------------------------------------------------------- entry point
    def compile(self, x: E.Expr, mode: str) -> dict:
        self.mode = mode
        return self._c(x)

    def _c_L(self, x: E.Expr) -> dict:
        saved, self.mode = self.mode, "L"
        try:
            return self._c(x)
        finally:
            self.mode = saved

    def _c(self, x: E.Expr) -> dict:
        V, ev = self.vars, self.P.eigenvalue
        if not x.depends_on(V) and not isinstance(x, E.TimeDerivative):
            raise DiscretizationError(f"'{x}' does not depend on the unknowns: it belongs on the RHS")
        if isinstance(x, E.Field):
            if self.mode == "M":
                # A bare unknown has no dt() and no eigenvalue: it never enters M. Same defect as the
                # Cartesian compiler, found here by the dt(u) + u = 0
                # on the disk decayed as e^{-t/2} and s u + u = 0 gave s = -1/2.
                return {}
            return self._field(x)
        if isinstance(x, E.TimeDerivative):
            if self.mode != "M":
                return {}
            arg = x.args[0]
            if any(isinstance(n, E.TimeDerivative) for n in arg.walk()):
                raise DiscretizationError(f"nested time derivative in '{x}': not supported")
            out = self._c_L(arg)
            if not any(e.blocks for e in out.values()):
                raise DiscretizationError(f"'{x}' compiles to an empty mass operator")
            return out
        if self.mode == "M" and not any(isinstance(n, E.TimeDerivative) for n in x.walk()) and (
                ev is None or ev not in x.free_parameters()):
            return {}
        if isinstance(x, E.Add):
            parts = [self._c(t) for t in x.args if t.depends_on(V) or isinstance(t, E.TimeDerivative)]
            out = {}
            for c in {c for p in parts for c in p}:
                out[c] = self.add([p[c] for p in parts if c in p], self.m + spin_k(c))
            return out
        if isinstance(x, E.Mul):
            return self._mul(x)
        if isinstance(x, E.Gradient):
            inner = self._c(x.args[0])
            out = {}
            for c, e in inner.items():
                self._need_bulk(e, x)
                for si, sigma in enumerate((-1, 1)):
                    A, lev, p = self.ops.G(sigma, e.k, e.level, self._p(e, x), self.n)
                    out[(si,) + c] = e.map(A, k=e.k + sigma, level=lev, p=p)
                if self.nc == 3:                                   # d/dz = i k_z
                    out[(2,) + c] = e.scale(1j * self.kz)
            return out
        if isinstance(x, E.Divergence):
            inner = self._c(x.args[0])
            out = {}
            for c in _comps(x.rank, self.nc):
                terms = []
                if self.nc == 3 and (2,) + c in inner:        # + d(u_z)/dz
                    terms.append(inner[(2,) + c].scale(1j * self.kz))
                for si, sigma in ((0, 1), (1, -1)):          # G_+ u_- , G_- u_+
                    e = inner.get((si,) + c)
                    if e is None:
                        continue
                    self._need_bulk(e, x)
                    A, lev, p = self.ops.G(sigma, e.k, e.level, self._p(e, x), self.n)
                    terms.append(e.map(0.5 * A, k=e.k + sigma, level=lev, p=p))
                if terms:
                    out[c] = self.add(terms, self.m + spin_k(c))
            return out
        if isinstance(x, E.Laplacian):
            out = {}
            for c, e in self._c(x.args[0]).items():
                self._need_bulk(e, x)
                A, lev, p = self.ops.L(e.k, e.level, self._p(e, x), self.n)
                out[c] = e.map(A, level=lev, p=p)
                if self.nc == 3 and self.kz != 0:              # - k_z^2 (r-multiplied like L)
                    out[c] = self.add([out[c], e.scale(-self.kz ** 2)], e.k)
            return out
        if isinstance(x, E.Curl):
            return self._curl(x)
        if isinstance(x, E.Component):
            inner = self._c(x.args[0])
            if (x.index < 2 and any(e.bulk for e in inner.values()) and self.ops.kind == "disk"):
                raise DiscretizationError(
                    f"'{x}': a physical component (u_r, u_phi) is not a regular function on the disk "
                    "(u_r = (u_+ + u_-)/2 mixes rho^|m+1| and rho^|m-1|); use it at a boundary "
                    "(interp) or write the equation with grad/div/lap/curl")
            return self._component(inner, x.index, x.rank)
        if isinstance(x, E.Differentiate):
            if self.nc == 3 and x.coord == self.P.space.zc:
                return {c: e.scale(1j * self.kz) for c, e in self._c(x.args[0]).items()}
            if x.coord == self.P.space.phi:
                if x.args[0].rank:
                    raise DiscretizationError(f"'{x}': d/dphi of a tensor (frame rotates): use grad")
                return {c: e.scale(1j * self.m) for c, e in self._c(x.args[0]).items()}
            raise DiscretizationError(
                f"'{x}': d/dr alone is not regular at the axis; use grad/div/lap, or "
                "interp(diff(., r), r, r0) for a boundary condition")
        if isinstance(x, E.Interpolate):
            if x.coord != self.P.space.r:
                raise DiscretizationError(f"'{x}': interpolation along {x.coord.name} couples all "
                                          "its wavenumbers (not separable)")
            return self._trace(x.args[0], x.position, 0)
        if isinstance(x, E.Integrate):
            return self._integrate(x)
        if isinstance(x, E.Lift):
            return self._lift(x)
        if isinstance(x, E.Dot):
            return self._dot(x)
        raise DiscretizationError(f"cannot discretize node {type(x).__name__} ('{x}') on a polar domain")

    def _need_bulk(self, e: _Ent, x: E.Expr) -> None:
        if not e.bulk:
            raise DiscretizationError(f"'{x}': derivative of a quantity without radial dependence")

    def _p(self, e: _Ent, x: E.Expr) -> int:
        if e.p is None:
            raise DiscretizationError(f"'{x}': derivative of a lifted tau term")
        return e.p

    def _field(self, x: E.Field) -> dict:
        out = {}
        bulk = self.P.space.kind(x.domain) in ("full", "radial")
        for c in _comps(x.rank, self.nc):
            k = self.m + spin_k(c)
            nv = self.P.size_of(x.domain, c, self.g)
            nr = self.nrow(k) if bulk else 1
            blk = sp.identity(nv, format="csr") if nv == nr else sp.csr_matrix((nr, nv))
            out[c] = _Ent(bulk, k, 0 if bulk else None, 0 if bulk else None, {(x, c): blk})
        return out

    def _component(self, inner: dict, i: int, rank: int) -> dict:
        out = {}
        for c in _comps(rank, self.nc):
            terms = [inner[(s,) + c].scale(PH[i, s]) for s in range(self.nc)
                     if (s,) + c in inner and PH[i, s] != 0]
            if terms:
                out[c] = self.add(terms, self.m + spin_k(c))
        return out

    def _curl(self, x: E.Curl) -> dict:
        if x.coordsys.dim != 2:
            raise DiscretizationError("curl on cylindrical domains: not implemented")
        inner = self._c(x.args[0])
        rin = x.args[0].rank

        def G(e, sigma, coef):
            self._need_bulk(e, x)
            A, lev, p = self.ops.G(sigma, e.k, e.level, self._p(e, x), self.n)
            return e.map(coef * A, k=e.k + sigma, level=lev, p=p)
        if rin == 0:              # grad(psi) x e_z: (+i G_- psi, -i G_+ psi)
            e = inner[()]
            return {(0,): G(e, -1, 1j), (1,): G(e, +1, -1j)}
        terms = []
        if (0,) in inner:
            terms.append(G(inner[(0,)], +1, 0.5j))
        if (1,) in inner:
            terms.append(G(inner[(1,)], -1, -0.5j))
        return {(): self.add(terms, self.m)} if terms else {}

    def _trace(self, x: E.Expr, pos: float, d: int) -> dict:
        """Point functional: value (d = 0) or radial derivative (d = 1) at r = pos."""
        if isinstance(x, E.Component):
            return self._component(self._trace(x.args[0], pos, d), x.index, x.rank)
        if isinstance(x, E.Differentiate) and x.coord == self.P.space.r:
            if d:
                raise DiscretizationError(f"'{x}': second radial derivative at a point")
            return self._trace(x.args[0], pos, d + 1)
        if isinstance(x, E.Add):
            parts = [self._trace(t, pos, d) for t in x.args if t.depends_on(self.vars)]
            return {c: _Ent(False, self.m + spin_k(c), None, None,
                            _sum_blocks([p[c] for p in parts if c in p]))
                    for c in {c for p in parts for c in p}}
        if isinstance(x, E.Mul):
            dep = [a for a in x.args if a.depends_on(self.vars)]
            rest = [a for a in x.args if not a.depends_on(self.vars)]
            vals = [self._const(a) for a in rest]
            if len(dep) == 1 and all(v is not None for v in vals):
                s = np.prod(vals) if vals else 1.0
                return {c: e.scale(s) for c, e in self._trace(dep[0], pos, d).items()}
        if isinstance(x, E.Dot) and not x.args[0].depends_on(self.vars) and \
                self.P.space.kind(x.args[0].domain) in ("edge", "const"):
            return self._trace_dot(x, pos, d)
        out = {}
        for c, e in self._c(x).items():
            if not e.bulk:
                if d:
                    raise DiscretizationError(f"radial derivative of '{x}' (no radial dependence)")
                out[c] = e
                continue
            row = sp.csr_matrix(self.ops.point_row(e.k, e.level, self._p(e, x), pos, d, self.n)[None, :])
            out[c] = e.map(row, bulk=False, level=None, p=None)
        return out

    def _trace_dot(self, x: E.Dot, pos: float, d: int) -> dict:
        """dot(C, trace) with C a rotation-invariant rank-2 tensor given on the boundary circle
        (constant spin components, plan.boundary_tensor): out_s' = sum_{s,t} C[s', s] G[s, t] X_t
        (point rows; the effective wavenumber of each output component is relabelled). Used by
        the curved free-slip condition (model/conditions.py)."""
        C, X = x.args
        if C.rank != 2 or X.rank != 1:
            raise DiscretizationError(f"'{x}': at a boundary, dot(coefficient, vector) needs a "
                                      "rank-2 coefficient and a vector")
        M = self.P.boundary_tensor(C) @ self.P.space.G
        # entries below round-off *of this tensor* are dropped: a threshold relative to its
        # largest entry (review finding 35: the absolute 1e-14 dropped every row of 1e-15 I)
        scale = float(np.max(np.abs(M), initial=0.0))
        inner = self._trace(X, pos, d)
        out = {}
        for co in _comps(1, self.nc):
            terms = [inner[ci].scale(M[co[0], ci[0]]) for ci in inner
                     if abs(M[co[0], ci[0]]) > 1e-14 * scale]
            if terms:
                out[co] = _Ent(False, self.m + spin_k(co), None, None, _sum_blocks(terms))
        return out

    def _integrate(self, x: E.Integrate) -> dict:
        S = self.P.space
        coords = set(x.coords)
        inner = self._c(x.args[0])
        if x.args[0].rank:
            raise DiscretizationError(f"'{x}': integral of a tensor on a polar domain")
        e = inner[()]
        full = {S.phi, S.r} | ({S.zc} if S.cyl else set())
        if coords == full:
            if not e.bulk:
                raise DiscretizationError(f"'{x}': area integral of a quantity without r-dependence")
            if self.m != 0 or self.kz != 0:
                return {(): _Ent(False, 0, None, None, {key: sp.csr_matrix((1, b.shape[1]))
                                                        for key, b in e.blocks.items()})}
            row = self.ops.integral_row(e.level, self._p(e, x), self.n)
            if S.cyl:
                row = row * S.bz.length
            return {(): e.map(sp.csr_matrix(row[None, :]), bulk=False, level=None, p=None)}
        if coords == {S.phi} and not e.bulk:
            return {(): e.scale(2 * np.pi if self.m == 0 else 0.0)}
        raise DiscretizationError(f"'{x}': on a polar/cylindrical domain integ() is over the whole "
                                  "domain (element r dr dphi (dz)) or over phi on a circle")

    def _lift(self, x: E.Lift) -> dict:
        S = self.P.space
        if x.basis.coord != S.r or x.basis.size != S.N:
            raise DiscretizationError(f"'{x}': the lift basis must be the radial basis of the problem "
                                      f"(size {S.N})")
        level = int(getattr(x, "lam", 0))
        out = {}
        for c, e in self._c(x.args[0]).items():
            if e.bulk:
                raise DiscretizationError(f"'{x}': the lifted quantity depends on r")
            col = sp.csr_matrix(self.ops.lift_col(e.k, level, x.n, self.n)[:, None])
            out[c] = e.map(col, bulk=True, level=level, p=None)
        return out

    def _mul(self, x: E.Mul) -> dict:
        V, ev = self.vars, self.P.eigenvalue
        dep = [a for a in x.args if a.depends_on(V)]
        rest = [a for a in x.args if not a.depends_on(V)]
        if len(dep) != 1:
            raise DiscretizationError(f"'{x}' is not linear in the unknowns")
        has_ev = ev is not None and any(a == ev for a in rest)
        if ev is not None:
            if sum(1 for a in rest if a == ev) > 1 or any(a != ev and ev in a.free_parameters() for a in rest):
                raise DiscretizationError(f"'{x}': eigenvalue must appear linearly")
            if has_ev and ev in dep[0].free_parameters():
                raise DiscretizationError(f"'{x}': eigenvalue must appear linearly")
            if has_ev and self.mode != "M":
                return {}
            rest = [a for a in rest if a != ev]
        # s * (...): the factor goes to M as an L-form; c * (s u + ...) (eigenvalue nested in the
        # unknown factor): compiled in the current mode (a nested s u was dropped from M)
        form = self._c_L(dep[0]) if (self.mode == "M" and has_ev) else self._c(dep[0])
        scalar, nccs = 1.0, []
        for f in rest:
            v = self._const(f)
            if v is not None:
                scalar = scalar * v
            else:
                nccs.append(f)
        out = {c: e.scale(scalar) for c, e in form.items()}
        for f in nccs:
            if isinstance(f, E.UnitVector) and f.index == 2 and self.nc == 3 and out.keys() == {()}:
                # e_z (constant, spin 0) times a scalar unknown: its z component, unchanged. Also
                # for a quantity without radial dependence (e.g. a pressure-gradient unknown G of a
                # constant-flux pipe, G e_z, broadcast as a constant in r by ``add``: m = 0 only)
                out = {(2,): out[()]}
                continue
            if f.rank:                       # tensor coefficient times a scalar unknown
                e = out[()]
                self._need_bulk(e, x)
                out = {cf: e.map(self.ops.ncc_spin(vec, spin_k(cf), e.k, e.level, self.n),
                                 k=e.k + spin_k(cf))
                       for cf, vec in self.P.ncc_spin_coeffs(f).items()}
                continue
            g = self.P.ncc_radial(f)
            out = {c: (e.map(self.ops.ncc(g, e.k, e.level, self.n)) if e.bulk else
                       self._ncc_point(f, e)) for c, e in out.items()}
        return out

    def _dot(self, x: E.Dot) -> dict:
        """dot(coefficient, unknown) or dot(unknown, coefficient): contraction with the spin-basis
        metric (e_- . e_+ = 1/2, e_z . e_z = 1), each term an axisymmetric coefficient product."""
        a, b = x.args
        V = self.vars
        if a.depends_on(V) and b.depends_on(V):
            raise DiscretizationError(f"'{x}' is quadratic in the unknowns")
        left_unknown = a.depends_on(V)
        form = self._c(a if left_unknown else b)
        coef = self.P.ncc_spin_coeffs(b if left_unknown else a)
        G = self.P.space.G
        ra = a.rank
        out = {}
        for c in _comps(x.rank, self.nc):
            ca, cb = c[:ra - 1], c[ra - 1:]
            terms = []
            for s in range(self.nc):
                for t in range(self.nc):
                    if not G[s, t]:
                        continue
                    ua, ub = ca + (s,), (t,) + cb
                    fu, fc = (ua, ub) if left_unknown else (ub, ua)
                    if fu not in form or fc not in coef:
                        continue
                    e = form[fu]
                    self._need_bulk(e, x)
                    M = self.ops.ncc_spin(coef[fc], spin_k(fc), e.k, e.level, self.n)
                    terms.append(e.map(G[s, t] * M, k=e.k + spin_k(fc)))
            if terms:
                out[c] = self.add(terms, self.m + spin_k(c))
        return out

    def _ncc_point(self, f: E.Expr, e: _Ent) -> _Ent:
        raise DiscretizationError(f"non-constant coefficient '{f}' multiplying a quantity without "
                                  "radial dependence")


def _sum_blocks(ents: list[_Ent]) -> dict:
    out: dict = {}
    for e in ents:
        for key, b in e.blocks.items():
            out[key] = (out[key] + b).tocsr() if key in out else b
    return out


# ----------------------------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------------------------

class PolarPlan:
    """Discretized problem on a polar domain (see module docstring); same public face as Plan."""

    is_polar = True

    def __init__(self, problem: Problem, ncc: Mapping[E.Field, Any] | None = None,
                 params: Mapping[str, Any] | None = None, ncc_cutoff: float | None = None,
                 assembly: str = "direct", ncc_tail_tol: float = NCC_TAIL_TOL) -> None:
        if assembly != "direct":
            raise OptionError(f"polar plans assemble every group directly, got assembly={assembly!r}",
                              fix="assembly='direct' (the only assembly of curvilinear plans)")
        self.problem = problem
        self.params = dict(params or {})
        # constructor data kept to rebuild the plan when solve(data) changes a coefficient
        self._ncc_src = data_snapshot(ncc)
        self._plan_kw = dict(ncc_cutoff=ncc_cutoff, assembly=assembly, ncc_tail_tol=ncc_tail_tol)
        self.ncc_cutoff = us.check_cutoff(ncc_cutoff)
        # resolution of the non-constant coefficients (same semantics as the Cartesian plan:
        # a sampled tail above ncc_tail_tol, or the declared ncc_cutoff if larger, is refused)
        self.ncc_tail_tol = us.check_tail_tol(ncc_tail_tol)
        dom = None
        for v in problem.variables:
            dom = Domain.union(dom, v.domain)
        for eq in problem.equations:
            dom = Domain.union(dom, eq.domain)
        self.domain = dom
        self.space = PolarSpace(dom)
        self.variables = problem.variables
        self.eigenvalue = problem.eigenvalue if isinstance(problem, EVP) else None
        S = self.space
        if S.cyl:
            self.groups = tuple(PolarGroup(int(m), i, float(S.kz_all[j]), int(j), True)
                                for i, m in enumerate(S.ms) for j in S.zkeep)
        else:
            self.groups = tuple(PolarGroup(int(m), i) for i, m in enumerate(S.ms))
        empty = [int(m) for m in self.space.ms if self.space.ops.size(int(m)) <= 0]
        if empty:
            # a group without radial coefficients would silently drop its boundary data
            raise DiscretizationError(
                f"azimuthal wavenumbers |m| >= {2 * self.space.N} have no radial coefficient in the "
                f"disk truncation N - |m|//2 (N = {self.space.N}): groups m = {empty[:3]}... would be "
                f"empty; use Nphi/2 - 1 < 2N (here Nphi = {self.space.bphi.size})")
        used = frozenset(f for eq in problem.equations for side in (eq.lhs, eq.rhs)
                         for f in side.free_fields())
        stray = [f.name for f in (ncc or {}) if f not in used]
        if stray:
            raise DiscretizationError(
                f"ncc given for field(s) not used by the problem: {', '.join(sorted(stray))}")
        # same rule for parameter values: an unknown key (typo) used to be ignored and the value
        # declared on the Parameter used instead
        pnames = {p.name for eq in problem.equations for side in (eq.lhs, eq.rhs)
                  for p in side.free_parameters()}
        stray_p = sorted(set(self.params) - pnames)
        from .param_keys import check_param_values
        check_param_values(self.params, where="discretize")
        if stray_p:
            raise DiscretizationError(
                f"params given for parameter(s) not used by the problem: "
                f"{', '.join(repr(n) for n in stray_p)} (used: {sorted(pnames)})")
        from .param_keys import PlanParams     # later writes checked too (review, inputs)
        self.params = PlanParams(self.params, pnames, "plan.params")
        self._params_assembled: dict = {}   # the values of the first assembly (param_keys)
        self._ncc_coeffs = {f: self.coeffs_from_data(f, d) for f, d in (ncc or {}).items()}
        self._ncc_data = self._ncc_coeffs        # same role as Plan._ncc_data (presence test)
        self.ncc_reports: dict = {}
        self._cache: dict = {}
        self.evaluator = PolarEvaluator(self.space)
        self.taus = frozenset(f for eq in problem.equations for n in eq.lhs.walk()
                              if isinstance(n, E.Lift) for f in n.free_fields() & frozenset(problem.variables))
        self.eq_kind = []
        for eq in problem.equations:
            kind = self.space.kind(eq.domain)
            if kind in ("full", "radial"):
                self.eq_kind.append("bulk")
            elif any(isinstance(n, E.Interpolate) and n.coord == self.space.r for n in eq.lhs.walk()):
                self.eq_kind.append("boundary")
            else:
                self.eq_kind.append("constraint")
        # symmetry subspace (AzimuthalMFold, cylinder Reflection(z)): orbits of the signed
        # permutations on the coefficients, curvilinear_symmetry
        self.subspace = CurvilinearReduction(self) if problem.symmetry is not None else None

    # ---------------------------------------------------------------- sizes and data
    def size_of(self, dom: Domain | None, comp: tuple, group: PolarGroup) -> int:
        """Number of coefficients of one component of a variable on ``dom`` in ``group`` (0 if absent)."""
        kind, m = self.space.kind(dom), group.m
        if kind in ("radial", "const") and m != 0:
            return 0
        if self.space.cyl and not self.space.hz(dom) and group.kz != 0:
            return 0
        if kind in ("full", "radial"):
            return self.space.ops.size(m)
        return 1

    def group(self, m: int, kz: float | None = None) -> PolarGroup:
        """The group of azimuthal wavenumber m (and axial wavenumber kz on a cylinder)."""
        for g in self.groups:
            if g.m == m and (kz is None or abs(g.kz - kz) < 1e-12 * max(1.0, abs(kz))):
                if self.space.cyl and kz is None:
                    raise KeyError("cylinder: give kz as well")
                return g
        raise KeyError(f"no group m={m}" + (f", kz={kz}" if kz is not None else ""))

    def coeff_shape(self, dom: Domain | None, rank: int) -> tuple:
        """Full coefficient-array shape of a field of this rank on ``dom``."""
        return self.space.coeff_shape(dom, rank)

    def coeffs_from_data(self, f: E.Field, d: Any) -> np.ndarray:
        """Coefficients (spin, level 0) of a field from data.

        - callable of the grids (phi (Mphi, 1), r (1, Mr)) returning physical components: sampled on
          the dealiased grid (at least N + 1 radial points) and projected (exact for data in the
          discrete space, also for the vector components of odd m, polar_radial.DiskOps.forward);
        - FieldData(layout="grid"): physical values on the non-dealiased grid (plan.space.grids());
        - FieldData(layout="coeff"): spin coefficients in the plan's layout."""
        S = self.space
        dealias = "data" if (callable(d) and not isinstance(d, FieldData)) else False
        shape = S.grid_shape(f.domain, f.rank, dealias)
        if isinstance(d, FieldData):
            if d.layout == "coeff":
                return np.broadcast_to(np.asarray(d.data), S.coeff_shape(f.domain, f.rank)).astype(complex)
            g = np.broadcast_to(np.asarray(d.data), shape)
        elif callable(d):
            grids = S.grids("data")
            kind = S.kind(f.domain)
            nd = 3 if S.cyl else 2
            sh = lambda a, ax: a.reshape([-1 if i == ax else 1 for i in range(nd)])  # noqa: E731
            Pg = sh(grids[0], 0) if kind in ("full", "edge") else np.zeros((1,) * nd)
            Rg = sh(grids[1], 1) if kind in ("full", "radial") else np.full((1,) * nd, S.br.bounds[1])
            args = [Pg, Rg]
            if S.cyl:
                args.append(sh(grids[2], 2) if S.hz(f.domain) else np.full((1,) * nd, S.bz.bounds[0]))
            g = np.broadcast_to(np.asarray(d(*args)), shape)
        else:
            raise InputTypeError(f"data for {f.name}: pass a FieldData (explicit layout) or a callable "
                                 "of the grids (phi, r); bare arrays are ambiguous",
                                 fix="FieldData(array, layout='grid') or FieldData(array, layout='coeff')")
        if S.real and np.iscomplexobj(g) and np.max(np.abs(np.imag(g))) > 0:
            raise OptionError(f"data for {f.name}: complex values on a RealFourier (real-field) domain",
                              fix="pass the real field (np.real(...) if the imaginary part is "
                                  "rounding), or use a ComplexFourier azimuth")
        return S.forward(g, f.domain, f.rank, dealias)

    def to_grid(self, name: str, coeffs: np.ndarray, exact_state: bool = False) -> np.ndarray:
        """Physical components of a variable on its non-dealiased grid (phi, r); imaginary parts of
        real fields checked against space.state_real_tol (a solver state, polar_eval.STATE_REAL_TOL), or
        space.real_tol with ``exact_state`` (a projected solver state, verified exactly real)."""
        v = next(v for v in self.variables if v.name == name)
        # exact_state: m = 0 coefficients verified exactly real (projected solver state):
        # the transform bound space.real_tol applies instead of the state bound
        return self.space.backward(coeffs, v.domain, v.rank, False,
                                   real_tol=None if exact_state else self.space.state_real_tol)

    def ncc_radial(self, f: E.Expr) -> np.ndarray:
        """Radial coefficients (m = 0, level 0: Legendre in z on the disk, T in t on the annulus)
        of an axisymmetric coefficient expression; tail reported, never truncated silently."""
        key = ("ncc", f)
        if key in self._cache:
            return self._cache[key]
        missing = [x for x in f.free_fields() if x not in self._ncc_coeffs]
        if missing:
            raise DiscretizationError(
                f"non-constant coefficient '{f}': no data for {', '.join(m.name for m in missing)} "
                "(pass ncc={field: data})")
        a = self._axisymmetric(f).get((), np.zeros(self.space.N))
        if np.max(np.abs(a.imag)) <= 1e-14 * max(np.max(np.abs(a)), 1e-300):
            a = a.real
        if self.space.ops.kind == "disk":              # normalized Zernike -> Legendre in z
            a = self.space.ops.legendre_from(a)
        a, rep = us._ncc_coeffs(a, str(f), self.ncc_cutoff, tail_tol=self.ncc_tail_tol, axis="r")
        self.ncc_reports[(str(f), ())] = rep
        self._cache[key] = a
        return a

    def _axisymmetric(self, f: E.Expr) -> dict:
        """Radial level-0 coefficients {spin comp: vector} of an axisymmetric (m = 0, k_z = 0)
        coefficient expression; any phi or z dependence is refused (it would couple groups)."""
        key = ("axi", f)
        if key in self._cache:
            return self._cache[key]
        missing = [x for x in f.free_fields() if x not in self._ncc_coeffs]
        if missing:
            raise DiscretizationError(
                f"non-constant coefficient '{f}': no data for {', '.join(m.name for m in missing)} "
                "(pass ncc={field: data})")
        S = self.space
        c = self.evaluator.evaluate(f, self._ncc_coeffs, self.params)
        kind = S.kind(f.domain)
        if kind in ("edge", "const"):
            raise DiscretizationError(f"coefficient '{f}' has no radial dependence: use a constant")
        c3 = c if S.cyl else c[..., None]
        scale = max(np.abs(c3).max(), 1e-300)
        other = c3.copy()
        other[..., 0, :, 0] = 0
        if np.abs(other).max() > 1e-12 * scale:
            raise DiscretizationError(
                f"non-constant coefficient '{f}' depends on phi or z (|m, k_z != 0 part| = "
                f"{np.abs(other).max():.1e}): it couples wavenumbers; not separable")
        out = {}
        for comp in _comps(f.rank, S.nc):
            v = (c3[comp] if comp else c3)[0, :, 0]
            if np.abs(v).max() > 1e-14 * scale:
                out[comp] = v
        self._cache[key] = out
        return out

    def boundary_tensor(self, f: E.Expr) -> np.ndarray:
        """Spin components (constants, nc x nc) of a rank-2 coefficient given on the boundary
        circle (no r): only rotation-invariant tensors (m = 0, k_z = 0) are accepted (e_r e_r,
        I - e_r e_r, ...), so a product with a boundary trace stays in the group. Used by the
        curved free-slip condition (model/conditions.py)."""
        key = ("bt", f)
        if key in self._cache:
            return self._cache[key]
        missing = [x for x in f.free_fields() if x not in self._ncc_coeffs]
        if missing:
            raise DiscretizationError(
                f"boundary coefficient '{f}': no data for {', '.join(m.name for m in missing)} "
                "(pass ncc={field: data})")
        if f.rank != 2 or self.space.kind(f.domain) in ("full", "radial"):
            raise DiscretizationError(f"boundary coefficient '{f}': a rank-2 tensor without radial "
                                      "dependence is expected")
        c = np.asarray(self.evaluator.evaluate(f, self._ncc_coeffs, self.params))
        v = c.reshape(c.shape[:2] + (-1,))
        scale = max(np.abs(v).max(), 1e-300)
        if np.abs(v[..., 1:]).max(initial=0.0) > 1e-12 * scale:
            raise DiscretizationError(
                f"boundary coefficient '{f}' is not rotation invariant (content at m != 0 or "
                "k_z != 0): its product with a boundary trace would couple wavenumbers")
        out = v[..., 0]
        self._cache[key] = out
        return out

    def ncc_spin_coeffs(self, f: E.Expr) -> dict:
        """{spin comp: normalized level-0 radial coefficients} of a tensor coefficient (zero comps
        omitted); tails reported like scalar NCCs, never truncated unless ncc_cutoff is given."""
        out = {}
        for comp, v in self._axisymmetric(f).items():
            a = v.real if np.max(np.abs(v.imag)) <= 1e-14 * max(np.max(np.abs(v)), 1e-300) else v
            a, rep = us._ncc_coeffs(a, f"{f}{list(comp)}", self.ncc_cutoff, tail_tol=self.ncc_tail_tol, axis="r")
            self.ncc_reports[(str(f), comp)] = rep
            out[comp] = a
        return out

    # ---------------------------------------------------------------- group layouts
    def var_slots(self, group: PolarGroup) -> list[tuple]:
        """[(variable index, comp, offset, size)] in the group vector (variable-major)."""
        out, off = [], 0
        for vi, v in enumerate(self.variables):
            for c in _comps(v.rank, self.space.nc):
                n = self.size_of(v.domain, c, group)
                out.append((vi, c, off, n))
                off += n
        return out

    def eq_slots(self, group: PolarGroup) -> list[tuple]:
        """Row slots (equation, component, offset, size) of ``group``."""
        out, off = [], 0
        for i, eq in enumerate(self.problem.equations):
            for c in _comps(eq.lhs.rank, self.space.nc):
                n = self.size_of(eq.domain, c, group)
                out.append((i, c, off, n))
                off += n
        return out

    # ---------------------------------------------------------------- assembly
    def _assemble(self, group: PolarGroup) -> GroupMatrices:
        from .param_keys import check_assembled_params
        check_assembled_params(self)          # cached groups hold the first values
        key = ("full", group)
        if key in self._cache:
            return self._cache[key]
        vs, es = self.var_slots(group), self.eq_slots(group)
        vpos = {(self.variables[vi], c): (off, n) for vi, c, off, n in vs}
        nrow, ncol = sum(s[3] for s in es), sum(s[3] for s in vs)
        comp = _PolarCompiler(self, group)
        pieces: dict = {"L": [], "M": []}
        rows: list = []
        info: dict = {}
        forms: dict = {}
        for i, c, r0, nr in es:
            eq = self.problem.equations[i]
            if i not in forms:
                fL = comp.compile(eq.lhs, "L")
                fM = comp.compile(eq.lhs, "M") if isinstance(self.problem, (IVP, EVP)) else {}
                if (isinstance(self.problem, IVP) and any(isinstance(n, E.TimeDerivative) for n in eq.lhs.walk())
                        and not any(e.blocks for e in fM.values())):
                    raise DiscretizationError(f"equation [{i}] contains a time derivative but its "
                                              "mass operator M is empty: refusing")
                forms[i] = (fL, fM)
            fL, fM = forms[i]
            rows += [(i, c, self.eq_kind[i])] * nr
            if nr == 0:
                continue
            k = group.m + spin_k(c)
            bulk = self.space.kind(eq.domain) in ("full", "radial")
            ents = [f[c] for f in (fL, fM) if c in f]
            level = p = None
            if bulk:
                level = max([e.level for e in ents if e.bulk], default=0)
                p = max([e.p for e in ents if e.bulk and e.p is not None], default=0)
            info[(i, c)] = (r0, nr, k, level, p)
            for f, target in ((fL, "L"), (fM, "M")):
                if c not in f:
                    continue
                e = f[c]
                if bulk:
                    e = comp.normalize(e, k, level, p)
                elif e.bulk:
                    raise DiscretizationError(f"equation [{i}] lives on a boundary/constant domain but "
                                              "its left-hand side depends on r (missing interp?)")
                for (v, vc), blk in e.blocks.items():
                    c0, nv = vpos[(v, vc)]
                    if blk.shape != (nr, nv):
                        raise DiscretizationError(
                            f"internal shape mismatch in equation [{i}]{list(c)} for {v.name}{list(vc)}"
                            f" at m={group.m}: {blk.shape} vs {(nr, nv)}")
                    if blk.nnz:
                        pieces[target].append((r0, c0, blk.tocoo()))

        def build(ps):
            if not ps:
                return sp.csr_matrix((nrow, ncol), dtype=complex)
            r = np.concatenate([b.row + r0 for r0, _, b in ps])
            cc = np.concatenate([b.col + c0 for _, c0, b in ps])
            d = np.concatenate([b.data.astype(complex) for _, _, b in ps])
            A = sp.coo_matrix((d, (r, cc)), shape=(nrow, ncol)).tocsr()
            A.sum_duplicates()
            return A
        L, M = build(pieces["L"]), build(pieces["M"])
        cols = []
        for vi, c, _off, n in vs:
            v = self.variables[vi]
            kind = self.space.kind(v.domain)
            kd = "tau" if v in self.taus else ("field" if kind in ("full", "radial") else "constant")
            cols += [(v.name, c, kd)] * n
        lams = {key: val[3] for key, val in info.items()}
        gm = GroupMatrices(group, group.label, L, M, rows, cols, np.arange(nrow), np.arange(ncol), lams)
        gm.row_info = info          # (i, c) -> (row0, nrows, k, level, p): used for right-hand sides
        self._cache[key] = gm
        return gm

    def matrices(self, group: PolarGroup | None = None, reduced: bool | None = None) -> GroupMatrices:
        """Public L, M, boundary rows and tau columns of one azimuthal group (default: the first);
        in the declared symmetry subspace by default when the problem has one (``reduced``)."""
        return plan_matrices(self, group, reduced)

    def embed(self, x_red: np.ndarray, group: PolarGroup | None = None) -> np.ndarray:
        """E: symmetry-subspace group vector -> full group vector."""
        return plan_embed(self, x_red, group)

    def project(self, x_full: np.ndarray, group: PolarGroup | None = None) -> np.ndarray:
        """P: full group vector -> subspace coefficients (P E = I)."""
        return plan_project(self, x_full, group)

    rank_report = Plan.rank_report
    check = Plan.check

    # ---------------------------------------------------------------- right-hand sides
    def rhs_coeffs(self, coeffs: Mapping[E.Field, np.ndarray]) -> list[np.ndarray | None]:
        """Coefficients (level 0, spin) of every equation's right-hand side (None if zero)."""
        data = dict(self._ncc_coeffs)
        data.update(coeffs)
        out = []
        for k, eq in enumerate(self.problem.equations):
            out.append(None if eq.rhs.is_zero() else self.space.embed(
                self.evaluator.evaluate(eq.rhs, data, self.params), eq.domain, eq.lhs.rank,
                f"right-hand side of equation {k}"))
        return out

    def group_rhs(self, group: PolarGroup, rhs: list) -> np.ndarray:
        """Group right-hand side: each row block converted to its level and multiplied by r^p."""
        gm = self._assemble(group)
        b = np.zeros(gm.L.shape[0], dtype=complex)
        ops = self.space.ops
        for (i, c), (r0, nr, k, level, p) in gm.row_info.items():
            r = rhs[i]
            if r is None:
                continue
            dom = self.problem.equations[i].domain
            x = self.space._3(r[c] if c else r)
            mi, _, zi = self.space.cindex(dom, group) if self.space.cyl else self.space.cindex(dom, group) + (0,)
            if level is None:
                b[r0] = x[mi, 0, zi]
                continue
            vec = x[mi, :nr, zi]
            b[r0:r0 + nr] = ops.multiply_r(k, level, p, nr) @ (ops.convert(k, 0, level, nr) @ vec)
        return b

    def solve(self, data: Mapping[E.Field, Any] | None = None, reduced: bool | None = None) -> LBVPSolution:
        """LBVP: one sparse LU per group, relative residual of every solve reported."""
        if not isinstance(self.problem, LBVP):
            raise DiscretizationError("solve() is for LBVPs (EVP: solvers.eigenvalues, IVP: IVPSolver)")
        fresh = rebind_for_data(self, data)       # a new operator coefficient: rebuilt plan
        if fresh is not None:
            return fresh.solve(data, reduced)
        if reduced is None:
            reduced = self.problem.symmetry is not None
        self.check(reduced)
        coeffs = {f: self.coeffs_from_data(f, d) for f, d in (data or {}).items()}
        rhs = self.rhs_coeffs(coeffs)
        out = {v.name: np.zeros(self.space.coeff_shape(v.domain, v.rank), dtype=complex)
               for v in self.variables}
        if reduced:
            return LBVPSolution(out, reduced_lbvp_solve(self, rhs, out), self, True)
        residuals = []
        for g in self.groups:
            gm = self._assemble(g)
            if gm.L.shape[0] == 0:
                continue
            b = self.group_rhs(g, rhs)
            A = gm.L.tocsc()
            x = spla.splu(A).solve(b)
            residuals.append((gm.label, float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))))
            self.scatter_group(g, x, out)
        return LBVPSolution(out, residuals, self, False)

    def scatter_group(self, group: PolarGroup, x: np.ndarray, out: dict) -> None:
        """Write the group solution ``x`` into the full coefficient arrays ``out`` (by variable name)."""
        S = self.space
        for vi, c, off, n in self.var_slots(group):
            v = self.variables[vi]
            if n == 0:
                continue
            arr = S._3(out[v.name][c] if c else out[v.name])        # a view: writes go through
            mi, _, zi = S.cindex(v.domain, group) if S.cyl else S.cindex(v.domain, group) + (0,)
            if S.kind(v.domain) in ("full", "radial"):
                arr[mi, :n, zi] = x[off:off + n]
            else:
                arr[mi, 0, zi] = x[off]

    # ---------------------------------------------------------------- explain / methods
    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import explain_html
        return explain_html(self)

    def explain(self, group: PolarGroup | None = None) -> str:
        """Bases, groups, equation spaces and rank report of ``group`` (default the first one)."""
        S = self.space
        g0 = self.groups[0] if group is None else group
        gm = self._assemble(g0)
        lines = [f"Discretization plan for {self.problem.kind} '{self.problem.name}' (polar)",
                 f"  domain: {self.domain!r}",
                 f"  {S.phi.name}: {type(S.bphi).__name__}, {len(self.groups)} groups m = "
                 f"{S.ms.min()}..{S.ms.max()} (separable, one block per azimuthal wavenumber)"
                 + (f"; {S.zc.name}: ComplexFourier({S.bz.size}), one block per (m, k_z)" if S.cyl else "")
                 + (" ; real fields: m >= 0 stored, f_-m = conj(f_m) with spins flipped" if S.real else ""),
                 f"  {S.r.name}: {S.ops.explain()}",
                 "  tensors: spin components (u_- = u_r - i u_phi, u_+ = u_r + i u_phi), effective "
                 "wavenumber k = m + sum of spins; nothing divided by r"]
        if S.real:
            lines.append(f"  real fields: imaginary parts on the grid refused above {S.real_tol:g} x max|value| "
                         "(plan.space.real_tol, declared; never dropped silently); a pointwise function "
                         f"in the RHS (T**2, exp) acts on Re of its argument, refused above "
                         f"{S.func_real_tol:g} relative (plan.space.func_real_tol: the non-real m = 0 part "
                         "of a state, solve rounding, grows relative to a decaying field); state outputs "
                         f"(to_grid) refuse above {S.state_real_tol:g} relative")
        if S.ops.kind == "annulus":
            lines.append("  annulus rows: multiplied by r^p (p = largest r-power of the row's terms, "
                         "declared); right-hand sides multiplied identically; lifted taus are not")
        lines.append(f"  group [{gm.label}]: {gm.shape[0]} x {gm.shape[1]}, nnz(L) = {gm.L.nnz}, "
                     f"nnz(M) = {gm.M.nnz}")
        for (i, c), (_r0, nr, k, level, p) in sorted(gm.row_info.items(), key=lambda t: (t[0][0], t[0][1])):
            space = ("point (1 row)" if level is None else
                     (f"P^({level}, {abs(k)}) of z, {nr} rows" if S.ops.kind == "disk"
                      else f"C^({level}) of t, x r^{p}, {nr} rows"))
            lines.append(f"  equation [{i}]{list(c) if c else ''}: {self.eq_kind[i]}, k={k}, {space}")
        lines.append("  NCC resolution: " + (
            us.ncc_policy(self.ncc_tail_tol, self.ncc_cutoff)))
        if self.ncc_reports:
            lines.append("  non-constant coefficients (axisymmetric):")
            for rep in self.ncc_reports.values():
                lines.append("    " + rep.explain())
        Ms = S.sizes(True)
        lines.append(f"  nonlinear terms: dealiased grid {' x '.join(str(v) for v in Ms)} = floor(N * dealias) "
                     f"({S.bphi.dealias:g}, {S.br.dealias:g}); spectral derivatives, grid products, "
                     "projection back (azimuthal truncation, radial " +
                     ("Gauss-Legendre projection)" if S.ops.kind == "disk" else "Chebyshev truncation)"))
        if self.subspace is not None:          # declared symmetry subspace: reduction and warning
            lines.append("  " + self.subspace.explain())
            lines += ["  " + ln for ln in self.problem.symmetry.explain().splitlines()]
        lines.append(Plan.rank_report(self, g0).explain())
        return "\n".join(lines)

    def methods(self, solver: Any = None, **kw):
        """Numerical-methods summary (conatus.methods) of this plan."""
        from ..methods import methods
        return methods(self, solver, **kw)
