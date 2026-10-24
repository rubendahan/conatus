"""IR -> sparse blocks, for one mode group (the core of the discretization).

A linear expression is compiled into a *form*: for each output tensor component,
an ``Entry`` = (ultraspherical index lam along z or None if constant in z,
Fourier dependence, and one sparse block per (variable, component)). Terms are
brought to a common lam by the conversions S (Olver–Townsend); constants in a
direction are broadcast explicitly (T_0 along z, zero mode along Fourier).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from itertools import product
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..errors import DiscretizationError, ParameterError  # public classes: conatus.errors
from ..ir import expr as E
from ..ir.coords import Coordinate
from ..tolerances import NCC_TAIL_TOL  # declared default
from . import ultraspherical as us
from .layout import GroupItem, Layout, lam_of

__all__ = ["Entry", "Compiler", "DiscretizationError"]

Key = tuple  # (Field, component tuple)


@dataclass
class Entry:
    """One output tensor component of a compiled form: its level ``lam`` along z (None: constant
    there), its Fourier dependence and one sparse block per (variable, component)."""
    lam: int | None                      # None: constant along z
    fdep: tuple[bool, ...]
    blocks: dict[Key, sp.spmatrix] = dc_field(default_factory=dict)

    def copy_with(self, blocks: dict, lam: Any = "same", fdep: Any = "same") -> Entry:
        """A copy with new ``blocks`` (and optionally a new level or Fourier dependence)."""
        return Entry(self.lam if lam == "same" else lam,
                     self.fdep if fdep == "same" else fdep, blocks)


def _comps(tensorsig: tuple) -> list[tuple]:
    return list(product(*[range(cs.dim) for cs in tensorsig])) if tensorsig else [()]


class Compiler:
    """Compile LHS expressions of one problem for one group."""

    def __init__(self, layout: Layout, group: tuple[GroupItem, ...], variables: tuple[E.Field, ...],
                 ncc_coeffs: Callable[[E.Field, tuple], np.ndarray],
                 ncc_reports: dict, params: Mapping[str, Any], eigenvalue: E.Parameter | None,
                 ncc_cutoff: float | None = None, ncc_tail_tol: float = NCC_TAIL_TOL,
                 ncc_exact: frozenset = frozenset(), ncc_floor: dict | None = None) -> None:
        self.L = layout
        self.g = group
        self.vars = frozenset(variables)
        self.ncc_coeffs = ncc_coeffs
        self.ncc_reports = ncc_reports
        self.params = params
        self.eigenvalue = eigenvalue
        self.ncc_cutoff = ncc_cutoff
        self.ncc_tail_tol = ncc_tail_tol      # refusal threshold of a sampled NCC tail
        self.ncc_exact = ncc_exact            # fields given as coefficients (exact, no tail)
        self.ncc_floor = {} if ncc_floor is None else ncc_floor   # rounding level per NCC (plan)
        self.nz = layout.nz
        self.zscale = 2.0 / layout.zbasis.length if layout.zbasis is not None else None
        # mapped Chebyshev (ir/maps.py): d/dy = mu(xi) d/dxi, mu expanded in T_n (exact if polynomial)
        self.zmap = getattr(layout.zbasis, "map", None)
        if self.zmap is not None:
            self.zscale = 1.0
        self._metric_cache: dict[int, sp.csr_matrix] = {}
        # Hermite / Laguerre / compound Chebyshev along z: every z-operator comes from the family
        # object (discretize/families.py); None keeps the Chebyshev T code below unchanged.
        self.zfam = layout.zfam
        self.mode = "L"

    # ------------------------------------------------------------------ helpers
    def _nf(self, fdep) -> int:
        return self.L.nf(fdep, self.g)

    def _F(self, fdep, per_dir: dict[int, np.ndarray]) -> sp.spmatrix:
        """kron over Fourier dirs: given matrices on some dirs, identity elsewhere."""
        out = sp.identity(1, format="csr")
        for i, (dep, item) in enumerate(zip(fdep, self.g, strict=True)):
            m = per_dir.get(i)
            if m is None:
                m = sp.identity(item.size(dep), format="csr")
            out = sp.kron(out, sp.csr_matrix(m), format="csr")
        return out

    def _apply(self, e: Entry, F: sp.spmatrix | None, Z: sp.spmatrix | None) -> dict:
        nzr = self.nz if e.lam is not None else 1
        if F is None:
            F = sp.identity(self._nf(e.fdep), format="csr")
        if Z is None:
            Z = sp.identity(nzr, format="csr")
        K = sp.kron(F, Z, format="csr")
        return {k: (K @ b).tocsr() for k, b in e.blocks.items()}

    def convert(self, e: Entry, lam: int) -> Entry:
        """The entry converted to level ``lam`` (unchanged when constant along z or already
        there)."""
        if e.lam is None or e.lam == lam:
            return e
        S = (self.zfam.conversion_chain(e.lam, lam) if self.zfam is not None
             else us.conversion_chain(self.nz, e.lam, lam))
        return e.copy_with(self._apply(e, None, S), lam=lam)

    def broadcast(self, e: Entry, fdep: tuple[bool, ...], z: bool) -> Entry:
        """The entry made dependent on the Fourier directions ``fdep`` and, if ``z``, on the
        non-periodic one."""
        per = {i: item.embed_const() for i, (item, a, b) in enumerate(zip(self.g, e.fdep, fdep, strict=True))
               if b and not a}
        for a, b in zip(e.fdep, fdep, strict=True):
            if a and not b:
                raise DiscretizationError("internal: cannot broadcast to fewer Fourier directions")
        F = self._F(e.fdep, per) if per else None  # rows follow the target dependence
        Z = None
        lam = e.lam
        if z and e.lam is None:
            # constant = T_0 (one T_0 per piece on a compound interval; refused on Hermite /
            # Laguerre axes, whose functions decay: families._no_constant)
            Z = (self.zfam.constant_column() if self.zfam is not None
                 else sp.csr_matrix(np.eye(self.nz, 1)))
            lam = 0
        if F is None and Z is None:
            return e
        return Entry(lam, tuple(fdep), self._apply(e, F, Z))

    def add_entries(self, entries: list[Entry]) -> Entry:
        """Sum of entries, broadcast to a common Fourier and non-periodic dependence and the
        highest level."""
        fdep = tuple(any(e.fdep[i] for e in entries) for i in range(len(self.g)))
        z = any(e.lam is not None for e in entries)
        entries = [self.broadcast(e, fdep, z) for e in entries]
        lam = max(e.lam for e in entries) if z else None
        out: dict = {}
        for e in entries:
            e = self.convert(e, lam) if z else e
            for k, b in e.blocks.items():
                out[k] = (out[k] + b).tocsr() if k in out else b
        return Entry(lam, fdep, out)

    def scale(self, e: Entry, c: complex) -> Entry:
        """The entry multiplied by the number ``c``."""
        return e.copy_with({k: (c * b).tocsr() for k, b in e.blocks.items()})

    # ------------------------------------------------------------------ compile
    def compile(self, x: E.Expr, mode: str = "L") -> dict[tuple, Entry]:
        """Sparse blocks (per unknown) of the expression ``x``; ``mode`` 'L' or 'M' picks the
        operator, and an EVP splits off the eigenvalue."""
        self.mode = mode
        ev = self.eigenvalue
        if ev is not None and ev in x.free_parameters() and not any(
                isinstance(n, E.TimeDerivative) for n in x.walk()):
            # EVP: split x = x0 + ev * x1 symbolically (products of sums are distributed, e.g.
            # (U - c) L(v)), then L <- x0 and M <- x1. Refuses any other dependence on ev.
            parts = eigen_split(x, ev)
            part = parts.get(1 if mode == "M" else 0)
            return {} if part is None else self._c_L(part)
        return self._c(x)

    def _param(self, p: E.Parameter) -> Any:
        if self.eigenvalue is not None and p == self.eigenvalue:
            raise DiscretizationError("internal: eigenvalue handled in Mul")
        v = self.params.get(p.name, p.value)
        if v is None:
            raise ParameterError(
                f"parameter {p.name} has no value", where=f"Parameter('{p.name}') in the LHS",
                why="the LHS matrices are assembled with numbers; a Parameter without value is a "
                    "free symbol (only the eigenvalue of an EVP or the parameters of a "
                    "ParametricEVP stay symbolic)",
                fix=f"give it a value, Parameter('{p.name}', 1e-2), or pass "
                    f"discretize(pb, params={{'{p.name}': ...}})")
        return v

    def _c(self, x: E.Expr) -> dict[tuple, Entry]:
        V = self.vars
        if not x.depends_on(V) and not isinstance(x, E.TimeDerivative):
            raise DiscretizationError(f"'{x}' does not depend on the unknowns: it belongs on the RHS")
        if isinstance(x, E.Field):
            if self.mode == "M":
                # A bare unknown has no dt() and no eigenvalue, so it never enters M.
                # It was returned as the identity in M mode too, so dt(u) + u = 0 gave M = 2 (u decaying
                # as e^{-t/2}) and s*u + u = 0 gave s = -1/2: growth rates silently wrong.
                return {}
            fdep = self.L.fdep_of(x.domain)
            z = self.L.zdep_of(x.domain)
            n = self._nf(fdep) * (self.nz if z else 1)
            return {c: Entry(0 if z else None, fdep, {(x, c): sp.identity(n, format="csr")})
                    for c in _comps(x.tensorsig)}
        if isinstance(x, E.TimeDerivative):
            if self.mode != "M":
                return {}
            arg = x.args[0]
            if any(isinstance(n, E.TimeDerivative) for n in arg.walk()):
                raise DiscretizationError(f"nested time derivative in '{x}': not supported")
            if self.eigenvalue is not None and self.eigenvalue in arg.free_parameters():
                raise DiscretizationError(f"'{x}': eigenvalue inside a time derivative")
            # dt commutes with time-independent linear spatial operators: M is the full
            # operator applied to the argument (compiled as an L-form, never truncated)
            out = self._c_L(arg)
            if not any(e.blocks for e in out.values()):
                raise DiscretizationError(f"'{x}' compiles to an empty mass operator")
            return out
        if self.mode == "M" and not any(isinstance(n, E.TimeDerivative) for n in x.walk()) and (
                self.eigenvalue is None or self.eigenvalue not in x.free_parameters()):
            return {}
        if isinstance(x, E.Add):
            parts = [self._c(t) for t in x.args if t.depends_on(V) or isinstance(t, E.TimeDerivative)]
            out: dict = {}
            for c in {c for p in parts for c in p}:
                ents = [p[c] for p in parts if c in p]
                out[c] = self.add_entries(ents)
            return out
        if isinstance(x, E.Mul):
            return self._mul(x)
        if isinstance(x, E.Differentiate):
            return {c: self._diff(e, x.coord) for c, e in self._c(x.args[0]).items()}
        if isinstance(x, E.Gradient):
            inner = self._c(x.args[0])
            coords = x.coordsys.coords
            return {(i,) + c: self._diff(e, coords[i]) for c, e in inner.items()
                    for i in range(len(coords))}
        if isinstance(x, E.Divergence):
            inner = self._c(x.args[0])
            coords = x.args[0].tensorsig[0].coords
            out = {}
            for c in _comps(x.tensorsig):
                ents = [self._diff(inner[(i,) + c], coords[i]) for i in range(len(coords))
                        if (i,) + c in inner]
                if ents:
                    out[c] = self.add_entries(ents)
            return out
        if isinstance(x, E.Laplacian):
            inner = self._c(x.args[0])
            out = {}
            for c, e in inner.items():
                out[c] = self.add_entries([self._diff(self._diff(e, q), q)
                                           for q in x.coordsys.coords])
            return out
        if isinstance(x, E.Curl):
            return self._curl(x)
        if isinstance(x, E.Component):
            inner = self._c(x.args[0])
            return {c[1:]: e for c, e in inner.items() if c[0] == x.index}
        if isinstance(x, E.Dot):
            return self._dot(x)
        if isinstance(x, E.Cross):
            return self._cross(x)
        if isinstance(x, E.Interpolate):
            return {c: self._interp(e, x.coord, x.position, getattr(x, "side", None))
                    for c, e in self._c(x.args[0]).items()}
        if isinstance(x, E.Integrate):
            out = self._c(x.args[0])
            for q in x.coords:
                out = {c: self._integ(e, q) for c, e in out.items()}
            return out
        if isinstance(x, E.Lift):
            return {c: self._lift(e, x) for c, e in self._c(x.args[0]).items()}
        raise DiscretizationError(f"cannot discretize node {type(x).__name__} ('{x}') in the LHS")

    # ------------------------------------------------------------------ node rules
    def _diff(self, e: Entry, q: Coordinate) -> Entry:
        for i, d in enumerate(self.L.fdirs):
            if d.coord == q:
                if not e.fdep[i]:
                    return Entry(e.lam, e.fdep, {})
                F = self._F(e.fdep, {i: self.g[i].deriv(1)})
                return e.copy_with(self._apply(e, F, None))
        if self.L.zcoord == q:
            if e.lam is None:
                return Entry(None, e.fdep, {})
            if self.zfam is not None:
                return e.copy_with(self._apply(e, None, self.zfam.differentiation(e.lam, 1)),
                                   lam=e.lam + 1)
            D = us.differentiation(self.nz, e.lam, 1) * self.zscale
            if self.zmap is not None:           # chain rule: mu(xi) d/dxi, in C^(lam+1)
                D = self._metric(e.lam + 1) @ D
            return e.copy_with(self._apply(e, None, D), lam=e.lam + 1)
        return Entry(e.lam, e.fdep, {})  # no basis along q: constant

    def _metric(self, lam: int) -> sp.csr_matrix:
        """Multiplication by the map's metric mu = dxi/dy in C^(lam) (reported like an NCC)."""
        if lam not in self._metric_cache:
            coeffs, how = self.zmap.metric_coeffs(self.nz)
            # an exact polynomial metric ((1 - xi^2)/L: 3 coefficients) has no tail to measure;
            # judged as a grid sample it raised a false NCCResolutionWarning (tail = 1)
            M, rep = us.multiplication(coeffs, self.nz, lam, name=f"metric dxi/dy of {self.zmap!r}",
                                       tail_tol=self.ncc_tail_tol,
                                       exact=how.startswith("exact polynomial"))
            self.ncc_reports[("metric", ())] = _MetricReport(rep, how)
            self._metric_cache[lam] = M
        return self._metric_cache[lam]

    def _interp(self, e: Entry, q: Coordinate, pos: float, side: str | None = None) -> Entry:
        if self.L.zcoord == q:
            if e.lam is None:
                return e
            if self.zfam is not None:
                row = sp.csr_matrix(self.zfam.evaluation_row(e.lam, pos, 0, side))
                return e.copy_with(self._apply(e, None, row), lam=None)
            zb = self.L.zbasis
            t = (self.zmap.xi_of(pos) if self.zmap is not None
                 else 2 * (pos - zb.bounds[0]) / zb.length - 1)
            row = sp.csr_matrix(us.evaluation_row(self.nz, e.lam, t))
            return e.copy_with(self._apply(e, None, row), lam=None)
        for d in self.L.fdirs:
            if d.coord == q:
                raise DiscretizationError(
                    f"interp along Fourier direction {q.name} couples all wavenumbers: not a "
                    "separable operator (not supported in the LHS)")
        return e

    def _integ(self, e: Entry, q: Coordinate) -> Entry:
        if self.L.zcoord == q and self.zmap is not None:
            raise DiscretizationError(
                f"integ along {q.name} on the mapped interval {self.L.zbasis.bounds}: the weight "
                "dy/dxi is not integrable at infinity; not implemented (integrate a decaying "
                "quantity by hand or use a finite interval)")
        if self.L.zcoord == q:
            if e.lam is None:
                if self.zfam is not None and not self.zfam.has_constant:
                    raise DiscretizationError(
                        f"integ along the unbounded axis {q.name} of a quantity constant along it "
                        "diverges")
                return self.scale(e, self.L.zbasis.length)
            if self.zfam is not None:
                row = sp.csr_matrix(self.zfam.integration_row(e.lam))
                return e.copy_with(self._apply(e, None, row), lam=None)
            row = sp.csr_matrix(us.integration_row(self.nz, e.lam) * self.L.zbasis.length / 2)
            return e.copy_with(self._apply(e, None, row), lam=None)
        for i, d in enumerate(self.L.fdirs):
            if d.coord == q:
                if not e.fdep[i]:
                    return self.scale(e, d.basis.length)
                F = self._F(e.fdep, {i: self.g[i].integrate(d.basis.length)})
                fdep = tuple(False if j == i else f for j, f in enumerate(e.fdep))
                return Entry(e.lam, fdep, self._apply(e, F, None))
        raise DiscretizationError(f"integration along {q.name}: no basis there")

    def _lift(self, e: Entry, x: E.Lift) -> Entry:
        b = x.basis
        if self.L.zcoord != b.coord:
            raise DiscretizationError(f"lift along {b.coord.name}: not the non-periodic direction")
        if b.size != self.nz:
            raise DiscretizationError(
                f"lift basis size {b.size} differs from the problem basis size {self.nz} along "
                f"{b.coord.name}")
        if e.lam is not None:
            raise DiscretizationError(f"lifted quantity '{x.args[0]}' depends on {b.coord.name}")
        col = np.zeros((self.nz, 1))
        piece = getattr(x, "piece", None)
        if self.zfam is not None:          # compound: mode N_p + n of piece p (families.lift_index)
            col[self.zfam.lift_index(x.n, piece), 0] = 1.0
        else:
            col[self.nz + x.n, 0] = 1.0
        # lam >= 1: the tau polynomial is C^(lam)_{N+n} (derivative basis), a unit column there
        from .jacobi_family import is_jacobi_trial
        if is_jacobi_trial(self.L.zbasis):
            # Legendre / Jacobi / Ultraspherical trial basis: the lift basis is the problem's own
            # basis and the level is lift(..., lam=q) (0 by default), as on Laguerre axes; an
            # Ultraspherical lift basis means C^(lam) only on a Chebyshev T problem
            if b != self.L.zbasis:
                raise DiscretizationError(
                    f"lift basis {b!r} on a {type(self.L.zbasis).__name__} problem",
                    fix=f"lift(tau, {self.L.zbasis!r}, n, lam=q): the problem basis, with the "
                        "level q of the equation (its derivative order)")
            lam = int(getattr(x, "lam", 0) or 0)
        else:
            lam = x.lam if getattr(x, "lam", 0) else lam_of(b)
        return e.copy_with(self._apply(e, None, sp.csr_matrix(col)), lam=lam)

    def _const_factor(self, f: E.Expr) -> Any:
        """Numeric value of a variable-free, grid-free scalar factor, or None."""
        if isinstance(f, E.Constant) and f.rank == 0:
            return f.value
        if isinstance(f, E.Parameter):
            return self._param(f)
        if isinstance(f, E.Conj) and not f.free_fields():     # conj of a complex parameter
            b = self._const_factor(f.args[0])
            return None if b is None else np.conj(b)
        if isinstance(f, E.Power) and not f.free_fields():
            b = self._const_factor(f.args[0])
            return None if b is None else b ** f.exponent
        if isinstance(f, E.Mul) and not f.free_fields():
            v = 1.0
            for a in f.args:
                w = self._const_factor(a)
                if w is None:
                    return None
                v = v * w
            return v
        return None

    def _mul(self, x: E.Mul) -> dict:
        V = self.vars
        dep = [a for a in x.args if a.depends_on(V)]
        rest = [a for a in x.args if not a.depends_on(V)]
        if len(dep) != 1:
            raise DiscretizationError(f"'{x}' is not linear in the unknowns")
        # eigenvalue: L gets the sigma-free terms, M the coefficient of sigma
        ev = self.eigenvalue
        has_ev = ev is not None and any(a == ev for a in rest)
        if ev is not None:
            if sum(1 for a in rest if a == ev) > 1 or any(
                    a != ev and ev in a.free_parameters() for a in rest):
                raise DiscretizationError(f"'{x}': eigenvalue must appear linearly")
            if has_ev != (self.mode == "M"):
                return {}
            rest = [a for a in rest if a != ev]
        form = self._c(dep[0]) if not (self.mode == "M" and ev is not None) else self._c_L(dep[0])
        scalar = 1.0
        tensor_factors = []
        for f in rest:
            v = self._const_factor(f)
            if v is not None:
                scalar = scalar * v
            else:
                tensor_factors.append(f)
        out = {c: self.scale(e, scalar) for c, e in form.items()}
        for f in tensor_factors:
            out = self._mul_factor(f, out, dep[0])
        return out

    def _c_L(self, x: E.Expr) -> dict:
        saved = self.mode
        self.mode = "L"
        try:
            return self._c(x)
        finally:
            self.mode = saved

    def _mul_factor(self, f: E.Expr, form: dict, operand: E.Expr) -> dict:
        if isinstance(f, E.UnitVector):
            if operand.rank:
                raise DiscretizationError("unit vector times a tensor: use dot/outer")
            return {(f.index,) + c: e for c, e in form.items()}
        if f.rank != 0:
            raise DiscretizationError(f"tensor coefficient '{f}' in a product: use dot()")
        return {c: self._ncc(f, e) for c, e in form.items()}

    def _ncc(self, f: E.Expr, e: Entry, comp: tuple = ()) -> Entry:
        """Multiply an entry by a non-constant coefficient (function of z only)."""
        fields = f.free_fields()
        for fl in fields:
            if any(self.L.fdep_of(fl.domain)):
                raise DiscretizationError(
                    f"non-constant coefficient '{f}' varies along a Fourier direction: it couples "
                    "wavenumbers (not separable); not supported")
        if not self.L.zdep_of(f.domain):
            v = self.ncc_coeffs(f, comp)
            return self.scale(e, complex(v[0]) if np.iscomplexobj(v) else float(v[0]))
        a_T = self.ncc_coeffs(f, comp)
        if e.lam is None:
            e = self.broadcast(e, e.fdep, True)
        mult = self.zfam.multiplication if self.zfam is not None else (
            lambda a, lam, **kw: us.multiplication(a, self.nz, lam, axis=self.L.zcoord.name, **kw))
        kw = {"exact": True} if isinstance(f, E.Field) and f in self.ncc_exact else {}
        kw["floor"] = self.ncc_floor.get((f, comp), 0.0)
        M, rep = mult(a_T, e.lam, name=str(f), cutoff=self.ncc_cutoff, tail_tol=self.ncc_tail_tol,
                      **kw)
        self.ncc_reports[(str(f), comp)] = rep
        return e.copy_with(self._apply(e, None, M))

    def _dot(self, x: E.Dot) -> dict:
        a, b = x.args
        V = self.vars
        if a.depends_on(V) and b.depends_on(V):
            raise DiscretizationError(f"'{x}' is quadratic in the unknowns")
        if a.depends_on(V):
            form, coef, left = self._c(a), b, False
        else:
            form, coef, left = self._c(b), a, True
        out: dict = {}
        dim = (a.tensorsig[-1] if not left else b.tensorsig[0]).dim
        for c in _comps(x.tensorsig):
            ents = []
            for i in range(dim):
                if left:   # coef . form : contract coef's last index with form's first
                    cc, fc = c[:coef.rank - 1] + (i,), (i,) + c[coef.rank - 1:]
                else:      # form . coef
                    fc, cc = c[:form_rank(a) - 1] + (i,), (i,) + c[form_rank(a) - 1:]
                if fc not in form:
                    continue
                ent = self._coef_times(coef, cc, form[fc])
                if ent is not None:
                    ents.append(ent)
            if ents:
                out[c] = self.add_entries(ents)
        return out

    def _const_vector(self, f: E.Expr) -> np.ndarray | None:
        """Components of a constant vector (unit vectors times numbers or parameters, and sums
        of them), or None when f is not such a constant."""
        if isinstance(f, E.UnitVector):
            v = np.zeros(f.coordsys.dim)
            v[f.index] = 1.0
            return v
        if f.free_fields() or f.rank != 1:
            return None
        if isinstance(f, E.Add):
            parts = [self._const_vector(a) for a in f.args]
            return None if any(p is None for p in parts) else sum(parts)
        if isinstance(f, E.Mul):
            vec = [a for a in f.args if a.rank]
            if len(vec) != 1:
                return None
            v = self._const_vector(vec[0])
            for a in f.args:
                if a is vec[0]:
                    continue
                c = self._const_factor(a)
                if v is None or c is None:
                    return None
                v = v * c
            return v
        return None

    def _cross(self, x: E.Cross) -> dict:
        """cross(W, u) or cross(u, W) with W a constant vector (implicit Coriolis): each
        component of the result is a combination of the components of u with constant
        coefficients, so it couples the components inside the groups of the banded solve."""
        from ..ir.expr_algebra import cartesian_cross_table
        a, b = x.args
        V = self.vars
        if a.depends_on(V) and b.depends_on(V):
            raise DiscretizationError(f"'{x}' is quadratic in the unknowns")
        left = not a.depends_on(V)
        coef, operand = (a, b) if left else (b, a)
        w = self._const_vector(coef)
        if w is None:
            raise DiscretizationError(
                f"'{x}': a cross product on the left-hand side needs a constant vector, "
                f"'{coef}' is not one",
                why="an implicit term is assembled into the matrices once: its coefficients "
                    "must be numbers (unit vectors times constants or parameters)",
                fix=f"write the vector as a combination of unit vectors (e.g. 2*Omega*ez), or "
                    f"treat the term explicitly: explicit({x}) on the right-hand side",
                doc="docs/site/guide/operators.md")
        if np.iscomplexobj(w) and not np.any(np.imag(w)):
            w = np.real(w)
        form = self._c(operand)
        out: dict = {}
        for (i, j), (k, sg) in cartesian_cross_table(a.tensorsig[0]).items():
            wv, fc = (w[i], (j,)) if left else (w[j], (i,))
            if wv == 0 or fc not in form:
                continue
            out.setdefault((k,), []).append(self.scale(form[fc], sg * wv))
        return {c: self.add_entries(es) for c, es in out.items()}

    def _coef_times(self, coef: E.Expr, cc: tuple, e: Entry) -> Entry | None:
        if isinstance(coef, E.UnitVector):
            return e if cc == (coef.index,) else None
        if isinstance(coef, E.Field) or not coef.depends_on(self.vars):
            # a coefficient field, or an expression of coefficient fields (e.g. grad(T0)):
            # evaluated by the reference evaluator, expanded as an NCC (reported, never truncated)
            return self._ncc(coef, e, cc)
        raise DiscretizationError(f"coefficient '{coef}' in dot() depends on the unknowns")

    def _curl(self, x: E.Curl) -> dict:
        inner = self._c(x.args[0])
        cs = x.coordsys.coords
        rin = x.args[0].rank

        def d(comp, j):
            return self._diff(inner[comp], cs[j]) if comp in inner else None

        def comb(*terms):
            ents = [self.scale(e, s) for s, e in terms if e is not None]
            return self.add_entries(ents) if ents else None

        if len(cs) == 2 and rin == 0:
            res = {(0,): comb((1, d((), 1))), (1,): comb((-1, d((), 0)))}
        elif len(cs) == 2 and rin == 1:
            res = {(): comb((1, d((1,), 0)), (-1, d((0,), 1)))}
        else:
            res = {(0,): comb((1, d((2,), 1)), (-1, d((1,), 2))),
                   (1,): comb((1, d((0,), 2)), (-1, d((2,), 0))),
                   (2,): comb((1, d((1,), 0)), (-1, d((0,), 1)))}
        return {c: e for c, e in res.items() if e is not None}


class _MetricReport:
    """NCCReport of a map's metric plus how its coefficients were obtained (exact or expanded)."""

    def __init__(self, rep: us.NCCReport, how: str) -> None:
        self.rep, self.how = rep, how

    def explain(self) -> str:
        return f"{self.rep.name}: {self.how}"


class _Constants:
    """The constant evaluators of :class:`Compiler` without a group (parameters only)."""

    _const_factor = Compiler._const_factor
    _const_vector = Compiler._const_vector
    _param = Compiler._param

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params, self.eigenvalue = params, None


def implicit_rotation_rate(problem: Any, params: Mapping[str, Any]) -> tuple[float, str] | None:
    """Largest |c W| over the terms c cross(W, u) (or c cross(u, W)) with a constant vector W on
    the left-hand sides of ``problem``, and the text of that term; None if there is none. |c W| bounds the frequency of the waves the implicit term carries (the inertial
    waves 2 Omega.k/|k| <= 2 |Omega| of a Coriolis term 2 Omega cross(ez, u)): with |c W| dt > 1
    the implicit stages damp and slow them (RK443: 1.7 % of amplitude per step at omega dt = 1,
    13 % at 2), which explain() and a warning of the solver state."""
    ks = _Constants(params)
    V = frozenset(problem.variables)
    best: list = []

    def walk(x: E.Expr, scale: complex, text: str | None = None) -> None:
        if isinstance(x, E.Add):
            for a in x.args:
                walk(a, scale)
        elif isinstance(x, E.Mul):
            dep = [a for a in x.args if a.depends_on(V)]
            if len(dep) != 1:
                return
            for a in x.args:
                if a is not dep[0]:
                    c = ks._const_factor(a)
                    if c is None:
                        return
                    scale = scale * c
            walk(dep[0], scale, text or str(x))
        elif isinstance(x, E.Cross):
            a, b = x.args
            if a.depends_on(V) == b.depends_on(V):
                return
            w = ks._const_vector(b if a.depends_on(V) else a)
            if w is not None:
                best.append((float(abs(scale) * np.linalg.norm(w)), text or str(x)))

    for eq in problem.equations:
        try:
            walk(eq.lhs, 1.0)
        except (ParameterError, DiscretizationError):  # a free symbol: no number to report
            continue
    return max(best) if best else None


def form_rank(a: E.Expr) -> int:
    """Tensor rank of the expression ``a``."""
    return a.rank


# Operators linear in their single argument, which commute with a scalar constant such as the
# eigenvalue: op(x0 + c x1) = op(x0) + c op(x1).
_LINEAR_UNARY = (E.Differentiate, E.Gradient, E.Divergence, E.Laplacian, E.Curl, E.Component,
                 E.Integrate, E.Interpolate, E.Lift)
_MULTILINEAR = (E.Mul, E.Dot, E.Outer) + ((E.Cross,) if hasattr(E, "Cross") else ())


def eigen_split(x: E.Expr, ev: E.Parameter) -> dict[int, E.Expr]:
    """Write ``x`` as sum_d ev**d * x_d (x_d free of ev) and require d in {0, 1}.

    Sums and products are expanded only where they contain ev (so (U - c) L(v) gives
    {0: U L(v), 1: -L(v)}), linear operators are applied to each part. Any dependence on ev
    that is not polynomial of degree <= 1 is refused with the offending term: ev**2 (quadratic
    pencil), 1/ev, f(ev) for a pointwise function f, and conj(ev) (not complex-linear).
    """
    parts = _esplit(x, ev)
    bad = sorted(d for d in parts if d not in (0, 1))
    if bad:
        raise DiscretizationError(
            f"'{x}': the eigenvalue {ev.name} must appear linearly (found degree {bad[-1]}); a "
            "polynomial eigenproblem needs a companion linearization (not implemented)")
    return parts


def _esplit(x: E.Expr, ev: E.Parameter) -> dict[int, E.Expr]:
    if x == ev:
        return {1: E.Constant(1)}
    if ev not in x.free_parameters():
        return {0: x}
    if isinstance(x, E.Add):
        out: dict[int, list] = {}
        for t in x.args:
            for d, p in _esplit(t, ev).items():
                out.setdefault(d, []).append(p)
        return {d: E.add(*ps) for d, ps in out.items()}
    if isinstance(x, _MULTILINEAR):
        acc: dict[int, list] = {0: []}             # degree -> list of argument tuples
        for a in x.args:
            pa = _esplit(a, ev)
            new: dict[int, list] = {}
            for d0, tuples in acc.items():
                for d1, p in pa.items():
                    new.setdefault(d0 + d1, []).extend(t + [p] for t in (tuples or [[]]))
            acc = new
        return {d: E.add(*(x.rebuild(t) for t in ts)) for d, ts in acc.items()}
    if isinstance(x, E.Power):
        e = x.exponent
        if float(e).is_integer() and e > 0:
            return _esplit(E.Mul([x.args[0]] * int(e)), ev) if int(e) > 1 else _esplit(x.args[0], ev)
        raise DiscretizationError(f"'{x}': the eigenvalue {ev.name} enters a power {e} (not a "
                                  "linear pencil)")
    if isinstance(x, _LINEAR_UNARY):
        return {d: x.rebuild([p]) for d, p in _esplit(x.args[0], ev).items()}
    raise DiscretizationError(
        f"'{x}': the eigenvalue {ev.name} enters a {type(x).__name__} node, which is not linear "
        "over C in it (conj, pointwise function, time derivative): not a linear pencil")
