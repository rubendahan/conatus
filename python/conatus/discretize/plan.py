"""Discretization plan: per-group sparse systems, structural rank report, LBVP solve.

``discretize(problem, ...)`` compiles the LHS of every equation, group by group
(one group = one Fourier wavenumber per periodic direction), into sparse
matrices in Olver–Townsend coefficient spaces:

    M dX/dt + L X = F        (IVP)      L X = F      (LBVP)      L X + sigma M X = 0   (EVP)

The matrices, the boundary rows and the tau columns are public
(``plan.matrices(group)``), so a propagator, a monodromy or an adjoint never
requires reverse engineering (DEDALUS.md §4). Before any factorization, the
rank report (spec §14) refuses non-square or structurally singular systems,
and small blocks get a numerical null-space check (missing gauge).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import structural_rank

from ..errors import RankError  # public class: conatus.errors
from ..ir import expr as E
from ..ir.bases import RealFourier
from ..ir.domain import Domain
from ..ir.problem import EVP, IVP, LBVP, Problem
from ..ir.symmetry import SymmetryError, component_parity
from ..reference import transforms as T
from ..reference.evaluate import FieldData, ReferenceEvaluator
from . import ultraspherical as us
from .compiler import Compiler, DiscretizationError, _comps
from .layout import GroupItem, Layout, ObjLayout
from ..tolerances import NCC_ROUNDING, NCC_TAIL_TOL  # declared defaults
from .param_keys import check_param_keys, problem_parameter_names
from .._repr import compact_repr

__all__ = ["discretize", "Plan", "GroupMatrices", "GroupRank", "RankError", "LBVPSolution"]

NUMERICAL_RANK_MAX = 2000   # dense SVD null-space check for blocks up to this size


@dataclass
class GroupMatrices:
    """Public sparse matrices of one group (full space, or the symmetry subspace if reduced)."""
    __repr__ = compact_repr   # one line, arrays by shape

    group: tuple[GroupItem, ...]
    label: str
    L: sp.csr_matrix
    M: sp.csr_matrix
    rows: list[tuple[int, tuple, str]]      # (equation index, component, kind)
    cols: list[tuple[str, tuple, str]]      # (variable name, component, kind)
    row_index: np.ndarray                   # for each row: position in the full group system
    col_index: np.ndarray
    lam: dict[tuple[int, tuple], int | None]
    reduced: bool = False

    @property
    def bc_rows(self) -> np.ndarray:
        """Positions of the boundary-condition rows."""
        return np.array([i for i, r in enumerate(self.rows) if r[2] == "boundary"], dtype=int)

    @property
    def tau_cols(self) -> np.ndarray:
        """Positions of the tau columns."""
        return np.array([i for i, c in enumerate(self.cols) if c[2] == "tau"], dtype=int)

    @property
    def shape(self) -> tuple[int, int]:
        return self.L.shape

    def explain(self) -> str:
        """Shapes, non-zeros, boundary rows, tau columns and unknowns by kind."""
        kinds: dict = {}
        for _, _, k in self.cols:
            kinds[k] = kinds.get(k, 0) + 1
        return (f"GroupMatrices [{self.label}]{' (symmetry subspace)' if self.reduced else ''}: "
                f"L, M of shape {self.shape}, nnz(L)={self.L.nnz}, nnz(M)={self.M.nnz}; "
                f"{len(self.bc_rows)} boundary rows, {len(self.tau_cols)} tau columns; "
                "unknowns by kind: " + ", ".join(f"{k}={n}" for k, n in sorted(kinds.items())))


@dataclass
class GroupRank:
    """Rank report of one group (spec §14): unknowns and equations by kind, structural rank and
    the dense numerical rank of small blocks; ``ok`` is the verdict, ``explain()`` the table."""

    label: str
    unknowns: dict[str, int]
    equations: dict[str, int]
    structural_rank: int
    numerical: str        # "full", "deficient (k)", "skipped (size n > max)"
    null_hint: str = ""

    @property
    def n_unknowns(self) -> int:
        return sum(self.unknowns.values())

    @property
    def n_equations(self) -> int:
        return sum(self.equations.values())

    @property
    def ok(self) -> bool:
        return (self.n_unknowns == self.n_equations == self.structural_rank
                and not self.numerical.startswith("deficient"))

    def explain(self) -> str:
        """The rank table and its PASS/FAIL verdict."""
        u, q = self.unknowns, self.equations
        lines = [
            f"System rank analysis [{self.label}]",
            "-" * 44,
            f"Unknown coefficient DOFs   {u.get('field', 0):8d}",
            f"Tau DOFs                   {u.get('tau', 0):8d}",
            f"Gauge/constant DOFs        {u.get('constant', 0):8d}",
            f"PDE equations              {q.get('bulk', 0):8d}",
            f"Boundary equations         {q.get('boundary', 0):8d}",
            f"Gauge/constraint equations {q.get('constraint', 0):8d}",
            f"Structural rank            {self.structural_rank:8d} / {self.n_unknowns}",
            f"Numerical rank             {self.numerical}",
        ]
        if self.null_hint:
            lines.append(f"  null space: {self.null_hint}")
        lines.append("PASS" if self.ok else "FAIL")
        return "\n".join(lines)

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import group_rank_html
        return group_rank_html(self)


def _rank_diagnosis(r: GroupRank, problem: Problem) -> tuple[str, str, str]:
    """Plain-words reading of a failed rank report: (why, fix, where). Counts only, no guess
    is applied; the IR count per non-periodic direction names missing / extra conditions."""
    u, q = r.unknowns, r.equations
    taus, bcs = u.get("tau", 0), q.get("boundary", 0)
    gauge_u, gauge_q = u.get("constant", 0), q.get("constraint", 0)
    where = f"group [{r.label}] of {problem.kind} '{problem.name}'"
    try:
        dirs = [d for d in problem.rank_report().directions if not d.consistent]
    except Exception as err:  # noqa: BLE001 -- the IR count is advisory; its failure is said
        dirs = []
        where += f"; IR count per direction unavailable ({type(err).__name__}: {err})"
    for d in dirs:
        where += (f"; along {d.coord.name}: total differential order {d.order_sum}, "
                  f"{d.boundary_components} boundary-condition and {d.tau_variable_components} "
                  "tau component(s)")
    for d in dirs:
        if d.boundary_components < d.order_sum:
            k = d.order_sum - d.boundary_components
            return (f"{k} boundary condition(s) missing along {d.coord.name}: a problem of total "
                    f"order {d.order_sum} in {d.coord.name} needs {d.order_sum} conditions (a "
                    "difference can be legitimate with a pressure gauge; here the discrete "
                    "system is singular)",
                    f"add the missing condition(s) with pb.add_bc(interp(f, {d.coord.name}, "
                    "'left'), interp(f, " + d.coord.name + ", 'right'), ...); for a fourth-order "
                    "equation give f and its derivative at each wall", where)
        if d.boundary_components > d.order_sum:
            return (f"{d.boundary_components - d.order_sum} boundary condition(s) too many along "
                    f"{d.coord.name} (total order {d.order_sum})",
                    "remove the extra condition(s); keep as many as the differential order", where)
    if r.n_equations > r.n_unknowns:
        if bcs > taus:
            return (f"{bcs} boundary equations but {taus} tau DOFs: every boundary condition needs "
                    "a tau term lifted into a bulk equation, otherwise the system has more "
                    f"equations ({r.n_equations}) than unknowns ({r.n_unknowns})",
                    "add the conditions with pb.add_bc(...) (it creates and lifts the taus), or "
                    "declare tau fields and add lift(tau, basis, -k) terms to the bulk equation",
                    where)
        return (f"{r.n_equations} equations for {r.n_unknowns} unknowns",
                "remove the extra equation(s) or declare the missing variable(s)", where)
    if r.n_equations < r.n_unknowns:
        if gauge_u > gauge_q:
            return (f"{gauge_u} constant (gauge) unknown(s) but {gauge_q} constraint equation(s)",
                    "add the gauge condition, e.g. integ(p) = 0 for the pressure level", where)
        if taus > bcs:
            return (f"{taus} tau DOFs for {bcs} boundary equations: {taus - bcs} condition(s) "
                    "missing", "add the boundary conditions that close the tau terms", where)
        return (f"{r.n_equations} equations for {r.n_unknowns} unknowns",
                "add the missing equation(s) or remove unused variable(s)", where)
    if r.numerical.startswith("deficient"):
        return ("the square system is numerically singular" + (f" ({r.null_hint})" if r.null_hint
                                                              else ""),
                "pure Neumann conditions or a pressure fix a level only up to a constant: add a "
                "gauge (integ(p) = 0 with a constant tau); if a parameter is exactly on an "
                "eigenvalue of the operator, move it", where)
    return (f"the system is square but structurally singular (rank {r.structural_rank} of "
            f"{r.n_unknowns}): some unknown is not reached by any equation, or two conditions "
            "fix the same thing (e.g. only Neumann conditions, or both conditions at one wall)",
            "check pb.explain() (rank report per direction); for pure Neumann add a gauge "
            "(integ(u) = 0 with a constant tau)", where)


@dataclass
class LBVPSolution:
    """Solution of an LBVP (``plan.solve()``): coefficients by variable and the relative residual
    of every group solve (always reported)."""
    __repr__ = compact_repr   # one line, arrays by shape

    coeffs: dict[str, np.ndarray]           # variable name -> coefficient array
    residuals: list[tuple[str, float]]      # (group label, relative residual)
    plan: Plan
    reduced: bool

    @property
    def max_residual(self) -> float:
        """Largest relative residual over the groups."""
        return max((r for _, r in self.residuals), default=0.0)

    def grid(self, name: str) -> np.ndarray:
        """Values of a variable on its (non-dealiased) collocation grid."""
        return self.plan.to_grid(name, self.coeffs[name])

    def coordinate(self, name: str) -> np.ndarray:
        """Points of coordinate ``name`` of the grid of ``grid()`` (same as Simulation.coordinate)."""
        c = next((c for c in self.plan.domain.coordsys.coords if c.name == name), None)
        if c is None or self.plan.domain.basis(c) is None:
            raise KeyError(f"no basis along {name!r} in {self.plan.domain!r}")
        return T.grid(self.plan.domain.basis(c), dealias=False)

    def explain(self) -> str:
        """Space solved in, number of groups and the worst residual."""
        from ..tolerances import ncc_floor_text
        worst = max(self.residuals, key=lambda r: r[1]) if self.residuals else ("-", 0.0)
        text = (f"LBVP solution ({'symmetry subspace' if self.reduced else 'full space'}): "
                f"{len(self.residuals)} group(s), max relative residual {worst[1]:.2e} "
                f"[{worst[0]}] (sparse LU, scipy SuperLU)")
        floor = ncc_floor_text(self.plan, worst[1])        # digits claimed honestly
        return text + (f"\n  {floor}" if floor else "")


def _ncomp(tensorsig) -> int:
    n = 1
    for cs in tensorsig:
        n *= cs.dim
    return n


class Plan:
    """Discretized problem: layout, groups, matrices on demand, rank checks, LBVP solve."""

    ASSEMBLY = ("direct", "wavenumber_interpolation")

    def __init__(self, problem: Problem, ncc: Mapping[E.Field, Any] | None = None,
                 params: Mapping[str, Any] | None = None, ncc_cutoff: float | None = None,
                 assembly: str = "direct", ncc_tail_tol: float = NCC_TAIL_TOL) -> None:
        if assembly not in self.ASSEMBLY:
            from ..errors import OptionError
            raise OptionError(f"discretize(assembly={assembly!r}): assembly must be one of "
                              f"{self.ASSEMBLY}",
                              fix="assembly='direct' compiles every group")
        from .terrain_plan import has_terrain
        if has_terrain(problem):
            # the flat discretizer would read xi as z (a flat channel): never silently
            raise DiscretizationError("Plan(...) of a problem on a terrain-following mapped "
                                      "domain: use discretize(problem) (TerrainPlan, metric "
                                      "terms included)")
        self.problem = problem
        # "direct": every group compiled from the IR. "wavenumber_interpolation": the group
        # matrices are polynomials in the Fourier wavenumbers (degree = differential order along
        # each periodic direction), so each class of groups is compiled at (d+1)^m nodes and
        # interpolated, verified at an independent group.
        self.assembly = assembly
        self._interp: dict = {}
        self.params = dict(params or {})
        self.ncc_cutoff = ncc_cutoff
        # any NCC whose sampled Chebyshev tail exceeds ncc_tail_tol (or the
        # declared ncc_cutoff if larger) is refused (DiscretizationError when the group is assembled).
        self.ncc_tail_tol = us.check_tail_tol(ncc_tail_tol)
        dom = None
        for v in problem.variables:
            dom = Domain.union(dom, v.domain)
        for eq in problem.equations:
            dom = Domain.union(dom, eq.domain)
        self.domain = dom
        self.layout = Layout(dom)
        zmap = getattr(self.layout.zbasis, "map", None)
        if zmap is not None and (isinstance(problem, IVP) or problem.symmetry is not None):
            # the time steppers' transforms and the parity masks assume an affine interval
            raise DiscretizationError(
                f"mapped basis {self.layout.zbasis!r}: supported for EVP, LBVP and NLBVP only "
                "(IVP and symmetry subspaces on mapped intervals are not implemented)")
        zfam = self.layout.zfam
        if zfam is not None and problem.symmetry is not None and any(
                g.coord == self.layout.zcoord for g in problem.symmetry.generators):
            raise DiscretizationError(
                f"symmetry subspace along {self.layout.zcoord.name} on {self.layout.zbasis!r}: "
                "parity masks are implemented for Fourier and Chebyshev T axes only",
                fix="declare the reflection on a Fourier axis, or drop the subspace")
        self.variables = problem.variables
        self.eigenvalue = problem.eigenvalue if isinstance(problem, EVP) else None
        # every supplied coefficient must appear in the problem: a stray key is almost always a
        # typo or a stale field, and ignoring it would silently change the physics
        used = frozenset(f for eq in problem.equations for side in (eq.lhs, eq.rhs)
                         for f in side.free_fields())
        stray = [f.name for f in (ncc or {}) if f not in used]
        if stray:
            raise DiscretizationError(
                f"ncc given for field(s) not used by the problem: {', '.join(sorted(stray))}")
        # same rule for parameter values: a key that names no Parameter of the
        # problem used to be ignored, and the value declared on the Parameter used instead
        check_param_keys(self.params, problem_parameter_names(problem), where="discretize")
        self._ncc_data = {f: self._as_grid(f, d) for f, d in (ncc or {}).items()}
        # coefficient data of a field is an exact polynomial (not a grid sample): no tail to judge
        # when it multiplies an unknown as is (e.g. the Newton iterate of an NLBVP)
        self._ncc_exact = frozenset(f for f, d in (ncc or {}).items()
                                    if isinstance(d, FieldData) and d.layout == "coeff")
        self._ncc_floor: dict = {}       # (expr, comp) -> absolute rounding level of its coefficients
        self.ncc_reports: dict = {}
        self._cache: dict = {}
        self.taus = frozenset(
            f for eq in problem.equations for n in eq.lhs.walk() if isinstance(n, E.Lift)
            for f in n.free_fields() & frozenset(problem.variables))
        zc = self.layout.zcoord
        self.eq_kind = []
        for eq in problem.equations:
            if self.layout.zdep_of(eq.domain):
                self.eq_kind.append("bulk")
            elif zc is not None and any(isinstance(n, E.Interpolate) and n.coord == zc
                                        for n in eq.lhs.walk()):
                self.eq_kind.append("boundary")
            elif self.layout.zbasis is None and any(self.layout.fdep_of(eq.domain)):
                self.eq_kind.append("bulk")
            else:
                self.eq_kind.append("constraint")

    # ------------------------------------------------------------------ data helpers
    def _grid_shape(self, dom: Domain | None, tensorsig: tuple) -> tuple:
        comps = tuple(cs.dim for cs in tensorsig)
        return comps + tuple(dom.basis(c).size if (dom is not None and dom.has(c)) else 1
                             for c in self.domain.coordsys.coords)

    def _as_grid(self, f: E.Field, d: Any) -> np.ndarray:
        """Data -> grid values on the (non-dealiased) N-point grid of the field's domain."""
        coords = self.domain.coordsys.coords
        shape = self._grid_shape(f.domain, f.tensorsig)
        if callable(d) and not isinstance(d, (np.ndarray, FieldData)):
            grids = []
            for i, c in enumerate(coords):
                b = f.domain.basis(c) if f.domain is not None else None
                g = T.grid(b, dealias=False) if b is not None else np.zeros(1)
                sh = [1] * len(coords)
                sh[i] = g.size
                grids.append(g.reshape(sh))
            raw = d(*grids)
            if f.rank == 1 and isinstance(raw, (tuple, list)):
                # components of different broadcast shapes, e.g. (y, 0*x): each broadcast to the
                # grid first (np.asarray of the tuple raised a bare ValueError)
                raw = [np.broadcast_to(np.asarray(r), shape[1:]) for r in raw]
            val = np.asarray(raw)
            return np.broadcast_to(val, shape).copy()
        if isinstance(d, FieldData):
            data = np.asarray(d.data)
            if d.layout == "grid":
                return np.broadcast_to(data, shape).copy()
            out = np.broadcast_to(data, shape).copy()
            for i, c in enumerate(coords):
                b = f.domain.basis(c)
                if b is not None:
                    out = T.backward(out, b, axis=f.rank + i, dealias=False)
            return out
        raise TypeError(f"data for {f.name}: pass a FieldData (explicit layout) or a callable of "
                        "the coordinate grids; bare arrays are ambiguous")

    def _eval_grid(self, e: E.Expr, dom: Domain | None, data: Mapping[E.Field, np.ndarray]) -> np.ndarray:
        shape = self._grid_shape(dom, e.tensorsig)
        if not e.free_fields() and e.domain is None:
            v = ReferenceEvaluator(self.domain).evaluate(e, {}, self.params)
            return np.broadcast_to(v, shape).copy()
        ev = ReferenceEvaluator(Domain.union(dom, e.domain) or self.domain)
        v = ev.evaluate(e, data, self.params)
        return np.broadcast_to(v, shape).copy()

    def _to_coeffs(self, g: np.ndarray, dom: Domain | None, rank: int) -> np.ndarray:
        for i, c in enumerate(self.domain.coordsys.coords):
            b = dom.basis(c) if dom is not None else None
            if b is not None:
                g = T.forward(g, b, axis=rank + i, dealias=False)
        return g

    def to_grid(self, name: str, coeffs: np.ndarray) -> np.ndarray:
        """Grid values (non-dealiased grid) of variable ``name`` from its coefficient array."""
        v = next(v for v in self.variables if v.name == name)
        g = coeffs
        for i, c in enumerate(self.domain.coordsys.coords):
            b = v.domain.basis(c)
            if b is not None:
                g = T.backward(g, b, axis=v.rank + i, dealias=False)
        return g

    def _rounding_floor(self, f: E.Expr) -> float:
        """Absolute rounding level of the Chebyshev coefficients of the NCC ``f``: NCC_ROUNDING
        x eps x the largest value of its data, amplified by ~(N^2 2/L) per derivative along the
        bounded axis. A tail below it is rounding (e.g. U'' of a linear U), not under-resolution."""
        S = max((float(np.max(np.abs(self._ncc_data[x]))) for x in f.free_fields()
                 if x in self._ncc_data and np.size(self._ncc_data[x])), default=0.0)
        zc, zb = self.layout.zcoord, self.layout.zbasis
        amp = 1.0
        if zc is not None and zb is not None:
            k = sum(1 for n in f.walk() if isinstance(n, E.Differentiate) and n.coord == zc)
            # the amplified floor is rounding only if the data the derivatives
            # act on are themselves exact to rounding. Data resolved to, say, 1e-10 give a U''
            # whose error is that interpolation error x N^4 (not rounding): it is then judged by
            # its relative tail like any coefficient, never exempted as "rounding".
            if k and any(self._data_tail(x) > NCC_ROUNDING * np.finfo(float).eps
                         for x in f.free_fields() if x in self._ncc_data):
                return 0.0
            per = self.layout.nz ** 2 * (2.0 / zb.length if np.isfinite(zb.length) else 1.0)
            amp = max(per, 1.0) ** k
        return NCC_ROUNDING * np.finfo(float).eps * S * amp

    def _data_tail(self, x: E.Field) -> float:
        """Relative Chebyshev tail (last 3 / max) of the data of the NCC field ``x`` along the
        bounded axis; 0 if the data do not depend on it (rounding floor, see above)."""
        zc = self.layout.zcoord
        if x.domain is None or not x.domain.has(zc):
            return 0.0
        g = np.asarray(self._ncc_data[x])
        ax = x.rank + self.domain.coordsys.coords.index(zc)
        if self.layout.zfam is not None:
            c = self.layout.zfam.ncc_forward(g, ax)
        else:
            c = T.forward(g, x.domain.basis(zc), axis=ax, dealias=False)
        c = np.abs(np.moveaxis(c, ax, 0).reshape(c.shape[ax], -1))
        amax = float(c.max()) if c.size else 0.0
        return float(c[-3:].max()) / amax if amax > 0 else 0.0

    def _ncc_coeffs(self, f: E.Expr, comp: tuple) -> np.ndarray:
        key = (f, comp)
        if key in self._cache:
            return self._cache[key]
        missing = [x for x in f.free_fields() if x not in self._ncc_data]
        if missing:
            raise DiscretizationError(
                f"non-constant coefficient '{f}': no data for {', '.join(m.name for m in missing)} "
                "(pass ncc={field: FieldData(...)})")
        dom = f.domain
        g = self._eval_grid(f, dom, self._ncc_data)
        g = g[comp] if comp else g
        self._ncc_floor[key] = self._rounding_floor(f)
        if not np.all(np.isfinite(g)):
            # e.g. 1/cosh(y)**2 overflowing on a mapped grid (y up to ~1e4): NaN would reach the
            # matrices and only surface (if at all) as a failed factorization downstream
            raise DiscretizationError(
                f"non-constant coefficient '{f}' has non-finite values on the grid "
                f"({int(np.sum(~np.isfinite(g)))} points): rewrite its data (e.g. 1 - tanh(y)**2 "
                "instead of 1/cosh(y)**2)")
        zc = self.layout.zcoord
        axis = self.domain.coordsys.coords.index(zc) if zc is not None else None
        if zc is not None and dom is not None and dom.has(zc):
            if self.layout.zfam is not None:      # polynomial (Hermite/Laguerre) or per-piece data
                c = self.layout.zfam.ncc_forward(g, axis)
            else:
                c = T.forward(g, dom.basis(zc), axis=axis, dealias=False)
            c = np.moveaxis(c, axis, 0).reshape(c.shape[axis], -1)
            # constant along the Fourier directions to the rounding of the transform (an explicit
            # bound: np.allclose's default rtol = 1e-5 took a 1e-6 variation for a constant)
            if c.shape[1] != 1 and not np.allclose(c, c[:, :1], rtol=0.0,
                                                   atol=1e-12 * max(np.abs(c).max(), 1e-300)):
                raise DiscretizationError(f"NCC '{f}' varies along a Fourier direction")
            out = c[:, 0]
        else:
            out = np.asarray(g).reshape(-1)[:1]
        self._cache[key] = out
        return out

    # ------------------------------------------------------------------ layout per group
    def var_layouts(self, group) -> list[ObjLayout]:
        """Where each variable sits in the group vector (columns)."""
        out, off = [], 0
        for v in self.variables:
            fdep = self.layout.fdep_of(v.domain)
            z = self.layout.zdep_of(v.domain)
            ol = ObjLayout(v.name, _ncomp(v.tensorsig), fdep, z, off, self.layout.nf(fdep, group),
                           self.layout.nz if z else 1)
            out.append(ol)
            off += ol.size
        return out

    def eq_layouts(self, group) -> list[ObjLayout]:
        """Where each equation sits in the group vector (rows)."""
        out, off = [], 0
        for i, eq in enumerate(self.problem.equations):
            fdep = self.layout.fdep_of(eq.domain)
            z = self.layout.zdep_of(eq.domain)
            ol = ObjLayout(f"eq{i}", eq.n_components, fdep, z, off, self.layout.nf(fdep, group),
                           self.layout.nz if z else 1)
            out.append(ol)
            off += ol.size
        return out

    @property
    def groups(self) -> tuple:
        """The mode groups (one Fourier wavenumber per periodic direction)."""
        return self.layout.groups

    # ------------------------------------------------------------------ assembly
    def _assemble(self, group) -> GroupMatrices:
        key = ("full", group)
        if key in self._cache:
            return self._cache[key]
        if self.assembly == "wavenumber_interpolation":
            gm = self._assemble_interpolated(group)
            if gm is not None:
                self._cache[key] = gm
                return gm
        return self._assemble_direct(group, key)

    # -- wavenumber interpolation -------------------------------------------------------
    def _fourier_degrees(self) -> tuple[int, ...]:
        """Polynomial degree of the group matrices in each periodic wavenumber: the largest
        differential order of the equations (LHS, unknown-dependent terms) along that direction."""
        fs = self.problem.variables
        return tuple(max(eq.lhs.order_in(d.coord, fs) for eq in self.problem.equations)
                     for d in self.layout.fdirs)

    def _interp_class(self, cls: tuple[bool, ...]):
        """Nodes and node matrices of one class of groups (which directions have k = 0)."""
        if cls in self._interp:
            return self._interp[cls]
        degs = self._fourier_degrees()
        nodes = []
        for d, zero, deg in zip(self.layout.fdirs, cls, degs, strict=True):
            if zero:
                nodes.append(None)
                continue
            ks = sorted({it.k for it in d.items if it.k != 0.0 and (it.k > 0 or d.complex)})
            if len(ks) < deg + 2:     # need the nodes plus one independent check point
                self._interp[cls] = None
                return None
            # Chebyshev (second-kind) nodes on [k_min, k_max]: the node matrices are compiled at
            # synthetic wavenumbers, so the nodes need not be groups; equispaced nodes (first
            # version) lost ~1e5 x eps at degree 6 over k in [0.5, 32]
            a, b = ks[0], ks[-1]
            t = np.cos(np.pi * np.arange(deg + 1) / deg) if deg else np.array([1.0])
            nodes.append(sorted(float(v) for v in 0.5 * (a + b) + 0.5 * (b - a) * t))
        self._interp[cls] = dict(nodes=nodes, mats={}, checked=False)
        return self._interp[cls]

    def _synthetic(self, group, ks) -> tuple:
        """Group with the same index structure as ``group`` and wavenumbers ``ks``."""
        return tuple(it if k is None else GroupItem(it.kind, it.index, float(k), it.indices)
                     for it, k in zip(group, ks, strict=True))

    def _assemble_interpolated(self, group) -> GroupMatrices | None:
        cls = tuple(it.zero for it in group)
        if all(cls):
            return None
        info = self._interp_class(cls)
        if info is None:
            return None
        from itertools import product as _prod
        axes = [range(len(n)) if n is not None else [None] for n in info["nodes"]]
        combos = list(_prod(*axes))
        for c in combos:          # node matrices, compiled once per class
            if c not in info["mats"]:
                ks = [None if i is None else info["nodes"][a][i] for a, i in enumerate(c)]
                g = self._synthetic(group, ks)
                info["mats"][c] = self._assemble_direct(g, ("node", cls, c))
        ref = info["mats"][combos[0]]
        # Lagrange weights of the target wavenumbers, per periodic direction
        w = {}
        for a, (n, it) in enumerate(zip(info["nodes"], group, strict=True)):
            if n is None:
                continue
            for i, ki in enumerate(n):
                others = [kj for j, kj in enumerate(n) if j != i]
                w[(a, i)] = float(np.prod([(it.k - kj) / (ki - kj) for kj in others]))
        L = M = None
        for c in combos:
            wc = float(np.prod([w[(a, i)] for a, i in enumerate(c) if i is not None]))
            if wc == 0.0:
                continue
            gm = info["mats"][c]
            L = gm.L * wc if L is None else L + gm.L * wc
            M = gm.M * wc if M is None else M + gm.M * wc
        L, M = L.tocsr(), M.tocsr()
        L.eliminate_zeros()
        M.eliminate_zeros()
        gm = GroupMatrices(group, self.layout.group_label(group) + " [interpolated]", L, M,
                           ref.rows, ref.cols, ref.row_index, ref.col_index, ref.lam)
        if not info["checked"]:     # once per class: exact reconstruction at a real group
            is_node = all(n is None or it.k in n for n, it in zip(info["nodes"], group, strict=True))
            if not is_node:
                direct = self._assemble_direct(group, ("check", group))
                for A, B, name in ((gm.L, direct.L, "L"), (gm.M, direct.M, "M")):
                    scale = max(abs(B).max(), 1e-300)
                    err = abs(A - B).max() / scale if (A - B).nnz else 0.0
                    if err > 1e-11:
                        raise DiscretizationError(
                            f"wavenumber interpolation of {name} failed at {direct.label}: "
                            f"relative error {err:.1e} > 1e-11 (matrices not polynomial of the "
                            "declared degree?): use assembly='direct'")
                info["checked"] = True
                info["check_group"] = direct.label
        return gm

    def _assemble_direct(self, group, key) -> GroupMatrices:
        vls, els = self.var_layouts(group), self.eq_layouts(group)
        nrow = sum(e.size for e in els)
        ncol = sum(v.size for v in vls)
        vpos = {v: (vl, _comps(v.tensorsig)) for v, vl in zip(self.variables, vls, strict=True)}
        comp_index = {v: {c: k for k, c in enumerate(cl)} for v, (vl, cl) in vpos.items()}
        pieces: dict[str, list] = {"L": [], "M": []}   # (row0, col0, block); dtype chosen after
        comp = Compiler(self.layout, group, self.variables, self._ncc_coeffs, self.ncc_reports,
                        self.params, self.eigenvalue, self.ncc_cutoff, self.ncc_tail_tol,
                        ncc_exact=self._ncc_exact, ncc_floor=self._ncc_floor)
        lams: dict = {}
        rows: list = []
        for i, (eq, el) in enumerate(zip(self.problem.equations, els, strict=True)):
            fL = comp.compile(eq.lhs, "L")
            fM = comp.compile(eq.lhs, "M") if isinstance(self.problem, (IVP, EVP)) else {}
            if (isinstance(self.problem, IVP)
                    and any(isinstance(n, E.TimeDerivative) for n in eq.lhs.walk())
                    and not any(e.blocks for e in fM.values())):
                raise DiscretizationError(
                    f"equation [{i}] contains a time derivative but its mass operator M is "
                    "empty: refusing (would silently drop dt)")
            for ci, c in enumerate(_comps(eq.lhs.tensorsig)):
                ents = [f[c] for f in (fL, fM) if c in f]
                lam = None
                if el.zdep:
                    lam = max([e.lam for e in ents if e.lam is not None], default=0)
                lams[(i, c)] = lam
                r0 = el.offset + ci * el.block
                rows += [(i, c, self.eq_kind[i])] * el.block
                for f, target in ((fL, "L"), (fM, "M")):
                    if c not in f:
                        continue
                    e = comp.broadcast(f[c], el.fdep, el.zdep)
                    if el.zdep:
                        e = comp.convert(e, lam)
                    for (v, vc), blk in e.blocks.items():
                        vl, _ = vpos[v]
                        c0 = vl.offset + comp_index[v][vc] * vl.block
                        if blk.shape != (el.block, vl.block):
                            raise DiscretizationError(
                                f"internal shape mismatch in equation [{i}] for {v.name}: "
                                f"{blk.shape} vs {(el.block, vl.block)}")
                        if blk.nnz:
                            pieces[target].append((r0, c0, blk.tocoo()))
        # complex as soon as the layout or any coefficient is complex: never drop an imaginary part
        cplx = self.layout.complex or any(np.iscomplexobj(b.data)
                                          for ps in pieces.values() for _, _, b in ps)
        dtype = complex if cplx else float

        def build(ps):
            if not ps:
                return sp.csr_matrix((nrow, ncol), dtype=dtype)
            r = np.concatenate([b.row + r0 for r0, _, b in ps])
            c = np.concatenate([b.col + c0 for _, c0, b in ps])
            d = np.concatenate([b.data.astype(dtype) for _, _, b in ps])
            A = sp.coo_matrix((d, (r, c)), shape=(nrow, ncol)).tocsr()
            A.sum_duplicates()
            return A
        Lb, Mb = build(pieces["L"]), build(pieces["M"])
        cols = []
        for v, vl in zip(self.variables, vls, strict=True):
            kind = "tau" if v in self.taus else ("field" if (vl.zdep or any(vl.fdep)) else "constant")
            for c in _comps(v.tensorsig):
                cols += [(v.name, c, kind)] * vl.block
        gm = GroupMatrices(group, self.layout.group_label(group), Lb, Mb, rows,
                           cols, np.arange(nrow), np.arange(ncol), lams)
        if key[0] == "full":
            self._cache[key] = gm
        return gm

    # ------------------------------------------------------------------ symmetry subspace
    def _keep_masks(self, group) -> tuple[np.ndarray, np.ndarray]:
        pb = self.problem
        sym = pb.symmetry
        vls, els = self.var_layouts(group), self.eq_layouts(group)
        col_keep = np.ones(sum(v.size for v in vls), bool)
        row_keep = np.ones(sum(e.size for e in els), bool)
        dropped_eqs = {j for (_, j, _) in pb.redundant_bcs}
        for R in sym.generators:
            kind, info = self._reflection_kind(R)
            for v, vl in zip(self.variables, vls, strict=True):
                p = sym.parity(v, R)
                for ci, c in enumerate(_comps(v.tensorsig)):
                    pc = component_parity(p, v.tensorsig, c, R)
                    sl = slice(vl.offset + ci * vl.block, vl.offset + (ci + 1) * vl.block)
                    col_keep[sl] &= self._parity_ok(pc, vl, group, kind, info)
            for i, (eq, el) in enumerate(zip(pb.equations, els, strict=True)):
                p = pb.equation_parities[i][R]
                # a trace at a non-fixed point along R (e.g. u(x=-1)) is not an R-symmetric
                # object: its image is the mirror BC. Keep one of each mirror pair.
                paired = any(isinstance(n, E.Interpolate) and n.coord == R.coord
                             and not R.is_fixed(n.position, n.args[0].domain.basis(n.coord))
                             for n in eq.lhs.walk())
                for ci, c in enumerate(_comps(eq.lhs.tensorsig)):
                    sl = slice(el.offset + ci * el.block, el.offset + (ci + 1) * el.block)
                    if i in dropped_eqs or (p is None and not paired):
                        row_keep[sl] = False
                        continue
                    if paired:
                        continue
                    pc = component_parity(p, eq.lhs.tensorsig, c, R)
                    row_keep[sl] &= self._parity_ok(pc, el, group, kind, info)
        return row_keep, col_keep

    def _reflection_kind(self, R) -> tuple[str, Any]:
        L = self.layout
        if L.zcoord == R.coord:
            R.centre(L.zbasis)
            return "z", None
        for i, d in enumerate(L.fdirs):
            if d.coord == R.coord:
                if not isinstance(d.basis, RealFourier):
                    raise NotImplementedError(
                        "reflection along a ComplexFourier direction: coefficients k and -k are "
                        "linked, use RealFourier (cos/sin) for symmetry subspaces")
                a, Lx = d.basis.bounds[0], d.basis.length
                x0 = R.centre(d.basis)
                s = (x0 - a) % Lx
                if min(abs(s), abs(s - Lx / 2), abs(s - Lx)) > 1e-12 * Lx:
                    raise NotImplementedError(
                        f"{R!r}: reflection centre must be x = a or a + L/2 on the periodic "
                        "interval (other centres need a phase shift, not implemented)")
                return "fourier", i
        return "none", None

    def _parity_ok(self, p, ol: ObjLayout, group, kind: str, info) -> np.ndarray:
        n = ol.block
        if kind == "z":
            if not ol.zdep:
                return np.full(n, p == 1)
            z = np.arange(ol.nz)
            par = np.where(z % 2 == 0, 1, -1)
            return np.tile(par == p, ol.nf)
        if kind == "fourier":
            i = info
            if not ol.fdep[i]:
                return np.full(n, p == 1)
            fidx = self.layout.fourier_indices(ol.fdep, group)
            ok = np.array([(1 if fi[i] % 2 == 0 else -1) == p for fi in fidx], bool)
            return np.repeat(ok, ol.nz)
        return np.full(n, p == 1)   # constant along the mirror direction

    def matrices(self, group=None, reduced: bool | None = None) -> GroupMatrices:
        """Public M, L, boundary rows, tau columns of one group (default: the first)."""
        group = self.groups[0] if group is None else group
        full = self._assemble(group)
        if reduced is None:
            reduced = self.problem.symmetry is not None
        if not reduced:
            return full
        if self.problem.symmetry is None:
            raise SymmetryError("no symmetry subspace declared on the problem")
        rk, ck = self._keep_masks(group)
        ri, ci = np.nonzero(rk)[0], np.nonzero(ck)[0]
        return GroupMatrices(group, full.label + " [symmetry subspace]",
                             full.L[ri][:, ci].tocsr(), full.M[ri][:, ci].tocsr(),
                             [full.rows[i] for i in ri], [full.cols[j] for j in ci], ri, ci,
                             full.lam, reduced=True)

    def embed(self, x_red: np.ndarray, group=None) -> np.ndarray:
        """E: symmetry-subspace group vector -> full group vector (zeros elsewhere)."""
        group = self.groups[0] if group is None else group
        gm = self.matrices(group, reduced=True)
        out = np.zeros(self._assemble(group).L.shape[1], dtype=np.result_type(x_red, float))
        out[gm.col_index] = x_red
        return out

    def project(self, x_full: np.ndarray, group=None) -> np.ndarray:
        """P: full group vector -> subspace coefficients. P E = I; E P = I on the subspace."""
        group = self.groups[0] if group is None else group
        return np.asarray(x_full)[self.matrices(group, reduced=True).col_index]

    # ------------------------------------------------------------------ rank (spec §14)
    def rank_report(self, group=None, reduced: bool | None = None,
                    operator: str = "L") -> GroupRank:
        """Rank report of one group (default: the first) for ``operator`` "L" or "M"."""
        gm = self.matrices(group, reduced)
        A = gm.L if operator == "L" else gm.M
        unk: dict = {}
        for _, _, k in gm.cols:
            unk[k] = unk.get(k, 0) + 1
        eqs: dict = {}
        for _, _, k in gm.rows:
            eqs[k] = eqs.get(k, 0) + 1
        srank = int(structural_rank(sp.csr_matrix(abs(A) > 0, dtype=float))) if A.shape[0] else 0
        numerical, hint = "not checked (not square)", ""
        if A.shape[0] == 0 and A.shape[1] == 0:
            numerical = "empty block (nothing in this group)"
        elif A.shape[0] == A.shape[1] and srank == A.shape[0]:
            if A.shape[0] <= NUMERICAL_RANK_MAX:
                Ad = A.toarray()
                s = np.linalg.svd(Ad, compute_uv=False)
                tol = s[0] * max(A.shape) * 1e-13 if s.size else 0
                k = int(np.sum(s <= tol))
                if k:
                    numerical = f"deficient ({k})"
                    _, _, vh = np.linalg.svd(Ad)
                    null = vh[-1]
                    names = sorted({gm.cols[j][0] for j in np.nonzero(abs(null) > 1e-6 * abs(null).max())[0]})
                    hint = (f"dimension {k}, involves {', '.join(names)}: missing gauge condition "
                            "(pure Neumann, pressure level) or a parameter sitting on an eigenvalue")
                else:
                    numerical = f"full (sigma_min/sigma_max = {s[-1] / s[0]:.1e})"
            else:
                numerical = f"skipped (size {A.shape[0]} > {NUMERICAL_RANK_MAX}; structural only)"
        return GroupRank(gm.label, unk, eqs, srank, numerical, hint)

    def check(self, reduced: bool | None = None) -> list[GroupRank]:
        """Rank report of every group; raises RankError on the first failure."""
        reps = []
        for g in self.groups:
            r = self.rank_report(g, reduced)
            if not r.ok:
                why, fix, where = _rank_diagnosis(r, self.problem)
                raise RankError("system refused before factorization:\n" + r.explain(),
                                where=where, why=why, fix=fix)
            reps.append(r)
        return reps

    # ------------------------------------------------------------------ LBVP solve
    def _rhs_coeffs(self, data: Mapping[E.Field, np.ndarray]) -> list[np.ndarray]:
        out = []
        for eq in self.problem.equations:
            dom = eq.domain
            g = self._eval_grid(eq.rhs, dom, data)
            out.append(self._to_coeffs(g, dom, eq.rhs.rank))
        return out

    def _group_rhs(self, group, rhs: list[np.ndarray], gm_full: GroupMatrices) -> np.ndarray:
        els = self.eq_layouts(group)
        dt = complex if (self.layout.complex or any(np.iscomplexobj(r) for r in rhs)) else float
        b = np.zeros(sum(e.size for e in els), dtype=dt)
        coords = self.domain.coordsys.coords
        zc = self.layout.zcoord
        fax = [coords.index(d.coord) for d in self.layout.fdirs]
        for i, (eq, el) in enumerate(zip(self.problem.equations, els, strict=True)):
            r = rhs[i]
            fidx = self.layout.fourier_indices(el.fdep, group)
            for ci, c in enumerate(_comps(eq.lhs.tensorsig)):
                rc = r[c] if c else r
                r0 = el.offset + ci * el.block
                for fj, fi in enumerate(fidx):
                    idx = [0] * len(coords)
                    for a, j in zip(fax, fi, strict=True):
                        idx[a] = j
                    if el.zdep:
                        idx[coords.index(zc)] = slice(None)
                        zv = np.asarray(rc[tuple(idx)])
                        zv = self.layout.zconv(gm_full.lam[(i, c)]) @ zv
                        b[r0 + fj * el.nz: r0 + (fj + 1) * el.nz] = zv
                    else:
                        b[r0 + fj] = rc[tuple(idx)]
        return b

    def solve(self, data: Mapping[E.Field, Any] | None = None,
              reduced: bool | None = None) -> LBVPSolution:
        """Solve an LBVP group by group (sparse LU, after the rank check): ``data`` gives the
        right-hand-side fields (those given at discretization are used too); the residual of every
        group is returned with the solution."""
        if not isinstance(self.problem, LBVP):
            raise DiscretizationError("solve() is for LBVPs; use IVPSolver(plan, ...) for an IVP and "
                                      "conatus.solvers.eigenvalues(plan, ...) for an EVP")
        if reduced is None:
            reduced = self.problem.symmetry is not None
        gdata = {f: self._as_grid(f, d) for f, d in (data or {}).items()}
        # data supplied at discretization (ncc=...) also feeds the right-hand side: a native LBVP
        # with a given source (m.given(f=...); lap(u) = f) failed with "no data bound for field f"
        # although f had data; explicit data passed here take precedence
        for f, g in self._ncc_data.items():
            gdata.setdefault(f, g)
        self.check(reduced)                       # refuse before any factorization
        rhs = self._rhs_coeffs(gdata)
        vls_all = {g: self.var_layouts(g) for g in self.groups}
        coords = self.domain.coordsys.coords
        zc = self.layout.zcoord
        fax = [coords.index(d.coord) for d in self.layout.fdirs]
        cplx = self.layout.complex or any(np.iscomplexobj(r) for r in rhs)
        out = {v.name: np.zeros(self._grid_shape(v.domain, v.tensorsig),
                                dtype=complex if cplx else float) for v in self.variables}
        residuals = []
        for g in self.groups:
            full = self._assemble(g)
            gm = self.matrices(g, reduced)
            b_full = self._group_rhs(g, rhs, full)
            if reduced:
                rk, ck = self._keep_masks(g)
                dropped = b_full[~rk]
                if dropped.size and np.max(np.abs(dropped)) > 1e-12 * max(1.0, np.max(np.abs(b_full))):
                    raise SymmetryError(
                        f"[{gm.label}] the data has components outside the symmetry subspace "
                        f"(max {np.max(np.abs(dropped)):.2e}): forcing/BC data break the symmetry")
                b = b_full[rk]
            else:
                b = b_full
            if gm.L.shape[0] == 0:
                continue
            A = gm.L.tocsc()
            if np.iscomplexobj(b) and not np.iscomplexobj(A.data):
                A = A.astype(complex)
            x = spla.splu(A).solve(b)
            res = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
            residuals.append((gm.label, float(res)))
            xf = np.zeros(full.L.shape[1], dtype=x.dtype)
            xf[gm.col_index] = x                                    # E: subspace -> full space
            if np.iscomplexobj(xf) and not cplx:
                # A complex LHS (e.g. i*omega*u) with real data and a real layout gives a complex
                # solution (gallery, "complex LBVP"); it used to be cast into real arrays (imaginary part dropped
                # with only a numpy ComplexWarning). Promote instead of discarding.
                cplx = True
                out = {k: a.astype(complex) for k, a in out.items()}
            for v, vl in zip(self.variables, vls_all[g], strict=True):
                fidx = self.layout.fourier_indices(vl.fdep, g)
                for ci, c in enumerate(_comps(v.tensorsig)):
                    r0 = vl.offset + ci * vl.block
                    for fj, fi in enumerate(fidx):
                        idx = list(c) + [0] * len(coords)
                        for a, j in zip(fax, fi, strict=True):
                            idx[len(c) + a] = j
                        if vl.zdep:
                            idx[len(c) + coords.index(zc)] = slice(None)
                            out[v.name][tuple(idx)] = xf[r0 + fj * vl.nz: r0 + (fj + 1) * vl.nz]
                        else:
                            out[v.name][tuple(idx)] = xf[r0 + fj]
        if not cplx:
            out = {k: np.real(v) for k, v in out.items()}
        return LBVPSolution(out, residuals, self, reduced)

    def resolution_report(self, state, **kw):
        """Spectral-tail resolution report of ``state`` (``conatus.resolution``); read-only."""
        from ..resolution import resolution_report
        return resolution_report(self, state, **kw)

    # ------------------------------------------------------------------ explain
    def explain(self) -> str:
        """Bases, groups, equation spaces, NCC reports and the rank report of the first group."""
        L = self.layout
        lines = [f"Discretization plan for {self.problem.kind} '{self.problem.name}'",
                 f"  domain: {self.domain!r}"]
        if getattr(self.problem, "doubling_note", None):
            lines.append("  " + self.problem.doubling_note)
        for d in L.fdirs:
            lines.append(f"  {d.coord.name}: {type(d.basis).__name__}, {len(d.items)} groups "
                         "(separable, one block per wavenumber)")
        if L.zfam is not None:
            lines.append("  " + L.zfam.explain())
        elif L.zbasis is not None:
            lines.append(f"  {L.zbasis.coord.name}: Chebyshev T, N={L.nz}, Olver–Townsend "
                         "ultraspherical operators (banded), generalized tau with lifts")
            if getattr(L.zbasis, "map", None) is not None:
                lines.append(f"  mapped coordinate: {L.zbasis.map.explain()}; d/d{L.zbasis.coord.name}"
                             " = (dxi/dy) d/dxi, the metric multiplied as a non-constant coefficient")
            lines.append("  NCC resolution: " + (
                us.ncc_policy(self.ncc_tail_tol, self.ncc_cutoff)))
        g0 = self.groups[0]
        gm = self._assemble(g0)
        lines.append(f"  group size (first group): {gm.shape[0]} x {gm.shape[1]}, "
                     f"nnz(L) = {gm.L.nnz}")
        for (i, c), lam in sorted(gm.lam.items(), key=lambda t: (t[0][0], t[0][1])):
            zname = L.zbasis.coord.name if L.zbasis is not None else "z"
            if L.zfam is None:
                space = f"constant along {zname}" if lam is None else ("T" if lam == 0 else f"C^({lam})")
            else:
                space = (f"constant along {zname}" if lam is None else
                         {"Hermite (whole line)": "psi_n",
                          "Laguerre (half line)": f"L^({lam})_n e^(-y/2)"}.get(
                              L.zfam.kind, (f"P^({L.zfam.a + lam:g},{L.zfam.b + lam:g})"
                                            if lam else L.zfam.coeffs_name)
                              if hasattr(L.zfam, "coeffs_name") else
                              ("T per piece" if lam == 0 else f"C^({lam}) per piece")))
            lines.append(f"  equation [{i}]{list(c) if c else ''}: {self.eq_kind[i]}, rows in {space}")
        if self.ncc_reports:
            lines.append("  non-constant coefficients:")
            for rep in self.ncc_reports.values():
                lines.append("    " + rep.explain())
        lines.append("  RHS data: projected by interpolation on N Gauss–Chebyshev / uniform nodes"
                     if L.zfam is None else
                     "  RHS data: projected by Gauss quadrature on the N-point grid of each axis")
        if self.problem.symmetry is not None:
            red = self.matrices(g0, True)
            lines.append(f"  symmetry subspace: first group {red.shape[0]} x {red.shape[1]} "
                         f"(full {gm.shape[0]} x {gm.shape[1]}); symmetry-breaking modes are "
                         "not represented")
        lines.append(self.rank_report(g0).explain())
        return "\n".join(lines)

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import plan_html
        return plan_html(self)


def discretize(problem: Problem, ncc: Mapping[E.Field, Any] | None = None,
               params: Mapping[str, Any] | None = None, ncc_cutoff: float | None = None,
               assembly: str = "direct", ncc_tail_tol: float = NCC_TAIL_TOL) -> Plan:
    """Build the discretization plan of a problem (matrices are assembled lazily per group).
    ``assembly="wavenumber_interpolation"`` compiles each class of Fourier groups at a few
    wavenumbers and interpolates (exact for these polynomial matrices; checked once per class).
    Problems on a terrain-following mapped domain (``TerrainChebyshev``) get a
    ``TerrainPlan`` (all Fourier modes coupled by the metric, one sparse system).
    Problems on PolarCoordinates get a ``PolarPlan`` (groups by azimuthal wavenumber m, regular
    disk/annulus bases; same public face). An LHS with conj() of an unknown (R-linear) is
    discretized as the equivalent doubled system (X, conj X), declared (``conjugate.py``).
    Non-constant coefficients: one whose sampled Chebyshev tail exceeds ``ncc_tail_tol``
    (declared default ``conatus.tolerances.NCC_TAIL_TOL``) is refused with the resolution to use;
    ``ncc_cutoff=c`` declares a truncation (coefficients below c max|a| dropped, sampled tails up
    to c accepted), reported by ``plan.explain()`` (every plan: Cartesian, polar, spherical)."""
    from .conjugate import conjugate_doubling, has_lhs_conj
    from .terrain_plan import TerrainPlan, has_terrain
    if has_terrain(problem):          # mapped domain: one coupled system, metric implicit
        return TerrainPlan(problem, ncc=ncc, params=params, ncc_cutoff=ncc_cutoff,
                           assembly=assembly, ncc_tail_tol=ncc_tail_tol)
    if has_lhs_conj(problem):
        problem = conjugate_doubling(problem)
    if problem.variables and problem.variables[0].domain.coordsys.kind in ("Polar", "Cylindrical"):
        from .polar_plan import PolarPlan
        return PolarPlan(problem, ncc=ncc, params=params, ncc_cutoff=ncc_cutoff, assembly=assembly,
                         ncc_tail_tol=ncc_tail_tol)
    if problem.variables and problem.variables[0].domain.coordsys.kind in ("S2", "Spherical"):
        from .sphere_plan import SpherePlan  # sphere, shell, ball
        return SpherePlan(problem, ncc=ncc, params=params, ncc_cutoff=ncc_cutoff, assembly=assembly,
                          ncc_tail_tol=ncc_tail_tol)
    return Plan(problem, ncc=ncc, params=params, ncc_cutoff=ncc_cutoff, assembly=assembly,
                ncc_tail_tol=ncc_tail_tol)
