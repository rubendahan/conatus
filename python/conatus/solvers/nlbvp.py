"""Newton solver for nonlinear boundary value problems (NLBVP, parity with Dedalus).

Every term t of F = LHS - RHS is linearized with the IR's Fréchet derivative around the current
iterate U (base fields ``U_<name>``, passed as non-constant coefficients). Newton is written for
the new iterate X directly:

    F'(U) X = F'(U) U - F(U)

Terms linear in the unknowns cancel on the right (t'(U) U = t(U)), so the right-hand side only
holds the nonlinear and constant terms, evaluated on the grid from U: no lift/tau term is ever
evaluated pointwise. Each iteration compiles the linear system with the plan (public
matrices, rank report that refuses a singular Jacobian), and the update is

    x_{k+1} = x_k + damping * (X - x_k)            (damping declared, never adapted silently)

The residual reported at each iterate is the discrete one, r_k = A_k x_k - b_k = F_h(x_k).
Nonlinear products are interpolated on N nodes (as in RHS data) while the Jacobian uses
banded multiplication by the interpolated coefficient: the two differ at the aliasing level, so
convergence is quadratic down to ~round-off / aliasing and the final residual is reported as is.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..discretize.compiler import _comps
from ..discretize.plan import Plan, RankError, discretize
from ..ir import expr as E
from ..ir.problem import LBVP, NLBVP, ProblemError, _terms
from ..reference.evaluate import FieldData
from .._deprecation import renames
from ..tolerances import NCC_TAIL_TOL, REQUIRED, require
from .._repr import compact_repr

__all__ = ["NewtonSolver", "NewtonIteration", "NLBVPSolution", "NewtonNotConverged", "pack_group",
           "unpack_group"]

ROUNDOFF_FLOOR = 1e-13   # residuals below this are round-off: excluded from observed orders


def _real_part(a: np.ndarray, name: str) -> np.ndarray:
    """Real part of an iterate on a real layout; refuse a genuine imaginary part."""
    a = np.asarray(a)
    if np.iscomplexobj(a):
        im, re = float(np.abs(a.imag).max(initial=0.0)), float(np.abs(a.real).max(initial=0.0))
        # relative to the iterate's own size (critic review of the invariance audit: max(re, 1)
        # let a genuine imaginary part of an iterate of amplitude 1e-12 be dropped)
        if im > 1e3 * np.finfo(float).eps * re or (re == 0.0 and im > 0.0):
            raise OptionError(f"iterate {name} has an imaginary part {im:.1e} on a real layout",
                              fix="give a real initial guess, or declare the problem complex")
        return a.real
    return a


from ..errors import NewtonNotConverged, OptionError  # noqa: E402  (public classes: conatus.errors)

#: Declared defaults of the Newton solvers of boundary-value problems (rule
#: algorithm controls default and are declared, the accuracy ``tol`` stays required): the
#: full Newton step and an iteration cap; the residual of every iterate is reported and a
#: solve that does not reach ``tol`` raises NewtonNotConverged.
NEWTON_DAMPING = 1.0
NEWTON_MAX_ITERATIONS = 50
#: Trivial-solution note: a converged solve whose nonlinear unknowns
#: end below NEWTON_TRIVIAL_FRACTION times the size of the initial guess (max |coefficient|) has
#: found the trivial state u = 0, not the branch the guess was aimed at (a small guess falls into
#: its basin). Flagged ``trivial=True`` with a note in ``explain()``; never refused (u = 0 may be
#: the wanted answer). Same value as the ECS finders' TRIVIAL_TOL.
NEWTON_TRIVIAL_FRACTION = 1e-8


@dataclass
class NewtonIteration:
    k: int
    residual: float          # max |F_h(x_k)|
    rel_residual: float      # max_i |F_h|_i / r_i over max_i max(|A_k x_k|_i, |b_k|_i) / r_i,
    #                          r_i = max_j |A_k|_ij: each row in its own unit (audit invariance)
    step: float | None       # max |x_{k+1} - x_k| (None for the last iterate)

    def explain(self) -> str:
        s = "-" if self.step is None else f"{self.step:.3e}"
        return f"  k={self.k:2d}  |F|={self.residual:.3e}  |F|/scale={self.rel_residual:.3e}  |dx|={s}"


@dataclass
class NLBVPSolution:
    __repr__ = compact_repr   # one line, arrays by shape
    coeffs: dict[str, np.ndarray]
    iterations: list[NewtonIteration]
    converged: bool
    plan: Plan
    settings: dict = dc_field(default_factory=dict)
    #: the iterate converged to the trivial state (NEWTON_TRIVIAL_FRACTION)
    trivial: bool = False
    notes: list = dc_field(default_factory=list)

    def grid(self, name: str) -> np.ndarray:
        return self.plan.to_grid(name, self.coeffs[name])

    def value(self, name: str) -> float:
        """Value of a constant (scalar) unknown."""
        return float(np.asarray(self.coeffs[name]).reshape(-1)[0])

    @property
    def residual(self) -> float:
        return self.iterations[-1].rel_residual

    def observed_orders(self) -> list[float]:
        """q_k = log(r_{k+1}/r_k) / log(r_k/r_{k-1}) on the residual sequence (2 = quadratic)."""
        r = [it.rel_residual for it in self.iterations]
        out = []
        for k in range(1, len(r) - 1):
            # skip triples that touch round-off (the last residual is then noise, not the rate)
            if min(r[k - 1], r[k], r[k + 1]) <= ROUNDOFF_FLOOR or r[k] >= r[k - 1]:
                continue
            out.append(math.log(r[k + 1] / r[k]) / math.log(r[k] / r[k - 1]))
        return out

    def as_guess(self) -> dict:
        """This solution as an initial guess (continuation), by variable, in coefficient layout."""
        return {v: FieldData(v, self.coeffs[v.name], "coeff") for v in self.plan.variables}

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import newton_html
        return newton_html(self)

    def explain(self) -> str:
        lines = [f"Newton on NLBVP '{self.plan.problem.name}': "
                 + ("converged" if self.converged else "NOT converged")
                 + f" in {len(self.iterations) - 1} iteration(s)",
                 "  settings: " + ", ".join(f"{k}={v}" for k, v in self.settings.items())]
        lines += [it.explain() for it in self.iterations]
        q = self.observed_orders()
        if q:
            lines.append("  observed orders: " + ", ".join(f"{x:.2f}" for x in q))
        lines += [f"  note: {n}" for n in self.notes]
        from ..tolerances import ncc_floor_text
        floor = ncc_floor_text(self.plan, self.residual)
        if floor:
            lines.append(f"  {floor}")
        return "\n".join(lines)


def _pack(plan: Plan, group, coeffs: Mapping[str, np.ndarray], dtype) -> np.ndarray:
    """Coefficient arrays per variable -> full group vector (inverse of Plan.solve's unpacking)."""
    vls = plan.var_layouts(group)
    coords = plan.domain.coordsys.coords
    zc = plan.layout.zcoord
    fax = [coords.index(d.coord) for d in plan.layout.fdirs]
    x = np.zeros(sum(v.size for v in vls), dtype=dtype)
    for v, vl in zip(plan.variables, vls, strict=True):
        arr = coeffs[v.name]
        fidx = plan.layout.fourier_indices(vl.fdep, group)
        for ci, c in enumerate(_comps(v.tensorsig)):
            r0 = vl.offset + ci * vl.block
            for fj, fi in enumerate(fidx):
                idx = list(c) + [0] * len(coords)
                for a, j in zip(fax, fi, strict=True):
                    idx[len(c) + a] = j
                if vl.zdep:
                    idx[len(c) + coords.index(zc)] = slice(None)
                    x[r0 + fj * vl.nz: r0 + (fj + 1) * vl.nz] = arr[tuple(idx)]
                else:
                    x[r0 + fj] = arr[tuple(idx)]
    return x


def _unpack(plan: Plan, group, x: np.ndarray, out: dict[str, np.ndarray]) -> None:
    vls = plan.var_layouts(group)
    coords = plan.domain.coordsys.coords
    zc = plan.layout.zcoord
    fax = [coords.index(d.coord) for d in plan.layout.fdirs]
    for v, vl in zip(plan.variables, vls, strict=True):
        fidx = plan.layout.fourier_indices(vl.fdep, group)
        for ci, c in enumerate(_comps(v.tensorsig)):
            r0 = vl.offset + ci * vl.block
            for fj, fi in enumerate(fidx):
                idx = list(c) + [0] * len(coords)
                for a, j in zip(fax, fi, strict=True):
                    idx[len(c) + a] = j
                if vl.zdep:
                    idx[len(c) + coords.index(zc)] = slice(None)
                    out[v.name][tuple(idx)] = x[r0 + fj * vl.nz: r0 + (fj + 1) * vl.nz]
                else:
                    out[v.name][tuple(idx)] = x[r0 + fj]


pack_group, unpack_group = _pack, _unpack   # public names (continuation adapters)


class NewtonSolver:
    """Newton for an NLBVP. The tolerance ``tol`` is required; damping and iteration budget have declared defaults.

    ``ncc`` / ``data``: fixed coefficient fields of the problem (e.g. the coordinate r used as a
    weight), passed as ``FieldData`` or callables of the grids, exactly as for ``discretize``.
    """

    @renames("conatus.solvers.nlbvp:NewtonSolver.__init__")
    def __init__(self, problem: NLBVP, *, tol: float = REQUIRED, damping: float = NEWTON_DAMPING,
                 max_iterations: int = NEWTON_MAX_ITERATIONS,
                 ncc: Mapping[E.Field, Any] | None = None, params: Mapping[str, Any] | None = None,
                 ncc_cutoff: float | None = None, ncc_tail_tol: float = NCC_TAIL_TOL) -> None:
        if not isinstance(problem, NLBVP):
            from ..errors import InputTypeError
            raise InputTypeError(f"NewtonSolver takes an NLBVP, got {type(problem).__name__}",
                                 fix="NLBVP([u, ...]) with its equations; an IVP is time-stepped, an "
                                     "LBVP solved by discretize(pb).solve()")
        if not (0 < damping <= 1):
            from ..errors import OptionError
            raise OptionError(f"NewtonSolver: damping={damping!r} must be in (0, 1]",
                              fix="damping=1.0 (full Newton step) or a smaller step, e.g. 0.5")
        tol = require(tol, "tol", where="NewtonSolver()", suggest="tol=1e-10",
                      why="tol is the accuracy of the solution: Newton stops when max |F_h| / "
                          "scale <= tol, and the residual of every iterate is reported")
        if not tol > 0 or isinstance(max_iterations, bool) or not isinstance(
                max_iterations, (int, np.integer)) or max_iterations < 1:
            from ..errors import OptionError
            raise OptionError(f"NewtonSolver: tol={tol!r} > 0 and an integer "
                              f"max_iterations={max_iterations!r} >= 1 are required",
                              fix="tol=1e-10, max_iterations=20 (the residual of every "
                                  "iterate is reported)")
        self.problem = problem
        # self.tolerance: the v0.1 attribute name, kept (read by scripts); tol is the argument
        self.tol = self.tolerance = tol
        self.damping, self.max_iterations = damping, max_iterations
        self.fixed = dict(ncc or {})
        self.params = dict(params or {})
        self.ncc_cutoff, self.ncc_tail_tol = ncc_cutoff, ncc_tail_tol      # as discretize
        V = frozenset(problem.variables)
        self.nonlinear = problem.nonlinear_variables()
        self.base = {v: E.Field(f"U_{v.name}", v.domain, v.tensorsig, v.dtype)
                     for v in self.nonlinear}
        around = dict(self.base)
        self.newton = LBVP(problem.variables, name=f"Newton step of {problem.name}")
        self.linear_terms = 0
        self.nonlinear_terms = 0
        for eq in problem.equations:
            lhs_parts, rhs_parts = [], []
            for t in _terms(E.add(eq.lhs, E.neg(eq.rhs))):
                d = t.degrees(V)
                if d == frozenset({1}):
                    lhs_parts.append(t)                         # linear: unchanged, cancels on RHS
                    self.linear_terms += 1
                    continue
                self.nonlinear_terms += 1
                if d != frozenset({0}):
                    lhs_parts.append(t.linearize(around, {v: v for v in V}))
                    t_lin_U = t.linearize(around, dict(self.base))
                    rhs_parts.append(E.add(t_lin_U, E.neg(t.substitute(self.base))))
                else:
                    rhs_parts.append(E.neg(t))
            lhs = E.add(*lhs_parts) if lhs_parts else None
            if lhs is None or lhs.is_zero():
                raise ProblemError(f"equation '{eq}' has a vanishing Jacobian row")
            rhs = E.add(*rhs_parts) if rhs_parts else 0
            self.newton.add_equation((lhs, rhs))

    # ------------------------------------------------------------------
    def _plan(self, base_coeffs: Mapping[str, np.ndarray],
              params: Mapping[str, Any] | None = None) -> tuple[Plan, dict]:
        data = {b: FieldData(b, base_coeffs[v.name], "coeff") for v, b in self.base.items()}
        ncc = {**self.fixed, **data}
        return discretize(self.newton, ncc=ncc, params={**self.params, **(params or {})},
                          ncc_cutoff=self.ncc_cutoff, ncc_tail_tol=self.ncc_tail_tol), ncc

    def discrete_system(self, coeffs: Mapping[str, np.ndarray],
                        params: Mapping[str, Any] | None = None, *,
                        check: bool = True) -> tuple[Plan, list]:
        """Newton system at the iterate ``coeffs``: per Fourier group (g, A, b, x) with
        F_h(x) = A x - b the discrete residual and A = F'(U) the Jacobian (used by continuation).
        ``params`` overrides parameter values (e.g. the continuation parameter). ``check=False``
        skips the rank report of A: only for callers that solve a bordered system which stays
        regular where A is singular (folds in continuation), and that say so."""
        plan, ncc = self._plan(coeffs, params)
        if check:
            plan.check()
        gdata = {f: plan._as_grid(f, d) for f, d in ncc.items()}
        rhs = plan._rhs_coeffs(gdata)
        dtype = complex if plan.layout.complex else float
        out = []
        for g in plan.groups:
            gm = plan._assemble(g)
            if gm.L.shape[0] == 0:
                continue
            b = plan._group_rhs(g, rhs, gm)
            x = _pack(plan, g, coeffs, np.result_type(b, gm.L.dtype, dtype))
            out.append((g, gm.L, b, x))
        return plan, out

    def initial_coeffs(self, guess: Mapping[E.Field, Any]) -> dict[str, np.ndarray]:
        """Coefficient arrays of an initial guess (linear-only unknowns start at 0)."""
        probe = discretize(self.newton, params=self.params)
        dtype = complex if probe.layout.complex else float
        coeffs = {}
        for v in self.problem.variables:
            if v in guess:
                g = probe._as_grid(v, guess[v])
                coeffs[v.name] = np.asarray(probe._to_coeffs(g, v.domain, v.rank), dtype=dtype)
            else:
                coeffs[v.name] = np.zeros(probe._grid_shape(v.domain, v.tensorsig), dtype=dtype)
        return coeffs

    def solve(self, guess: Mapping[E.Field, Any], *, require_convergence: bool = True) -> NLBVPSolution:
        """Newton from ``guess`` (every nonlinear variable needs one; linear ones start at 0)."""
        missing = [v.name for v in self.nonlinear if v not in guess]
        if missing:
            raise ProblemError(f"initial guess required for {missing} (they enter nonlinearly)")
        probe = discretize(self.newton, params=self.params)
        cplx = probe.layout.complex
        dtype = complex if cplx else float
        coeffs = {}
        for v in self.problem.variables:
            shape = probe._grid_shape(v.domain, v.tensorsig)
            if v in guess:
                g = probe._as_grid(v, guess[v])
                coeffs[v.name] = np.asarray(probe._to_coeffs(g, v.domain, v.rank), dtype=dtype)
            else:
                coeffs[v.name] = np.zeros(shape, dtype=dtype)
        guess0 = self._size(coeffs)
        iterations: list[NewtonIteration] = []
        converged = False
        plan = probe
        for k in range(self.max_iterations + 1):
            plan, ncc = self._plan(coeffs)
            try:
                plan.check()                               # refuses a singular Jacobian
            except RankError as e:
                hist = "\n".join(it.explain() for it in iterations)
                raise RankError(f"Newton iterate k={k}: Jacobian refused (at k>0 the iterate "
                                f"probably left the basin: better guess, continuation or damping)"
                                f"\n{hist}\n{e}") from None
            gdata = {f: plan._as_grid(f, d) for f, d in ncc.items()}
            rhs = plan._rhs_coeffs(gdata)
            new = {n: a.copy() for n, a in coeffs.items()}
            res = scale = raw = 0.0
            targets = []
            for g in plan.groups:
                gm = plan._assemble(g)
                if gm.L.shape[0] == 0:
                    continue
                b = plan._group_rhs(g, rhs, gm)
                x = _pack(plan, g, coeffs, np.result_type(b, gm.L.dtype, dtype))
                Ax = gm.L @ x
                # each row in units of its own largest coefficient (audit invariance,
                #): the relative residual of c F(u) = 0 is that of F(u) = 0 whatever
                # c and whichever equation it multiplies; with one global scale an equation
                # written times 1e-8 was measured against its O(1) boundary rows and Newton
                # never reached tol (stalled at 6e-11, reported not converged)
                rn = np.asarray(abs(gm.L).max(axis=1).todense()).ravel() if sp.issparse(gm.L) \
                    else np.abs(gm.L).max(axis=1)
                rn = np.where(rn > 0, rn, 1.0)
                raw = max(raw, float(np.max(np.abs(Ax - b))))
                res = max(res, float(np.max(np.abs(Ax - b) / rn)))
                scale = max(scale, float(np.max(np.abs(Ax) / rn)), float(np.max(np.abs(b) / rn)))
                targets.append((g, gm, b, x))
            rel = res / max(scale, 1e-300)
            iterations.append(NewtonIteration(k, raw, rel, None))
            if rel <= self.tolerance:
                converged = True
                break
            if k == self.max_iterations:
                break
            step = 0.0
            for g, gm, b, x in targets:
                A = gm.L.tocsc()
                if np.iscomplexobj(b) and not np.iscomplexobj(A.data):
                    A = A.astype(complex)
                X = spla.splu(A).solve(b)
                xn = x + self.damping * (X - x)
                step = max(step, float(np.max(np.abs(xn - x))))
                _unpack(plan, g, xn, new)
            iterations[-1].step = step
            coeffs = {n: (a if cplx else _real_part(a, n)) for n, a in new.items()}
        sol = NLBVPSolution(coeffs, iterations, converged, plan,
                            {"tolerance": self.tolerance, "damping": self.damping,
                             "max_iterations": self.max_iterations})
        if converged:
            self._trivial_note(sol, guess0)
        if require_convergence and not converged:
            raise NewtonNotConverged(
                sol.explain(), where=f"NewtonSolver on NLBVP '{self.problem.name}'",
                why=f"the residual did not fall below tol={self.tol:g} within "
                    f"max_iterations={self.max_iterations} (damping={self.damping:g}); the table "
                    "above shows whether it stalled, oscillated or diverged",
                fix="start closer (a solution at a nearby parameter, then continuation), damp the "
                    "steps (damping=0.5), or raise max_iterations if the residual still decreases; "
                    "require_convergence=False returns the last iterate, flagged not converged")
        return sol

    def _size(self, coeffs: Mapping[str, np.ndarray]) -> float:
        """max |coefficient| over the nonlinear unknowns (the state Newton is solving for)."""
        vals = [float(np.max(np.abs(coeffs[v.name]))) for v in self.nonlinear
                if v.name in coeffs and np.size(coeffs[v.name])]
        return max(vals, default=0.0)

    def _trivial_note(self, sol: NLBVPSolution, guess0: float) -> None:
        """Flag a converged solution that is the trivial state (NEWTON_TRIVIAL_FRACTION: a 1e-6 guess converged to u = 0 with converged=True and no word)."""
        end = self._size(sol.coeffs)
        thr = NEWTON_TRIVIAL_FRACTION * guess0
        sol.notes.append(f"trivial-state check: max |u| = {end:.3e} vs {NEWTON_TRIVIAL_FRACTION:g} "
                         f"x the guess ({guess0:.3e})")
        if self.nonlinear and end <= thr:
            sol.trivial = True
            sol.notes.append("converged to the trivial state u = 0 (the unknowns "
                             + ", ".join(v.name for v in self.nonlinear) + " vanish): not the "
                             "branch the guess was aimed at; a non-trivial solution needs a "
                             "larger guess or continuation from a known branch")

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import explain_html
        return explain_html(self)

    def explain(self) -> str:
        lines = [f"NewtonSolver for NLBVP '{self.problem.name}'",
                 f"  unknowns: {', '.join(v.name for v in self.problem.variables)}",
                 f"  nonlinear in: {', '.join(v.name for v in self.nonlinear) or '(none)'}; base "
                 f"fields {', '.join(b.name for b in self.base.values())} (current iterate, NCC)",
                 f"  terms: {self.linear_terms} linear (kept, cancel on the RHS), "
                 f"{self.nonlinear_terms} nonlinear/constant (exact Fréchet derivative of the IR)",
                 "  update: x <- x + damping (X - x), with F'(U) X = F'(U) U - F(U)",
                 f"  tolerance (max |F_h| / scale, rows in their own unit) = {self.tolerance:g}, "
                 f"damping = {self.damping:g} (fixed, never adapted), "
                 f"max_iterations = {self.max_iterations}",
                 "  linear-only unknowns (taus...) start at 0: Newton solves for them exactly",
                 "  Newton system per iteration:"]
        lines += ["    " + s for s in self.newton.explain().splitlines()]
        return "\n".join(lines)
