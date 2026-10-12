"""Newton for NLBVPs on periodic (all-Fourier) domains, with a declared phase condition.

The Newton step of ``NewtonSolver`` is a per-wavenumber LBVP: it refuses a Jacobian whose
coefficients vary along a Fourier direction ("couples wavenumbers"), which is exactly the case of
any x-dependent steady state (rolls, fronts, cellular patterns). This solver treats the coupled
problem directly (gallery gap, Swift–Hohenberg rolls):

* unknowns: all represented Fourier coefficients of every variable (RealFourier pairs
  [cos_n, -sin_n] without msin_0; ComplexFourier without the Nyquist entry), plus the scalar
  unknowns (fields without any coordinate, e.g. an amplitude);
* residual F_h(c): every equation F = LHS - RHS evaluated by the reference evaluator on the
  dealiased grid of the bases (M = floor(N * dealias) points, the declared factor, never changed)
  and projected back on the N retained modes (the truncation is the dealiasing projection);
* Jacobian: the IR's exact Fréchet derivative F'(U)[dv], evaluated column by column on the same
  grid, i.e. the exact derivative of F_h: a **dense coupled Jacobian** (declared, size n); fine for
  the 1D and small 2D patterns it is meant for, not for DNS-size states;
* translation invariance: along a Fourier direction on which no coefficient field depends, every
  x-shift of a solution is a solution and F_h' is singular. A ``PhaseCondition`` must then be
  declared (refused otherwise, with the fix in the message). It adds one equation p(u) = 0 and one
  unknown mu, the Lagrange multiplier of a co-moving-frame term mu * d_x(u) added to one declared
  equation: F(u) + mu d_x u = 0, p(u) = 0. For a steady state mu -> 0 (reported); a nonzero mu is
  a travelling wave and is refused as a steady solution (``allow_drift=True`` keeps it, declared).

Update x <- x + damping * dx with the declared damping (never adapted); the residual of every
iterate is reported (``explain()``). Chebyshev directions are refused explicitly (use
``NewtonSolver``, whose Newton step is banded, for Fourier x Chebyshev problems with
x-independent iterates).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..discretize.param_keys import check_param_keys, problem_parameter_names
from ..ir import expr as E
from ..ir.bases import ComplexFourier, RealFourier
from ..ir.coords import Coordinate
from ..ir.domain import Domain
from ..ir.problem import NLBVP, ProblemError
from ..reference import transforms as T
from ..reference.evaluate import FieldData, ReferenceEvaluator
from .._deprecation import renames
from ..tolerances import REQUIRED, require
from .nlbvp import NEWTON_DAMPING, NEWTON_MAX_ITERATIONS, NewtonIteration, NewtonNotConverged
from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["PhaseCondition", "PeriodicNewtonSolver", "PeriodicNLBVPSolution"]


@dataclass(frozen=True)
class PhaseCondition:
    """Fix the shift along a periodic ``coord`` by one linear condition on ``field``.

    kind="sin1": the (-sin) coefficient of the first harmonic along ``coord`` (mean mode along the
    other directions) vanishes, i.e. the first harmonic is a pure cosine (RealFourier only).
    kind="reference": int field * d_coord(reference) dV = 0 (the classical orthogonality to the
    shifted reference; ``reference`` is a callable of the grids, or None for the initial guess).
    ``equation``: index of the equation receiving the multiplier term mu * d_coord(field); it
    must be given when the problem has several equations on the full domain (never guessed).
    """

    field: E.Field
    coord: Coordinate
    kind: str = "sin1"
    reference: Any = None
    equation: int | None = None

    def explain(self) -> str:
        what = (f"(-sin) coefficient of mode 1 of {self.field.name} along {self.coord.name} = 0"
                if self.kind == "sin1" else
                f"int {self.field.name} d_{self.coord.name}(u_ref) dV = 0 (u_ref = "
                + ("initial guess" if self.reference is None else "given reference") + ")")
        return f"phase condition along {self.coord.name}: {what}"


@dataclass
class PeriodicNLBVPSolution:
    __repr__ = compact_repr   # one line, arrays by shape
    coeffs: dict[str, np.ndarray]
    iterations: list[NewtonIteration]
    converged: bool
    multipliers: dict[str, float]
    solver: PeriodicNewtonSolver

    @property
    def residual(self) -> float:
        return self.iterations[-1].residual

    def grid(self, name: str) -> np.ndarray:
        """Values of a variable on its N-point (non-dealiased) grid."""
        v = self.solver._var(name)
        g = self.coeffs[name]
        for i, b in self.solver._axes(v):
            g = T.backward(g, b, axis=v.rank + i, dealias=False)
        return g

    def value(self, name: str) -> complex | float:
        a = np.asarray(self.coeffs[name]).ravel()
        if a.size != 1:
            raise OptionError(f"{name} is not a scalar unknown", fix='name a scalar (constant) unknown')
        return a[0]

    def harmonic(self, name: str, coord: Coordinate, n: int) -> float:
        """Amplitude |c_n| of the n-th harmonic of a scalar field along ``coord`` (mean mode along
        the others): sqrt(cos_n^2 + sin_n^2) for RealFourier, 2|c_n| for ComplexFourier."""
        v = self.solver._var(name)
        c = np.asarray(self.coeffs[name])
        idx = [0] * c.ndim
        ax = self.solver.coords.index(coord)
        b = v.domain.basis(coord)
        if isinstance(b, RealFourier):
            idx[ax] = 2 * n
            a = c[tuple(idx)]
            idx[ax] = 2 * n + 1
            return float(np.hypot(a, c[tuple(idx)]))
        idx[ax] = n
        return float(2 * abs(c[tuple(idx)]))

    def explain(self) -> str:
        lines = [f"Periodic NLBVP solution: {'converged' if self.converged else 'NOT converged'} "
                 f"in {len(self.iterations) - 1} Newton iterations, final |F_h| = "
                 f"{self.residual:.2e} (relative {self.iterations[-1].rel_residual:.2e})"]
        for n, m in self.multipliers.items():
            lines.append(f"  multiplier {n} (co-moving drift) = {m:.2e}  (0 for a steady state)")
        lines += [it.explain() for it in self.iterations]
        return "\n".join(lines)


class PeriodicNewtonSolver:
    """Newton with a dense coupled Jacobian for an NLBVP on an all-Fourier domain.

    ``tolerance`` (on max |F_h| / scale), ``damping`` and ``max_iterations`` are required.
    ``phase``: one PhaseCondition per translation-invariant periodic direction (required there).
    ``ncc``: fixed coefficient fields (callables of the grids or FieldData), as for discretize.
    """

    @renames("conatus.solvers.periodic:PeriodicNewtonSolver.__init__")
    def __init__(self, problem: NLBVP, *, tol: float = REQUIRED, damping: float = NEWTON_DAMPING,
                 max_iterations: int = NEWTON_MAX_ITERATIONS,
                 phase: Sequence[PhaseCondition] | PhaseCondition | None = None,
                 ncc: Mapping[E.Field, Any] | None = None, params: Mapping[str, Any] | None = None,
                 allow_drift: bool = False) -> None:
        if not isinstance(problem, NLBVP):
            from ..errors import InputTypeError
            raise InputTypeError(f"PeriodicNewtonSolver takes an NLBVP, got {type(problem).__name__}",
                                 fix="NLBVP([u, ...]) with its equations; an IVP is time-stepped, an "
                                     "LBVP solved by discretize(pb).solve()")
        if not (0 < damping <= 1):
            from ..errors import OptionError
            raise OptionError(f"PeriodicNewtonSolver: damping={damping!r} must be in (0, 1]",
                              fix="damping=1.0 (full Newton step) or a smaller step, e.g. 0.5")
        tol = require(tol, "tol", where="PeriodicNewtonSolver()", suggest="tol=1e-10",
                      why="tol is the accuracy of the solution: Newton stops when max |F_h| / "
                          "scale <= tol, and the residual of every iterate is reported")
        if not tol > 0 or isinstance(max_iterations, bool) or not isinstance(
                max_iterations, (int, np.integer)) or max_iterations < 1:
            from ..errors import OptionError
            raise OptionError(f"PeriodicNewtonSolver: tol={tol!r} > 0 and an integer "
                              f"max_iterations={max_iterations!r} >= 1 are required",
                              fix="tol=1e-10, max_iterations=20 (the residual of every "
                                  "iterate is reported)")
        self.problem = problem
        # self.tolerance: the v0.1 attribute name, kept (read by scripts); tol is the argument
        self.tol = self.tolerance = tol
        self.damping, self.max_iterations = damping, max_iterations
        self.params = dict(params or {})
        # this solver evaluates the IR directly (no discretize): check the keys here
        check_param_keys(self.params, problem_parameter_names(problem), where="PeriodicNewtonSolver")
        self.allow_drift = allow_drift
        self.phase = [phase] if isinstance(phase, PhaseCondition) else list(phase or [])
        V = frozenset(problem.variables)
        dom = None
        for v in problem.variables:
            dom = Domain.union(dom, v.domain)
        for eq in problem.equations:
            dom = Domain.union(dom, eq.domain)
        if dom is None or not dom.bases:
            raise ProblemError("PeriodicNewtonSolver: the problem has no periodic direction")
        bad = [b for b in dom.bases if not isinstance(b, (RealFourier, ComplexFourier))]
        if bad:
            raise NotImplementedError(
                f"PeriodicNewtonSolver: non-periodic basis {bad[0]!r}. Only all-Fourier domains "
                "are handled (dense coupled Jacobian); use NewtonSolver (banded Newton step) when "
                "the iterate does not vary along the Fourier directions")
        kinds = {type(b) for b in dom.bases}
        if len(kinds) > 1:
            raise NotImplementedError("PeriodicNewtonSolver: mixed RealFourier/ComplexFourier "
                                      "domains are not supported (use one kind)")
        self.domain = dom
        self.coords = dom.coordsys.coords
        self.complex = ComplexFourier in kinds
        for v in problem.variables:
            if v.domain is not None and v.domain.bases and v.domain != dom:
                raise ProblemError(f"variable {v.name}: must live on the full domain or be a "
                                   "scalar (no coordinate); partial domains are not supported")
        # fine (dealiased) evaluation domain: same bases with M = grid_size points, dealias 1
        fine = [type(b)(b.coord, b.grid_size, b.bounds, dealias=1) for b in dom.bases]
        self.fine = Domain(dom.coordsys, fine)
        self.ev = ReferenceEvaluator(self.fine)
        # fixed coefficient fields: every non-variable field of the equations needs data
        used = frozenset(f for eq in problem.equations for s in (eq.lhs, eq.rhs)
                         for f in s.free_fields()) - V
        given = dict(ncc or {})
        stray = [f.name for f in given if f not in used]
        if stray:
            raise ProblemError(f"ncc given for field(s) not used by the problem: {sorted(stray)}")
        missing = [f.name for f in used if f not in given]
        if missing:
            raise ProblemError(f"no data for coefficient field(s) {sorted(missing)} (pass ncc=)")
        self.fixed = {f: self._fine_data(f, d) for f, d in given.items()}
        # translation invariance, direction by direction (only fields can carry x-dependence)
        self.invariant = []
        for b in dom.bases:
            c = b.coord
            dep = any(f.domain is not None and f.domain.has(c) for f in used)
            pinned = any(isinstance(n, (E.Interpolate, E.Lift)) and getattr(n, "coord", None) == c
                         for eq in problem.equations for n in eq.lhs.walk())
            if not (dep or pinned):
                self.invariant.append(c)
        declared = [p.coord for p in self.phase]
        missing = [c.name for c in self.invariant if c not in declared]
        if missing:
            raise ProblemError(
                f"the problem is invariant under translations along {', '.join(missing)} (no "
                "coefficient depends on it): every shift of a solution is a solution and the "
                "Jacobian is singular. Declare one phase=PhaseCondition(field, coord, ...) per "
                "such direction (e.g. kind='sin1': pure-cosine first harmonic)")
        if len(set(declared)) != len(declared):
            raise ProblemError("two phase conditions along the same direction")
        full_eqs = [i for i, eq in enumerate(problem.equations)
                    if eq.domain is not None and eq.domain == dom]
        self._phase_eq = []
        for p in self.phase:
            if p.field not in V or p.field.domain != dom:
                raise ProblemError(f"phase condition: {p.field.name} must be an unknown on the "
                                   "full domain")
            if p.kind not in ("sin1", "reference"):
                raise OptionError(f"phase condition kind must be 'sin1' or 'reference', got {p.kind!r}",
                                  fix="PhaseCondition(kind='sin1', ...) or PhaseCondition(kind='reference', ...)")
            if p.kind == "sin1" and not isinstance(dom.basis(p.coord), RealFourier):
                raise ProblemError("phase kind 'sin1' needs RealFourier along the direction; use "
                                   "kind='reference' on ComplexFourier")
            i = p.equation
            if i is None:
                if len(full_eqs) != 1:
                    raise ProblemError(f"{len(full_eqs)} equations on the full domain: say which one "
                                       "receives the multiplier term with PhaseCondition(equation=i)")
                i = full_eqs[0]
            if problem.equations[i].lhs.tensorsig != p.field.tensorsig:
                raise ProblemError(f"equation [{i}] and {p.field.name} have different tensor ranks")
            self._phase_eq.append(i)
        # Fréchet derivative of every equation in the direction of perturbation fields d<name>
        self.pert = {v: E.Field("d_" + v.name, v.domain, v.tensorsig, v.dtype)
                     for v in problem.variables}
        self.F = [E.add(eq.lhs, E.neg(eq.rhs)) for eq in problem.equations]
        self.dF = [f.linearize(None, dict(self.pert)) for f in self.F]
        # unknown / equation layouts (masks of represented coefficients)
        self.var_slices, n = {}, 0
        for v in problem.variables:
            m = self._mask(v.domain, v.tensorsig)
            self.var_slices[v.name] = (slice(n, n + int(m.sum())), m)
            n += int(m.sum())
        self.n_fields = n
        self.eq_masks = [self._mask(eq.domain, eq.lhs.tensorsig) for eq in problem.equations]
        n_eq = sum(int(m.sum()) for m in self.eq_masks)
        if n_eq != n:
            raise ProblemError(f"non-square system: {n_eq} equation coefficients for {n} unknown "
                               "coefficients (phase conditions add one equation and one "
                               "multiplier each)")
        self.size = n + len(self.phase)

    # ------------------------------------------------------------------ layout helpers
    def _var(self, name: str) -> E.Field:
        return next(v for v in self.problem.variables if v.name == name)

    def _axes(self, x) -> list[tuple[int, Any]]:
        dom = x.domain if isinstance(x, E.Expr) else x
        return [(i, dom.basis(c)) for i, c in enumerate(self.coords)
                if dom is not None and dom.has(c)]

    def _shape(self, dom: Domain | None, tensorsig) -> tuple:
        comps = tuple(cs.dim for cs in tensorsig)
        return comps + tuple(dom.basis(c).size if (dom is not None and dom.has(c)) else 1
                             for c in self.coords)

    def _mask(self, dom: Domain | None, tensorsig) -> np.ndarray:
        """Represented coefficients: RealFourier drops msin_0 (index 1), ComplexFourier Nyquist."""
        m = np.ones(self._shape(dom, tensorsig), bool)
        r = len(tensorsig)
        for i, b in self._axes(dom):
            sl = [slice(None)] * m.ndim
            sl[r + i] = 1 if isinstance(b, RealFourier) else b.size // 2
            m[tuple(sl)] = False
        return m

    def _to_fine(self, c: np.ndarray, x) -> np.ndarray:
        for i, b in self._axes(x):
            c = T.backward(c, b, axis=x.rank + i, dealias=True)
        return c

    def _from_fine(self, g: np.ndarray, dom: Domain | None, rank: int) -> np.ndarray:
        out = np.asarray(g)
        if not self.complex and np.iscomplexobj(out):
            _negligible_imag(out)                       # refuses a genuine imaginary part
            out = out.real
        for i, b in self._axes(dom):
            out = T.forward(out, b, axis=rank + i, dealias=True)
        return out

    def _fine_data(self, f: E.Field, d: Any) -> np.ndarray:
        """Coefficient-field data on the fine grid (callable of the fine grids, or FieldData)."""
        if isinstance(d, FieldData):
            c = np.asarray(d.data)
            if d.layout == "grid":
                for i, b in self._axes(f):
                    c = T.forward(c, b, axis=f.rank + i, dealias=False)
            return self._to_fine(np.broadcast_to(c, self._shape(f.domain, f.tensorsig)), f)
        if callable(d):
            grids = [self.ev.grid(c) if f.domain is not None and f.domain.has(c)
                     else np.zeros([1] * len(self.coords)) for c in self.coords]
            return np.asarray(d(*grids))
        return np.asarray(d)

    def _coeffs_of_guess(self, v: E.Field, d: Any) -> np.ndarray:
        shape = self._shape(v.domain, v.tensorsig)
        dtype = complex if self.complex else float
        if d is None:
            return np.zeros(shape, dtype)
        if not v.domain or not v.domain.bases:
            return np.full(shape, d, dtype=dtype)
        g = np.broadcast_to(self._fine_data(v, d), self._fine_shape(v)).astype(dtype)
        return self._from_fine(g, v.domain, v.rank).astype(dtype)

    def _fine_shape(self, v: E.Field) -> tuple:
        comps = tuple(cs.dim for cs in v.tensorsig)
        return comps + tuple(b.size if (v.domain is not None and v.domain.has(b.coord)) else 1
                             for b in self.fine.bases)

    def pack(self, coeffs: Mapping[str, np.ndarray]) -> np.ndarray:
        x = np.zeros(self.n_fields, complex if self.complex else float)
        for v in self.problem.variables:
            sl, m = self.var_slices[v.name]
            x[sl] = np.asarray(coeffs[v.name])[m]
        return x

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for v in self.problem.variables:
            sl, m = self.var_slices[v.name]
            a = np.zeros(m.shape, x.dtype)
            a[m] = x[sl]
            out[v.name] = a
        return out

    # ------------------------------------------------------------------ residual and Jacobian
    def _bindings(self, coeffs: Mapping[str, np.ndarray]) -> dict:
        b = dict(self.fixed)
        for v in self.problem.variables:
            b[v] = self._to_fine(np.asarray(coeffs[v.name]), v)
        return b

    def _project(self, vals: list[np.ndarray]) -> np.ndarray:
        """Equation values on the fine grid -> retained coefficients, stacked."""
        out = []
        for eq, m, g in zip(self.problem.equations, self.eq_masks, vals, strict=True):
            shape = tuple(cs.dim for cs in eq.lhs.tensorsig) + tuple(
                b.grid_size if (eq.domain is not None and eq.domain.has(b.coord)) else 1
                for b in self.domain.bases)
            g = np.broadcast_to(g, shape)
            c = self._from_fine(g, eq.domain, eq.lhs.rank)
            out.append(np.asarray(c)[m])
        return np.concatenate(out)

    def residual(self, coeffs: Mapping[str, np.ndarray], mu: Sequence[float] = ()) -> np.ndarray:
        """F_h(c) (retained coefficients of every equation), with the multiplier terms."""
        bind = self._bindings(coeffs)
        vals = [self.ev.evaluate(f, bind, self.params) for f in self.F]
        for m, p, i in zip(mu, self.phase, self._phase_eq, strict=True):
            vals[i] = vals[i] + m * self.ev.evaluate(E.diff(p.field, p.coord), bind, self.params)
        return self._project(vals)

    def jacobian(self, coeffs: Mapping[str, np.ndarray], mu: Sequence[float] = ()) -> np.ndarray:
        """Exact dF_h/dc, column by column (dense, n x n), including mu * d_x on the phase rows."""
        bind = self._bindings(coeffs)
        n = self.n_fields
        J = np.zeros((n, n), complex if self.complex else float)
        zero = {self.pert[v]: np.zeros(self._fine_shape(v)) for v in self.problem.variables}
        for v in self.problem.variables:
            sl, m = self.var_slices[v.name]
            idx = np.argwhere(m)
            for col, ix in zip(range(sl.start, sl.stop), idx, strict=True):
                e = np.zeros(m.shape, complex if self.complex else float)
                e[tuple(ix)] = 1.0
                b2 = {**bind, **zero, self.pert[v]: self._to_fine(e, v)}
                vals = [self.ev.evaluate(d, b2, self.params) for d in self.dF]
                for mm, p, i in zip(mu, self.phase, self._phase_eq, strict=True):
                    if p.field == v:
                        vals[i] = vals[i] + mm * self.ev.evaluate(E.diff(self.pert[v], p.coord),
                                                                  b2, self.params)
                J[:, col] = self._project(vals)
        return J

    def _phase_rows(self, coeffs0: Mapping[str, np.ndarray]) -> np.ndarray:
        """Linear phase functionals P (one row per condition, over the field unknowns)."""
        n = self.n_fields
        P = np.zeros((len(self.phase), n), complex if self.complex else float)
        for k, p in enumerate(self.phase):
            sl, m = self.var_slices[p.field.name]
            ax = p.field.rank + self.coords.index(p.coord)
            if p.kind == "sin1":
                sel = np.zeros(m.shape, bool)
                idx = [0] * m.ndim
                idx[ax] = 3                              # msin_1 along coord, mode 0 elsewhere
                sel[tuple(idx)] = True
                P[k, sl] = sel[m].astype(float)
            else:
                ref = (coeffs0[p.field.name] if p.reference is None
                       else self._coeffs_of_guess(p.field, p.reference))
                dref = self.ev.evaluate(E.diff(p.field, p.coord),
                                        {p.field: self._to_fine(np.asarray(ref), p.field)}, self.params)
                w = np.prod([b.length / b.grid_size for b in self.fine.bases])
                idx = np.argwhere(m)
                for j, ix in zip(range(sl.start, sl.stop), idx, strict=True):
                    e = np.zeros(m.shape, P.dtype)
                    e[tuple(ix)] = 1.0
                    P[k, j] = np.sum(self._to_fine(e, p.field) * dref) * w
        return P

    # ------------------------------------------------------------------ Newton
    def solve(self, guess: Mapping[E.Field, Any], *,
              require_convergence: bool = True) -> PeriodicNLBVPSolution:
        """Newton from ``guess`` (field -> callable of the grids / FieldData / number)."""
        missing = [v.name for v in self.problem.nonlinear_variables() if v not in guess]
        if missing:
            raise ProblemError(f"initial guess required for {missing} (they enter nonlinearly)")
        coeffs = {v.name: self._coeffs_of_guess(v, guess.get(v)) for v in self.problem.variables}
        P = self._phase_rows(coeffs)
        n, q = self.n_fields, len(self.phase)
        x = self.pack(coeffs)
        mu = np.zeros(q, x.dtype)
        iterations: list[NewtonIteration] = []
        converged = False
        for k in range(self.max_iterations + 1):
            cf = self.unpack(x)
            R = np.concatenate([self.residual(cf, mu), P @ x])
            J = self.jacobian(cf, mu)
            G = np.zeros((n, q), x.dtype)
            for j, (p, i) in enumerate(zip(self.phase, self._phase_eq, strict=True)):
                vals = [np.zeros(1)] * len(self.F)
                vals[i] = self.ev.evaluate(E.diff(p.field, p.coord), self._bindings(cf), self.params)
                G[:, j] = self._project(vals)
            A = np.block([[J, G], [P, np.zeros((q, q))]]) if q else J
            # scale of the operator's terms: ||J||_inf ||x||_inf (includes the k^p growth of
            # high derivatives, where round-off lives), or the residual itself
            scale = max(float(np.abs(J).sum(axis=1).max(initial=0.0) * np.abs(x).max(initial=0.0)),
                        float(np.abs(R).max(initial=0.0)))
            res = float(np.abs(R).max(initial=0.0))
            rel = res / max(scale, 1e-300)
            iterations.append(NewtonIteration(k, res, rel, None))
            if rel <= self.tolerance:
                converged = True
                break
            if k == self.max_iterations:
                break
            s = np.linalg.svd(A, compute_uv=False)
            if s[-1] <= s[0] * A.shape[0] * np.finfo(float).eps * 10:
                hist = "\n".join(it.explain() for it in iterations)
                raise ProblemError(
                    f"Newton iterate k={k}: bordered Jacobian numerically singular (sigma_min/sigma_max "
                    f"= {s[-1] / s[0]:.1e}): a continuous symmetry without phase condition, or a "
                    f"phase condition that does not fix it (e.g. 'sin1' on a zero first harmonic)\n{hist}")
            d = np.linalg.solve(A, -R)
            x = x + self.damping * d[:n]
            mu = mu + self.damping * d[n:]
            iterations[-1].step = float(np.abs(d).max())
        mult = {f"mu_{p.coord.name}": complex(m) if self.complex else float(np.real(m))
                for p, m in zip(self.phase, mu, strict=True)}
        sol = PeriodicNLBVPSolution(self.unpack(x), iterations, converged, mult, self)
        if require_convergence and not converged:
            raise NewtonNotConverged(sol.explain())
        if converged and not self.allow_drift:
            drift = max((abs(m) for m in mult.values()), default=0.0)
            if drift > 1e3 * self.tolerance * max(1.0, float(np.abs(x).max())):
                raise ProblemError(
                    f"converged to a travelling state, not a steady one: drift multiplier "
                    f"{drift:.3e} (pass allow_drift=True to accept it as a travelling wave)\n"
                    + sol.explain())
        return sol

    def explain(self) -> str:
        dims = ", ".join(f"{b.coord.name}: {type(b).__name__} N={b.size}, dealias {b.dealias:g} "
                         f"(grid M={b.grid_size})" for b in self.domain.bases)
        lines = [f"PeriodicNewtonSolver for NLBVP '{self.problem.name}'",
                 f"  domain: {dims}",
                 f"  unknowns: {self.n_fields} represented coefficients"
                 + (f" + {len(self.phase)} multiplier(s)" if self.phase else ""),
                 "  residual: F = LHS - RHS on the dealiased grid, projected on the N retained modes",
                 f"  Jacobian: exact Fréchet derivative of the IR, dense coupled matrix of size "
                 f"{self.size} (wavenumbers coupled by the x-dependent iterate)",
                 "  translation-invariant directions: "
                 + (", ".join(c.name for c in self.invariant) or "none")]
        for p, i in zip(self.phase, self._phase_eq, strict=True):
            lines.append(f"  {p.explain()}; multiplier term mu*d_{p.coord.name}({p.field.name}) "
                         f"in equation [{i}] (co-moving frame; mu reported, ~0 for a steady state)")
        lines.append(f"  tolerance (max |F_h| / (||J||_inf ||x||_inf)) = {self.tolerance:g}, damping = "
                     f"{self.damping:g} (fixed), max_iterations = {self.max_iterations}")
        return "\n".join(lines)


def _negligible_imag(a: np.ndarray) -> bool:
    """Real-layout values that came out complex only by round-off (e.g. FFT derivatives)."""
    re = float(np.abs(a.real).max(initial=0.0))
    im = float(np.abs(a.imag).max(initial=0.0))
    if im > 1e3 * np.finfo(float).eps * max(re, 1.0):
        raise OptionError(f"residual has an imaginary part {im:.1e} on a real layout (refused)",
                          fix="give real data (guess, forcing) for a real layout, or declare the problem complex")
    return True

