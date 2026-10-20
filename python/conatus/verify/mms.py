"""Method of manufactured solutions, driven by the IR (level D).

``to_sympy`` translates an IR expression into a sympy expression given sympy
expressions for its fields; ``manufacture`` applies each equation's LHS to a
manufactured solution to produce the exact forcing and boundary data. The
forcing is therefore derived from the *same* IR the solver discretizes: no
hand-written right-hand side.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sy

from ..ir import expr as E
from ..ir.coords import Coordinate
from ..ir.problem import Problem
from ..errors import OptionError

__all__ = ["to_sympy", "manufacture", "Manufactured", "observed_order"]

# Pointwise functions of the IR (``UnaryFunction``) and their sympy counterparts. Kept in sync with
# ``ir.expr._FUNCS``: a function missing here is refused by ``to_sympy`` instead of being skipped.
_SYMPY_FUNCS = {"exp": sy.exp, "log": sy.log, "sin": sy.sin, "cos": sy.cos, "tanh": sy.tanh,
                "sqrt": sy.sqrt, "abs": sy.Abs, "sign": sy.sign}


def _sym(c: Coordinate) -> sy.Symbol:
    return sy.Symbol(c.name, real=True)


def to_sympy(e: E.Expr, fields: Mapping[E.Field, Any], params: Mapping[str, Any] | None = None) -> Any:
    """IR -> sympy. Tensors become nested lists (component-major). Lifts are zero (exact solution)."""
    params = dict(params or {})

    def rec(x: E.Expr) -> Any:
        if isinstance(x, E.Field):
            if x not in fields:
                raise KeyError(f"no manufactured expression for field {x.name}")
            return fields[x]
        if isinstance(x, E.Parameter):
            v = params.get(x.name, x.value)
            return sy.nsimplify(v) if v is not None else sy.Symbol(x.name)
        if isinstance(x, E.Constant):
            if x.rank:
                return _zeros(x.tensorsig)
            return sy.nsimplify(x.value)
        if isinstance(x, E.UnitVector):
            return [sy.Integer(1 if i == x.index else 0) for i in range(x.coordsys.dim)]
        if isinstance(x, E.Add):
            out = rec(x.args[0])
            for a in x.args[1:]:
                out = _zip(out, rec(a), lambda p, q: p + q)
            return out
        if isinstance(x, E.Mul):
            out: Any = sy.Integer(1)
            for a in x.args:
                out = _scal_mul(out, rec(a))
            return out
        if isinstance(x, E.Power):
            return rec(x.args[0]) ** sy.nsimplify(x.exponent)
        if isinstance(x, E.UnaryFunction):
            if x.func not in _SYMPY_FUNCS:
                raise NotImplementedError(f"to_sympy: pointwise function '{x.func}' has no sympy "
                                          "counterpart")
            return _SYMPY_FUNCS[x.func](rec(x.args[0]))
        if isinstance(x, E.Differentiate):
            return _map(rec(x.args[0]), lambda f: sy.diff(f, _sym(x.coord)))
        if isinstance(x, E.Gradient):
            f = rec(x.args[0])
            return [_map(f, lambda g, c=c: sy.diff(g, _sym(c))) for c in x.coordsys.coords]
        if isinstance(x, E.Divergence):
            f = rec(x.args[0])
            cs = x.args[0].tensorsig[0].coords
            out = _map(f[0], lambda g: sy.diff(g, _sym(cs[0])))
            for i in range(1, len(cs)):
                out = _zip(out, _map(f[i], lambda g, i=i: sy.diff(g, _sym(cs[i]))), lambda p, q: p + q)
            return out
        if isinstance(x, E.Laplacian):
            f = rec(x.args[0])
            return _map(f, lambda g: sum(sy.diff(g, _sym(c), 2) for c in x.coordsys.coords))
        if isinstance(x, E.Component):
            return rec(x.args[0])[x.index]
        if isinstance(x, E.Dot):
            a, b = rec(x.args[0]), rec(x.args[1])
            return _dot(a, b)
        if isinstance(x, E.Interpolate):
            return _map(rec(x.args[0]), lambda g: g.subs(_sym(x.coord), sy.nsimplify(x.position)))
        if isinstance(x, E.Integrate):
            out = rec(x.args[0])
            for c in x.coords:
                b = x.args[0].domain.basis(c)
                out = _map(out, lambda g, c=c, b=b: sy.integrate(
                    g, (_sym(c), sy.nsimplify(b.bounds[0]), sy.nsimplify(b.bounds[1]))))
            return out
        if isinstance(x, E.Lift):
            return _zeros(x.tensorsig)
        if isinstance(x, E.Curl):
            f = rec(x.args[0])
            cs = [_sym(c) for c in x.coordsys.coords]
            if len(cs) == 2 and x.args[0].rank == 0:
                return [sy.diff(f, cs[1]), -sy.diff(f, cs[0])]
            if len(cs) == 2:
                return sy.diff(f[1], cs[0]) - sy.diff(f[0], cs[1])
            return [sy.diff(f[2], cs[1]) - sy.diff(f[1], cs[2]),
                    sy.diff(f[0], cs[2]) - sy.diff(f[2], cs[0]),
                    sy.diff(f[1], cs[0]) - sy.diff(f[0], cs[1])]
        raise NotImplementedError(f"to_sympy: {type(x).__name__}")

    return rec(e)


def _zeros(sig) -> Any:
    if not sig:
        return sy.Integer(0)
    return [_zeros(sig[1:]) for _ in range(sig[0].dim)]


def _map(f: Any, fn) -> Any:
    return [_map(g, fn) for g in f] if isinstance(f, list) else fn(f)


def _zip(a: Any, b: Any, fn) -> Any:
    if isinstance(a, list):
        return [_zip(p, q, fn) for p, q in zip(a, b, strict=True)]
    return fn(a, b)


def _scal_mul(a: Any, b: Any) -> Any:
    if isinstance(a, list) and isinstance(b, list):
        return [_scal_mul(p, b) for p in a]  # outer product
    if isinstance(a, list):
        return _map(a, lambda p: p * b)
    if isinstance(b, list):
        return _map(b, lambda q: a * q)
    return a * b


def _dot(a: Any, b: Any) -> Any:
    # contract last index of a with first of b
    if isinstance(a[0], list):
        return [_dot(p, b) for p in a]
    out = None
    for i in range(len(a)):
        term = _scal_mul(a[i], b[i])
        out = term if out is None else _zip(out, term, lambda p, q: p + q)
    return out


@dataclass
class Manufactured:
    solution: dict[E.Field, Any]      # sympy expressions of the unknowns
    rhs: list[Any]                    # sympy expression of each equation's RHS
    params: dict
    nonpolynomial: tuple[str, ...] = ()   # pointwise functions of the problem (aliasing not exact)

    def explain(self) -> str:
        lines = [f"Manufactured solution for {', '.join(v.name for v in self.solution)}; "
                 f"{len(self.rhs)} exact right-hand sides (derived from the IR)"]
        if self.nonpolynomial:
            lines.append("  non-polynomial functions " + ", ".join(self.nonpolynomial)
                         + ": the forcing is exact but the discrete operator carries an aliasing "
                         "error at any dealias factor -- judge the MMS by convergence in N, not by "
                         "a round-off-level error at one resolution")
        return "\n".join(lines)

    def callable(self, expr: Any, coords: list[Coordinate]):
        syms = [_sym(c) for c in coords]
        f = sy.lambdify(syms, expr, "numpy")

        def g(*grids):
            v = np.asarray(f(*grids), dtype=complex)
            if np.all(v.imag == 0):
                v = v.real
            shape = np.broadcast_shapes(*[np.shape(x) for x in grids])
            return np.broadcast_to(v, shape)
        return g


def manufacture(problem: Problem, solution: Mapping[E.Field, Any],
                params: Mapping[str, Any] | None = None) -> Manufactured:
    """Exact RHS of every equation for a manufactured solution.

    Tau variables (operands of ``lift``) default to 0, as they must for an exact
    solution. Every other variable must be given: a missing physical unknown is
    refused instead of being set to zero.
    """
    sol = dict(solution)
    taus = {n.args[0] for eq in problem.equations for side in (eq.lhs, eq.rhs)
            for n in side.walk() if isinstance(n, E.Lift) and isinstance(n.args[0], E.Field)}
    missing = [v.name for v in problem.variables if v not in sol and v not in taus]
    if missing:
        raise OptionError(f"manufactured solution missing for {missing} (only tau variables "
                          "default to 0)",
                          fix="give a manufactured solution for every unknown that is not a tau variable")
    for v in problem.variables:
        sol.setdefault(v, _zeros(v.tensorsig))
    rhs = [to_sympy(eq.lhs, sol, params) for eq in problem.equations]
    rhs = [_map(r, sy.simplify) if isinstance(r, list) else sy.simplify(r) for r in rhs]
    nonpoly = tuple(sorted({f for eq in problem.equations
                            for f in E.has_nonpolynomial(eq.lhs) + E.has_nonpolynomial(eq.rhs)}))
    return Manufactured(sol, rhs, dict(params or {}), nonpoly)


def observed_order(Ns: list[int], errors: list[float]) -> dict:
    """Fit log(err) ~ a - b N (spectral: exponential decay). Returns rate and a verdict string."""
    Ns_a, e = np.asarray(Ns, float), np.maximum(np.asarray(errors, float), 1e-300)
    b, a = np.polyfit(Ns_a, np.log(e), 1)
    return {"rate": -b, "intercept": a,
            "report": f"log(err) ~ {a:.2f} - {-b:.3f} N over N in {list(Ns)} (exponential if rate > 0 "
                      "and residual of the fit small)"}
