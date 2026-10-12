"""Exact derivative of an expression with respect to a scalar parameter.

Used for time-dependent boundary data: ``bc_time="rk"`` imposes at each Runge–Kutta stage the
value obtained by applying the scheme's implicit tableau to g'(t), which needs g'(t) exactly.
The derivative is symbolic (a new expression, evaluated by the same evaluators as g): sums,
products (Leibniz rule), dot/cross/outer products, numeric powers, the pointwise functions with
a derivative (exp, log, sin, cos, tanh, sqrt, abs) and every spatial operator linear in its
single argument (derivatives, gradients, interpolation, integrals, components, conjugation).
Fields and constants do not depend on a parameter (data are static). Anything else (``sign``,
an unknown node) raises :class:`ParameterDerivativeError` with the reason: the caller declares
its fallback, nothing is approximated.
"""
from __future__ import annotations

from .expr_algebra import (_FUNCS, Add, Component, Conj, Cross, Dot, Mul, Outer, Power,
                           UnaryFunction, add, conj, cross, dot, mul, outer, power)
from .expr_base import Constant, Expr, Field, Parameter, UnitVector, Zero
from .expr_calculus import (Curl, Differentiate, Divergence, Gradient, Integrate, Interpolate,
                            Laplacian, Lift, TimeDerivative)

__all__ = ["diff_parameter", "ParameterDerivativeError"]

# operators linear in their single argument: d/dp commutes with them
_LINEAR = (Differentiate, Gradient, Divergence, Laplacian, Curl, Interpolate, Integrate, Lift,
           Component)


class ParameterDerivativeError(NotImplementedError):
    """The derivative with respect to a parameter is not available exactly (reason in str)."""


def diff_parameter(e: Expr, name: str) -> Expr:
    """d e / d p for the scalar parameter named ``name`` (exact, symbolic). Raises
    :class:`ParameterDerivativeError` when a node has no exact derivative."""
    memo: dict = {}
    return _d(e, name, memo)


def _zero(e: Expr) -> Expr:
    return Zero(e.tensorsig)


def _d(e: Expr, p: str, memo: dict) -> Expr:
    if e in memo:
        return memo[e]
    out = _d1(e, p, memo)
    memo[e] = out
    return out


def _d1(e: Expr, p: str, memo: dict) -> Expr:
    if isinstance(e, Parameter):
        return Constant(1) if e.name == p else Constant(0)
    if not e.args:
        if isinstance(e, (Field, Constant, UnitVector)):
            return _zero(e)
        raise ParameterDerivativeError(f"no derivative rule for the leaf {type(e).__name__}")
    if p not in {q.name for q in e.free_parameters()}:
        return _zero(e)                          # does not depend on p
    if isinstance(e, TimeDerivative):
        raise ParameterDerivativeError("dt() of an unknown in a boundary value")
    if isinstance(e, Add):
        return add(*[_d(a, p, memo) for a in e.args])
    if isinstance(e, Mul):
        terms = []
        for i, a in enumerate(e.args):
            da = _d(a, p, memo)
            if not da.is_zero():
                terms.append(mul(*[da if j == i else b for j, b in enumerate(e.args)]))
        return add(*terms) if terms else _zero(e)
    if isinstance(e, (Dot, Cross, Outer)):
        f = {Dot: dot, Cross: cross, Outer: outer}[type(e)]
        a, b = e.args
        da, db = _d(a, p, memo), _d(b, p, memo)
        terms = ([] if da.is_zero() else [f(da, b)]) + ([] if db.is_zero() else [f(a, db)])
        return add(*terms) if terms else _zero(e)
    if isinstance(e, Power):
        b = e.args[0]
        db = _d(b, p, memo)
        return _zero(e) if db.is_zero() else mul(e.exponent, power(b, e.exponent - 1), db)
    if isinstance(e, UnaryFunction):
        u = e.args[0]
        du = _d(u, p, memo)
        if du.is_zero():
            return _zero(e)
        rule = _FUNCS[e.func][3]
        if rule is None:
            raise ParameterDerivativeError(f"{e.func}() has no derivative (0 almost everywhere, "
                                           "undefined at 0)")
        return mul(rule(u), du)
    if isinstance(e, Conj):
        return conj(_d(e.args[0], p, memo))
    if isinstance(e, _LINEAR) and len(e.args) == 1:
        da = _d(e.args[0], p, memo)
        return _zero(e) if da.is_zero() else e.rebuild([da])
    raise ParameterDerivativeError(f"no derivative rule for the node {type(e).__name__} ({e})")
