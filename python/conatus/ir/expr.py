"""Immutable, structurally hashed expression tree.

Every node knows, symbolically:
  * its tensor signature, domain and dtype;
  * ``degrees(fields)``   -> polynomial degrees in a set of fields (linearity checks);
  * ``order_in(coord)``   -> differential order along a coordinate (tau counting);
  * ``linearize(around, direction)`` -> Fréchet derivative, as a new expression;
  * ``transpose(u, w)``   -> formal transpose w.r.t. the bilinear pairing ∫ a·b dV,
    with the boundary terms of the integrations by parts kept explicit;
  * ``explain()`` / ``inspect()`` / ``tree()``.

Canonical forms: ``Add`` and ``Mul`` are n-ary, flattened, with numeric
constants folded and operands sorted by structural hash, so that structurally
equal expressions compare equal regardless of construction order.

Layout (pure split, every name importable from here as before): ``expr_base`` (the ``Expr``
base class, leaves, ``TransposeResult``), ``expr_algebra`` (sums, products, tensor products,
powers, pointwise functions), ``expr_calculus`` (differential operators, integrals, traces,
lifts).
"""
from __future__ import annotations

from .expr_base import (  # noqa: F401  (re-exported: public face)
    TensorSig, NONPOLY, _sig_key, _sig_str, _num_key, _num_str, _combine_degrees, Expr, Field,
    VectorField, TensorField, UnitVector, unit_vector, Parameter, Constant, Zero, as_expr,
    _union_domains, _result_dtype, TransposeResult)

from .expr_algebra import (  # noqa: F401  (re-exported: public face)
    Add, _split_coeff, add, Mul, mul, neg, _check_cs, Dot, dot, Cross, cross, Outer, outer, pair,
    Component, comp, Power, power, Conj, conj, _d_exp, _d_log, _d_sin, _d_cos, _d_tanh, _d_sqrt,
    _d_abs, _FUNCS, UnaryFunction, _unary, exp, log, sin, cos, tanh, sqrt, absolute, sign,
    has_nonpolynomial, pointwise, is_user_function)

from .expr_calculus import (  # noqa: F401  (re-exported: public face)
    _jump, Differentiate, diff, _cartesian, _no_curvilinear_transpose, Gradient, grad, Divergence,
    div, Laplacian, lap, Curl, curl, TimeDerivative, dt, Integrate, integ, Interpolate, interp,
    Lift, lift)

__all__ = [
    "Expr", "Field", "VectorField", "TensorField", "Parameter", "Constant", "Zero", "UnitVector",
    "Add", "Mul", "Dot", "Outer", "Power", "Conj", "Component",
    "Differentiate", "Gradient", "Divergence", "Laplacian", "Curl", "TimeDerivative",
    "Integrate", "Interpolate", "Lift", "TransposeResult",
    "as_expr", "add", "mul", "neg", "dot", "outer", "power", "conj", "comp",
    "diff", "grad", "div", "lap", "curl", "dt", "integ", "interp", "lift", "pair", "unit_vector",
    "UnaryFunction", "exp", "log", "sin", "cos", "tanh", "sqrt", "absolute", "sign",
    "has_nonpolynomial", "NONPOLY", "pointwise", "is_user_function",
]
