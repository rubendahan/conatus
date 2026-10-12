"""Expression tree, part 2: algebraic nodes (public face: ``conatus.ir.expr``).

Canonical n-ary ``Add`` / ``Mul`` (flattened, constants folded, operands sorted by structural hash),
tensor products (``Dot``, ``Cross``, ``Outer``, ``pair``), ``Component``, ``Power``, ``Conj`` and the
pointwise non-polynomial functions (``UnaryFunction``: exp, log, sin, cos, tanh, sqrt, abs, sign).
"""
from __future__ import annotations

import numbers
from typing import Any
from collections.abc import Sequence

import numpy as np

from .expr_base import (NONPOLY, Constant, Expr, Field, Parameter, Zero, _combine_degrees, _num_key, _num_str,
                        _result_dtype, _sig_str, _union_domains, as_expr)
from ..errors import EquationError, OptionError


# --------------------------------------------------------------------------- Add / Mul


class Add(Expr):
    """n-ary sum (canonical: flattened, like terms combined, sorted)."""

    _prec = 10

    def __init__(self, terms: Sequence[Expr]) -> None:
        self._setup(terms, terms[0].tensorsig, _union_domains(terms), _result_dtype(terms))

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return add(*args)

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        return frozenset().union(*(a._degrees(fields) for a in self.args))

    def _lin(self, around: dict, direction: dict) -> Expr:
        return add(*(a._lin(around, direction) for a in self.args))

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        adjs, bts = [], []
        for t in self.args:
            a, b = t._tr(u, w)
            adjs.append(a)
            bts.append(b)
        return add(*adjs), add(*bts)

    def __str__(self) -> str:
        parts = [self._wrap(self.args[0])]
        for t in self.args[1:]:
            s = self._wrap(t)
            parts.append(f"- {s[1:]}" if s.startswith("-") else f"+ {s}")
        return " ".join(parts)


def _split_coeff(t: Expr) -> tuple[complex, Expr]:
    if isinstance(t, Mul) and isinstance(t.args[0], Constant):
        rest = t.args[1:]
        return t.args[0].value, (rest[0] if len(rest) == 1 else Mul(rest))
    return 1, t


def add(*terms: Any) -> Expr:
    """Sum of terms (what ``a + b`` builds); zero terms are dropped, constants folded."""
    terms_e = [as_expr(t) for t in terms]
    if not terms_e:
        return Zero(())
    nonzero = [t for t in terms_e if not t.is_zero()]
    sig = (nonzero or terms_e)[0].tensorsig
    for t in nonzero:
        if t.tensorsig != sig:
            from ..errors import TensorError
            raise TensorError(f"cannot add {_sig_str(sig)} and {_sig_str(t.tensorsig)}",
                              why="the terms of a sum have the same tensor signature",
                              fix="take a component (u@ex), a divergence or a gradient so that "
                                  "every term has the same rank", doc="docs/site/guide/operators.md")
    flat: list[Expr] = []
    for t in terms_e:
        flat.extend(t.args if isinstance(t, Add) else [t])
    const: complex = 0
    coeffs: dict[Expr, complex] = {}
    for t in flat:
        if isinstance(t, Constant):
            const += t.value
            continue
        c, rest = _split_coeff(t)
        coeffs[rest] = coeffs.get(rest, 0) + c
    out = [mul(c, r) for r, c in coeffs.items() if c != 0]
    if const != 0:
        if sig:
            raise TypeError("cannot add a nonzero scalar constant to a tensor")
        out.append(Constant(const))
    out = [o for o in out if not o.is_zero()]
    if not out:
        return Zero(sig or ())
    if len(out) == 1:
        return out[0]
    out.sort(key=lambda e: e.structural_hash)
    return Add(out)


class Mul(Expr):
    """n-ary product: [numeric coefficient] * scalar factors (sorted) * [one tensor factor]."""

    _prec = 20

    def __init__(self, factors: Sequence[Expr]) -> None:
        tensors = [f for f in factors if f.rank > 0]
        sig = tensors[0].tensorsig if tensors else ()
        self._setup(factors, sig, _union_domains(factors), _result_dtype(factors))

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return mul(*args)

    @property
    def is_nonlinear_product(self) -> bool:
        """True if two or more factors depend on fields (a nonlinearity)."""
        return sum(1 for f in self.args if f.free_fields()) >= 2

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        out = frozenset({0})
        for f in self.args:
            out = _combine_degrees(out, f._degrees(fields))
        return out

    def _lin(self, around: dict, direction: dict) -> Expr:
        base = [f.substitute(around) for f in self.args]
        terms = []
        for i, f in enumerate(self.args):
            df = f._lin(around, direction)
            if df.is_zero():
                continue
            terms.append(mul(*(base[:i] + [df] + base[i + 1:])))
        return add(*terms) if terms else Zero(self.tensorsig)

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        dep = [f for f in self.args if f._degrees(frozenset({u})) != frozenset({0})]
        if len(dep) != 1:
            raise EquationError("transpose: product is not linear in the field",
                                fix="transpose a product that is linear in the field (linearize first)")
        a = dep[0]
        coef_factors = [f for f in self.args if f is not a]
        coef = mul(*coef_factors) if coef_factors else Constant(1)
        if a.rank == self.rank:
            w2 = mul(coef, w)
        else:  # scalar a times tensor coefficient
            w2 = pair(coef, w)
        return a._tr(u, w2)

    def __str__(self) -> str:
        args = list(self.args)
        prefix = ""
        if isinstance(args[0], Constant) and len(args) > 1:
            v = args[0].value
            if v == -1:
                prefix = "-"
                args = args[1:]
        return prefix + "*".join(self._wrap(a) for a in args)


def mul(*factors: Any) -> Expr:
    """Product of factors (what ``a * b`` builds); scalar times tensor allowed."""
    fs = [as_expr(f) for f in factors]
    flat: list[Expr] = []
    for f in fs:
        flat.extend(f.args if isinstance(f, Mul) else [f])
    coef: complex = 1
    others: list[Expr] = []
    zero = False
    for f in flat:
        if isinstance(f, Constant):
            if f.is_zero():
                zero = True
            else:
                coef *= f.value
        else:
            others.append(f)
    tensors = [f for f in flat if f.rank > 0]
    if len(tensors) > 1:
        from ..errors import TensorError
        raise TensorError("product of two tensors is ambiguous: use dot() or outer()",
                          fix="dot(u, v) (or u @ v) for the contraction, outer(u, v) for the "
                              "tensor product", doc="docs/site/guide/operators.md")
    sig = tensors[0].tensorsig if tensors else ()
    if zero or coef == 0:
        return Zero(sig)
    scalars = sorted((f for f in others if f.rank == 0), key=lambda e: e.structural_hash)
    tens = [f for f in others if f.rank > 0]
    items: list[Expr] = ([Constant(coef)] if coef != 1 else []) + scalars + tens
    if not items:
        return Constant(coef)
    if len(items) == 1:
        return items[0]
    return Mul(items)


def neg(x: Any) -> Expr:
    """Negation, ``-x``."""
    return mul(-1, x)


# --------------------------------------------------------------------------- tensor products


def _check_cs(a: Expr, b: Expr) -> None:
    if a.tensorsig[-1] != b.tensorsig[0]:
        raise TypeError("contracted indices live on different coordinate systems")


class Dot(Expr):
    """Contraction of the last index of a with the first index of b."""

    _prec = 90

    def __init__(self, a: Expr, b: Expr) -> None:
        _check_cs(a, b)
        self._setup((a, b), a.tensorsig[:-1] + b.tensorsig[1:], _union_domains((a, b)),
                    _result_dtype((a, b)))

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return dot(*args)

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        return _combine_degrees(self.args[0]._degrees(fields), self.args[1]._degrees(fields))

    def _lin(self, around: dict, direction: dict) -> Expr:
        a, b = self.args
        return add(dot(a._lin(around, direction), b.substitute(around)),
                   dot(a.substitute(around), b._lin(around, direction)))

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        a, b = self.args
        U = frozenset({u})
        a_dep = a._degrees(U) != frozenset({0})
        b_dep = b._degrees(U) != frozenset({0})
        if a_dep == b_dep:
            raise EquationError("transpose: dot product is not linear in the field",
                                fix="transpose a dot product that is linear in the field (linearize first)")
        if b_dep:  # dot(c, x): (c.x)_J = c_i x_iJ  ->  x : (c outer w)
            c, x = a, b
            if c.rank != 1:
                raise NotImplementedError("transpose of dot(c, x) with rank(c) > 1")
            return x._tr(u, outer(c, w))
        c, x = b, a  # dot(x, c)
        if c.rank == 1:  # (x.c)_I = x_Ii c_i -> x : (w outer c)
            return x._tr(u, outer(w, c))
        if c.rank == 2 and x.rank == 1:  # (x.C)_j = x_i C_ij -> x . (C w)
            return x._tr(u, dot(c, w))
        raise NotImplementedError("transpose of dot(x, C) for this rank combination")

    def __str__(self) -> str:
        return f"dot({self.args[0]}, {self.args[1]})"


def dot(a: Any, b: Any) -> Expr:
    """Contraction of the last index of ``a`` with the first index of ``b`` (rank >= 1)."""
    a, b = as_expr(a), as_expr(b)
    if a.rank == 0 or b.rank == 0:
        from ..errors import TensorError
        raise TensorError("dot needs two tensors of rank >= 1 (use * for scalars)",
                          fix="a * b for a scalar factor; dot(u, v) / u @ v for two vectors",
                          doc="docs/site/guide/operators.md")
    _check_cs(a, b)
    if a.is_zero() or b.is_zero():
        return Zero(a.tensorsig[:-1] + b.tensorsig[1:])
    return Dot(a, b)


class Cross(Expr):
    """Cross product a x b of two 3D vectors (right-handed frame of the coordinate system; in
    spherical coordinates the frame (e_r, e_theta, e_phi); in Cartesian coordinates named x, y, z
    the physical frame e_x x e_y = e_z in any order of declaration, ``cartesian_cross_table``)."""

    _prec = 90

    def __init__(self, a: Expr, b: Expr) -> None:
        self._setup((a, b), (a.tensorsig[0],), _union_domains((a, b)), _result_dtype((a, b)))

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return cross(*args)

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        return _combine_degrees(self.args[0]._degrees(fields), self.args[1]._degrees(fields))

    def _lin(self, around: dict, direction: dict) -> Expr:
        a, b = self.args
        return add(cross(a._lin(around, direction), b.substitute(around)),
                   cross(a.substitute(around), b._lin(around, direction)))

    def __str__(self) -> str:
        return f"cross({self.args[0]}, {self.args[1]})"


def cartesian_cross_table(coordsys) -> dict[tuple[int, int], tuple[int, float]]:
    """(i, j) -> (k, sign) with e_i x e_j = sign e_k for the unit vectors of a 3D Cartesian
    system, indices in the order of the system (feat2). Coordinates named x, y, z, declared in
    any order: the physical right-handed frame (e_x x e_y = e_z whatever the order of the axes);
    other names: the order of declaration is the right-handed frame (e_0 x e_1 = e_2)."""
    from ..errors import TensorError
    names = [c.name for c in coordsys.coords]
    if len(names) != 3:
        raise TensorError("cross product: 3D vectors only",
                          fix="in 2D write the components (e.g. the curl of a stream function)",
                          doc="docs/site/guide/operators.md")
    if getattr(coordsys, "curvilinear", False):
        # the declaration order of a curvilinear frame is not its right-handed order (cylindrical
        # (phi, r, z): e_phi x e_r = -e_z); the spherical evaluator has its own spin-basis table
        #: refused here rather than signed by the declaration order
        raise TensorError(f"cross product: no Cartesian frame table for {coordsys.name} "
                          "coordinates", why="this table orders a Cartesian frame",
                          fix="write the components of the cross product explicitly",
                          doc="docs/site/guide/operators.md")
    sign = 1.0
    if sorted(names) == ["x", "y", "z"]:
        pos = ["xyz".index(n) for n in names]
        inv = sum(1 for a in range(3) for b in range(a + 1, 3) if pos[a] > pos[b])
        sign = -1.0 if inv % 2 else 1.0
    out = {}
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        out[(i, j)] = (k, sign)
        out[(j, i)] = (k, -sign)
    return out


def cross(a: Any, b: Any) -> Expr:
    """a x b for two vectors of the same 3D coordinate system."""
    a, b = as_expr(a), as_expr(b)
    if a.rank != 1 or b.rank != 1:
        raise TypeError("cross needs two vectors (rank 1)")
    _check_cs(a, b)
    if a.tensorsig[0].dim != 3:
        raise TypeError("cross product: 3D vectors only (2D: use skew conventions of curl)")
    if a.is_zero() or b.is_zero():
        return Zero(a.tensorsig)
    return Cross(a, b)


class Outer(Expr):
    """Tensor (outer) product a ⊗ b."""

    _prec = 90

    def __init__(self, a: Expr, b: Expr) -> None:
        self._setup((a, b), a.tensorsig + b.tensorsig, _union_domains((a, b)),
                    _result_dtype((a, b)))

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return outer(*args)

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        return _combine_degrees(self.args[0]._degrees(fields), self.args[1]._degrees(fields))

    def _lin(self, around: dict, direction: dict) -> Expr:
        a, b = self.args
        return add(outer(a._lin(around, direction), b.substitute(around)),
                   outer(a.substitute(around), b._lin(around, direction)))

    def __str__(self) -> str:
        return f"outer({self.args[0]}, {self.args[1]})"


def outer(a: Any, b: Any) -> Expr:
    """Outer (tensor) product; a plain product when one factor is a scalar."""
    a, b = as_expr(a), as_expr(b)
    if a.rank == 0 or b.rank == 0:
        return mul(a, b)
    if a.is_zero() or b.is_zero():
        return Zero(a.tensorsig + b.tensorsig)
    return Outer(a, b)


def pair(a: Any, b: Any) -> Expr:
    """Full contraction a : b of two tensors of equal rank (a*b for scalars)."""
    a, b = as_expr(a), as_expr(b)
    if a.tensorsig != b.tensorsig:
        raise TypeError("pair needs equal tensor signatures")
    if a.rank == 0:
        return mul(a, b)
    if a.rank == 1:
        return dot(a, b)
    dim = a.tensorsig[0].dim
    return add(*(pair(comp(a, i), comp(b, i)) for i in range(dim)))


class Component(Expr):
    """Component i of the first tensor index (Cartesian; in polar/cylindrical frames the physical
    component along e_phi, e_r (, e_z), in the coordinate order)."""

    def __init__(self, operand: Expr, index: int) -> None:
        self._setup((operand,), operand.tensorsig[1:], operand.domain, operand.dtype,
                    params=(index,), index=index)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return comp(args[0], self.index)

    def __str__(self) -> str:
        return f"{self._wrap(self.args[0])}[{self.index}]"

    def _tree_label(self) -> str:
        return str(self.index)


def comp(x: Any, index: int) -> Expr:
    """Component ``index`` of a tensor along its first index."""
    x = as_expr(x)
    if x.rank == 0:
        from ..errors import TensorError
        raise TensorError("cannot take a component of a scalar",
                          fix="comp(u, i) / u @ ex applies to a vector or tensor", doc="docs/site/guide/operators.md")
    if not 0 <= index < x.tensorsig[0].dim:
        raise IndexError(f"component {index} out of range for dim {x.tensorsig[0].dim}")
    if x.is_zero():
        return Zero(x.tensorsig[1:])
    return Component(x, index)


class Power(Expr):
    """Scalar power base**exponent (numeric exponent)."""

    _prec = 30

    def __init__(self, base: Expr, exponent: float) -> None:
        self._setup((base,), (), base.domain, base.dtype, params=_num_key(exponent),
                    exponent=exponent)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return power(args[0], self.exponent)

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        d = self.args[0]._degrees(fields)
        if d == frozenset({0}):
            return d
        e = self.exponent
        if isinstance(e, numbers.Integral) or (isinstance(e, float) and e.is_integer()):
            e = int(e)
            if e >= 0:
                out = frozenset({0})
                for _ in range(e):
                    out = _combine_degrees(out, d)
                return out
        return frozenset({NONPOLY})

    def _lin(self, around: dict, direction: dict) -> Expr:
        b = self.args[0]
        return mul(self.exponent, power(b.substitute(around), self.exponent - 1),
                   b._lin(around, direction))

    def __str__(self) -> str:
        return f"{self._wrap(self.args[0])}**{_num_str(self.exponent)}"


def power(base: Any, exponent: Any) -> Expr:
    """``base ** exponent`` for scalars."""
    base = as_expr(base)
    if base.rank != 0:
        from ..errors import TensorError
        raise TensorError("power is defined for scalars only",
                          fix="u @ u for the squared norm of a vector, then ** of that scalar",
                          doc="docs/site/guide/operators.md")
    if not isinstance(exponent, numbers.Number):
        from ..errors import InputTypeError
        raise InputTypeError(f"exponent must be a number, got {exponent!r}",
                             fix="u ** 2, u ** 0.5; a field exponent is exp(log(u) * v)", doc="docs/site/guide/operators.md")
    if exponent == 0:
        return Constant(1)
    if exponent == 1:
        return base
    if isinstance(base, Constant):
        if base.is_zero() and exponent < 0:
            raise ZeroDivisionError("0 ** negative")
        return Constant(base.value ** exponent)
    if isinstance(base, Power):
        return power(base.args[0], base.exponent * exponent)
    return Power(base, exponent)


class Conj(Expr):
    """Complex conjugate (R-linear, C-antilinear)."""

    def __init__(self, operand: Expr) -> None:
        self._setup((operand,), operand.tensorsig, operand.domain, operand.dtype)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return conj(args[0])

    def __str__(self) -> str:
        return f"conj({self.args[0]})"

    def _notes(self) -> list[str]:
        return ["conjugation is antilinear over C: 'linear' here means R-linear"]


def conj(x: Any) -> Expr:
    """Complex conjugate (the identity on real expressions)."""
    x = as_expr(x)
    if x.dtype.kind != "c":
        return x
    if isinstance(x, Conj):
        return x.args[0]
    if isinstance(x, Constant):
        return Constant(np.conj(x.value))
    return Conj(x)


# --------------------------------------------------------------------------- pointwise functions

# name -> (numpy function, parity kind, f(0) or None if undefined, derivative builder f'(u))
#   parity kind: "odd" (f(-u) = -f(u)), "even" (f(-u) = f(u)), "none" (no parity: an odd
#   argument is refused, e.g. exp(u) for u odd has no parity under the reflection).
def _d_exp(u: Expr) -> Expr:
    return exp(u)


def _d_log(u: Expr) -> Expr:
    return power(u, -1)


def _d_sin(u: Expr) -> Expr:
    return cos(u)


def _d_cos(u: Expr) -> Expr:
    return neg(sin(u))


def _d_tanh(u: Expr) -> Expr:
    return add(1, neg(power(tanh(u), 2)))


def _d_sqrt(u: Expr) -> Expr:
    return mul(0.5, power(u, -0.5))


def _d_abs(u: Expr) -> Expr:
    return sign(u)


_FUNCS: dict[str, tuple] = {
    "exp": (np.exp, "none", 1.0, _d_exp),
    "log": (np.log, "none", None, _d_log),
    "sin": (np.sin, "odd", 0.0, _d_sin),
    "cos": (np.cos, "even", 1.0, _d_cos),
    "tanh": (np.tanh, "odd", 0.0, _d_tanh),
    "sqrt": (np.sqrt, "none", 0.0, _d_sqrt),
    "abs": (np.abs, "even", 0.0, _d_abs),
    "sign": (np.sign, "odd", 0.0, None),   # derivative 0 a.e., undefined at 0: not linearizable
}
_BUILTIN_FUNCS = frozenset(_FUNCS)
#: Functions whose derivative is not continuous: refused by the tangent and adjoint maps
#:; sign() is also refused by linearization.
NONDIFFERENTIABLE = ("abs", "sign")
NONDIFFERENTIABLE_FIX = ("replace it by a smooth approximation whose derivative exists everywhere, "
                         "e.g. abs(T) -> sqrt(T**2 + eps**2) or sign(T) -> tanh(T/eps) with eps a "
                         "declared parameter (the model then means that); or leave the term out of "
                         "the model you differentiate (adjoint of the remaining dynamics)")
#: Name of the marker left by the linearization of abs() about a zero base (refused by
#: Expr.linearize when it survives the simplification; review of).
ABS_AT_ZERO = "abs'(0)"
_USER_FUNCS: dict[str, Any] = {}       # name -> the user's callable (pointwise())


def pointwise(name: str, f: Any, *, derivative: Any = None, parity: str = "none",
              value_at_zero: float | None = None):
    """Register a user pointwise scalar function f(u) and return its builder ``g`` (``g(u)`` is an
    IR node like ``sin(u)``): the role of Dedalus' ``GeneralFunction`` for the common case of a
    pointwise function of one scalar field.

    ``f`` maps an array of grid values to an array of the same shape (NumPy on CPU; on the GPU
    backend it receives CuPy arrays and must accept them); ``derivative`` (optional) is f' in the
    same form: with it, linearization, tangent and Newton work exactly (f'(u) v), without it they
    are refused. ``parity``: "odd", "even" or "none" (used by the symmetry checks). Like the
    built-in functions the node is evaluated on the dealiased grid and is not alias-free
    (declared by explain()); non-finite values are refused at evaluation. A name means one thing:
    registering a built-in name, or the same name with another function, is refused."""
    import re
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise OptionError(f"pointwise({name!r}): the name must be an identifier",
                          fix="a Python identifier, e.g. pointwise('softplus', f)")
    if name in _BUILTIN_FUNCS:
        raise OptionError(f"pointwise({name!r}): {name} is a built-in function",
                          fix="another name: built-in functions are already available")
    if not callable(f) or (derivative is not None and not callable(derivative)):
        raise TypeError(f"pointwise({name!r}): f and derivative must be callables of arrays")
    if parity not in ("odd", "even", "none"):
        raise OptionError(f"pointwise({name!r}, parity={parity!r}): 'odd', 'even' or 'none'",
                          fix="parity='odd', 'even' or 'none'")
    old = _USER_FUNCS.get(name)
    if old is not None and old != (f, derivative, parity):
        raise OptionError(f"pointwise({name!r}): already registered with another function "
                          "(a name means one thing; choose another name)",
                          fix="register the function once, or use another name")
    dname = f"{name}_prime"
    deriv_rule = None
    if derivative is not None:
        # the derivative node "<name>_prime" used to overwrite a
        # user function registered under that name, silently changing what it computes
        oldd = _USER_FUNCS.get(dname)
        if dname in _BUILTIN_FUNCS or (oldd is not None and oldd != (derivative, None, "none")):
            raise OptionError(f"pointwise({name!r}, derivative=...): the name {dname!r} of its "
                              "derivative already means another function (choose another name)",
                              fix="name the derivative function with an identifier that is not built in")
        _FUNCS[dname] = (derivative, "none", None, None)
        _USER_FUNCS[dname] = (derivative, None, "none")

        def deriv_rule(u: Expr) -> Expr:
            return _unary(dname, u)
    _FUNCS[name] = (f, parity, value_at_zero, deriv_rule)
    _USER_FUNCS[name] = (f, derivative, parity)

    def builder(x: Any) -> Expr:
        return _unary(name, x)
    builder.__name__ = name
    builder.__doc__ = f"User pointwise function {name}(u) (conatus.ir.pointwise)."
    return builder


def is_user_function(name: str) -> bool:
    """True for a function registered by :func:`pointwise` (not a NumPy built-in)."""
    return name in _USER_FUNCS


class UnaryFunction(Expr):
    """Pointwise scalar function f(u) (exp, log, sin, cos, tanh, sqrt, abs, sign).

    Non-polynomial: a pseudospectral product rule no longer holds, so dealiasing (3/2 or any
    factor) is **not exact** for these nodes; the aliasing error is not zero and this is stated
    by ``explain()`` and by the problem/methods reports, never hidden. Linearization is exact
    (f'(u) v); a formal transpose is refused (the node is nonlinear); non-finite values at
    evaluation are refused by the evaluators (e.g. log of a non-positive value).
    """

    _prec = 100

    def __init__(self, name: str, operand: Expr) -> None:
        if name not in _FUNCS:
            raise OptionError(f"unknown pointwise function '{name}' (known: {sorted(_FUNCS)})",
                              fix="register it first with pointwise(name, f), or use a known name")
        if operand.rank != 0:
            raise TypeError(f"{name}() is defined for scalars only")
        if name in ("abs", "sign") and operand.dtype.kind == "c":
            raise TypeError(f"{name}() of a complex quantity is not holomorphic: refused")
        self._setup((operand,), (), operand.domain, operand.dtype, params=(name,), func=name)

    @property
    def numpy_func(self):
        return _FUNCS[self.func][0]

    @property
    def parity_kind(self) -> str:
        return _FUNCS[self.func][1]

    @property
    def value_at_zero(self) -> float | None:
        return _FUNCS[self.func][2]

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return _unary(self.func, args[0])

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        d = self.args[0]._degrees(fields)
        return d if d == frozenset({0}) else frozenset({NONPOLY})

    def _lin(self, around: dict, direction: dict) -> Expr:
        u = self.args[0]
        du = u._lin(around, direction)
        if du.is_zero():
            return Zero()
        deriv = _FUNCS[self.func][3]
        if deriv is None:
            # typed refusals; UnsupportedError is still a
            # NotImplementedError, the historical type
            from ..errors import UnsupportedError
            if self.func in _USER_FUNCS:
                raise UnsupportedError(f"{self.func}() was registered without derivative=: "
                                       "linearization refused",
                                       fix="pass derivative=f' to cn.pointwise(...)")
            raise UnsupportedError(f"{self.func}({u}) is not differentiable: linearization refused",
                                   why=f"the derivative of {self.func} is 0 away from 0 and "
                                       "undefined (a jump) at 0: the linearized operator does "
                                       "not exist where the argument changes sign",
                                   fix=NONDIFFERENTIABLE_FIX)
        base = u.substitute(around)
        if self.func == "abs" and base.is_zero():
            # abs at a zero base: no derivative there (|du| is not linear in du). sign(0) = 0
            # would drop the term silently (review of): a marker instead, which survives
            # only if no other factor vanishes with the base (|u| u at u = 0 is differentiable,
            # its marker is multiplied by 0); Expr.linearize refuses a result that still holds it
            return mul(Parameter(ABS_AT_ZERO), du)
        return mul(deriv(base), du)

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        raise NotImplementedError(f"{self.func}() is nonlinear: no transpose (linearize first)")

    def __str__(self) -> str:
        return f"{self.func}({self.args[0]})"

    def _tree_label(self) -> str:
        return self.func

    @property
    def on_grid(self) -> bool:
        """True if the argument varies in space (fields or coordinates): only then is the function
        evaluated on a grid, with an aliasing error. f(parameters) is a number, evaluated exactly."""
        a = self.args[0]
        return a.domain is not None or bool(a.free_fields())

    def _notes(self) -> list[str]:
        if not self.on_grid:
            return [f"{self.func}() of parameters/constants only: a number, evaluated exactly "
                    "(no grid, no aliasing)"]
        return [f"{self.func}() is non-polynomial: pseudospectral evaluation is not alias-free "
                "for any dealias factor (aliasing error nonzero, reported, not hidden)"]


def _unary(name: str, x: Any) -> Expr:
    x = as_expr(x)
    if isinstance(x, Constant):
        if x.rank:
            raise TypeError(f"{name}() is defined for scalars only")
        v = _FUNCS[name][0](x.value)
        if not np.all(np.isfinite(v)):
            raise OptionError(f"{name}({x.value}) is not finite",
                              fix="check the argument range of the function (values are never clipped)")
        return Constant(v)
    return UnaryFunction(name, x)


def exp(x: Any) -> Expr:
    """Pointwise exponential (non-polynomial: evaluated on the grid, aliasing reported)."""
    return _unary("exp", x)


def log(x: Any) -> Expr:
    """Pointwise natural logarithm (non-polynomial: evaluated on the grid, aliasing reported)."""
    return _unary("log", x)


def sin(x: Any) -> Expr:
    """Pointwise sine (non-polynomial: evaluated on the grid, aliasing reported)."""
    return _unary("sin", x)


def cos(x: Any) -> Expr:
    """Pointwise cosine (non-polynomial: evaluated on the grid, aliasing reported)."""
    return _unary("cos", x)


def tanh(x: Any) -> Expr:
    """Pointwise hyperbolic tangent (non-polynomial: evaluated on the grid, aliasing reported)."""
    return _unary("tanh", x)


def sqrt(x: Any) -> Expr:
    """Pointwise square root (non-polynomial: evaluated on the grid, aliasing reported)."""
    return _unary("sqrt", x)


def absolute(x: Any) -> Expr:
    """Pointwise absolute value (non-polynomial: evaluated on the grid, aliasing reported)."""
    return _unary("abs", x)


def sign(x: Any) -> Expr:
    """Pointwise sign (non-polynomial: evaluated on the grid, aliasing reported)."""
    return _unary("sign", x)


def has_nonpolynomial(e: Expr) -> list[str]:
    """Names of the non-polynomial pointwise functions of grid data in ``e`` (aliasing reports).

    A function of parameters only (cos(omega t), sqrt(1 - q^2)) is a number and is not listed: it
    was, and every such problem was wrongly declared "NOT alias-free" (gallery, BACKLOG wave 1)."""
    return sorted({x.func for x in e.walk() if isinstance(x, UnaryFunction) and x.on_grid})
