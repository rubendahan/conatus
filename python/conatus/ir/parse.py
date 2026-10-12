"""String equations: ``"dt(u) - nu*lap(u) + grad(p) = -u@grad(u)"`` -> the IR (v0.1 UX).

The Python API stays the reference: a string is parsed into exactly the expression the same
Python code would build (``add``/``mul``/``dot``/... canonical forms), never into anything else.
Nothing is evaluated with ``eval``: the text is tokenized and parsed by the recursive-descent
parser below, whose grammar is declared in ``GRAMMAR``; every name is looked up in a declared
``Namespace`` (the problem's variables, the user's parameters and numbers, the coordinates of
their domains) or in the fixed table of operators ``OPERATORS``. Calling a Python function is
possible only if the user put that function in the namespace (e.g. Dedalus-style
``lift = lambda A: ...``, translated by ``conatus.translate``).

Errors are :class:`conatus.errors.EquationSyntaxError` (or ``UndeclaredVariableError`` for an
unknown name) with the text, a caret at the offending column and a suggestion (never applied).

Round trip: ``to_string(eq)`` is ``str(eq)`` checked to be parseable (identifier names, no name
that collides with an operator), and ``parse_equation(to_string(eq), ns) == eq`` structurally
for every equation of the examples and tutorials (tests/parse/).
"""
from __future__ import annotations

import numbers
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from ..errors import (EquationSyntaxError, InputTypeError, UndeclaredVariableError, UnsupportedError,
                      did_you_mean)
from .bases import Basis
from .coords import Coordinate, CoordinateSystem
from .expr import (Expr, Field, Parameter, absolute, as_expr, comp, conj, cos, cross, curl, diff, div, dot,
                   dt, exp, grad, integ, interp, lap, lift, log, outer, power, sign, sin, sqrt, tanh,
                   unit_vector)

__all__ = ["GRAMMAR", "OPERATORS", "Namespace", "parse_equation", "parse_expr", "to_string",
           "roundtrip"]

GRAMMAR = r"""
equation  := expr "=" expr
expr      := term (("+" | "-") term)*
term      := unary (("*" | "/" | "@") unary)*          # "@" = dot (contraction)
unary     := ("-" | "+") unary | power
power     := postfix ("**" unary)?                     # right-associative; exponent: a number
postfix   := atom (call | "[" INTEGER "]")*            # u[0] = component 0
call      := "(" [arg ("," arg)*] ")"
arg       := NAME "=" expr | expr | STRING            # keyword, positional, 'left'/'right'
atom      := NUMBER | NAME | "(" expr ")"
NUMBER    := digits ["." digits] [("e"|"E") ["+"|"-"] digits] ["j"]   (also ".5", "1.")
NAME      := [A-Za-z_][A-Za-z0-9_]*
STRING    := "'" chars "'" | '"' chars '"'

A call on an operator name applies the operator (OPERATORS); a call on a namespace function
calls it; a call with keyword arguments on an expression is a trace: u(z=0) = interp(u, z, 0),
u(z=left) = interp(u, z, 'left'). Names d<coord> (dx, dz, dr...) are partial derivatives along
the declared coordinates; e<coord> (ex, ez) the Cartesian unit vectors. Coordinate arguments
(integ, lift, diff, interp and trace keywords) are resolved among the declared coordinates.
"""

# Operators: name -> (python callable, how its arguments are read). Kinds of arguments:
# "e" expression, "c" coordinate, "b" basis or coordinate (lift), "i" integer, "p" position.
OPERATORS: dict[str, tuple[Callable, str]] = {
    "dt": (dt, "e"), "grad": (grad, "e"), "div": (div, "e"), "lap": (lap, "e"),
    "curl": (curl, "e"), "dot": (dot, "ee"), "cross": (cross, "ee"), "outer": (outer, "ee"),
    "conj": (conj, "e"), "exp": (exp, "e"), "log": (log, "e"), "sin": (sin, "e"),
    "cos": (cos, "e"), "tanh": (tanh, "e"), "sqrt": (sqrt, "e"), "abs": (absolute, "e"),
    "absolute": (absolute, "e"), "sign": (sign, "e"), "comp": (comp, "ei"),
    "diff": (diff, "ec"), "integ": (integ, "ec*"), "interp": (interp, "ecp"),
    "lift": (lift, "ebi"),
}
# Dedalus operator names that Conatus does not have as nodes: refused with the equivalent form.
_DEDALUS_ONLY = {
    "trace": "write div(u) for trace(grad(u)) and dot(a, b) for trace(a*b); in a Dedalus "
             "import, conatus.from_dedalus rewrites trace() of these forms and reports it",
    "skew": "write the rotated components explicitly (2D: skew(u) = -ez x u), or curl(u)",
    "transpose": "no transpose-of-components node yet: write the components explicitly",
    "Average": "use integ(u, x)/Lx (the length is explicit)", "ave": "use integ(u, x)/Lx",
    "Laplacian": "the Conatus name is lap", "Gradient": "the Conatus name is grad",
    "Divergence": "the Conatus name is div", "Curl": "the Conatus name is curl",
    "Integrate": "the Conatus name is integ", "Lift": "the Conatus name is lift",
    "Differentiate": "the Conatus names are diff(u, x) or dx(u)",
    "Interpolate": "write u(z=0) or interp(u, z, 0)",
    "TimeDerivative": "the Conatus name is dt",
    "radial": "no radial-component node in strings yet: use comp(u, index)",
    "angular": "no angular-component node in strings yet: use comp(u, index)",
    "azimuthal": "no azimuthal-component node in strings yet: use comp(u, index)",
}
_CONSTANTS = {"pi": float(np.pi), "inf": float("inf")}
# spatial operators: refused on an expression without any field (_Parser._check_spatial)
_SPATIAL = {"grad", "div", "lap", "curl", "diff"}
_POSITIONS = ("left", "right")
_RESERVED = set(OPERATORS) | set(_CONSTANTS) | set(_POSITIONS)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


# --------------------------------------------------------------------------- namespace


@dataclass
class Namespace:
    """The declared names a string equation may use.

    ``values`` maps names to Fields, Parameters, numbers, coordinates, coordinate systems, bases
    or Python callables (anything else is refused *when used*, with the reason). Coordinates and
    bases are also collected from the domains of every Field in ``values``. A value under an
    operator name shadows the operator only if it *is* that operator or a callable (Dedalus
    style, e.g. a user ``lift``); every shadowing is recorded in ``shadowed``.
    """

    values: dict[str, Any] = dc_field(default_factory=dict)
    tensor_product: str = "refuse"   # "refuse" (Conatus) or "outer" (Dedalus: u*v = outer(u, v))
    shadowed: set[str] = dc_field(default_factory=set)
    notes: list[str] = dc_field(default_factory=list)   # declared rewrites (e.g. u*v -> outer)
    # which layer reads the strings: "expert" (namespace= mapping) or "native" (cn.Model
    # declarations): only the wording of the fixes differs
    layer: str = "expert"

    def __post_init__(self) -> None:
        if self.tensor_product not in ("refuse", "outer"):
            raise InputTypeError(f"tensor_product={self.tensor_product!r}",
                                 fix="tensor_product='refuse' or 'outer'")
        self.values = {k: v for k, v in dict(self.values).items() if isinstance(k, str)}
        self.coords: dict[str, Coordinate] = {}
        self.bases: dict[Coordinate, list[Basis]] = {}
        for v in self.values.values():
            self._collect(v)

    @classmethod
    def of(cls, *sources: Any, tensor_product: str = "refuse", layer: str = "expert",
           **values: Any) -> Namespace:
        """Namespace from mappings (``locals()``), Problems (their variables), iterables of
        Fields/Parameters (by their ``.name``) and keywords."""
        out: dict[str, Any] = {}
        for s in sources:
            if s is None:
                continue
            if isinstance(s, Namespace):
                out.update(s.values)
            elif isinstance(s, Mapping):
                out.update(s)
            elif hasattr(s, "variables") and hasattr(s, "equations"):     # a Problem
                out.update({v.name: v for v in s.variables})
            else:
                for o in s:
                    out[o.name] = o
        out.update(values)
        return cls(out, tensor_product=tensor_product, layer=layer)

    def _add_coord(self, c: Coordinate) -> None:
        old = self.coords.get(c.name)
        if old is not None and old != c:
            # two different coordinates with one name: resolving the name would be a guess
            self.coords[c.name] = None      # type: ignore[assignment]  (marked ambiguous)
        elif old is None and c.name not in self.coords:
            self.coords[c.name] = c

    def _add_basis(self, b: Basis) -> None:
        lst = self.bases.setdefault(b.coord, [])
        if all(b.key != o.key for o in lst):
            lst.append(b)

    def _collect(self, v: Any) -> None:
        if isinstance(v, Field):
            for c in v.domain.coordsys.coords:
                self._add_coord(c)
            for b in v.domain.bases:
                self._add_basis(b)
        elif isinstance(v, Coordinate):
            for c in v.system.coords:
                self._add_coord(c)
        elif isinstance(v, CoordinateSystem):
            for c in v.coords:
                self._add_coord(c)
        elif isinstance(v, Basis):
            self._add_basis(v)
            for c in v.coord.system.coords:
                self._add_coord(c)

    # lookups ---------------------------------------------------------------------------------
    def coordinate(self, name: str) -> Coordinate | None:
        c = self.coords.get(name, False)
        return None if c is False else c    # None = ambiguous or absent (callers tell which)

    def names(self) -> list[str]:
        """Every name a string may use (for suggestions)."""
        derived = [f"d{n}" for n in self.coords] + [f"e{n}" for n in self.coords]
        return sorted(set(self.values) | _RESERVED | set(derived) | set(self.coords))


# --------------------------------------------------------------------------- tokenizer

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?j?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<string>'[^']*'|"[^"]*")
  | (?P<op>\*\*|[-+*/@=(),\[\]])
""", re.VERBOSE)


@dataclass
class _Tok:
    kind: str
    text: str
    pos: int


def _tokenize(text: str) -> list[_Tok]:
    toks, i = [], 0
    while i < len(text):
        m = _TOKEN.match(text, i)
        if m is None:
            ch = text[i]
            hint = {"^": "powers are written '**' (u**2)", "{": "use parentheses", "}": "use "
                    "parentheses", "'": "unterminated string", '"': "unterminated string",
                    "!": "'!=' is not an equation", "<": "inequalities are not equations",
                    ">": "inequalities are not equations",
                    # "remove it" turned u.grad(u) into ugrad(u)
                    ".": "a dot product is written u@grad(u) (or u·grad(u)), a method call is "
                         "not part of an equation; a decimal number has a digit next to its "
                         "point (0.5)"}.get(
                ch, "this character is not part of the grammar: write the term with the "
                    "operators of guide/equations (+ - * / ** @, calls f(u))")
            raise EquationSyntaxError(f"unexpected character {ch!r}", text=text, position=i,
                                      doc="docs/site/guide/equations.md", why="it is not part of the declared grammar "
                                          "(conatus.ir.parse.GRAMMAR)", fix=hint)
        kind = m.lastgroup
        if kind != "ws":
            toks.append(_Tok(kind, m.group(), i))
        i = m.end()
    toks.append(_Tok("end", "", len(text)))
    return toks


# --------------------------------------------------------------------------- parser / evaluator

@dataclass
class _Pos:
    """A value with the position of its first character (for error messages)."""
    value: Any
    pos: int
    text: str = ""          # source text of the sub-expression (for messages)


def _unclosed(text: str) -> int | None:
    """Index of the first '(' or '[' of ``text`` that is never closed, else None."""
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch in "([":
            stack.append(i)
        elif ch in ")]" and stack:
            stack.pop()
    return stack[0] if stack else None


class _Parser:
    """Recursive descent over GRAMMAR; builds IR nodes directly (no intermediate code)."""

    def __init__(self, text: str, ns: Namespace) -> None:
        self.text, self.ns = text, ns
        self.toks = _tokenize(text)
        self.i = 0
        self.used_shadowed: set[str] = set()

    # token helpers ---------------------------------------------------------------------------
    @property
    def tok(self) -> _Tok:
        return self.toks[self.i]

    def _next(self) -> _Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _accept(self, text: str) -> _Tok | None:
        if self.tok.kind == "op" and self.tok.text == text:
            return self._next()
        return None

    def _expect(self, text: str, context: str) -> _Tok:
        t = self._accept(text)
        if t is None:
            got = "end of text" if self.tok.kind == "end" else repr(self.tok.text)
            fix = f"insert {text!r}" if text in ")]" else None
            open_at = _unclosed(self.text)
            if open_at is not None:          # say which bracket is open
                o = self.text[open_at]
                c = ")" if o == "(" else "]"
                start = open_at
                while start > 0 and (self.text[start - 1].isalnum() or self.text[start - 1] == "_"):
                    start -= 1
                fix = (f"close the {o!r} of {self.text[start:open_at + 1]!r} at column "
                       f"{open_at + 1} with {c!r}: every {o!r} needs its {c!r}")
            raise self._err(f"expected {text!r} {context}, found {got}", self.tok.pos, fix=fix)
        return t

    def _err(self, msg: str, pos: int, *, why: str | None = None, fix: str | None = None,
             cls=EquationSyntaxError) -> Exception:
        if cls is EquationSyntaxError:
            return EquationSyntaxError(msg, text=self.text, position=pos, why=why, fix=fix,
                                       doc="docs/site/guide/equations.md")
        caret = f"{self.text}\n{' ' * pos}^ (column {pos + 1})"
        return cls(msg, where=caret, why=why, fix=fix, doc="docs/site/guide/equations.md")

    def _src(self, start: int) -> str:
        return self.text[start:self.tok.pos].strip()

    # grammar ---------------------------------------------------------------------------------
    def equation(self):
        from .problem import Equation
        lhs = self.expr()
        if self.tok.kind == "end":
            raise self._err("an equation needs '=' (write 'lhs = rhs'; '= 0' for a homogeneous "
                            "one)", self.tok.pos, fix=f"{self.text.strip()} = 0")
        if self.tok.kind in ("name", "number") or (self.tok.kind == "op" and self.tok.text == "("):
            raise self._err(f"unexpected {self.tok.text!r}: an operator is missing", self.tok.pos,
                            fix="write '*' for a product (Pr*lap(T), 2*u), '+' or '-' between "
                                "terms")
        eq_tok = self._expect("=", "between the two sides")
        if self._accept("="):
            raise self._err("'==' is not an equation sign", eq_tok.pos, fix="write a single '='")
        rhs = self.expr()
        self._end()
        left, right = self._as_expr(lhs), self._as_expr(rhs)
        try:
            return Equation(left, right)
        except Exception as err:  # noqa: BLE001 -- tensor mismatch: re-raised, located
            raise self._err(f"invalid equation: {err}", eq_tok.pos,
                            why="both sides must have the same tensor signature") from None

    def expression(self) -> _Pos:
        e = self.expr()
        self._end()
        return e

    def _end(self) -> None:
        if self.tok.kind != "end":
            t = self.tok
            fix = None
            if t.text == "=":
                fix = "an equation has one '='; parse it with parse_equation"
            elif t.kind in ("name", "number") or t.text == "(":
                fix = "an operator is missing: write '*' for a product (2*u, not 2u)"
            raise self._err(f"unexpected {t.text!r}", t.pos, fix=fix)

    def expr(self) -> _Pos:
        start = self.tok.pos
        left = self.term()
        while self.tok.kind == "op" and self.tok.text in "+-" and self.tok.text:
            op = self._next()
            right = self.term()
            left = _Pos(self._binary(op, left, right), start)
            left.text = self._src(start)
        return left

    def term(self) -> _Pos:
        start = self.tok.pos
        left = self.unary()
        while self.tok.kind == "op" and self.tok.text in ("*", "/", "@"):
            op = self._next()
            right = self.unary()
            left = _Pos(self._binary(op, left, right), start)
            left.text = self._src(start)
        return left

    def unary(self) -> _Pos:
        t = self.tok
        if t.kind == "op" and t.text in "+-" and t.text:
            self._next()
            v = self.unary()
            if t.text == "-":
                return _Pos(self._apply(lambda a: -a, v, t.pos), t.pos, self._src(t.pos))
            return _Pos(v.value, t.pos, v.text)
        return self.power()

    def power(self) -> _Pos:
        base = self.postfix()
        if self.tok.kind == "op" and self.tok.text == "**":
            op = self._next()
            ex = self.unary()
            if not _is_number(ex.value):
                raise self._err("the exponent of '**' must be a number", ex.pos,
                                why="power() is defined for numeric exponents; a field exponent "
                                    "would be exp(b*log(a))", fix="write exp(b*log(a)) explicitly")
            if _is_number(base.value):
                return _Pos(base.value ** ex.value, base.pos)
            return _Pos(self._apply(lambda a: power(a, _real_if_close(ex.value)), base, op.pos),
                        base.pos)
        return base

    def postfix(self) -> _Pos:
        v = self.atom()
        while True:
            if self.tok.kind == "op" and self.tok.text == "(":
                v = self.call(v)
            elif self.tok.kind == "op" and self.tok.text == "[":
                lb = self._next()
                t = self._next()
                if t.kind != "number" or not re.fullmatch(r"\d+", t.text):
                    raise self._err("a component index is a non-negative integer (u[0])", t.pos)
                self._expect("]", "after the component index")
                v = _Pos(self._apply(lambda a, t=t: comp(a, int(t.text)), v, lb.pos), v.pos)
            else:
                return v

    def atom(self) -> _Pos:
        t = self._next()
        if t.kind == "number":
            s = t.text
            if s.endswith("j"):
                return _Pos(complex(s), t.pos, s)
            return _Pos(int(s) if re.fullmatch(r"\d+", s) else float(s), t.pos, s)
        if t.kind == "name":
            return _Pos(_NameRef(t.text), t.pos, t.text)
        if t.kind == "op" and t.text == "(":
            e = self.expr()
            self._expect(")", "to close the parenthesis")
            return _Pos(e.value, t.pos, e.text)
        if t.kind == "string":
            raise self._err("a string is only allowed as a trace position ('left'/'right')",
                            t.pos, fix="remove the quotes")
        got = "end of text" if t.kind == "end" else repr(t.text)
        raise self._err(f"expected a number, a name or '(', found {got}", t.pos,
                        fix="an operand is missing" if t.kind != "end" else
                        "the expression is incomplete")

    # calls -----------------------------------------------------------------------------------
    def _args(self) -> list[tuple[str | None, _Pos, _Tok]]:
        args = []
        if self._accept(")"):
            return args
        while True:
            t = self.tok
            if (t.kind == "name" and self.toks[self.i + 1].kind == "op"
                    and self.toks[self.i + 1].text == "=" and not (
                        self.toks[self.i + 2].kind == "op" and self.toks[self.i + 2].text == "=")):
                self.i += 2
                args.append((t.text, self._arg_value(), t))
            else:
                args.append((None, self._arg_value(), t))
            if self._accept(")"):
                return args
            self._expect(",", "between arguments (or ')' to close the call)")

    def _arg_value(self) -> _Pos:
        t = self.tok
        if t.kind == "string":
            self._next()
            return _Pos(_Str(t.text[1:-1]), t.pos, t.text)
        return self.expr()

    def call(self, f: _Pos) -> _Pos:
        lp = self._next()
        args = self._args()
        ref = f.value
        if isinstance(ref, _NameRef):
            name = ref.name
            user = self.ns.values.get(name, _MISSING)
            if name in OPERATORS or name in _DEDALUS_ONLY or _derivative_coord(name, self.ns):
                if user is not _MISSING and user is not OPERATORS.get(name, (None,))[0]:
                    if callable(user) and not isinstance(user, (Expr, Basis)):
                        self.used_shadowed.add(name)
                        self.ns.shadowed.add(name)
                        return self._call_user(name, user, args, f.pos)
                    raise self._err(
                        f"'{name}' is an operator but the namespace defines {name} = "
                        f"{_short(user)}", f.pos,
                        why="a name must mean one thing in an equation",
                        fix=f"rename the namespace entry (e.g. {name}_) or drop it from the "
                            "namespace")
                if name in _DEDALUS_ONLY:
                    raise self._err(f"'{name}' is a Dedalus operator that Conatus strings do "
                                    "not provide", f.pos, cls=UnsupportedError,
                                    fix=_DEDALUS_ONLY[name])
                d = _derivative_coord(name, self.ns)
                if d is not None and name not in OPERATORS:
                    if len(args) != 1 or args[0][0] is not None:
                        raise self._err(f"{name}(u) takes one expression", lp.pos)
                    self._check_spatial(name, args[0][1])
                    return _Pos(self._apply(lambda a: diff(a, d), args[0][1], f.pos), f.pos)
                return self._call_op(name, args, f.pos, lp)
            if user is not _MISSING:
                if isinstance(user, (Expr,)) or _is_number(user):
                    return self._trace(_Pos(self._value(f), f.pos, name), args, lp)
                if callable(user) and not isinstance(user, (Basis, Coordinate, CoordinateSystem)):
                    return self._call_user(name, user, args, f.pos)
            # unknown name or unusable value: _value raises the right error
            return self._trace(_Pos(self._value(f), f.pos, name), args, lp)
        return self._trace(f, args, lp)

    def _call_user(self, name: str, fn: Callable, args, pos: int) -> _Pos:
        pos_args, kw = [], {}
        for key, a, _ in args:
            val = self._value(a)
            if key is None:
                pos_args.append(val)
            else:
                kw[key] = val
        try:
            return _Pos(fn(*pos_args, **kw), pos)
        except Exception as err:  # noqa: BLE001 -- re-raised below, located
            raise self._err(f"{name}(...) failed: {type(err).__name__}: {err}", pos,
                            why=f"'{name}' is a function of the namespace, called with the "
                                "parsed arguments") from None

    def _call_op(self, name: str, args, pos: int, lp: _Tok) -> _Pos:
        fn, kinds = OPERATORS[name]
        pos_args = [a for a in args if a[0] is None]
        kw = {k: a for k, a, _ in args if k is not None}
        allowed_kw = {"lift": {"lam"}}.get(name, set())
        bad = set(kw) - allowed_kw
        if bad:
            k = sorted(bad)[0]
            t = next(t for kk, _, t in args if kk == k)
            raise self._err(f"{name}() has no keyword argument '{k}'", t.pos,
                            fix=("keywords accepted: " + ", ".join(sorted(allowed_kw)))
                            if allowed_kw else f"pass the arguments of {name} by position")
        star = kinds.endswith("*")
        base = kinds.rstrip("*")
        nmin = 1 if star else len(base)
        if not (nmin <= len(pos_args) <= (len(pos_args) if star else len(base))):
            raise self._err(f"{name}() takes {_signature(name)}, got {len(pos_args)} "
                            "positional argument(s)", lp.pos, fix=f"write {_signature(name)}")
        if name in _SPATIAL and pos_args:
            self._check_spatial(name, pos_args[0][1])
        vals = []
        for j, (_, a, _t) in enumerate(pos_args):
            kind = base[min(j, len(base) - 1)]
            vals.append(self._read(kind, a, name))
        kwv = {k: self._read("i", a, name) for k, a in kw.items()}
        if name == "integ" and len(vals) > 1:
            vals = [vals[0], tuple(vals[1:])]
        if name == "lift" and isinstance(vals[1], _LiftBasis):
            lb = vals[1]
            vals[1] = lb.basis
            if lb.lam and "lam" in kwv and kwv["lam"] != lb.lam:
                raise self._err("lift: two derivative levels given", pos)
            kwv.setdefault("lam", lb.lam)
        try:
            return _Pos(fn(*vals, **kwv), pos)
        except Exception as err:  # noqa: BLE001 -- re-raised below, located
            raise self._err(f"{name}(...): {type(err).__name__}: {err}", pos,
                            why=f"the operator {name} refused these arguments") from None

    def _trace(self, f: _Pos, args, lp: _Tok) -> _Pos:
        """``u(z=0)``: trace along each keyword coordinate (Dedalus syntax, and str() of the IR)."""
        e = self._value(f)
        if not args or any(k is None for k, _, _ in args):
            raise self._err(f"'{f.text or 'this expression'}' is not a function", lp.pos,
                            why="a call on a field or an expression is a trace and takes "
                                "keyword arguments only: u(z=0), u(z=left)",
                            fix="write u(z=value); for a product write '*' (a*(b), not a(b))")
        for key, a, t in args:
            c = self._coord(key, t.pos)
            p = self._read("p", a, "trace")
            try:
                e = interp(e, c, p)
            except Exception as err:  # noqa: BLE001 -- re-raised below, located
                raise self._err(f"trace {key}={a.text}: {err}", t.pos) from None
        return _Pos(e, f.pos, f.text)

    def _check_spatial(self, name: str, a: _Pos) -> None:
        """A spatial derivative of an expression without any field (``lap(t)``, ``dx(Ra)``,
        ``dz(2)``) is identically zero: refused with a did-you-mean of the fields (``lap(t)`` for ``lap(T)`` silently removed the diffusion term)."""
        v = self._value(a)
        if isinstance(v, Expr) and v.free_fields():
            return
        if not (isinstance(v, Expr) or _is_number(v)):
            return                                   # not an expression: the operator refuses
        used = _names_used(a.text or "")
        if not a.text or any(callable(self.ns.values.get(n)) and not isinstance(
                self.ns.values.get(n), Expr) for n in used):
            # a namespace function may return a constant on purpose (the native layer re-parses
            # with its explicit()/implicit() markers set to 0): only what is written is judged
            return
        fields = sorted(k for k, f in self.ns.values.items() if isinstance(f, Field))
        used = [n for n in used if n not in fields]
        what = []
        for n in used:
            val = self.ns.values.get(n)
            if n == "t" and isinstance(val, Parameter) and self.ns.layer == "native":
                what.append("t is the time")
            elif isinstance(val, Parameter):
                what.append(f"{n} is a parameter")
        hint = "".join(did_you_mean(n, fields) for n in used)
        arg = a.text or "this expression"
        raise self._err(
            f"{name}({arg}) is identically zero: '{arg}' contains no field"
            + (f" ({', '.join(what)})" if what else "") + "." + hint, a.pos,
            why="a spatial derivative of a number, a parameter or the time vanishes; no "
                "equation is meant to contain one, and the term would disappear without a word",
            fix=("write the field you mean" + (f" ({', '.join(fields)})" if fields else "")
                 + "; a constant term needs no derivative"))

    # value resolution ------------------------------------------------------------------------
    def _coord(self, name: str, pos: int) -> Coordinate:
        c = self.ns.coordinate(name)
        if c is None:
            if name in self.ns.coords:
                raise self._err(f"coordinate name '{name}' is ambiguous (two coordinate "
                                "systems use it)", pos, fix="rename one of the coordinates")
            raise self._err(f"'{name}' is not a declared coordinate", pos,
                            cls=UndeclaredVariableError,
                            why="coordinates are read from the domains of the namespace fields",
                            fix=("declared coordinates: " + ", ".join(sorted(self.ns.coords)))
                            + did_you_mean(name, self.ns.coords))
        return c

    def _read(self, kind: str, a: _Pos, opname: str) -> Any:
        v = a.value
        if kind == "e":
            return self._value(a)
        if kind in ("c", "b"):
            if isinstance(v, _NameRef):
                user = self.ns.values.get(v.name, _MISSING)
                if kind == "b" and isinstance(user, (Basis, _LiftBasis)):
                    return user
                if isinstance(user, Coordinate):
                    c = user
                else:
                    c = self._coord(v.name, a.pos)
                if kind == "c":
                    return c
                return self._basis_on(c, a.pos)
            if kind == "b" and isinstance(v, (Basis, _LiftBasis)):
                return v
            if isinstance(v, Coordinate):
                return v if kind == "c" else self._basis_on(v, a.pos)
            raise self._err(f"{opname}: expected a coordinate name here", a.pos)
        if kind == "i":
            val = self._value(a)
            if not (_is_number(val) and float(np.real(val)) == int(np.real(val))
                    and np.imag(val) == 0):
                raise self._err(f"{opname}: expected an integer", a.pos)
            return int(np.real(val))
        if kind == "p":
            if isinstance(v, _Str) or (isinstance(v, _NameRef) and v.name in _POSITIONS
                                       and v.name not in self.ns.values):
                s = v.s if isinstance(v, _Str) else v.name
                if s not in _POSITIONS:
                    raise self._err(f"trace position {s!r}: 'left', 'right' or a number",
                                    a.pos, fix=did_you_mean(s, _POSITIONS).strip() or None)
                return s
            val = self._value(a)
            if not _is_number(val) or np.imag(val) != 0:
                raise self._err("a trace position is a real number, 'left' or 'right'", a.pos)
            return float(np.real(val))
        raise AssertionError(kind)

    def _basis_on(self, c: Coordinate, pos: int) -> Basis:
        bs = self.ns.bases.get(c, [])
        if len(bs) != 1:
            listing = ", ".join(repr(b) for b in bs) or "none"
            raise self._err(f"lift along {c.name}: {len(bs)} basis/bases declared on "
                            f"{c.name} ({listing})", pos,
                            why="the lift basis fixes where the tau term lives; with several "
                                "candidates it is never guessed",
                            fix="put the basis in the namespace under a name and write "
                                "lift(tau, that_name, -1)")
        return bs[0]

    def _value(self, a: _Pos) -> Any:
        """Resolve a name (and leave numbers/expressions as they are)."""
        v = a.value
        if not isinstance(v, _NameRef):
            if isinstance(v, _Str):
                raise self._err("a string is only allowed as a trace position ('left'/'right')",
                                a.pos)
            return v
        name = v.name
        if name in self.ns.values:
            val = self.ns.values[name]
            if isinstance(val, (Expr,)) or _is_number(val):
                return _num(val)
            if isinstance(val, (Coordinate, CoordinateSystem, Basis, _LiftBasis)):
                raise self._err(f"'{name}' is a {type(val).__name__}, not a quantity", a.pos,
                                fix="coordinates/bases are arguments of integ, lift, diff, "
                                    "interp; a coordinate as a function is a Field (NCC)")
            what = _short(val)
            fix = None
            if isinstance(val, np.ndarray):
                fix = ("grid arrays are not symbols: declare a Field for this coefficient and "
                       "give its values at discretization (ncc=...)")
            elif callable(val):
                fix = f"call it: {name}(...)"
            raise self._err(f"'{name}' is {what}, which is not an expression", a.pos,
                            cls=InputTypeError, fix=fix)
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        if name.startswith("e") and self.ns.coordinate(name[1:]) is not None:
            c = self.ns.coordinate(name[1:])
            try:
                return unit_vector(c)
            except Exception as err:  # noqa: BLE001 -- re-raised below, located
                raise self._err(f"{name}: {err}", a.pos, cls=UnsupportedError) from None
        if name in OPERATORS or name in _DEDALUS_ONLY or _derivative_coord(name, self.ns):
            raise self._err(f"'{name}' is an operator: call it, {name}(...)", a.pos)
        if name in self.ns.coords:
            raise self._err(f"'{name}' is a coordinate, not a quantity", a.pos,
                            fix=f"to use the coordinate {name} as a coefficient, declare a Field "
                                "(non-constant coefficient) and give its values at "
                                "discretization; coordinates are otherwise arguments of "
                                "integ, lift, diff and traces")
        if self.ns.layer == "native":
            own = [k for k in self.ns.values if not callable(self.ns.values[k])
                   or isinstance(self.ns.values[k], Expr)]
            raise self._err(
                f"name '{name}' is not declared." + (did_you_mean(name, own)
                                                     # a one- or two-letter name is no typo
                                                     # of an operator ('s' -> 'sin'; critique
                                                     #)
                                                     or ("" if len(name) <= 2 else
                                                         did_you_mean(name, self.ns.names()))),
                a.pos,
                cls=UndeclaredVariableError,
                why="the equations of a model see its declarations only: unknowns, given "
                    "fields, parameters, the eigenvalue, operators of m.define, the time t",
                fix=(f"declare it: a parameter cn.Model(geometry, ..., {name}=value), a given "
                     f"field m.given({name}=lambda z: ...), an unknown m.scalar({name!r}), or "
                     f"the eigenvalue of an eigenvalue problem, m.eigenvalue({name!r}, "
                     f"growth=...) before m.equations"))
        raise self._err(f"name '{name}' is not declared", a.pos, cls=UndeclaredVariableError,
                        why="string equations only see the declared namespace (problem "
                            "variables, the namespace= mapping, operators)",
                        fix=(f"add {name} to the namespace (namespace=dict(..., {name}={name})"
                             " or namespace=locals())") + did_you_mean(name, self.ns.names()))

    # arithmetic --------------------------------------------------------------------------------
    def _as_expr(self, a: _Pos) -> Expr:
        v = self._value(a)
        try:
            return as_expr(v)
        except Exception:  # noqa: BLE001 -- re-raised below, located
            raise self._err(f"'{a.text}' is not an expression", a.pos) from None

    def _apply(self, fn, a: _Pos, pos: int):
        v = self._value(a)
        try:
            return fn(v)
        except Exception as err:  # noqa: BLE001 -- re-raised below, located
            raise self._err(f"{type(err).__name__}: {err}", pos) from None

    def _binary(self, op: _Tok, a: _Pos, b: _Pos):
        x, y = self._value(a), self._value(b)
        o = op.text
        try:
            if o == "+":
                return x + y
            if o == "-":
                return x - y
            if o == "/":
                if _is_number(y) and y == 0:
                    raise ZeroDivisionError("division by zero")
                return x / y
            if o == "@":
                return dot(x, y)
            # "*": the Conatus Python rule, or the Dedalus tensor product when declared
            if (isinstance(x, Expr) and isinstance(y, Expr) and x.rank > 0 and y.rank > 0):
                if self.ns.tensor_product == "outer":
                    self.ns.notes.append(f"'{a.text}*{b.text}' of two tensors read as "
                                         f"outer({a.text}, {b.text}) (Dedalus convention)")
                    return outer(x, y)
                raise self._err(
                    f"product of two tensors '{a.text}' * '{b.text}' is ambiguous", op.pos,
                    cls=EquationSyntaxError,
                    why="both factors have tensor indices: '*' could mean a contraction or a "
                        "tensor product",
                    fix=f"write {a.text}@{b.text} (= dot) or outer({a.text}, {b.text})")
            return x * y
        except EquationSyntaxError:
            raise
        except Exception as err:  # noqa: BLE001 -- re-raised below, located
            raise self._err(f"'{a.text}' {o} '{b.text}': {type(err).__name__}: {err}",
                            op.pos) from None


# --------------------------------------------------------------------------- helpers

_MISSING = object()


@dataclass(frozen=True)
class _NameRef:
    name: str


@dataclass(frozen=True)
class _Str:
    s: str


@dataclass(frozen=True)
class _LiftBasis:
    """A basis with a derivative level: Dedalus ``basis.derivative_basis(lam)`` (C^(lam))."""
    basis: Basis
    lam: int


def _is_number(v: Any) -> bool:
    return isinstance(v, numbers.Number) and not isinstance(v, bool)


def _num(v: Any) -> Any:
    """numpy scalars -> Python numbers (the IR folds Python numbers)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _real_if_close(v: Any) -> Any:
    return v.real if isinstance(v, complex) and v.imag == 0 else v


def _short(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return f"a numpy array of shape {v.shape}"
    s = repr(v)
    return f"{type(v).__name__} {s[:40]}"


def _derivative_coord(name: str, ns: Namespace) -> Coordinate | None:
    if len(name) > 1 and name.startswith("d") and name not in OPERATORS:
        return ns.coordinate(name[1:])
    return None


def _signature(name: str) -> str:
    return {"dt": "dt(u)", "grad": "grad(u)", "div": "div(u)", "lap": "lap(u)",
            "curl": "curl(u)", "dot": "dot(a, b)", "cross": "cross(a, b)",
            "outer": "outer(a, b)", "comp": "comp(u, index)", "diff": "diff(u, x)",
            "integ": "integ(u) or integ(u, x, ...)", "interp": "interp(u, z, position)",
            "lift": "lift(tau, z, -k) or lift(tau, z, -k, lam=q)"}.get(name, f"{name}(u)")


def _namespace(namespace: Any) -> Namespace:
    if isinstance(namespace, Namespace):
        return namespace
    if namespace is None:
        return Namespace()
    return Namespace.of(namespace)


# --------------------------------------------------------------------------- public API

def parse_equation(text: str, namespace: Any = None):
    """Parse ``"lhs = rhs"`` into an :class:`~conatus.ir.problem.Equation` (no eval).

    ``namespace``: a :class:`Namespace`, a mapping (``locals()``, ``dict(u=u, Re=Re)``), a
    Problem (its variables) or an iterable of Fields/Parameters. Grammar: ``GRAMMAR``.
    """
    if not isinstance(text, str):
        raise InputTypeError(f"parse_equation needs a string, got {type(text).__name__}")
    return _Parser(text, _namespace(namespace)).equation()


def parse_expr(text: str, namespace: Any = None) -> Expr:
    """Parse one expression (no '=') into an IR expression (a number stays a Constant)."""
    p = _Parser(text, _namespace(namespace))
    return p._as_expr(p.expression())


def to_string(obj: Any) -> str:
    """The string form of an Equation or expression that ``parse_equation``/``parse_expr``
    read back into the same IR. Refuses (never mangles) names that are not identifiers or that
    collide with an operator, and nodes without a string form."""
    from .expr import Constant, Interpolate, UnaryFunction
    from .problem import Equation
    exprs = [obj.lhs, obj.rhs] if isinstance(obj, Equation) else [as_expr(obj)]
    for e in exprs:
        for n in e.walk():
            name = getattr(n, "name", None) if isinstance(n, (Field, Parameter)) else None
            if name is not None:
                if not _IDENT.match(name):
                    raise UnsupportedError(f"name {name!r} is not an identifier: it has no "
                                           "string form", fix="rename the field/parameter")
                if name in _RESERVED:
                    raise UnsupportedError(f"name {name!r} collides with an operator in string "
                                           "form", fix=f"rename it (e.g. {name}_)")
            if isinstance(n, Interpolate) and not _IDENT.match(n.coord.name):
                raise UnsupportedError(f"coordinate {n.coord.name!r} is not an identifier")
            if isinstance(n, UnaryFunction) and n.func not in OPERATORS:
                raise UnsupportedError(f"function {n.func} has no string form")
            if isinstance(n, Constant) and n.rank and not n.is_zero():
                raise UnsupportedError("non-zero tensor constant has no string form")
    return str(obj)


def roundtrip(obj: Any, namespace: Any = None) -> tuple[bool, str]:
    """(IR -> string -> IR equals the original, the string). The namespace defaults to every
    field and parameter appearing in ``obj``."""
    from .expr import Lift
    from .problem import Equation
    s = to_string(obj)
    if namespace is None:
        exprs = [obj.lhs, obj.rhs] if isinstance(obj, Equation) else [as_expr(obj)]
        vals: dict[str, Any] = {}
        for e in exprs:
            for f in list(e.free_fields()) + list(e.free_parameters()):
                vals[f.name] = f
        namespace = Namespace(vals)
        for e in exprs:                  # lift bases (the tau fields do not carry them)
            for n in e.walk():
                if isinstance(n, Lift):
                    namespace._add_basis(n.basis)
    back = parse_equation(s, namespace) if isinstance(obj, Equation) else parse_expr(s, namespace)
    if isinstance(obj, Equation):
        return (back.lhs == obj.lhs and back.rhs == obj.rhs), s
    return back == as_expr(obj), s


def _lift_basis(basis: Basis, lam: int) -> _LiftBasis:
    """Derivative basis marker used by the Dedalus translator (``derivative_basis(lam)``)."""
    return _LiftBasis(basis, int(lam))


def _names_used(text: str) -> Iterable[str]:
    return [t.text for t in _tokenize(text) if t.kind == "name"]
