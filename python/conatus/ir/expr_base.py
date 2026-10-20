"""Expression tree, part 1: the ``Expr`` base class, leaves and the transpose result.

Structural hashing, immutability, traversal, degrees / differential orders, linearization and
formal-transpose protocols, display (``explain`` / ``inspect`` / ``tree``); the leaves ``Field``,
``UnitVector``, ``Parameter``, ``Constant``; ``TransposeResult``. Public face: ``conatus.ir.expr``
(module docstring there). The arithmetic builders used by the operators of ``Expr`` (``add``,
``mul``, ...) live in ``expr_algebra`` and are imported at the bottom (circular by design: the
base class needs them only at call time).
"""
from __future__ import annotations

import hashlib
import math
import numbers
from typing import Any
from collections.abc import Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..errors import CoordinateError, EquationError, OptionError, SymmetryError
from .coords import Coordinate, CoordinateSystem
from .domain import Domain
from .symmetry import ParityLookup, Reflection, _normalize_field_symmetry, component_parity, parity_of

TensorSig = tuple  # tuple[CoordinateSystem, ...]
NONPOLY = 10 ** 6  # degree marker for non-polynomial dependence (never linear)


def _sig_key(sig: TensorSig) -> tuple:
    return tuple(cs.key for cs in sig)


def _sig_str(sig: TensorSig) -> str:
    if not sig:
        return "scalar"
    names = {"1": "vector"}.get(str(len(sig)), f"rank-{len(sig)} tensor")
    return f"{names} on {sig[0].name}"


def _num_key(v: complex) -> tuple:
    c = complex(v)
    return (float(c.real), float(c.imag))


def _num_str(v: complex) -> str:
    c = complex(v)
    if c.imag == 0:
        r = c.real
        if not np.isfinite(r):              # traces at infinity (mapped bases, ir/maps.py)
            return repr(r)
        return str(int(r)) if r == int(r) and abs(r) < 1e15 else repr(r)
    return repr(c)


def _combine_degrees(a: frozenset, b: frozenset) -> frozenset:
    return frozenset(min(x + y, NONPOLY) for x in a for y in b)


# --------------------------------------------------------------------------- base


def _canonical_key(key: Any, renamed: Mapping[str, str]) -> Any:
    """``key`` with the registry keys of renamed user functions replaced by their names."""
    if not isinstance(key, tuple):
        return key
    if (len(key) == 4 and key[0] == "UnaryFunction" and isinstance(key[1], tuple) and key[1]
            and key[1][0] in renamed):
        return (key[0], (renamed[key[1][0]],) + key[1][1:], key[2],
                tuple(_canonical_key(a, renamed) for a in key[3]))
    return tuple(_canonical_key(k, renamed) for k in key)


class Expr:
    """Base class of all IR nodes. Instances are immutable."""

    args: tuple[Expr, ...]
    tensorsig: TensorSig
    domain: Domain | None
    dtype: np.dtype

    def _setup(self, args: Sequence[Expr], tensorsig: TensorSig, domain: Domain | None,
               dtype: Any, params: tuple = (), **attrs: Any) -> None:
        s = object.__setattr__
        s(self, "args", tuple(args))
        s(self, "tensorsig", tuple(tensorsig))
        s(self, "domain", domain)
        s(self, "dtype", np.dtype(dtype))
        for k, v in attrs.items():
            s(self, k, v)
        key = (type(self).__name__, params, _sig_key(self.tensorsig),
               tuple(a._key for a in self.args))
        s(self, "_key", key)
        s(self, "_hash", hash(key))

    # immutability ------------------------------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    # structural identity -----------------------------------------------------
    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expr) and self._hash == other._hash and self._key == other._key

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @property
    def structural_hash(self) -> str:
        """Process-independent hash (sha256 of the structural key). A user function enters by
        the user's name, not by its registry key (``g__2`` when another model of the process
        declared its own ``g``, F105): the hash, the order of terms sorted by it and the identity
        of a computation do not depend on what the process declared before (numerics review);
        the function's code enters the identity separately (``io._identity.user_functions``)."""
        from .expr_algebra import _RENAMED
        key = _canonical_key(self._key, _RENAMED) if _RENAMED else self._key
        return hashlib.sha256(repr(key).encode()).hexdigest()[:16]

    @property
    def rank(self) -> int:
        return len(self.tensorsig)

    def is_zero(self) -> bool:
        return False

    # arithmetic --------------------------------------------------------------
    def __add__(self, other: Any) -> Expr:
        return add(self, other)

    def __radd__(self, other: Any) -> Expr:
        return add(other, self)

    def __sub__(self, other: Any) -> Expr:
        return add(self, neg(as_expr(other)))

    def __rsub__(self, other: Any) -> Expr:
        return add(other, neg(self))

    def __mul__(self, other: Any) -> Expr:
        return mul(self, other)

    def __rmul__(self, other: Any) -> Expr:
        return mul(other, self)

    def __truediv__(self, other: Any) -> Expr:
        other = as_expr(other)
        if isinstance(other, Constant):
            if other.is_zero():
                raise ZeroDivisionError("division by the constant 0")
            return mul(Constant(1 / other.value), self)
        return mul(self, power(other, -1))

    def __rtruediv__(self, other: Any) -> Expr:
        return mul(as_expr(other), power(self, -1))

    def __neg__(self) -> Expr:
        return neg(self)

    def __pow__(self, exponent: Any) -> Expr:
        return power(self, exponent)

    def __abs__(self) -> Expr:
        return absolute(self)

    def __matmul__(self, other: Any) -> Expr:
        return dot(self, other)

    # numpy functions of expressions: np.sin(t) calls t.sin() on an object operand, so a
    # native boundary value lambda t: np.sin(2 * t) builds the IR expression sin(2 t)
    def sin(self) -> Expr:
        from .expr_algebra import sin
        return sin(self)

    def cos(self) -> Expr:
        from .expr_algebra import cos
        return cos(self)

    def exp(self) -> Expr:
        from .expr_algebra import exp
        return exp(self)

    def log(self) -> Expr:
        from .expr_algebra import log
        return log(self)

    def tanh(self) -> Expr:
        from .expr_algebra import tanh
        return tanh(self)

    def sqrt(self) -> Expr:
        from .expr_algebra import sqrt
        return sqrt(self)

    # traversal ---------------------------------------------------------------
    def walk(self) -> Iterator[Expr]:
        """Pre-order traversal of unique sub-expressions."""
        seen: set = set()
        stack = [self]
        while stack:
            e = stack.pop()
            if e in seen:
                continue
            seen.add(e)
            yield e
            stack.extend(reversed(e.args))

    def free_fields(self) -> frozenset[Field]:
        return frozenset(e for e in self.walk() if isinstance(e, Field))

    def free_parameters(self) -> frozenset[Parameter]:
        return frozenset(e for e in self.walk() if isinstance(e, Parameter))

    def depends_on(self, fields: Iterable[Field]) -> bool:
        fs = frozenset(fields)
        return bool(self.free_fields() & fs)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """Rebuild this node with new children, through the canonicalising builders."""
        raise NotImplementedError(type(self).__name__)

    def substitute(self, mapping: Mapping[Expr, Any]) -> Expr:
        if self in mapping:
            return as_expr(mapping[self])
        if not self.args:
            return self
        new = [a.substitute(mapping) for a in self.args]
        if all(n is a for n, a in zip(new, self.args, strict=True)):
            return self
        return self.rebuild(new)

    # algebraic structure -----------------------------------------------------
    def degrees(self, fields: Iterable[Field]) -> frozenset[int]:
        """Set of polynomial degrees of the monomials in the given fields."""
        return self._degrees(frozenset(fields))

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        if not self.args:
            return frozenset({0})
        return frozenset().union(*(a._degrees(fields) for a in self.args))

    def is_linear_in(self, fields: Iterable[Field]) -> bool:
        """True iff every monomial has degree exactly 1 in ``fields`` (homogeneous linear)."""
        return self.degrees(fields) == frozenset({1})

    def order_in(self, coord: Coordinate, fields: Iterable[Field] | None = None) -> int:
        """Differential order along ``coord`` (restricted to terms depending on ``fields``)."""
        fs = None if fields is None else frozenset(fields)
        return self._order(coord, fs)

    def _order(self, coord: Coordinate, fields: frozenset | None) -> int:
        if fields is not None and not (self.free_fields() & fields):
            return 0
        return max((a._order(coord, fields) for a in self.args), default=0) + self._own_order(coord)

    def _own_order(self, coord: Coordinate) -> int:
        return 0

    def time_order(self) -> int:
        return max((a.time_order() for a in self.args), default=0)

    # differentiation ---------------------------------------------------------
    def linearize(self, around: Mapping[Field, Any] | None,
                  direction: Mapping[Field, Any]) -> Expr:
        """Fréchet derivative at ``around`` (field -> base state) in ``direction``.

        Fields absent from ``direction`` are held fixed; fields absent from
        ``around`` are evaluated at themselves (symbolic base state).
        """
        around = {f: as_expr(v) for f, v in (around or {}).items()}
        direction = {f: as_expr(v) for f, v in direction.items()}
        for f, v in list(around.items()) + list(direction.items()):
            if not isinstance(f, Field):
                raise TypeError(f"linearize keys must be Fields, got {f!r}")
            if v.tensorsig != f.tensorsig:
                raise TypeError(f"tensor signature mismatch for {f.name}")
        out = self._lin(around, direction)
        from .expr_algebra import ABS_AT_ZERO
        if any(isinstance(x, Parameter) and x.name == ABS_AT_ZERO for x in out.walk()):
            from ..errors import UnsupportedError
            from .expr_algebra import NONDIFFERENTIABLE_FIX
            zero = ", ".join(f.name for f, v in around.items() if v.is_zero())
            raise UnsupportedError(
                f"linearization of {self}: abs() of a quantity whose base state is zero"
                + (f" ({zero} = 0)" if zero else ""),
                why="abs has no derivative at 0 (a corner): its change is |du|, not linear in "
                    "du, so the linearized operator does not exist there (sign(0) = 0 would "
                    "drop the term silently)",
                fix="linearize about a base where the argument of abs is non-zero, or "
                    + NONDIFFERENTIABLE_FIX)
        return out

    def _lin(self, around: dict, direction: dict) -> Expr:
        # default for operators linear in their single argument
        if len(self.args) != 1:
            raise NotImplementedError(f"linearize not implemented for {type(self).__name__}")
        return self.rebuild([self.args[0]._lin(around, direction)])

    def transpose(self, u: Field, w: Expr) -> TransposeResult:
        """Formal transpose of the map u -> self, tested against ``w``.

        Returns A^T w and the boundary term B(u, w) such that
        ∫ (A u)·w dV = ∫ u·(A^T w) dV + B(u, w)  (bilinear, no conjugation;
        the adjoint for an inner product W is W^{-1} conj(A^T) W).
        """
        w = as_expr(w)
        if w.tensorsig != self.tensorsig:
            raise TypeError("test function must have the tensor signature of the expression")
        if self.degrees({u}) != frozenset({1}):
            raise EquationError(f"{self} is not linear (homogeneous) in {u.name}; "
                                "transpose is only defined for linear maps",
                                fix="transpose a linear (homogeneous) operator: linearize the expression "
                                    "about a state first")
        adj, bt = self._tr(u, w)
        return TransposeResult(self, u, w, adj, bt)

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        raise NotImplementedError(
            f"transpose of {type(self).__name__} is not implemented")

    # symmetry ----------------------------------------------------------------
    def parity_under(self, reflection: Reflection, lookup: ParityLookup | None = None):
        """Tensor parity under a reflection (None = identically zero). See ir/symmetry.py."""
        return parity_of(self, reflection, lookup)

    # display -----------------------------------------------------------------
    _prec = 100  # printing precedence (higher binds tighter)

    def _wrap(self, child: Expr) -> str:
        s = str(child)
        return f"({s})" if child._prec < self._prec else s

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self}>"

    def _notes(self) -> list[str]:
        return []

    def explain(self) -> str:
        fields = sorted(f.name for f in self.free_fields())
        params = sorted(p.name for p in self.free_parameters())
        lines = [
            f"Expression : {self}",
            f"  node     : {type(self).__name__}",
            f"  tensor   : {_sig_str(self.tensorsig)}",
            f"  domain   : {self.domain!r}",
            f"  dtype    : {self.dtype}",
            f"  fields   : {', '.join(fields) or '-'}",
            f"  params   : {', '.join(params) or '-'}",
        ]
        coords = self._all_coords()
        if coords:
            orders = ", ".join(f"{c.name}:{self.order_in(c)}" for c in coords)
            lines.append(f"  diff. order: {orders}; time order: {self.time_order()}")
        lines.append(f"  hash     : {self.structural_hash}")
        lines += [f"  note     : {n}" for n in self._notes()]
        return "\n".join(lines)

    def _all_coords(self) -> tuple[Coordinate, ...]:
        cs = None
        for e in self.walk():
            if e.domain is not None:
                cs = e.domain.coordsys
                break
        return cs.coords if cs is not None else ()

    def inspect(self) -> dict:
        return {
            "node": type(self).__name__,
            "expr": str(self),
            "tensor": _sig_str(self.tensorsig),
            "domain": repr(self.domain),
            "dtype": str(self.dtype),
            "hash": self.structural_hash,
            "args": [a.inspect() for a in self.args],
        }

    def tree(self, indent: int = 0) -> str:
        pad = "  " * indent
        head = f"{pad}{type(self).__name__}"
        label = self._tree_label()
        if label:
            head += f" [{label}]"
        return "\n".join([head] + [a.tree(indent + 1) for a in self.args])

    def _tree_label(self) -> str:
        return ""


# --------------------------------------------------------------------------- leaves


class Field(Expr):
    """A named unknown / coefficient field on a domain. Carries no data (see reference.FieldData)."""

    def __init__(self, name: str, domain: Domain, tensorsig: TensorSig | CoordinateSystem = (),
                 dtype: Any = np.float64, symmetry: Mapping | None = None) -> None:
        if not name or not isinstance(name, str):
            # ValueError kept (v0.1 type of this refusal); the message says what to write
            raise OptionError(f"field name must be a non-empty string, got {name!r}: Field('u', domain)",
                              fix="Field('u', domain)")
        if not isinstance(domain, Domain):
            from ..errors import InputTypeError
            raise InputTypeError(f"Field domain must be a Domain, got {type(domain).__name__}",
                                 fix="Field('u', Domain(cs, [basis_x, basis_z]))",
                                 doc="docs/site/guide/operators.md")
        if isinstance(tensorsig, CoordinateSystem):
            tensorsig = (tensorsig,)
        for cs in tensorsig:
            if cs != domain.coordsys:
                raise CoordinateError("tensor indices must live on the domain coordinate system",
                                      fix="build the tensor signature from the domain's own coordinate "
                                          "system (domain.coordsys)")
        # a field is real (float64) or complex (complex128): an integer or a single-precision
        # dtype was stored and every transform then cast it (audit inputs)
        from ..errors import DtypeError
        try:
            dt_ = np.dtype(dtype)
        except TypeError:
            dt_ = None
        if dt_ is None or dt_ not in (np.dtype(np.float64), np.dtype(np.complex128)):
            raise DtypeError(f"Field({name!r}, ..., dtype={dtype!r}): a field is float64 or complex128",
                             fix="dtype=float (default) or dtype=complex")
        sym = _normalize_field_symmetry(symmetry)
        for R, _ in sym:
            if R.coord.system != domain.coordsys:
                raise SymmetryError(f"{R!r} is not on the coordinate system of field {name}",
                                    fix="declare symmetries with generators of the field's coordinate system")
            if domain.has(R.coord):
                R.centre(domain.basis(R.coord))
        self._setup((), tensorsig, domain, dtype,
                    params=(name, domain.key, np.dtype(dtype).str,
                            tuple((R.key, int(p)) for R, p in sym)),
                    name=name, symmetry=sym)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return self

    def _degrees(self, fields: frozenset) -> frozenset[int]:
        return frozenset({1 if self in fields else 0})

    def _lin(self, around: dict, direction: dict) -> Expr:
        return direction.get(self, Zero(self.tensorsig))

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        if self != u:
            raise ValueError(f"unexpected field {self.name} in transpose w.r.t. {u.name}")
        return w, Zero(())

    def __str__(self) -> str:
        return self.name

    def _tree_label(self) -> str:
        return f"{self.name}: {_sig_str(self.tensorsig)}"

    def _notes(self) -> list[str]:
        out = ["a Field is a symbol; numerical data lives outside the IR (FieldData, explicit layout)"]
        for R, p in self.symmetry:
            line = f"symmetry: {p} under {R!r}"
            if self.rank == 1:
                comps = ", ".join(
                    f"{self.name}_{c.name} {component_parity(p, self.tensorsig, (i,), R)}"
                    for i, c in enumerate(self.tensorsig[0].coords))
                line += f" -> components: {comps}"
            out.append(line)
        return out


def VectorField(name: str, domain: Domain, dtype: Any = np.float64,
                symmetry: Mapping | None = None) -> Field:
    """Vector field: a Field with one index in the domain's coordinate system."""
    return Field(name, domain, (domain.coordsys,), dtype, symmetry)


def TensorField(name: str, domain: Domain, rank: int, dtype: Any = np.float64,
                symmetry: Mapping | None = None) -> Field:
    """Tensor field of the given rank on the domain's coordinate system."""
    from ..errors import check_count
    check_count(rank, "rank", f"TensorField({name!r}, ...)", minimum=0)   # audit inputs
    return Field(name, domain, (domain.coordsys,) * rank, dtype, symmetry)


class UnitVector(Expr):
    """Constant unit vector e_i of a Cartesian coordinate system."""

    def __init__(self, coordsys: CoordinateSystem, index: int) -> None:
        if coordsys.curvilinear and not (coordsys.kind == "Cylindrical" and index == 2):
            raise NotImplementedError("unit vectors of curvilinear frames vary in space (only e_z of "
                                      "cylindrical coordinates is constant)")
        self._setup((), (coordsys,), None, np.float64, params=(index,),
                    coordsys=coordsys, index=index)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return self

    def _lin(self, around: dict, direction: dict) -> Expr:
        return Zero(self.tensorsig)

    def __str__(self) -> str:
        return f"e{self.coordsys.coords[self.index].name}"


def unit_vector(coord: Coordinate) -> UnitVector:
    """Constant unit vector e_coord of a Cartesian (or the z of a cylindrical) system."""
    return UnitVector(coord.system, coord.index)


class Parameter(Expr):
    """Named scalar parameter (Re, Ra, eigenvalue...). Its value is not part of the hash."""

    def __init__(self, name: str, value: Any = None, dtype: Any = np.float64) -> None:
        # Name and value checked at declaration (audit inputs): Parameter("Re", nan),
        # Parameter("Re", "1") and the problem presets built on them (cn.problems.channel_evp(
        # Re=nan)) were accepted and failed, or gave NaN, deep inside the assembly.
        from ..errors import DtypeError, InputTypeError
        if not isinstance(name, str) or not name:
            raise InputTypeError(f"Parameter({name!r}, ...): the name is a non-empty string",
                                 fix="Parameter('Re', 1000.0)")
        try:
            kind = np.dtype(dtype).kind
        except TypeError as err:
            raise InputTypeError(f"Parameter({name!r}, dtype={dtype!r}): not a dtype ({err})",
                                 fix="dtype=float (default) or dtype=complex") from None
        if kind not in "fc":
            raise DtypeError(f"Parameter({name!r}, dtype={dtype!r}): a parameter is real (float) "
                             "or complex", fix="dtype=float (default) or dtype=complex")
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Number):
                raise InputTypeError(f"Parameter({name!r}, {value!r}): the value is a number",
                                     fix=f"Parameter({name!r}, 1.0)")
            c = complex(value)
            if not (math.isfinite(c.real) and math.isfinite(c.imag)):
                raise InputTypeError(f"Parameter({name!r}, {value!r}): the value is not finite",
                                     why="a NaN or infinite parameter makes every matrix and "
                                         "result NaN or meaningless",
                                     fix=f"a finite number for {name}")
        # The declared dtype enters the hash and explain(): a complex value must be declared.
        if (value is not None and np.iscomplexobj(value) and np.imag(value) != 0
                and kind != "c"):
            raise DtypeError(f"Parameter '{name}' has complex value {value!r} but real dtype "
                             f"{np.dtype(dtype)}; declare dtype=complex",
                             fix=f"Parameter({name!r}, {value!r}, dtype=complex)")
        self._setup((), (), None, dtype, params=(name, np.dtype(dtype).str), name=name, value=value)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return self

    def _lin(self, around: dict, direction: dict) -> Expr:
        return Zero(())

    def __str__(self) -> str:
        return self.name

    def _tree_label(self) -> str:
        return f"{self.name}={self.value}"


class Constant(Expr):
    """Numeric constant. Non-scalar constants may only be zero."""

    def __init__(self, value: Any, tensorsig: TensorSig = ()) -> None:
        from ..errors import InputTypeError
        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise InputTypeError(f"Constant needs a number, got {value!r}")
        c = complex(value)
        if not (math.isfinite(c.real) and math.isfinite(c.imag)):
            # audit inputs: as_expr(nan) and power(u, nan) built NaN terms
            raise InputTypeError(f"Constant({value!r}): not a finite number",
                                 why="a NaN or infinite coefficient makes every result NaN",
                                 fix="a finite number")
        if tensorsig and c != 0:
            raise EquationError("only the zero constant may carry a tensor signature")
        dtype = np.complex128 if c.imag != 0 else np.float64
        v = c if c.imag != 0 else c.real
        self._setup((), tensorsig, None, dtype, params=_num_key(c), value=v)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        return self

    def is_zero(self) -> bool:
        return self.value == 0

    def _lin(self, around: dict, direction: dict) -> Expr:
        return Zero(self.tensorsig)

    def __str__(self) -> str:
        return _num_str(self.value)

    def _tree_label(self) -> str:
        return str(self)


def Zero(tensorsig: TensorSig = ()) -> Constant:
    """The zero constant with tensor signature ``tensorsig`` (scalar by default)."""
    return Constant(0, tuple(tensorsig))


def as_expr(x: Any) -> Expr:
    """Wrap a number (or pass an expression through) as an IR expression."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, numbers.Number) and not isinstance(x, bool):
        return Constant(x)
    from ..errors import InputTypeError
    raise InputTypeError(f"cannot convert {x!r} to an expression",
                         fix="use numbers, Fields, Parameters and the operators of conatus.expert "
                             "(an array of values is data: pass it as a field's ncc= data)", doc="docs/site/guide/operators.md")


def _union_domains(exprs: Iterable[Expr]) -> Domain | None:
    d = None
    for e in exprs:
        d = Domain.union(d, e.domain)
    return d


def _result_dtype(exprs: Iterable[Expr]) -> np.dtype:
    return np.result_type(*[e.dtype for e in exprs]) if exprs else np.dtype(np.float64)


# --------------------------------------------------------------------------- transpose result


class TransposeResult:
    """Result of a formal transpose: ∫(A u)·w = ∫ u·(A^T w) + B(u, w)."""

    def __init__(self, op: Expr, u: Field, w: Expr, adjoint: Expr, boundary: Expr) -> None:
        self.op, self.u, self.w = op, u, w
        self.adjoint = adjoint
        self.boundary = boundary

    def assumptions(self) -> list[str]:
        coords = self.op._all_coords()
        order = max((self.op.order_in(c) for c in coords), default=0)
        dom = Domain.union(self.op.domain, self.w.domain)
        periodic = [c.name for c in dom.separable_coords] if dom else []
        out = [
            f"{self.u.name} and {self.w} regular enough for integration by parts "
            f"(e.g. H^{order} on the domain)",
            "Cartesian coordinates, Lebesgue measure, bilinear pairing (no conjugation)",
        ]
        if periodic:
            out.append(f"periodicity in {', '.join(periodic)}: boundary terms vanish there")
        return out

    def proof_obligation(self) -> str:
        """Statement to be discharged by the formal layer (track F)."""
        return "\n".join(
            ["Assumptions:"] + [f"  - {a}" for a in self.assumptions()] + [
                "Goal:",
                f"  integ(({self.op}) . {self.w}) = integ({self.u} . ({self.adjoint})) "
                f"+ [{self.boundary}]",
            ])

    def explain(self) -> str:
        return "\n".join([
            "Formal transpose (integration by parts, boundary terms explicit)",
            f"  A u      : {self.op}",
            f"  test w   : {self.w}",
            f"  A^T w    : {self.adjoint}",
            f"  B(u, w)  : {self.boundary}",
            "  identity : ∫ (A u)·w dV = ∫ u·(A^T w) dV + B(u, w)",
            "  status   : derived symbolically by IBP rules; numerically checked in tests; "
            "not formally proven (see proof_obligation())",
            "  warning  : transpose ≠ adjoint. The adjoint for an inner product W is "
            "W^-1 conj(A^T) W (conatus.linearize.InnerProduct).",
        ])


# Builders used by Expr's operators at call time (expr_algebra imports this module first).
from .expr_algebra import absolute, add, dot, mul, neg, power  # noqa: E402
