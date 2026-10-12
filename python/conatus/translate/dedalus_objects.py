"""Dedalus translation, part 1: report lines and the stand-ins of Dedalus objects.

``ReportLine`` (one source line, its status and note), the internal refusals (``_Unsupported``,
``_Skip``), the light objects the restricted interpreter builds in place of Dedalus's (distributor,
basis bundles, grids, problem and solver declarations, script lambdas, field data statements), and
small AST / IR helpers. Public face and design: ``conatus.translate.dedalus``.
"""
from __future__ import annotations

import ast
import numbers
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..ir import (Coordinate, CoordinateSystem, Field,
                  Parameter, div)
from ..ir import expr as E
from ..ir.bases import Basis
from ..ir.parse import _LiftBasis, to_string

if TYPE_CHECKING:
    from .dedalus_interpreter import _Interpreter


STATUSES = ("translated", "approximated", "unsupported", "skipped")
# Time steppers with the same name and tableau in Conatus (conatus.timestep.integrator.SCHEMES).
_SCHEMES = ("SBDF1", "SBDF2", "SBDF3", "SBDF4", "RK111", "RK222", "RK443",
            "CNAB1", "CNAB2", "MCNAB2", "CNLF2", "RKSMR", "RKGFY")
_DEDALUS_ONLY_SCHEMES = ("SBDF2_R", "RK443_IMEX")


# --------------------------------------------------------------------------- report


@dataclass
class ReportLine:
    """One source statement: its line number, text, status and the reason."""
    lineno: int
    source: str
    status: str
    note: str = ""

    def __str__(self) -> str:
        src = self.source.strip().splitlines()[0] if self.source.strip() else ""
        if len(src) > 70:
            src = src[:67] + "..."
        return f"{self.lineno:4d}  {self.status:12s} {src}" + (f"\n{'':18s}-> {self.note}"
                                                                if self.note else "")


class _Unsupported(Exception):
    """A construct the translator does not handle (the message is the report note)."""


class _Skip(Exception):
    """Output-only construct (plots, logs, analysis): reported as skipped."""


@dataclass
class _Opaque:
    """Value of an untranslated statement: any use of it is unsupported (or skipped, for an
    output-only value), with the origin."""
    reason: str
    line: int
    skip: bool = False


@dataclass
class _Module:
    name: str            # "d3", "numpy", "scipy.special", "ignored:<name>"


@dataclass
class _Dist:
    coordsys: CoordinateSystem
    dtype: Any


@dataclass
class _Bundle:
    """A Dedalus curvilinear basis (DiskBasis(coords, shape=...)) = several Conatus bases."""
    bases: list
    kind: str
    radial: Basis | None = None

    @property
    def edge(self) -> _Bundle:
        return _Bundle([b for b in self.bases if b is not self.radial], "edge")

    @property
    def radial_basis(self) -> _Bundle:
        return _Bundle([self.radial], "radial", self.radial)

    def derivative_basis(self, lam: int = 1) -> _LiftBasis:
        return _LiftBasis(self.radial, int(lam))


@dataclass
class _Grid:
    """``dist.local_grid(basis)``: the grid of one coordinate (used by initial data only)."""
    coord: Coordinate


@dataclass
class _Scheme:
    name: str


@dataclass
class _ProblemDecl:
    problem: Any
    kind: str
    time: Any = None


@dataclass
class _SolverDecl:
    problem: _ProblemDecl
    scheme: str | None
    line: int


@dataclass
class _Lambda:
    """A script lambda: evaluated by the restricted interpreter at call time (closure = scope)."""
    node: ast.Lambda
    scope: dict
    interp: _Interpreter

    def __call__(self, *args: Any, **kw: Any) -> Any:
        a = self.node.args
        names = [x.arg for x in a.args]
        if len(args) > len(names):
            raise _Unsupported(f"lambda called with {len(args)} arguments")
        local = dict(self.scope)
        defaults = a.defaults
        for j, n in enumerate(names):
            if j < len(args):
                local[n] = args[j]
            elif n in kw:
                local[n] = kw[n]
            else:
                k = j - (len(names) - len(defaults))
                if k < 0:
                    raise _Unsupported(f"lambda argument '{n}' missing")
                local[n] = self.interp.eval(defaults[k], self.scope)
        return self.interp.eval(self.node.body, local)


@dataclass
class _FieldData:
    """Grid data of a field written by the script: a list of (op, component, ast, scope)."""
    ops: list = dc_field(default_factory=list)
    unsupported: str | None = None


# --------------------------------------------------------------------------- IR helpers


def _is_num(v: Any) -> bool:
    return isinstance(v, (numbers.Number, np.number)) and not isinstance(v, bool)


def _trace(T: Any, notes: list[str]) -> Any:
    """Dedalus ``trace`` for the forms that appear in equations; exact identities, declared."""
    T = E.as_expr(T)
    if T.rank != 2:
        raise _Unsupported(f"trace of a rank-{T.rank} tensor")
    if isinstance(T, E.Gradient):
        notes.append(f"trace(grad({T.args[0]})) -> div({T.args[0]}) (exact identity)")
        return div(T.args[0])
    if isinstance(T, E.Add):
        return E.add(*(_trace(t, notes) for t in T.args))
    if isinstance(T, E.Outer):
        notes.append(f"trace(outer({T.args[0]}, {T.args[1]})) -> dot(...) (exact identity)")
        return E.dot(T.args[0], T.args[1])
    if isinstance(T, E.Mul):
        scal = [f for f in T.args if f.rank == 0]
        ten = [f for f in T.args if f.rank > 0]
        return E.mul(*scal, _trace(ten[0], notes))
    raise _Unsupported(f"trace({T}) (only trace of grad, outer products and their sums)")


def _rewrite_div_outer(e: E.Expr, notes: list[str]) -> E.Expr:
    """div(outer(e_c, V)) -> d_c(V) for a constant Cartesian unit vector e_c (exact identity).

    Dedalus first-order formulations write grad_u = grad(u) + ez*lift(tau) and then
    div(grad_u); Conatus discretizes div(grad(u)) but has no matrix for the Outer node."""
    def rw(x: E.Expr) -> E.Expr:
        if not x.args:
            return x
        new = [rw(a) for a in x.args]
        if any(n is not o for n, o in zip(new, x.args, strict=True)):
            x = x.rebuild(new)
        if isinstance(x, E.Divergence):
            a = x.args[0]
            if isinstance(a, E.Add) and any(_outer_unit(t) for t in a.args):
                return E.add(*(rw(div(t)) for t in a.args))
            hit = _outer_unit(a)
            if hit is not None:
                coef, c, V = hit
                notes.append(f"div(outer(e{c.name}, {V})) -> d{c.name}({V}) (exact identity, "
                             "constant unit vector)")
                return E.mul(coef, E.diff(V, c))
        return x
    return rw(e)


def _outer_unit(t: E.Expr):
    """(coefficient, coordinate, V) if t = coef * outer(e_c, V) with e_c a constant unit vector."""
    coef: list = []
    if isinstance(t, E.Mul):
        coef = [f for f in t.args if f.rank == 0]
        ten = [f for f in t.args if f.rank > 0]
        if len(ten) != 1 or any(f.free_fields() for f in coef):
            return None
        t = ten[0]
    if isinstance(t, E.Outer) and isinstance(t.args[0], E.UnitVector):
        uv = t.args[0]
        return (E.mul(*coef) if coef else 1, uv.coordsys.coords[uv.index], t.args[1])
    return None


# --------------------------------------------------------------------------- helpers


def _is_main_guard(test: ast.AST) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__")


def _is_field_data(t: ast.AST) -> bool:
    """u['g'] or u['g'][i]."""
    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) \
            and t.slice.value in ("g", "c"):
        return True
    return (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Subscript)
            and isinstance(t.value.slice, ast.Constant) and t.value.slice.value in ("g", "c"))


def _field_data_target(t: ast.Subscript, tr: _Interpreter, scope: dict):
    comp = None
    if isinstance(t.slice, ast.Constant) and t.slice.value in ("g", "c"):
        layout, base = t.slice.value, t.value
    else:
        comp = tr.eval(t.slice, scope)
        layout, base = t.value.slice.value, t.value.value
    if layout == "c":
        raise _Unsupported("coefficient data (Dedalus coefficient layout differs from Conatus)")
    return tr.eval(base, scope), comp


def _code_of(v: Any, name: str) -> str:
    """Conatus constructor text for a translated value (for to_script)."""
    if isinstance(v, (int, float, complex, np.number)):
        return repr(v.item() if isinstance(v, np.generic) else v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, CoordinateSystem):
        return f"{type(v).__name__}({', '.join(repr(x) for x in v.names)})"
    if isinstance(v, Coordinate):
        return f"{type(v.system).__name__}({', '.join(repr(x) for x in v.system.names)})" \
               f"[{v.name!r}]"
    if isinstance(v, _Dist):
        return f"None  # distributor: dtype {v.dtype}, coordinates {v.coordsys.name}"
    if isinstance(v, Basis):
        return repr(v)
    if isinstance(v, _Bundle):
        return "(" + ", ".join(repr(b) for b in v.bases) + ")  # " + v.kind
    if isinstance(v, _LiftBasis):
        return f"None  # lift basis: C^({v.lam}) of {v.basis!r} (lift(..., lam={v.lam}))"
    if isinstance(v, Field):
        dom = ", ".join(repr(b) for b in v.domain.bases)
        kind = "VectorField" if v.rank == 1 else "Field"
        dt_ = "" if v.dtype == np.float64 else f", dtype={v.dtype.name}"
        cs = v.domain.coordsys
        return f"{kind}({v.name!r}, Domain(cs, [{dom}]){dt_})  # on {cs.name}"
    if isinstance(v, Parameter):
        return f"Parameter({v.name!r}, dtype={v.dtype.name})"
    if isinstance(v, E.Expr):
        return f"parse_expr({to_string(v)!r}, namespace=locals())"
    if isinstance(v, _Lambda):
        return f"lambda ...  # Dedalus helper, inlined in the equations: {ast.unparse(v.node)}"
    if isinstance(v, _ProblemDecl):
        return f"{v.kind}([...])"
    if isinstance(v, _Scheme):
        return repr(v.name)
    if isinstance(v, tuple):
        return "(" + ", ".join(_code_of(x, name) for x in v) + ")"
    return f"None  # {type(v).__name__}"
