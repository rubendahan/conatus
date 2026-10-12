"""Compile a native :class:`~conatus.model.Model` into the IR (the expert layer) at a resolution.

Every automatic choice is made by a fixed rule and recorded (``Compiled.explain()``); anything the
rules cannot decide is refused with the explicit override. Rules (docs/API_NATIVE.md):

* **Split** (IVP): a term linear in the unknowns (constant, parameter or declared ``given``
  coefficients) goes to the implicit side, a nonlinear term or a source to the explicit side;
  ``dt(...)`` must be linear. ``implicit(...)`` / ``explicit(...)`` in the text override (a
  nonlinear term cannot be implicit). EVP: every term linear; LBVP: sources to the right side;
  NLBVP: kept as written (Newton linearizes everything).
* **Taus**: each condition constrains one unknown ``f``; its tau goes to the equation where ``f``
  has its highest order along the wall coordinate (unique, else refused unless ``into=``); taus
  are lifted into the modes N-1, N-2, ... of that equation in declaration order; vector
  conditions give vector taus (componentwise). Curvilinear recipes are those of the
  hand-written problems of the repository (``Geometry.recipe``).
* **Incompressibility**: a scalar unknown entering only through ``grad`` and without condition,
  paired with the one time-algebraic scalar equation holding ``div`` of a vector unknown of the
  same momentum equation, gets a gauge tau in that equation and ``integ(p) = 0``; with walls, the
  divergence tau of the recipe is added (Cartesian: ``dot(n, lift(tau_u_last, -1))``).
"""
from __future__ import annotations

import inspect
import numbers
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from ..errors import (BoundaryConditionError, EquationError, InputTypeError, NonlinearLHSError,
                      UnsupportedError)
from ..ir import (EVP, IVP, LBVP, NLBVP, Add, Constant, Divergence, Domain, Field, Gradient, Mul, Parameter,
                  Power, TensorField, TimeDerivative, UnaryFunction, VectorField, add, as_expr, div, dot, grad, integ, interp, lap, lift, neg,
                  outer, unit_vector)
from ..ir.bases import Hermite, Laguerre
from ..ir.parse import Namespace, parse_equation, parse_expr
from . import sincos as _sincos
from .conditions import (NavierSlip, WallCondition, curved_free_slip_operator, curved_wall_tensors,
                         free_slip_operator, navier_slip_operator, no_slip)
from ..ir.expr_base import NONPOLY
from .geometry import DEFAULT_DEALIAS, WEIGHTED_DEALIAS, Box

__all__ = ["Compiled", "compile_model", "TauRecord", "dealias_factor", "default_dealias_map"]

_DOC = "docs/site/guide/boundary_conditions.md"   # public guide page (never an internal design note)
_EQ_DOC = "docs/site/guide/equations.md"           # the time t in equations (_check_time)
# Unicode accepted in equation text, declared rewrite before parsing (never anything else).
_UNICODE = {"·": "@", "−": "-"}


def _text(s: str) -> str:
    for a, b in _UNICODE.items():
        s = s.replace(a, b)
    return s


def _terms(e):
    return list(e.args) if isinstance(e, Add) else ([] if e.is_zero() else [e])


def _coord_args(fn: Callable, names: tuple[str, ...], what: str) -> tuple[str, ...]:
    """Coordinates a callable depends on, read off its argument names (declared dependence)."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        raise InputTypeError(f"{what}: cannot read the arguments of {fn!r}",
                             fix="use a def or a lambda whose arguments are coordinate names",
                             doc=_DOC) from None
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        raise InputTypeError(f"{what}: *args/**kwargs callables do not declare their coordinates",
                             fix=f"write lambda {', '.join(names)}: ...", doc=_DOC)
    args = tuple(p.name for p in params)
    bad = [a for a in args if a not in names]
    if bad:
        raise InputTypeError(
            f"{what}: argument(s) {bad} are not coordinates of the geometry {names}",
            why="the argument names of a data function declare which coordinates it depends on",
            fix=f"name the arguments after the coordinates, e.g. lambda {', '.join(names)}: ...",
            doc=_DOC)
    return args


def _on_all_grids(fn: Callable, args: tuple[str, ...], names: tuple[str, ...]) -> Callable:
    """The callable in the IR convention (all coordinate grids, in coordinate order)."""
    if args == names:
        return fn
    idx = [names.index(a) for a in args]
    return lambda *grids: fn(*(grids[i] for i in idx))


@dataclass
class TauRecord:
    """One automatic tau (for explain() and the tests)."""
    name: str
    unknown: str
    condition: str
    boundary: str
    target: int
    mode: int              # lift index -k (mode N-k)
    lam: int
    rule: str
    piece: int | None = None   # compound interval: the piece whose last modes receive it


@dataclass
class Compiled:
    """The IR problem of a model at one resolution, with the record of every automatic choice.

    Expert access: ``problem`` (IR Problem), ``ncc`` (data of the given fields), ``fields``
    (name -> IR Field, taus included), ``params`` (name -> Parameter), ``bases``."""

    model: Any
    N: dict
    dealias: dict
    kind: str
    problem: Any
    ncc: dict
    fields: dict
    params: dict
    eigenvalue: Any
    coordsys: Any
    bases: dict
    domain: Any
    labels: list
    split: list = dc_field(default_factory=list)          # per equation: list of (term, side, why)
    taus: list = dc_field(default_factory=list)
    gauge: list = dc_field(default_factory=list)
    notes: list = dc_field(default_factory=list)
    dealias_default: bool = True                        # dealias=None: the degree rule
    degree: int = 1                 # highest polynomial degree of a term in the unknowns
    nonpoly: list = dc_field(default_factory=list)      # terms not polynomial in the unknowns
    overrides: dict = dc_field(default_factory=dict)    # per-call parameter values (simulate(Ra=))
    parity: list = dc_field(default_factory=list)       # sine/cosine axis: parity of each unknown
    symmetry: Any = None                                # its SymmetrySubspace (None otherwise)
    time: Any = None              # the time Parameter "t" when the model uses it, else None
    illposed: list = dc_field(default_factory=list)    # IVP: ill-posed implicit terms (wellposed.py)

    def discretize(self, **kw):
        """``conatus.discretize`` of the compiled problem with the given-field data."""
        from ..discretize import discretize
        return discretize(self.problem, ncc=self.ncc or None, **kw)

    def decisions(self, symbolic: bool = False) -> list[str]:
        """Every automatic decision, one line each (``symbolic``: modes as N_c - k)."""
        m = self.model
        out = [f"Problem kind: {self.kind} ({self._kind_reason()})"]
        head = {"IVP": ("implicit", "explicit"), "EVP": ("operator", "-"),
                "LBVP": ("operator (left)", "data (right)"), "NLBVP": ("as written", "as written")}
        imp, exp = head[self.kind]
        out.append("Term placement" + (" (IMEX split, automatic; override with implicit(...) / "
                                        "explicit(...))" if self.kind == "IVP" else "") + ":")
        for i, rows in enumerate(self.split):
            out.append(f"  [{i}] {self.labels[i]}")
            for term, side, why in rows:
                out.append(f"      {imp if side == 'L' else exp:9s} {term}   ({why})")
        out += [f"  warning: {t}" for t in self.illposed]
        wall = m.geometry.wall_coord
        if self.taus:
            out.append("Boundary conditions -> tau terms (automatic, deterministic):")
            for t in self.taus:
                if t.piece is not None:           # compound interval: mode of one piece
                    n_p = self.bases[wall].sizes[t.piece]
                    mode = (f"N_{wall},{t.piece}{t.mode}" if symbolic else
                            f"{n_p + t.mode} (N_{wall},{t.piece}{t.mode}) of piece {t.piece}")
                elif symbolic:
                    mode = f"N_{wall}{t.mode}"
                else:
                    mode = f"{self.bases[wall].size + t.mode} (N_{wall}{t.mode})"
                lam = f", derivative basis lam={t.lam}" if t.lam else ""
                out.append(f"  {t.condition} at {t.boundary}: {t.name} lifted into mode {mode}"
                           f"{lam} of [{t.target}] {self.labels[t.target]} -- {t.rule}")
        elif wall is None:
            out.append("No boundary (no tau term).")
        if self.time is not None:
            out.append(f"Time: {self.time.name} appears in the data of the equations or conditions "
                       "(right-hand sides evaluated at each stage time; time-dependent boundary "
                       "values: see the solver's 'boundary data' line)")
        out += self.parity
        for g in self.gauge:
            out.append(g)
        for n in self.notes:
            out.append(n)
        deal = ", ".join(f"{c}: {_frac(v)}" for c, v in self.dealias.items())
        out.append(f"Dealiasing: {deal}" + (f" ({dealias_rule_text(self.degree, self.bases)}; override with "
                                             "dealias=)" if self.dealias_default else " (given)"))
        if self.nonpoly:
            out.append("  not polynomial in the unknowns (no dealiasing factor removes their "
                       "aliasing; resolve them, check the spectral tails): "
                       + ", ".join(self.nonpoly[:4]) + (" ..." if len(self.nonpoly) > 4 else ""))
        if self.overrides:
            m = self.model
            out.append("Parameters for this call (the model's own values unchanged): " + ", ".join(
                f"{k} = {v:g} (model: {m.params[k]:g})" for k, v in self.overrides.items()))
        out.append("Bases: " + "; ".join(repr(b) for b in self.bases.values()))
        return out

    def summary(self) -> list[str]:
        """The decisions of :meth:`decisions` in a few lines, in the words of the model: the
        problem kind, which terms the time stepper treats explicitly (by equation label), the
        conditions by boundary, the gauge, the dealiasing and every warning; the per-term split,
        the tau names and modes and the bases are left to ``explain(full=True)`` (the default text led with tau lifting and solver buckets)."""
        out = [f"Problem: {self.kind} ({self._kind_reason()})"]
        rows = [(i, term, side, why) for i, rs in enumerate(self.split) for term, side, why in rs]
        if self.kind == "IVP":
            n_imp = sum(1 for _i, _t, side, _w in rows if side == "L")
            exp = [(i, t, w) for i, t, side, w in rows if side == "R"]
            nonlin = list(dict.fromkeys(self.labels[i] for i, _t, w in exp
                                        if w.startswith("nonlinear")))
            source = list(dict.fromkeys(self.labels[i] for i, _t, w in exp
                                        if w.startswith("source")))
            line = f"Time stepping: {n_imp} linear term(s) implicit"
            if nonlin:
                line += "; nonlinear terms explicit (" + ", ".join(nonlin) + ")"
            if source:
                line += "; source terms explicit (" + ", ".join(source) + ")"
            out.append(line)
            marked = [f"{t} [{self.labels[i]}]" for i, t, w in exp if w.startswith("linear")]
            if marked:
                out.append("  linear terms explicit by your explicit(...): " + ", ".join(marked))
        if self.taus:
            by_wall: dict[str, list[str]] = {}
            for t in self.taus:
                by_wall.setdefault(t.boundary, []).append(t.condition)
            out.append("Boundary conditions: " + "; ".join(f"{b}: {', '.join(c)}"
                                                         for b, c in by_wall.items())
                       + " (tau method, placed automatically)")
        gauge = " ".join(self.gauge)
        fixed = sorted(set(re.findall(r"integ\((\w+)\) = 0", gauge)))
        if fixed:
            out.append("Gauge: " + ", ".join(f"the mean of {v} is fixed (integ({v}) = 0)"
                                             for v in fixed))
        elif self.gauge:
            out.append(f"Gauge: {len(self.gauge)} automatic decision(s)")
        out += self.parity
        deal = ", ".join(f"{c}: {_frac(v)}" for c, v in self.dealias.items())
        out.append(f"Dealiasing: {deal}" + (f" (declared default for products of degree "
                                             f"{max(int(self.degree), 1)})"
                                             if self.dealias_default else " (given)"))
        weighted = _weighted_axes(self.bases)
        if weighted and self.degree > 1:
            out.append(f"  approximate along {', '.join(weighted)}: products of Hermite/Laguerre "
                       "functions are not alias-free at any factor (resolve the fields, check the "
                       "spectral tails)" + ("; default there max(2, (p+1)/2), which lowers the "
                                            "aliasing of a quadratic product 10-100x against 3/2"
                                            if self.dealias_default else ""))
        out += [f"  warning: {t}" for t in self.illposed]
        if self.nonpoly:
            out.append("  warning: not polynomial in the unknowns (no dealiasing factor removes "
                       "their aliasing): " + ", ".join(self.nonpoly[:4])
                       + (" ..." if len(self.nonpoly) > 4 else ""))
        if self.overrides:
            m = self.model
            out.append("Parameters for this call (the model's own values unchanged): " + ", ".join(
                f"{k} = {v:g} (model: {m.params[k]:g})" for k, v in self.overrides.items()))
        if self.time is not None:
            out.append(f"Time: {self.time.name} appears in the data (evaluated at each stage time)")
        if self.notes:
            out.append(f"{len(self.notes)} further note(s) on the conditions or the geometry: "
                       "explain(full=True)")
        return out

    def _kind_reason(self) -> str:
        return {"IVP": "a time derivative dt() appears",
                "EVP": f"eigenvalue {self.eigenvalue.name if self.eigenvalue else '?'} declared",
                "LBVP": "no dt(), linear in the unknowns",
                "NLBVP": "no dt(), nonlinear in the unknowns"}[self.kind]

    def explain(self, *, full: bool = False) -> str:
        """Resolution and the automatic decisions: a summary (:meth:`summary`) by default;
        ``full=True``: every decision (taus, gauge, term-by-term split) and the IR problem."""
        lines = [f"Compiled model '{self.problem.name}' at N = "
                 + ", ".join(f"{c}: {n}" for c, n in self.N.items())]
        if not full:
            return "\n".join(lines + self.summary()
                             + ["(every decision and the IR problem: explain(full=True))"])
        lines += self.decisions()
        lines.append("")
        lines.append("IR problem (expert layer):")
        lines.append(self.problem.explain())
        return "\n".join(lines)


def _frac(v: float) -> str:
    return {1.5: "3/2", 1.0: "1", 2.0: "2"}.get(float(v), f"{v:g}")


# ------------------------------------------------------------------------------------------------
# namespace
# ------------------------------------------------------------------------------------------------

def _core(x) -> tuple:
    """Non-numeric factors of a product (numbers dropped), to recognise a marked term whatever
    number multiplies it; () for a pure number."""
    args = list(x.args) if isinstance(x, Mul) else [x]
    return tuple(sorted(str(a) for a in args if not isinstance(a, Constant)))


class _Markers:
    """``explicit(...)`` / ``implicit(...)`` in equation text: record the marked expressions."""

    def __init__(self) -> None:
        self.marks: dict[str, list] = {"explicit": [], "implicit": []}
        self.used: set = set()

    def make(self, kind: str) -> Callable:
        def mark(e):
            e = as_expr(e)
            self.marks[kind].append(e)
            return e
        mark.__name__ = kind
        return mark

    def kind_of(self, term, record: bool = True) -> str | None:
        for kind, lst in self.marks.items():
            for j, e in enumerate(lst):
                parts = [e] + (list(e.args) if isinstance(e, Add) else [])
                facs = list(term.args) if isinstance(term, Mul) else []
                # a sign or number moved onto the marked product (-explicit(2*a*f/E) is the
                # term -2*E**-1*a*f) is still the marked term (native2)
                if any(term == p or p in facs or _core(term) == _core(p) for p in parts):
                    if record:
                        self.used.add((kind, j))
                    return kind
        return None

    def unused(self) -> list[str]:
        return [str(e) for kind, lst in self.marks.items() for j, e in enumerate(lst)
                if (kind, j) not in self.used]


def _macro(name: str, args: tuple[str, ...], body: str, ns_values: dict) -> Callable:
    def call(*vals):
        if len(vals) != len(args):
            raise TypeError(f"{name} takes {len(args)} argument(s), got {len(vals)}")
        return parse_expr(body, Namespace.of(ns_values, dict(zip(args, vals, strict=True)), layer="native"))
    call.__name__ = name
    return call


# ------------------------------------------------------------------------------------------------
# compilation
# ------------------------------------------------------------------------------------------------

def dealias_factor(degree: int, weighted: bool = False) -> float:
    """Default dealiasing factor of the native layer for a highest polynomial degree ``degree``
    of the terms in the unknowns: (p + 1)/2, so that a product of p fields
    truncated at N modes is computed without aliasing on Fourier and polynomial bases (3/2 for
    quadratic, 2 for cubic), and never below the historical 3/2 (linear models keep 3/2: their
    coefficient products are computed on the same grids as before).

    ``weighted=True``: the axis is Hermite or Laguerre. No factor makes products exact there
    (the functions carry a Gaussian or exponential weight: a product is not in the span of the
    first modes); the factor reduces aliasing only. For products of the unknowns (p >= 2) the
    floor is then 2 (the aliasing error of a quadratic product drops from
    ~1e-1 to ~1e-2 for unit spectra and from ~1e-3 to ~1e-5 for decaying ones); a linear
    model (p = 1) keeps 3/2 on every axis."""
    p = max(int(degree), 1)
    floor = WEIGHTED_DEALIAS if (weighted and p >= 2) else DEFAULT_DEALIAS
    return max(floor, (p + 1) / 2)


def default_dealias_map(geometry, degree: int) -> dict[str, float]:
    """Per-coordinate default factor (dealias=None): ``dealias_factor`` with the Hermite /
    Laguerre floor on the geometry's whole-line and half-line axes."""
    weighted = set(geometry._weighted_coords())
    return {c: dealias_factor(degree, c in weighted) for c in geometry.coord_names}


def _weighted_axes(bases: Mapping | None) -> list[str]:
    """Coordinates whose basis is Hermite or Laguerre (products not alias-free at any factor)."""
    out = []
    for c, b in (bases or {}).items():
        if isinstance(b, (Hermite, Laguerre)):
            out.append(str(c))
    return out


def dealias_rule_text(degree: int, bases: Mapping | None = None) -> str:
    """The sentence explain() prints for a default dealiasing factor; ``bases`` (coordinate ->
    basis) names the Hermite/Laguerre axes, where it is approximate (and floored at 2 for
    products of the unknowns)."""
    f = _frac(dealias_factor(degree))
    p = max(int(degree), 1)
    text = (f"declared default {f} of the native layer: max(3/2, (p+1)/2) for the highest "
            f"polynomial degree p = {p} of the equations in the unknowns, exact (alias-free) "
            "products on Fourier and polynomial bases")
    weighted = _weighted_axes(bases)
    if weighted:
        fw = _frac(dealias_factor(degree, True))
        text += (f"; approximate on the Hermite/Laguerre axis {', '.join(weighted)} (products of "
                 "Hermite or Laguerre functions are not alias-free at any factor: resolve the "
                 "fields and check the spectral tails)")
        if p >= 2:
            text += (f"; there the default is {fw} = max(2, (p+1)/2), because the factor only "
                     "reduces aliasing on these axes and 2 instead of 3/2 lowers the error of a "
                     "quadratic product by one to two decades")
    return text


def _max_degree(raw, V) -> tuple[int, list[str]]:
    """(highest polynomial degree of a term in the unknowns, terms that are not polynomial)."""
    p, nonpoly = 1, []
    for eq in raw:
        for side in (eq.lhs, eq.rhs):
            for t in _terms(side):
                for d in t.degrees(V):
                    if d >= NONPOLY:
                        nonpoly.append(str(t))
                    else:
                        p = max(p, d)
    return p, nonpoly


def compile_model(model, N, dealias=None, *, check_conditions: bool = True,
                  params: dict | None = None) -> Compiled:
    """Build the IR problem of ``model`` at resolution ``N`` (see module docstring).
    ``check_conditions=False``: parse and split only (declaration-time probe, no problem).
    ``params``: validated values of declared parameters for this compilation only.
    ``dealias=None``: the factor follows the highest polynomial degree of the equations,
    found by the declaration-time probe (``dealias_factor``)."""
    geo = model.geometry
    res = geo.resolution(N)
    degree = getattr(model, "_degree", None) or 1
    deal = (geo.dealias_map(dealias) if dealias is not None
            else default_dealias_map(geo, degree))
    cs, bases, D = geo.build(res, deal)
    names = geo.coord_names

    # unknowns, given fields, parameters ----------------------------------------------------
    fields: dict[str, Field] = {}
    unknowns: list[Field] = []
    for h in model.unknowns:
        f = VectorField(h.name, D) if h.rank == 1 else Field(h.name, D)
        fields[h.name] = f
        unknowns.append(f)
    V = frozenset(unknowns)
    ncc: dict = {}
    sincos = geo.sincos_coord
    probe = Compiled(model, res, deal, "?", None, {}, {}, {}, None, cs, bases, D, [])
    for g in model.givens:
        if callable(g.value):
            args = _coord_args(g.value, names, f"given {g.name}")
            dom = Domain(cs, [bases[c] for c in names if c in args])
            data = _on_all_grids(g.value, args, names)
        else:
            dom, data = Domain(cs, []), (lambda v: (lambda *grids: np.asarray(v)))(g.value)
        sym = None
        if sincos is not None:       # a coefficient on a sine/cosine axis has a checked parity
            R = _sincos.reflection(geo, cs)
            sym = {R: _sincos.data_parity(probe, data, g.rank, f"given {g.name}", R=R)}
        f = (VectorField(g.name, dom, symmetry=sym) if g.rank == 1
             else Field(g.name, dom, symmetry=sym))
        fields[g.name] = f
        ncc[f] = data
    overrides = dict(params or {})
    params = {k: Parameter(k, overrides.get(k, v),
                           dtype=complex if isinstance(overrides.get(k, v), complex) else np.float64)
              for k, v in model.params.items()}
    eig = Parameter(model.eig, dtype=complex) if model.eig else None
    markers = _Markers()
    values: dict[str, Any] = {}
    values.update(params)
    if eig is not None:
        values[eig.name] = eig
    values.update(fields)
    # the time "t": a name of the native layer unless the model uses it for something else
    # (coordinate, unknown, given field, parameter); used only in right-hand-side data
    tpar = None
    if "t" not in values and "t" not in names:
        tpar = values["t"] = Parameter("t", 0.0)
    values["explicit"], values["implicit"] = markers.make("explicit"), markers.make("implicit")
    for name, args, body, _ in model.macros:
        values[name] = _macro(name, args, _text(body), values)
    values.update(getattr(model, "functions", {}))       # m.function(...): pointwise builders
    ns = Namespace.of(values, layer="native")

    # equations and the split ----------------------------------------------------------------
    labels = [lab for lab, _ in model.eqs]
    raw, plain = [], []
    for lab, text in model.eqs:
        n_marks = sum(len(v) for v in markers.marks.values())
        eq = parse_equation(_text(text), ns)
        # the parser adds like terms, so a marked term and an
        # unmarked one of the same form become one term (explicit(2*dx(c)) + 3*dx(c) is 5*dx(c));
        # the cores of the equation parsed with the markers set to zero find such a merge
        if sum(len(v) for v in markers.marks.values()) > n_marks:
            saved = values["explicit"], values["implicit"]
            values["explicit"] = values["implicit"] = lambda e: as_expr(0)
            try:
                eq0 = parse_equation(_text(text), Namespace.of(values, layer="native"))
            finally:
                values["explicit"], values["implicit"] = saved
            plain.append(tuple(({str(t) for t in _terms(side)}, {_core(t) for t in _terms(side)})
                               for side in (eq0.lhs, eq0.rhs)))
        else:
            plain.append(None)
        if any(type(n).__name__ == "Lift" for side in (eq.lhs, eq.rhs) for n in side.walk()):
            raise BoundaryConditionError(
                f"equation '{lab}' contains lift(): the native layer places the tau terms itself",
                where=text, why="a hand-written lift would add a second tau for the same condition",
                fix="remove the lift() terms and declare the conditions with at(...); for a "
                    "hand-made tau formulation use the IR (expert layer): conatus.expert.IVP/EVP",
                doc=_DOC)
        raw.append(eq)
    has_dt = any(isinstance(n, TimeDerivative) for eq in raw for s in (eq.lhs, eq.rhs)
                 for n in s.walk())
    nonlinear = any(t.degrees(V) != frozenset({1}) and t.degrees(V) != frozenset({0})
                    for eq in raw for s in (eq.lhs, eq.rhs) for t in _terms(s))
    if eig is not None:
        kind = "EVP"
        if has_dt:
            raise EquationError("an eigenvalue problem cannot contain dt()",
                                why=f"the eigenvalue {eig.name} stands for the time derivative",
                                fix=f"write {eig.name}*u where the evolution equation has dt(u), "
                                    "or declare the IVP and use Model.linearize(...)", doc=_DOC)
    elif has_dt:
        kind = "IVP"
    else:
        kind = "NLBVP" if nonlinear else "LBVP"
    split: list = []
    eqs: list[tuple[Any, Any]] = []
    for i, (eq, lab) in enumerate(zip(raw, labels, strict=True)):
        lhs, rhs, rows = _split(eq, V, kind, markers, lab, fields, plain[i])
        eqs.append((lhs, rhs))
        split.append(rows)
    unused = markers.unused()
    if unused:
        raise EquationError(f"implicit(...)/explicit(...) must wrap whole terms; not applied to: "
                            f"{', '.join(unused)}",
                            fix="write explicit(...) around a complete term (a product with its "
                                "coefficients), e.g. -explicit(2*cross(ez, u))", doc=_DOC)

    pdeg, nonpoly = _max_degree(raw, V)
    comp = Compiled(model, res, deal, kind, None, ncc, fields, params, eig, cs, bases, D, labels,
                    split=split, dealias_default=dealias is None, degree=pdeg, nonpoly=nonpoly,
                    overrides=overrides)
    comp._time_candidate = tpar
    if not check_conditions:
        return comp
    if kind == "IVP":                # backward diffusion / negative viscosity (Y6)
        from .wellposed import check_ill_posed
        comp.illposed = check_ill_posed(comp, eqs, labels, _terms)
    sym = None
    if sincos is not None:           # sine/cosine axis: parities, no tau (model/sincos.py)
        variables, equations, sym = _sincos.place_parity(comp, eqs, unknowns, V, ns)
    else:
        variables, equations = _place_taus(comp, eqs, unknowns, V, ns)
    name = model.name
    if kind == "IVP":
        pb = IVP(variables, name=name, symmetry=sym)
    elif kind == "EVP":
        pb = EVP(variables, eigenvalue=eig, name=name, symmetry=sym)
    elif kind == "LBVP":
        pb = LBVP(variables, name=name, symmetry=sym)
    else:
        pb = NLBVP(variables, name=name, symmetry=sym)
    for lhs, rhs in equations:
        pb.add_equation((lhs, rhs))
    comp.problem = pb
    _check_time(comp, tpar)
    traced = comp.__dict__.get("_traced", set())
    if traced:                      # a given used only through its wall trace (A22): no data
        used = {f for eq in pb.equations for side in (eq.lhs, as_expr(eq.rhs))
                for f in side.free_fields()}
        for f in traced - used:
            comp.ncc.pop(f, None)
    return comp


def _check_time(comp: Compiled, tpar) -> None:
    """The time t of the native layer: allowed in the right-hand sides of a time-dependent
    model (forcing, boundary values), refused on the implicit side (L would change with time)
    and in models without dt()."""
    if tpar is None:
        return
    pb = comp.problem
    used = [k for k, eq in enumerate(pb.equations)
            if tpar in eq.lhs.free_parameters() or tpar in as_expr(eq.rhs).free_parameters()]
    if not used:
        return
    # a field named like the time (T, t_): the t may be a typo
    from ..errors import did_you_mean
    dym = did_you_mean("t", [n for n in comp.fields if not n.startswith("tau_")])
    if comp.kind != "IVP":
        raise EquationError(
            f"the time t appears in a {comp.kind} model (no dt())." + dym, where=comp.labels[0] if
            used[0] < len(comp.labels) else "conditions",
            why="t is the time of a time-dependent model; a boundary-value or eigenvalue problem "
                "has no time", fix="add dt(...) to make it an IVP, or declare a parameter: "
                                    "cn.Model(geometry, ..., t0=...) and use that name", doc=_EQ_DOC)
    for k in used:
        if tpar in pb.equations[k].lhs.free_parameters():
            raise EquationError(
                f"the time t multiplies an unknown in an implicit term of equation {k}." + dym,
                why="implicit terms build the matrix L, assembled and factorized once: it cannot "
                    "change with time", fix="mark the term explicit(...) (evaluated at each stage "
                                            "time with the right-hand side)", doc=_EQ_DOC)
    comp.time = tpar


def _why_linear(t, fields: dict, V) -> str:
    coeffs = sorted(f.name for f in t.free_fields() - V)
    if coeffs:
        return "linear, non-constant coefficient " + ", ".join(coeffs)
    return "linear, constant coefficients"


def _split(eq, V, kind: str, markers: _Markers, label: str, fields: dict, plain=None):
    """(implicit LHS, explicit RHS, records) of one equation (module docstring rules).
    ``plain``: for an equation with markers, per side (lhs, rhs), the terms and the cores of the
    terms of the equation parsed with its marked expressions set to zero."""
    if kind == "NLBVP":
        rows = [(str(t), "L", "NLBVP: every term kept as written") for t in _terms(eq.lhs)]
        rows += [(str(t), "R", "NLBVP: every term kept as written") for t in _terms(eq.rhs)]
        return eq.lhs, eq.rhs, rows
    left, right, rows = [], [], []
    for si, (sign, side) in enumerate(((1, eq.lhs), (-1, eq.rhs))):
        for t in _terms(side):
            deg = t.degrees(V)
            # a term found as such in the equation without its markers is unmarked (dt(c) -
            # 2*lap(c) = explicit(-lap(c)): -2*lap(c) has the core of the marked term)
            unmarked = plain is not None and str(t) in plain[si][0]
            mark = None if unmarked else markers.kind_of(t)
            has_dt = any(isinstance(n, TimeDerivative) for n in t.walk())
            moved = "; moved from the right side" if sign < 0 else ""
            where = f"equation '{label}', term '{t}'"
            if (plain is not None and mark is not None and deg == frozenset({1})
                    and _core(t) in plain[si][1]):
                raise EquationError(
                    f"{mark}(...) term merged with an unmarked term of the same form: '{t}'",
                    where=where,
                    why="the parser adds like terms (explicit(2*dx(c)) + 3*dx(c) is 5*dx(c)), so "
                        "the marked part and the unmarked part can no longer be told apart; "
                        "taking the sum as marked would move the unmarked part to the other side "
                        "of the time stepper",
                    fix="write the term once, with its whole coefficient, marked or unmarked",
                    doc=_DOC)
            if deg == frozenset({1}):
                if mark == "explicit":
                    if has_dt or kind != "IVP":
                        raise EquationError(f"explicit(...) on '{t}'", where=where,
                                            why="a time derivative, and every term of an "
                                                "EVP/LBVP operator, is implicit by definition",
                                            fix="remove the explicit(...) marker", doc=_DOC)
                    right.append(t if sign < 0 else neg(t))
                    rows.append((str(right[-1]), "R", "linear, marked explicit"
                                 + ("; moved from the left side" if sign > 0 else "")))
                else:
                    left.append(t if sign > 0 else neg(t))
                    why = _why_linear(t, fields, V) + (", marked implicit" if mark else "")
                    rows.append((str(left[-1]), "L", why + moved))
                continue
            if mark == "implicit":
                raise NonlinearLHSError(f"implicit(...) on the nonlinear term '{t}'", where=where,
                                        why="only terms linear in the unknowns have a matrix",
                                        fix="remove the marker (the term is explicit)", doc=_DOC)
            if has_dt:
                raise NonlinearLHSError(f"time derivative in a nonlinear term '{t}'", where=where,
                                        why="dt() must act linearly on the unknowns",
                                        fix="write dt(u) alone and move the rest to the right side",
                                        doc=_DOC)
            if kind == "EVP":
                raise EquationError(f"term '{t}' is not linear in the unknowns", where=where,
                                    why="an eigenvalue problem is linear and homogeneous",
                                    fix="linearize the model about a base state "
                                        "(Model.linearize(about=...)) or remove the term",
                                    doc=_DOC)
            right.append(t if sign < 0 else neg(t))
            why = ("source term (no unknown)" if deg == frozenset({0})
                   else f"nonlinear, degree(s) {sorted(deg)} in the unknowns")
            if mark == "explicit":
                why += ", marked explicit"
            rows.append((str(right[-1]), "R", why + ("; moved from the left side" if sign > 0
                                                      else "")))
    if not left:
        raise EquationError(f"equation '{label}' has no term linear in the unknowns",
                            where=str(eq),
                            why="each equation needs an implicit (matrix) part to be solved for",
                            fix="check the equation; a purely nonlinear relation is not a "
                                "time-stepped or eigenvalue equation", doc=_DOC)
    return add(*left), (add(*right) if right else 0), rows


# ------------------------------------------------------------------------------------------------
# boundary conditions, taus, incompressibility
# ------------------------------------------------------------------------------------------------

def _occurrences(e, target, parent=None):
    """(parent, node) for every occurrence of ``target`` in the tree (no deduplication)."""
    if e == target:
        yield parent, e
        return
    for a in e.args:
        yield from _occurrences(a, target, e)


def _pressure_like(q, eqs) -> bool:
    """q appears, and only as grad(q), in every equation."""
    seen = False
    for lhs, rhs in eqs:
        for side in (lhs, as_expr(rhs)):
            for parent, _ in _occurrences(side, q):
                if not isinstance(parent, Gradient):
                    return False
                seen = True
    return seen


def _normal_mode_pressure_like(q, eqs, geo, comp) -> bool:
    """q is the pressure of a normal-mode model on one wall coordinate (critique
    N1): a scalar unknown of a geometry with the wall coordinate only, written as on paper for
    modes exp(i k x): it enters algebraically (``1j*k*p``) and through its first derivative along
    the wall (``dz(p)``), never with a higher derivative, and only on the left-hand sides."""
    wall = geo.wall_coord
    if wall is None or q.rank != 0 or len(geo.coord_names) != 1:
        return False
    coord = comp.coordsys[wall]
    order, seen = 0, False
    for lhs, rhs in eqs:
        if any(True for _ in _occurrences(as_expr(rhs), q)):
            return False
        if any(True for _ in _occurrences(lhs, q)):
            seen = True
            order = max(order, lhs.order_in(coord, [q]))
    return seen and order == 1


def _resolve_into(into, labels: list[str], where: str) -> int:
    if isinstance(into, int):
        if 0 <= into < len(labels):
            return into
    elif into in labels:
        return labels.index(into)
    raise BoundaryConditionError(f"into={into!r} names no equation", where=where,
                                 fix="into=<index> or into=<label>; equations: "
                                     + ", ".join(f"[{i}] {lab}" for i, lab in enumerate(labels)),
                                 doc=_DOC)


def _condition_exprs(comp: Compiled, cond, coord, side, ns: Namespace):
    """(lhs, rhs, printable) of one condition at one side of the wall coordinate."""
    fields = comp.fields
    if cond.unknown is not None:
        if cond.unknown not in fields or fields[cond.unknown] not in comp._V:
            raise BoundaryConditionError(f"{cond.unknown}=...: not an unknown of the model",
                                         where=f"at({cond.boundary!r}, {cond.unknown}=...)",
                                         fix="unknowns: " + ", ".join(
                                             f.name for f in comp._unknowns), doc=_DOC)
        f = fields[cond.unknown]
        lhs = interp(f, coord, side)
        v = cond.value
        if isinstance(v, WallCondition):
            terrain = getattr(comp.model.geometry, "terrain", None)
            # mapped domain: refused only on a curved surface (a flat one has the normal e_z)
            wall_surface = None if terrain is None else (
                terrain.bottom if side == "left" else terrain.top)
            if v is not no_slip and wall_surface is not None and not wall_surface.flat:
                raise UnsupportedError(
                    f"{cond.unknown}={v!r} on a curved mapped wall",
                    where=f"at({cond.boundary!r}, {cond.unknown}={v!r})",
                    why="the free-slip and Navier-slip operators of the native layer use the "
                        "normal of a flat wall (e_z); on a curved surface the normal varies",
                    fix=f"no-slip ({cond.unknown}=0), or the components written as conditions "
                        "with the normal given as a field (n = m.given(...))",
                    doc="docs/site/guide/mapped.md")
            if f.rank != 1:
                raise BoundaryConditionError(
                    f"{cond.unknown}={v!r}: {v.name} is a condition on a velocity (a vector)",
                    where=f"at({cond.boundary!r}, {cond.unknown}={v!r})",
                    fix=f"declare {cond.unknown} with m.velocity(...), or write a scalar condition "
                        f"('{cond.unknown} = 0', 'd{coord.name}({cond.unknown}) = 0')", doc=_DOC)
            if v is no_slip:
                return lhs, 0, f"{cond.unknown} = 0 (no-slip)"
            if isinstance(v, NavierSlip) and v.length == 0:
                return lhs, 0, f"{cond.unknown} = 0 (Navier slip of length 0: no-slip)"
            if isinstance(v, NavierSlip):
                sgn = -1.0 if side == "left" else 1.0       # outward normal along the wall axis
                if comp.coordsys.curvilinear:
                    # curved wall r = R (model/conditions.py): with S(u) = R Pt·dr(u) + K·u (the
                    # free-slip operator: normal u_r, tangential R·2(E·n)_t), the Navier slip
                    # u_t + s l 2(E·n)_t = 0, u_r = 0 is, multiplied by R,
                    #   B(u) = s l S(u) + R Pt·u + (1 - s l) (I - Pt)·u = 0
                    Pt, K, _meaning = _curved_wall_tensors(comp, coord)
                    R = comp.bases[coord.name].bounds[0 if side == "left" else 1]
                    sl = sgn * v.length
                    B = (sl * curved_free_slip_operator(f, coord, R, Pt, K) + R * dot(Pt, f)
                         + (1 - sl) * (f - dot(Pt, f)))
                    return (interp(B, coord, side), 0,
                            f"{cond.unknown}: Navier slip on the curved wall ({cond.unknown}·n = 0, "
                            f"{cond.unknown}_t {'-' if sgn < 0 else '+'} {v.length:g} "
                            f"2(E·n)_t = 0)")
                return (interp(navier_slip_operator(f, coord, v.length, sgn), coord, side), 0,
                        f"{cond.unknown}: Navier slip ({cond.unknown}·n = 0, {cond.unknown}_t "
                        f"{'-' if sgn < 0 else '+'} {v.length:g} d{coord.name}({cond.unknown}_t) = 0)")
            if comp.coordsys.curvilinear:
                # one vector condition R Pt·dr(u) + K·u = 0 at r = R (model/conditions.py)
                Pt, K, meaning = _curved_wall_tensors(comp, coord)
                R = comp.bases[coord.name].bounds[0 if side == "left" else 1]
                return (interp(curved_free_slip_operator(f, coord, R, Pt, K), coord, side), 0,
                        f"{cond.unknown}: free-slip ({meaning.format(u=cond.unknown)})")
            # one vector condition B(u) = (I - n n)·dz(u) + n (n·u) = 0 (model/conditions.py)
            return (interp(free_slip_operator(f, coord), coord, side), 0,
                    f"{cond.unknown}: free-slip ({cond.unknown}·n = 0, d{coord.name}"
                    f"({cond.unknown}_t) = 0)")
        if isinstance(v, str):
            # a boundary value written as text: an expression of the parameters, given fields and
            # the time t ("sin(2*t)"; exact time derivative for bc_time='rk')
            rhs = parse_expr(_text(v), ns)
            if f.rank != as_expr(rhs).rank:
                raise BoundaryConditionError(f"{cond.unknown}={v!r}: rank {as_expr(rhs).rank} for an "
                                             f"unknown of rank {f.rank}", doc=_DOC)
            return lhs, rhs, f"{cond.unknown} = {v}"
        tpar = ns.values.get("t") if isinstance(ns.values.get("t"), Parameter) else None
        if callable(v) and tpar is not None and _takes_time(v):
            return lhs, _symbolic_value(comp, cond, coord, v, tpar), \
                f"{cond.unknown} = {getattr(v, '__name__', 'data')}(t)"
        if callable(v):
            names = comp.model.geometry.coord_names
            args = _coord_args(v, names, f"boundary value {cond.unknown}")
            if coord.name in args:
                raise BoundaryConditionError(
                    f"boundary value of {cond.unknown} depends on {coord.name}, the wall "
                    "coordinate", fix=f"a boundary value is a function of the other "
                                      f"coordinates: lambda {', '.join(a for a in names if a != coord.name)}: ...",
                    doc=_DOC)
            gname = f"{cond.unknown}_{cond.boundary.replace('=', '_').replace('.', 'p').replace('-', 'm')}_{side}"
            dom = Domain(comp.coordsys, [comp.bases[c] for c in names if c in args])
            g = VectorField(gname, dom) if f.rank == 1 else Field(gname, dom)
            comp.ncc[g] = _on_all_grids(v, args, names)
            rhs = g
            text = f"{cond.unknown} = {getattr(v, '__name__', 'data')}"
        elif isinstance(v, (tuple, list)):
            if f.rank != 1 or comp.coordsys.curvilinear:
                raise BoundaryConditionError(f"{cond.unknown}={v!r}: a tuple is a Cartesian vector",
                                             fix="give a number (0) or a vector for a vector "
                                                 "unknown of a Cartesian geometry", doc=_DOC)
            rhs = add(*[c * unit_vector(x) for c, x in zip(v, comp.coordsys.coords, strict=True) if c != 0])
            text = f"{cond.unknown} = {tuple(v)}"
        else:
            rhs = v
            text = f"{cond.unknown} = {v:g}" if isinstance(v, (int, float)) else f"{cond.unknown} = {v}"
        return lhs, rhs, text
    eq = parse_equation(_text(cond.text), ns)
    rhs = eq.rhs
    if comp.coordsys.curvilinear:          # exact traces of the given functions
        rhs = _wall_traces(comp, as_expr(rhs), coord, side, cond)
    return interp(eq.lhs, coord, side), interp(rhs, coord, side), cond.text.strip()


_TRACE_SAFE = (Add, Mul, Power, UnaryFunction)   # pointwise: trace(f(g)) = f(trace(g))


def _wall_traces(comp: Compiled, e, coord, side, cond):
    """Audit A22 (curved geometries): a given field that depends on the wall coordinate, in the
    right side of a condition, is replaced by its trace at the wall evaluated from its function
    (g(phi, R): exact, declared in explain()); the polar and spherical right-hand sides do not
    evaluate traces. Under a derivative (dr(g)) the trace of the function is not enough: refused
    with what to declare instead."""
    givens = {g.name: g for g in comp.model.givens}
    names = comp.model.geometry.coord_names
    R = comp.bases[coord.name].bounds[0 if side == "left" else 1]
    made: dict = {}

    def sub(x, pointwise: bool):
        if isinstance(x, Field) and x.name in givens and x.domain.has(coord):
            g = givens[x.name]
            if not pointwise or not callable(g.value) or x.rank:
                raise BoundaryConditionError(
                    f"condition '{cond.text.strip()}': {x.name} depends on {coord.name}, the wall "
                    f"coordinate, and appears {'under an operator' if not pointwise else 'as a tensor'}",
                    where=f"at({cond.boundary!r}, {cond.text.strip()!r})",
                    why="on a curved geometry a boundary value is evaluated on the wall only; the "
                        "trace of a function is taken exactly, not that of its derivatives",
                    fix=f"declare the wall value as a function of the other coordinates, e.g. "
                        f"m.given({x.name}_wall=lambda {', '.join(a for a in names if a != coord.name)}: "
                        f"...) and use {x.name}_wall in the condition", doc=_DOC)
            if x not in made:
                args = _coord_args(g.value, names, f"given {g.name}")
                rest = [a for a in args if a != coord.name]
                fn = g.value

                def data(*grids, fn=fn, args=args):
                    return fn(*[R if a == coord.name else grids[names.index(a)] for a in args])
                tf = Field(f"{x.name}_{coord.name}{_g(R).replace('.', 'p').replace('-', 'm')}",
                           Domain(comp.coordsys, [comp.bases[a] for a in rest]))
                comp.ncc[tf] = data
                comp.notes.append(f"{x.name} in the condition at {cond.boundary}: its trace at "
                                  f"{coord.name} = {_g(R)} evaluated from its function (exact)")
                made[x] = tf
                comp.__dict__.setdefault("_traced", set()).add(x)
            return made[x]
        if not x.args:
            return x
        inner = pointwise and isinstance(x, _TRACE_SAFE)
        new = [sub(a, inner) for a in x.args]
        return x if all(n is a for n, a in zip(new, x.args, strict=True)) else x.rebuild(new)

    return sub(e, True)


def _takes_time(fn) -> bool:
    """True if a data callable has an argument named t (the time of the native layer)."""
    try:
        return "t" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _symbolic_value(comp: Compiled, cond, coord, fn, tpar):
    """A boundary value ``lambda t: ...`` as an IR expression: called once with the time
    parameter, so that numpy functions of it (np.sin, np.exp, ...) build the expression; its
    time derivative is then exact (bc_time='rk'). A function of t and of coordinates is refused
    with the separable form to write instead: a coordinate on a periodic axis is not a smooth
    field (a sawtooth), so sin(x - t) cannot be built from it exactly."""
    names = comp.model.geometry.coord_names
    params = list(inspect.signature(fn).parameters)
    where = f"at({cond.boundary!r}, {cond.unknown}=...)"
    if params != ["t"]:
        others = [a for a in params if a != "t"]
        raise BoundaryConditionError(
            f"boundary value of {cond.unknown}: a function of t and of {others}", where=where,
            why="a time-dependent boundary value is kept as an expression of t (evaluated at "
                "every stage time, differentiated exactly); its spatial part must be declared "
                "data, and a coordinate itself is not smooth data on a periodic axis",
            fix=f"declare the spatial factors and combine them with t in text, e.g. for "
                f"sin(x - t): m.given(S=lambda x: np.sin(x), C=lambda x: np.cos(x)); "
                f"m.at({cond.boundary!r}, {cond.unknown}='S*cos(t) - C*sin(t)')"
                + ("" if all(a in names for a in others) else
                   f" (coordinates of this geometry: {names})"), doc=_DOC)
    try:
        val = fn(tpar)
    except Exception as err:  # noqa: BLE001 -- any failure of user code is reported with the fix
        raise BoundaryConditionError(
            f"boundary value of {cond.unknown}: {getattr(fn, '__name__', 'the function')} could "
            f"not be evaluated symbolically in t ({type(err).__name__}: {err})", where=where,
            why="a time-dependent boundary value is kept as an expression of t so that it is "
                "evaluated at every stage time and differentiated exactly",
            fix=f"use numpy functions (np.sin, np.cos, np.exp, np.tanh, np.sqrt, np.log) and "
                f"arithmetic only, or write it as text: at({cond.boundary!r}, "
                f"{cond.unknown}='sin(2*t)')", doc=_DOC) from None
    val = as_expr(val)
    if val.rank != comp.fields[cond.unknown].rank:
        raise BoundaryConditionError(f"boundary value of {cond.unknown}: rank {val.rank} does not "
                                     "match the unknown", where=where, doc=_DOC)
    return val


def _curved_wall_tensors(comp: Compiled, coord):
    """The wall tensors Pt, K of the curved free-slip condition (fields on the wall, without r,
    with their data in comp.ncc; created once per compilation, reserved names)."""
    if getattr(comp, "_wall", None) is not None:
        return comp._wall
    for name in ("Pt_wall", "K_wall"):
        if name in comp.fields:
            raise InputTypeError(f"the name {name!r} is reserved with cn.free_slip on a curved "
                                 "wall (tensor of the stress-free condition)", doc=_DOC)
    cs = comp.coordsys
    kind = {"Polar": "polar", "Cylindrical": "cylinder", "Spherical": "spherical"}.get(cs.kind)
    if kind is None:
        raise UnsupportedError(f"cn.free_slip on {cs.kind} coordinates", doc=_DOC)
    # the angular coordinates only (not r, nor the axis of a cylinder): constant tensors there
    dom = Domain(cs, [comp.bases[c.name] for c in cs.coords
                      if c != coord and not (kind == "cylinder" and c.index == 2)])
    pt_data, k_data, meaning = curved_wall_tensors(kind)
    Pt, K = TensorField("Pt_wall", dom, 2), TensorField("K_wall", dom, 2)
    comp.ncc[Pt], comp.ncc[K] = pt_data, k_data
    comp.fields["Pt_wall"], comp.fields["K_wall"] = Pt, K
    comp._wall = (Pt, K, meaning)
    comp.notes.append(
        "Free-slip on a curved wall r = R: one vector condition R Pt·dr(u) + K·u = 0 per wall "
        "(Pt = I - e_r e_r and K = e_r e_r - Pc, Pc the projector on the curved tangential "
        "directions, are the given wall tensors Pt_wall, K_wall; normal component u_r, tangential "
        "R dr(u_t) - u_t = R^2 dr(u_t / r)" + (", axial R dr(u_z)" if kind == "cylinder" else "")
        + "; model/conditions.py)")
    return Pt, K, meaning


def _interface_conditions(comp, geo, coord, wall, eqs, unknowns, pressures, conds, assign):
    """Compound interval: continuity of each unknown of order q along the wall coordinate at each
    interface (value and derivatives up to q - 1), appended to ``conds`` and to the equation that
    receives the unknown's wall conditions. Nothing is asked from the user; explain() lists them."""
    from ..ir import diff
    if pressures:
        return                     # refused by the caller with the reason
    for f in unknowns:
        q = max((lhs.order_in(coord, [f]) for lhs, _ in eqs), default=0)
        own = [c for c in conds if c["f"] == f]
        if q == 0 or not own:
            continue
        targets = sorted({c["target"] for c in own})
        if len(targets) != 1:
            raise BoundaryConditionError(
                f"{f.name}: its wall conditions go to several equations "
                f"({', '.join(str(t) for t in targets)}); on a compound interval its interface "
                "conditions need one target", where=", ".join(c["where"] for c in own),
                fix="send all its conditions to one equation (into=...)", doc=_DOC)
        i = targets[0]
        for cpos in geo.interfaces:
            d = f
            for m in range(q):
                jump = interp(d, coord, cpos, side="above") - interp(d, coord, cpos, side="below")
                what = f.name if m == 0 else f"d{wall}{'^' + str(m) if m > 1 else ''}({f.name})"
                c = dict(cond=None, f=f, lhs=jump, rhs=0, text=f"{what} continuous",
                         side=None, pos=cpos, interface=True, target=i,
                         where=f"interface {wall} = {_g(cpos)} (automatic)",
                         boundary=f"interface {wall}={_g(cpos)}",
                         rule=f"continuity of {what} across the interface (order {q} of {f.name})")
                conds.append(c)
                assign.setdefault(i, []).append(c)
                d = diff(d, coord)


def _lift_for(recipe: str, tau, basis, k: int):
    if recipe == "cartesian":
        return lift(tau, basis, -k)
    if recipe in ("regular", "annulus"):
        return lift(tau, basis, -k, lam=2)
    raise AssertionError(recipe)


# Shown by explain(): the rule itself, never an internal test or tutorial file name (audit
#); the provenance of each recipe is in docs/API_NATIVE.md §4.
_RULES = {"cartesian": "Chebyshev lift into the last modes",
          "regular": "lift in the derivative basis lam=2 (regular at the centre)",
          "annulus": "lift in the derivative basis lam=2",
          "shell": "first-order form: lap(f) -> div(grad(f) + rvec lift(tau_1, lam=1)), "
                   "lift(tau_2, lam=2) (one basis higher than tau_1: the Dedalus shell examples "
                   "lift both in lam=1, which has a spurious growing tau mode from N_r ~ 48)"}


def _place_taus(comp: Compiled, eqs, unknowns, V, ns):
    """Conditions -> tau variables, lifts, gauges; returns (variables, [(lhs, rhs)])."""
    model, geo = comp.model, comp.model.geometry
    comp._V, comp._unknowns = V, unknowns
    labels = comp.labels
    eqs = [(lhs, rhs) for lhs, rhs in eqs]
    wall = geo.wall_coord
    if model.conditions and wall is None:
        raise BoundaryConditionError(f"{geo!r} has no boundary: no condition can be declared",
                                     where=f"at({model.conditions[0].boundary!r}, ...)",
                                     fix="remove the conditions (periodic/closed geometry)",
                                     doc=_DOC)
    # expand conditions (declaration order; per call: positional texts, then keywords; per
    # condition: the sides of the boundary name)
    conds = []
    if wall is not None:
        coord = comp.coordsys[wall]
        basis = comp.bases[wall]
        for c in model.conditions:
            for _cname, side in geo.boundary(c.boundary):
                lhs, rhs, text = _condition_exprs(comp, c, coord, side, ns)
                where = f"at({c.boundary!r}, {text})"
                got = lhs.free_fields() & V
                if len(got) != 1:
                    raise BoundaryConditionError(
                        f"condition '{text}' at {c.boundary} involves "
                        f"{len(got)} unknowns ({', '.join(sorted(f.name for f in got)) or 'none'})",
                        where=where,
                        why="each condition closes the equation of one unknown; with several "
                            "unknowns the target of its tau is not determined",
                        fix="write one unknown per condition, or build this coupling in the IR "
                            "(expert layer) with explicit taus", doc=_DOC)
                if as_expr(rhs).depends_on(V):
                    raise BoundaryConditionError(f"right side of '{text}' depends on the unknowns",
                                                 where=where, fix="move every unknown to the left",
                                                 doc=_DOC)
                (f,) = got
                conds.append(dict(cond=c, f=f, lhs=lhs, rhs=rhs, text=text, side=side,
                                  where=where, boundary=c.boundary if len(geo.boundary(c.boundary)) == 1
                                  else f"{c.boundary} ({'left' if side == 'left' else 'right'}: "
                                       f"{wall}={_g(basis.bounds[0 if side == 'left' else 1])})"))
    conds = _merge_component_conditions(comp, conds)
    # pressure recognition (structural)
    cond_unknowns = {c["f"] for c in conds}
    pressures = []
    for q in unknowns:
        if q.rank == 0 and q not in cond_unknowns and _pressure_like(q, eqs):
            pressures.append(q)
    # normal-mode pressures: declared with m.pressure on a one-coordinate wall
    # geometry, written as 1j*k*p and dz(p); recognized only when declared, never guessed
    mode_pressures = []
    for h in model.unknowns:
        q = comp.fields[h.name]
        if h.role == "pressure" and q not in pressures:
            if q not in cond_unknowns and _normal_mode_pressure_like(q, eqs, geo, comp):
                mode_pressures.append(q)
                continue
            one_d = wall is not None and len(geo.coord_names) == 1
            raise BoundaryConditionError(
                f"{h.name} is declared as a pressure but does not have the structure of one",
                why="a pressure enters only through grad(p)"
                    + (" (or, in a normal-mode model on the single wall coordinate, as i k p and "
                       f"d{wall}(p): no higher derivative, no right-hand side)" if one_d else "")
                    + ", has no boundary condition, and is fixed by an algebraic constraint such "
                      "as div(u) = 0",
                fix="check the equations (a pressure takes no condition), or declare it with "
                    f"m.scalar({h.name!r})", doc=_DOC)
    # targets
    n_walls = geo.n_walls()
    assign: dict[int, list] = {}
    if wall is not None:
        for c in conds:
            f = c["f"]
            orders = [lhs.order_in(coord, [f]) for lhs, _ in eqs]
            qmax = max(orders, default=0)
            if c["cond"].into is not None:
                i = _resolve_into(c["cond"].into, labels, c["where"])
                if orders[i] == 0:
                    raise BoundaryConditionError(
                        f"into={c['cond'].into!r}: {f.name} has no {wall}-derivative in "
                        f"[{i}] {labels[i]}", where=c["where"],
                        why="a tau closes an equation where the constrained unknown is "
                            "differentiated along the wall coordinate",
                        fix="choose an equation where it is", doc=_DOC)
                c["rule"] = f"target given by into={c['cond'].into!r}"
            else:
                if qmax == 0:
                    raise BoundaryConditionError(
                        f"{f.name} is not differentiated along {wall} in any implicit term: the "
                        f"condition '{c['text']}' has no equation to close", where=c["where"],
                        why="a boundary condition replaces one row of an equation of positive "
                            "order along the wall coordinate",
                        fix=f"remove the condition, or check the equations of {f.name}", doc=_DOC)
                cands = [i for i, o in enumerate(orders) if o == qmax]
                if len(cands) > 1:
                    raise BoundaryConditionError(
                        f"{f.name} has its highest {wall}-order ({qmax}) in several equations: "
                        + ", ".join(f"[{i}] {labels[i]}" for i in cands),
                        where=c["where"],
                        why="the tau of a condition goes into one equation; the choice changes the "
                            "discrete system, so it is never guessed",
                        fix="say which one: at(..., into=<label or index>)", doc=_DOC)
                i = cands[0]
                c["rule"] = f"{f.name} has its highest {wall}-order ({qmax}) there"
            c["target"] = i
            assign.setdefault(i, []).append(c)
        # counts: each unknown needs order * n_walls / 2 conditions (vector = componentwise)
        for f in unknowns:
            if f in pressures or f in mode_pressures:
                continue
            q = max((lhs.order_in(coord, [f]) for lhs, _ in eqs), default=0)
            need = q * n_walls // 2
            have = sum(1 for c in conds if c["f"] == f)
            if have != need:
                sides = ", ".join(repr(k) for k in geo.boundaries())
                hint = ""
                if have == 0 and _normal_mode_pressure_like(f, eqs, geo, comp):
                    # the fix used to be only "declare 1 condition
                    # on p", physically wrong for the pressure of a normal-mode model
                    hint = (f"; if {f.name} is a pressure (i k {f.name} and d{wall}({f.name}) "
                            f"in a normal-mode model), declare it with m.pressure({f.name!r}) "
                            "instead: a pressure takes no condition")
                raise BoundaryConditionError(
                    f"{f.name}: {have} boundary condition(s) for {need} needed "
                    f"(order {q} along {wall}, {n_walls} wall(s))",
                    where="at(...) calls: " + (", ".join(c["where"] for c in conds if c["f"] == f)
                                               or "none for " + f.name),
                    why=("missing conditions leave the discrete system singular" if have < need else
                         "an extra condition would perturb the equation (one tau too many)"),
                    fix=f"declare {need} condition(s) on {f.name} (boundaries: {sides})" + hint,
                    doc=_DOC)
        for i, lst in assign.items():
            cap = eqs[i][0].order_in(coord, V) * n_walls // 2
            if len(lst) > cap:
                pq = sorted({c["f"].name for c in lst if c["f"].rank == 0
                             and _normal_mode_pressure_like(c["f"], eqs, geo, comp)})
                raise BoundaryConditionError(
                    f"[{i}] {labels[i]} receives {len(lst)} taus for order {cap} along {wall}",
                    where=", ".join(c["where"] for c in lst),
                    why="an equation of order q along the wall coordinate can absorb q taus",
                    fix=(f"{', '.join(pq)} looks like the pressure of a normal-mode model: remove "
                         f"its condition and declare it with m.pressure({pq[0]!r}) (Conatus then "
                         "places the taus itself); otherwise " if pq else "")
                        + "send some conditions to another equation with into=", doc=_DOC)
        for c in conds:
            c["pos"] = basis.bounds[0 if c["side"] == "left" else 1]
        if geo.interfaces:
            _interface_conditions(comp, geo, coord, wall, eqs, unknowns, pressures, conds, assign)
    if pressures and isinstance(geo, Box) and geo.bounded_coord is not None and (
            geo.interfaces or geo.bounded_coord != wall or geo.n_walls() == 1):
        raise UnsupportedError(
            f"incompressible flow ({', '.join(q.name for q in pressures)} recognized as a pressure) "
            f"on {geo!r}", why="the gauge and divergence-tau recipe of the native layer is written "
            "for walls(a, b) with Chebyshev T or sine/cosine bases; on a compound interval, a half "
            "line or the whole line it is not derived (the constant gauge is not even representable "
            "on an unbounded axis)",
            fix="use walls(a, b), or write the problem in the IR (conatus.expert) with explicit taus",
            doc=_DOC)
    # taus
    recipe = geo.recipe
    tau_vars: dict = {}                # f -> list of (tau, cond) in declaration order
    per_f: dict = {}
    rv = None
    for i in sorted(assign):
        lst = assign[i]
        lhs, rhs = eqs[i]
        sig = lhs.tensorsig
        tdom = Domain.union(lhs.domain, None).without(coord)
        new = []
        W = sum(1 for c in lst if not c.get("interface"))       # wall conditions = taus per piece
        if geo.interfaces:
            # compound interval: conditions ordered along the axis (bottom wall, interfaces, top
            # wall; declaration order at one place) and dealt W per piece, piece after piece
            lst = sorted(lst, key=lambda c: c["pos"])
        for idx, c in enumerate(lst):
            k, piece = (idx % W + 1, idx // W) if geo.interfaces else (idx + 1, None)
            f = c["f"]
            if c["lhs"].tensorsig != sig:
                raise BoundaryConditionError(
                    f"condition '{c['text']}' and equation [{i}] {labels[i]} have different tensor "
                    "ranks", where=c["where"],
                    why="the tau of a vector equation is a vector (componentwise conditions); a "
                        "scalar condition on one component of a vector unknown needs component taus",
                    fix="give the whole vector condition (u = 0, or u=cn.free_slip for a "
                        "stress-free wall), or build component taus in the IR (expert layer)",
                    doc=_DOC)
            j = per_f.get(f, 0) + 1
            per_f[f] = j
            tau = (VectorField if sig else Field)(f"tau_{f.name}{j}", tdom)
            tau_vars.setdefault(f, []).append((tau, c))
            new.append((tau, c, k, piece))
        if recipe == "shell":
            fs = {c["f"] for c in lst}
            if len(fs) != 1 or len(lst) != 2:
                raise UnsupportedError(
                    f"shell: [{i}] {labels[i]} receives {len(lst)} condition(s) on "
                    f"{sorted(f.name for f in fs)}", where=", ".join(c["where"] for c in lst),
                    why="the shell recipe (first-order form) takes one unknown with one condition "
                        "at each wall per equation", fix="use the IR (expert layer)", doc=_DOC)
            (f,) = fs
            if rv is None:
                rv = VectorField("rvec", Domain(comp.coordsys, [basis]))
                if "rvec" in comp.fields:
                    raise InputTypeError("the name 'rvec' is reserved on a Shell (r e_r)", doc=_DOC)
                comp.fields["rvec"] = rv
                comp.ncc[rv] = lambda P_, T_, R_: np.array([0 * R_, 0 * R_, R_ + 0 * P_])
            (t1, c1, _, _), (t2, c2, _, _) = new
            L1 = lift(t1, basis, -1, lam=1)
            target = lap(f)
            if not any(n == target for n in lhs.walk()):
                raise UnsupportedError(
                    f"shell: [{i}] {labels[i]} has no lap({f.name}) term to write in first-order "
                    "form", where=labels[i],
                    why="the shell recipe rewrites lap(f) as div(grad(f) + rvec lift(tau_1))",
                    fix=f"write the diffusion of {f.name} as lap({f.name}), or use the IR", doc=_DOC)
            first = div(grad(f) + (outer(rv, L1) if f.rank == 1 else rv * L1))
            lhs = lhs.substitute({target: first}) + lift(t2, basis, -1, lam=2)   # B6: lam_1 + 1
            comp.taus.append(TauRecord(t1.name, f.name, c1["text"], c1["boundary"], i, -1, 1,
                                       c1["rule"] + "; " + _RULES["shell"] + ": tau_1 in the gradient"))
            comp.taus.append(TauRecord(t2.name, f.name, c2["text"], c2["boundary"], i, -1, 2,
                                       c2["rule"] + "; lifted (lam=2, mode N-1)"))
        else:
            lifts = []
            for tau, c, k, piece in new:
                if piece is not None:
                    lifts.append(lift(tau, basis, -k, piece=piece))
                    rule = c["rule"] + f"; Chebyshev lift into the last modes of piece {piece}"
                else:
                    lifts.append(_lift_for(recipe, tau, basis, k))
                    rule = c["rule"] + "; " + (_RULES[recipe] if not isinstance(basis, Laguerre) else
                                               "Laguerre lift into the last modes of the derivative "
                                               "basis of the equation (discretize/families.py)")
                comp.taus.append(TauRecord(tau.name, c["f"].name, c["text"], c["boundary"], i, -k,
                                           0 if recipe == "cartesian" else 2, rule, piece))
            lhs = lhs + add(*lifts)
        eqs[i] = (lhs, rhs)
    # incompressibility: gauge, integral constraint, divergence tau
    gauges = []
    for q in pressures:
        mom = [i for i, (lhs, rhs) in enumerate(eqs)
               if any(True for _ in _occurrences(lhs, q)) or any(True for _ in _occurrences(as_expr(rhs), q))]
        ws = {w for i in mom for w in (eqs[i][0].free_fields() & V) if w.rank == 1}
        cons = []
        for i, (lhs, _rhs) in enumerate(eqs):
            if i in mom or lhs.rank != 0 or lhs.time_order() > 0:
                continue
            if comp.eigenvalue is not None and comp.eigenvalue in lhs.free_parameters():
                continue
            dv = {n.args[0] for n in lhs.walk() if isinstance(n, Divergence)} & ws
            if dv:
                cons.append((i, dv))
        if len(cons) != 1 or len(cons[0][1]) != 1:
            raise BoundaryConditionError(
                f"{q.name} enters only through grad({q.name}) but "
                + ("no" if not cons else "more than one") + " constraint equation pairs with it",
                why="a pressure is fixed by one algebraic equation holding div(u) of the velocity "
                    "of its momentum equation (no time derivative)",
                fix=f"write the constraint (div(u) = 0), or give {q.name} its own equation",
                doc=_DOC)
        (ci, (w,)), = [(cons[0][0], tuple(cons[0][1]))]
        tq = Field(f"tau_{q.name}", Domain(comp.coordsys, []))
        lhs, rhs = eqs[ci]
        extra = [tq]
        corr = None
        if wall is not None and w in tau_vars:
            ts = tau_vars[w]
            if recipe == "cartesian":
                last = ts[-1][0]
                corr = dot(unit_vector(coord), lift(last, basis, -1))
                corr_txt = (f"divergence tau dot(e{wall}, lift({last.name}, {wall}, -1)) (the "
                            f"velocity tau of the lowest lifted mode)")
            elif recipe == "shell":
                corr = dot(rv, lift(ts[0][0], basis, -1, lam=1))
                corr_txt = (f"divergence tau dot(rvec, lift({ts[0][0].name}, lam=1)) (first-order "
                            "form)")
            elif recipe == "annulus" and len(ts) >= 2:
                # lifted: without it the second-order form is
                # singular in the group m = 0 (k_z = 0) of an annulus (pressure-tau null mode);
                # r e_r . lift(tau_2, lam=1) also closes continuity, the generalized tau of the
                # hand-written Taylor-Couette IVP (problems.taylor_couette)
                rva = comp.fields.get("rvec")
                if rva is None:
                    if any(u.name == "rvec" for u in comp.model.unknowns) or any(
                            g.name == "rvec" for g in comp.model.givens):
                        raise InputTypeError("the name 'rvec' is reserved on an annulus (r e_r)",
                                             doc=_DOC)
                    rva = VectorField("rvec", Domain(comp.coordsys, [basis]))
                    comp.fields["rvec"] = rva
                    if comp.coordsys.dim == 3:          # (phi, r, z)
                        comp.ncc[rva] = lambda P_, R_, Z_: np.array([0 * R_, R_ + 0 * P_, 0 * R_])
                    else:                               # (phi, r)
                        comp.ncc[rva] = lambda P_, R_: np.array([0 * R_, R_ + 0 * P_])
                t2 = ts[1][0]
                corr = dot(rva, lift(t2, basis, -1, lam=1))
                corr_txt = (f"divergence tau dot(rvec, lift({t2.name}, r, -1, lam=1)) (generalized "
                            "tau of the annulus: keeps the group m = 0 regular)")
            else:
                corr_txt = ("no divergence tau (recipe of the hand-written disk/ball "
                            "problems: the lam=2 lift keeps the system square)")
        else:
            corr_txt = "no divergence tau (no wall)"
        if corr is not None:
            extra.append(corr)
        eqs[ci] = (lhs + add(*extra), rhs)
        gauges.append((tq, q))
        comp.gauge.append(
            f"Incompressibility recognized: {q.name} enters only through grad({q.name}) and has no "
            f"condition; [{ci}] {labels[ci]} is its constraint (div({w.name}), no time derivative):")
        comp.gauge.append(f"  gauge tau {tq.name} (a constant) added to [{ci}]; integral constraint "
                          f"integ({q.name}) = 0 fixes the mean {q.name}")
        comp.gauge.append(f"  {corr_txt}")
    # normal-mode pressure: the continuity equation of the components
    # (1j*k*u + dz(w) = 0) is first order in the wall coordinate and receives no condition; as the
    # vector recipe above, it gets the divergence tau of the wall-normal component: the last tau
    # of w lifted into its last mode. No gauge: for k != 0 the pressure is fixed (i k p is in the
    # tangential equation); at k = 0 the pencil is singular, which the rank report refuses.
    for q in mode_pressures:
        if recipe != "cartesian":
            raise UnsupportedError(
                f"normal-mode pressure {q.name} on a {recipe} axis",
                why="the divergence-tau recipe of the normal-mode form is written for walls(a, b) "
                    "with a Chebyshev basis", fix="use cn.walls(a, b) (Chebyshev), or the IR "
                    "(conatus.expert) with explicit taus", doc=_DOC)
        cons = []
        for i, (lhs, _rhs) in enumerate(eqs):
            if any(True for _ in _occurrences(lhs, q)) or lhs.rank != 0 or lhs.time_order() > 0:
                continue
            if comp.eigenvalue is not None and comp.eigenvalue in lhs.free_parameters():
                continue
            ws = [w for w in unknowns if w in tau_vars and lhs.order_in(coord, [w]) == 1]
            if lhs.order_in(coord, V) == 1 and ws:
                cons.append((i, ws))
        if len(cons) != 1 or len(cons[0][1]) != 1:
            raise BoundaryConditionError(
                f"{q.name} is the pressure of a normal-mode model, but "
                + ("no" if not cons else "more than one") + " continuity equation pairs with it",
                why="its constraint is one equation without time derivative or eigenvalue, of "
                    f"first order in {wall} through one velocity component (1j*k*u + d{wall}(w) "
                    "= 0)", fix="write the continuity equation once, e.g. '1j*k*u + dz(w) = 0'",
                doc=_DOC)
        (ci, (w,)), = cons
        last = tau_vars[w][-1][0]
        eqs[ci] = (eqs[ci][0] + lift(last, basis, -1), eqs[ci][1])
        comp.gauge.append(
            f"Normal-mode pressure recognized: {q.name} enters as i k {q.name} and "
            f"d{wall}({q.name}), no condition; [{ci}] {labels[ci]} is its constraint "
            f"(d{wall}({w.name}), no time derivative):")
        comp.gauge.append(f"  divergence tau lift({last.name}, {wall}, -1) (the last tau of the "
                          f"wall-normal component {w.name}) added to [{ci}]; no gauge (k != 0 "
                          f"fixes {q.name}; k = 0 is refused as a singular pencil)")
    # written gauge of a closed geometry (native2): an equation "integ(q) = c" on a
    # geometry without walls (a streamfunction on a torus, Kolmogorov flow) is one equation more
    # than the unknowns; it gets its Lagrange multiplier, a constant tau_q added to the only
    # other equation holding q (the mean mode of that equation then reads tau_q = <rhs>, zero
    # when the right-hand side has zero mean). With walls an integral replaces a condition
    # (conditions above), and is left alone here.
    if wall is None:
        for ci, (lhs, rhs) in enumerate(eqs):
            hit = [q for q in unknowns if q.rank == 0 and lhs == integ(q)]
            if not hit or as_expr(rhs).depends_on(V):
                continue
            (q,) = hit
            others = [i for i, (l2, r2) in enumerate(eqs) if i != ci and
                      (any(True for _ in _occurrences(l2, q)) or any(True for _ in _occurrences(as_expr(r2), q)))]
            if len(others) != 1:
                raise EquationError(
                    f"[{ci}] {labels[ci]}: integ({q.name}) fixes the mean of {q.name}, but "
                    f"{len(others)} other equations hold {q.name}",
                    why="its Lagrange multiplier (a constant tau) goes to the one equation whose "
                        "mean mode leaves the mean of the unknown free",
                    fix="write the gauge in the IR (expert layer) with an explicit tau", doc=_DOC)
            ti = others[0]
            tq = Field(f"tau_{q.name}", Domain(comp.coordsys, []))
            eqs[ti] = (eqs[ti][0] + tq, eqs[ti][1])
            gauges.append((tq, None))
            comp.gauge.append(f"Written gauge: [{ci}] {labels[ci]} fixes the mean of {q.name} (no "
                              f"wall); its multiplier tau_{q.name} (a constant) is added to [{ti}] "
                              f"{labels[ti]} (its mean mode then gives tau_{q.name} = mean of the "
                              "right-hand side, zero for a zero-mean forcing)")
    # assemble: unknowns, taus (by unknown, in order), gauges; equations, conditions, integrals
    variables = list(unknowns)
    for f in unknowns:
        variables += [t for t, _ in tau_vars.get(f, [])]
    variables += [tq for tq, _ in gauges]
    equations = list(eqs)
    for c in conds:
        equations.append((c["lhs"], c["rhs"]))
        comp.labels.append(f"{c['text']} at {c['boundary']}")
    for _tq, q in gauges:
        if q is None:                   # written gauge: its equation is already in eqs
            continue
        equations.append((integ(q), 0))
        comp.labels.append(f"integ({q.name}) = 0 (gauge)")
    for t, _ in [(t, c) for lst in tau_vars.values() for t, c in lst]:
        comp.fields[t.name] = t
    for tq, _ in gauges:
        comp.fields[tq.name] = tq
    return variables, equations


def _merge_component_conditions(comp: Compiled, conds: list) -> list:
    """Scalar conditions on a vector unknown at one wall (``"u@ez = 0"``, ``"dz(u@ex) = 0"``): a full set of ``dim`` of them is merged, in declaration order, into
    ONE vector condition sum_k e_k (lhs_k - rhs_k) = 0, so that the vector-tau machinery applies
    unchanged (one vector tau per wall, as for cn.free_slip). The k-th condition becomes the k-th
    row of the vector condition: a row order, the discrete problem is the same whichever condition
    is written first. Cartesian geometries only (curvilinear components are spin components)."""
    groups: dict = {}
    for c in conds:
        if c["f"].rank == 1 and as_expr(c["lhs"]).rank == 0:
            groups.setdefault((c["f"], c["side"], c["cond"].boundary), []).append(c)
    if not groups:
        return conds
    cs = comp.coordsys
    first = {id(lst[0]): key for key, lst in groups.items()}
    out = []
    for c in conds:
        key = first.get(id(c))
        if c["f"].rank == 1 and as_expr(c["lhs"]).rank == 0 and key is None:
            continue                                   # merged into the first of its group
        if key is None:
            out.append(c)
            continue
        f, side, boundary = key
        lst = groups[key]
        if cs.curvilinear:
            raise BoundaryConditionError(
                f"scalar condition(s) on the vector {f.name} at {boundary} on a curved geometry",
                where=" ; ".join(x["where"] for x in lst),
                why="the components of a vector there are spin components; a componentwise "
                    "condition is not written for them", fix="u = 0 or u=cn.free_slip", doc=_DOC)
        if len(lst) != cs.dim:
            raise BoundaryConditionError(
                f"{len(lst)} scalar condition(s) on the vector {f.name} at {boundary}: give all "
                f"{cs.dim} (one per component)", where=" ; ".join(x["where"] for x in lst),
                why="the conditions of a vector unknown at a wall form one vector condition (one "
                    "vector tau): componentwise conditions must cover every component",
                fix=f"e.g. at({boundary!r}, '{f.name}@ez = 0', 'dz({f.name}@ex) = 0') in 2D (x, z), "
                    f"or {f.name}=cn.free_slip / cn.navier_slip(l)", doc=_DOC)
        units = [unit_vector(x) for x in cs.coords]
        lhs = add(*[e * x["lhs"] for e, x in zip(units, lst, strict=True)])
        rterms = [e * x["rhs"] for e, x in zip(units, lst, strict=True)
                  if not (isinstance(x["rhs"], numbers.Number) and x["rhs"] == 0)
                  and not (not isinstance(x["rhs"], numbers.Number) and as_expr(x["rhs"]).is_zero())]
        rhs = add(*rterms) if rterms else 0
        merged = dict(lst[0])
        merged.update(lhs=lhs, rhs=rhs,
                      text=f"{f.name} componentwise: " + "; ".join(x["text"] for x in lst),
                      where=" ; ".join(x["where"] for x in lst))
        out.append(merged)
    return out


def _g(v: float) -> str:
    return f"{v:g}"

