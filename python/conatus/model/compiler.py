"""Compile a native :class:`~conatus.model.Model` into the IR (the expert layer) at a resolution.

Every automatic choice is made by a fixed rule and recorded (``Compiled.explain()``); anything the
rules cannot decide is refused with the explicit override. Rules:

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

The conditions, taus and gauge are placed by :mod:`conatus.model.taus`; helpers shared by both
modules are in :mod:`conatus.model._compile_base`. This module re-exports their names.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from ..errors import BoundaryConditionError, EquationError, NonlinearLHSError
from ..ir import (EVP, IVP, LBVP, NLBVP, Add, Constant, Domain, Field, Mul, Parameter, TimeDerivative,
                  VectorField, add, as_expr, neg)
from ..ir.bases import Hermite, Laguerre
from ..ir.parse import Namespace, parse_equation, parse_expr
from . import sincos as _sincos
from ..ir.expr_base import NONPOLY
from .geometry import DEFAULT_DEALIAS, WEIGHTED_DEALIAS
from ._compile_base import (  # noqa: F401 (re-export: moved to _compile_base.py)
    TauRecord, _DOC, _UNICODE, _coord_args, _g, _on_all_grids, _text,
)
from .taus import (  # noqa: F401 (re-export: moved to taus.py)
    _RULES, _TRACE_SAFE, _condition_exprs, _curved_wall_tensors, _interface_conditions, _lift_for,
    _merge_component_conditions, _normal_mode_pressure_like, _occurrences, _place_taus,
    _pressure_like, _resolve_into, _symbolic_value, _takes_time, _wall_traces,
)

__all__ = ["Compiled", "compile_model", "TauRecord", "dealias_factor", "default_dealias_map"]

_EQ_DOC = "docs/site/guide/equations.md"           # the time t in equations (_check_time)

def _terms(e):
    return list(e.args) if isinstance(e, Add) else ([] if e.is_zero() else [e])


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
    degree: int = 1                 # highest product degree of a term (unknowns + given fields)
    # highest polynomial degree in the unknowns only (Hermite/Laguerre floor); set by
    # compile_model, unannotated so that it is not a constructor argument (frozen API surface)
    unknown_degree = 1
    nonpoly: list = dc_field(default_factory=list)      # terms not polynomial in unknowns/givens
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
        out.append(f"Dealiasing: {deal}" + (f" ({dealias_rule_text(self.degree, self.bases, self.unknown_degree)}; override with "
                                             "dealias=)" if self.dealias_default else " (given)"))
        if self.nonpoly:
            out.append("  not polynomial in the unknowns and given fields (no dealiasing factor removes their "
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
        the tau names and modes and the bases are left to ``explain(full=True)`` (the default text does not lead with tau lifting and solver buckets)."""
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
        if weighted and self.unknown_degree > 1:
            out.append(f"  approximate along {', '.join(weighted)}: products of Hermite/Laguerre "
                       "functions are not alias-free at any factor (resolve the fields, check the "
                       "spectral tails)" + ("; default there max(2, (p+1)/2), which lowers the "
                                            "aliasing of a quadratic product 10-100x against 3/2"
                                            if self.dealias_default else ""))
        out += [f"  warning: {t}" for t in self.illposed]
        if self.nonpoly:
            out.append("  warning: not polynomial in the unknowns and given fields (no dealiasing factor removes "
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
                             + ["(compiler decisions and the IR problem: explain(full=True))"])
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

def dealias_factor(degree: int, weighted: bool = False, unknown_degree: int | None = None) -> float:
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
    model (p = 1) keeps 3/2 on every axis.

    ``degree`` counts the spatially varying given fields as factors;
    ``unknown_degree`` (default ``degree``) is the degree in the unknowns alone, which decides
    the weighted-axis floor (a model linear in the unknowns keeps 3/2 there)."""
    p = max(int(degree), 1)
    q = p if unknown_degree is None else max(int(unknown_degree), 1)
    floor = WEIGHTED_DEALIAS if (weighted and q >= 2) else DEFAULT_DEALIAS
    return max(floor, (p + 1) / 2)


def default_dealias_map(geometry, degree: int, unknown_degree: int | None = None) -> dict[str, float]:
    """Per-coordinate default factor (dealias=None): ``dealias_factor`` with the Hermite /
    Laguerre floor on the geometry's whole-line and half-line axes."""
    weighted = set(geometry._weighted_coords())
    return {c: dealias_factor(degree, c in weighted, unknown_degree) for c in geometry.coord_names}


def _weighted_axes(bases: Mapping | None) -> list[str]:
    """Coordinates whose basis is Hermite or Laguerre (products not alias-free at any factor)."""
    out = []
    for c, b in (bases or {}).items():
        if isinstance(b, (Hermite, Laguerre)):
            out.append(str(c))
    return out


def dealias_rule_text(degree: int, bases: Mapping | None = None,
                      unknown_degree: int | None = None) -> str:
    """The sentence explain() prints for a default dealiasing factor; ``bases`` (coordinate ->
    basis) names the Hermite/Laguerre axes, where it is approximate (and floored at 2 for
    products of the unknowns)."""
    f = _frac(dealias_factor(degree))
    p = max(int(degree), 1)
    text = (f"declared default {f} of the native layer: max(3/2, (p+1)/2) for the highest "
            f"polynomial degree p = {p} of the equations in the unknowns and the given fields; "
            "on Fourier and polynomial bases a product of p fields, each truncated at the N "
            "modes of the resolution, is then computed without aliasing (a given field that is "
            "not resolved at N modes is truncated first: check its spectral tail)")
    weighted = _weighted_axes(bases)
    if weighted:
        fw = _frac(dealias_factor(degree, True, unknown_degree))
        text += (f"; approximate on the Hermite/Laguerre axis {', '.join(weighted)} (products of "
                 "Hermite or Laguerre functions are not alias-free at any factor: resolve the "
                 "fields and check the spectral tails)")
        if (p if unknown_degree is None else unknown_degree) >= 2:
            text += (f"; there the default is {fw} = max(2, (p+1)/2), because the factor only "
                     "reduces aliasing on these axes and 2 instead of 3/2 lowers the error of a "
                     "quadratic product by one to two decades")
    return text


def _max_degree(raw, V, G=frozenset()) -> tuple[int, int, list[str]]:
    """(highest polynomial degree of a term in the unknowns and the spatially varying given
    fields ``G``, the same in the unknowns alone, terms that are not polynomial in them).

    A given field enters a pseudospectral product exactly like an unknown (its coefficients are
    truncated at the same N modes), so U*u*u is a cubic product (review finding 3: counting the
    unknowns only gave 3/2 and aliased cos(7x)^3 at N = 16). Counting every given field is
    conservative. A non-polynomial function of given fields alone (exp(G), 1/G) is one more
    factor, like a given field; a term not polynomial in the unknowns is listed."""
    p, q, nonpoly = 1, 1, []
    G = frozenset(G)
    vars_ = frozenset(V) | G
    for eq in raw:
        for side in (eq.lhs, eq.rhs):
            for t in _terms(side):
                t, stand_ins = _given_functions_as_factors(t, G)
                for d in t.degrees(vars_ | stand_ins):
                    if d >= NONPOLY:
                        nonpoly.append(str(t))
                    else:
                        p = max(p, d)
                q = max([q] + [d for d in t.degrees(V) if d < NONPOLY])
    return p, q, nonpoly


def _given_functions_as_factors(e, G: frozenset):
    """(``e`` with every non-polynomial function of given fields alone replaced by a stand-in
    field, the stand-ins).

    ``exp(G)``, ``sqrt(G)`` or ``1/G`` is a known coefficient like ``G`` itself: one factor of the
    product, evaluated on the grid. Without this, ``exp(G)*u**3`` was "not polynomial" and lost the
    degree 3 of its unknowns (default factor 3/2 instead of 2 on v0.1.0); it
    now counts as a product of four fields."""
    if not G:
        return e, frozenset()
    mapping: dict = {}

    def visit(x):
        ff = x.free_fields()
        if ff and ff <= G and max(x.degrees(G)) >= NONPOLY:
            if x not in mapping:
                f0 = min(ff, key=lambda f: f.name)
                mapping[x] = Field(f"_given_function_{len(mapping)}", f0.domain, x.tensorsig)
            return
        for a in x.args:
            visit(a)

    visit(e)
    if not mapping:
        return e, frozenset()
    return e.substitute(mapping), frozenset(mapping.values())


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
            else default_dealias_map(geo, degree, getattr(model, "_unknown_degree", None)))
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

    # given fields that vary in space count as factors of a product (constants do not)
    spatial_given = frozenset(f for f in ncc if f.domain is not None and f.domain.bases)
    pdeg, qdeg, nonpoly = _max_degree(raw, V, spatial_given)
    comp = Compiled(model, res, deal, kind, None, ncc, fields, params, eig, cs, bases, D, labels,
                    split=split, dealias_default=dealias is None, degree=pdeg, nonpoly=nonpoly,
                    overrides=overrides)
    comp._time_candidate = tpar
    comp.unknown_degree = qdeg
    if not check_conditions:
        return comp
    if kind == "IVP":                # backward diffusion / negative viscosity
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
    _warn_constant_t(comp)
    traced = comp.__dict__.get("_traced", set())
    if traced:                      # a given used only through its wall trace (A22): no data
        used = {f for eq in pb.equations for side in (eq.lhs, as_expr(eq.rhs))
                for f in side.free_fields()}
        for f in traced - used:
            comp.ncc.pop(f, None)
    return comp


def _warn_constant_t(comp: Compiled) -> None:
    """A time-dependent model that declares ``t`` itself, as a parameter or a given field, and
    uses it in its equations or boundary values: ``t`` is then that constant, not the time (the native time is ``t`` only when the model does not use the name), so a forcing ``sin(t)``
    stays frozen at its declared value for the whole run. Said with a DefaultChoiceWarning
    naming the value (review of the inputs audit: it ran without a word)."""
    if comp.kind != "IVP":
        return
    model = comp.model
    what = None
    if "t" in model.params:
        what = f"the parameter t = {model.params['t']!r}"
        used = any(p.name == "t" for eq in comp.problem.equations for side in (eq.lhs, eq.rhs)
                   for p in as_expr(side).free_parameters())
    elif any(g.name == "t" for g in model.givens):
        what = "the given field t"
        used = any(f.name == "t" for eq in comp.problem.equations for side in (eq.lhs, eq.rhs)
                   for f in as_expr(side).free_fields())
    if what is None or not used:
        return
    from ..errors import DefaultChoiceWarning, _warn_user
    _warn_user(f"model {model.name!r}: 't' in the equations or boundary values is {what}, a "
               "constant, not the time: a forcing such as sin(t) keeps its value for the whole "
               "run. The time of a model is 't' only when the model does not declare the name "
               "itself; rename the declaration (e.g. t0) to use the time, or keep it if a "
               "constant was meant", DefaultChoiceWarning)


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
