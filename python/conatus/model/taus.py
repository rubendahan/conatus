"""Boundary conditions, taus and incompressibility of the native compiler (the "Taus" and
"Incompressibility" rules of :mod:`conatus.model.compiler`): the conditions of a model are turned
into IR equations, each tau is placed in the equation where its unknown has its highest order
along the wall coordinate, and the pressure gauge is added. Split out of
:mod:`conatus.model.compiler`, which re-exports every name."""
from __future__ import annotations

import inspect
import numbers
from typing import TYPE_CHECKING

import numpy as np

from ..errors import BoundaryConditionError, EquationError, InputTypeError, UnsupportedError
from ..ir import (Add, Divergence, Domain, Field, Gradient, Mul, Parameter, Power, TensorField,
                  UnaryFunction, VectorField, add, as_expr, div, dot, grad, integ, interp, lap, lift,
                  outer, unit_vector)
from ..ir.bases import Laguerre
from ..ir.parse import Namespace, parse_equation, parse_expr
from ._compile_base import _DOC, TauRecord, _coord_args, _g, _on_all_grids, _text
from .conditions import (NavierSlip, WallCondition, curved_free_slip_operator, curved_wall_tensors,
                         free_slip_operator, navier_slip_operator, no_slip)
from .geometry import Box

if TYPE_CHECKING:
    from .compiler import Compiled


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
    """q is the pressure of a normal-mode model on one wall coordinate: a scalar unknown of a geometry with the wall coordinate only, written as on paper for
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
    """Curved geometries: a given field that depends on the wall coordinate, in the
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


# Shown by explain(): the rule itself, never an internal test or tutorial file name.
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
