"""Sine/cosine wall axis of the native layer (``walls(a, b, basis="sincos")``).

Discretization: the axis [a, b] (L = b - a) with N wavenumbers n pi / L is the reflection
subspace of the 2N-mode real Fourier series on the doubled period [a, 2b - a): an odd field
(parity -1 under the reflection z -> 2a - z) is a sine series sin(n pi (z - a)/L), an even field
a cosine series. The compiler declares ``SymmetrySubspace(Reflection(z, about=a), **parities)`` on
the IR problem; the plan and the solvers run in that subspace (``reduced=True``), with no tau and
no boundary row: the wall conditions hold exactly through the parity.

Parity of each unknown, never guessed:

* read off its declared wall conditions: ``f = 0`` -> odd (sine), ``dz(f) = 0`` -> even
  (cosine), ``dz^m(f) = 0`` -> parity (-1)^(m+1); a velocity with ``cn.free_slip`` -> tensor
  parity even (tangential components cosine, normal component sine; ir/symmetry.py convention).
  The conditions must be homogeneous, involve one unknown, agree on one parity and be as many as
  on a Chebyshev axis (order along z x 2 walls / 2), so the same model text runs with either basis;
* unknowns without condition (a pressure) get the unique parity that makes every equation
  equivariant (all 2^k assignments are tried; none or several -> refused with the list);
* given fields and base states are checked numerically on the doubled grid (even or odd about
  z = a, else refused).
"""
from __future__ import annotations

import itertools
import numbers

import numpy as np

from ..errors import BoundaryConditionError, EquationError, InputTypeError, UnsupportedError
from ..ir import (EVEN, ODD, Add, Differentiate, Domain, Field, Mul, Reflection, SymmetrySubspace,
                  as_expr, integ)
from ..ir.symmetry import SymmetryBreakingError, SymmetryError, component_parity, parity_of
from .conditions import WallCondition, free_slip, no_slip

__all__ = ["reflection", "place_parity", "data_parity", "series_name", "extend_to_period",
           "physical_slice"]

_DOC = "docs/site/guide/bases.md"   # public guide page (never an internal design note)
_CHEB = ("walls(a, b) (Chebyshev, tau method: any condition)")


def reflection(geo, coordsys) -> Reflection:
    """The reflection z -> 2a - z of the sine/cosine axis (about its first wall)."""
    c = geo.sincos_coord
    return Reflection(coordsys[c], about=geo._axes[c].bounds[0])


def series_name(p, rank: int, coordsys, wall: str) -> str:
    """'sine series' / 'cosine series', or per component for a vector."""
    if rank == 0:
        return "sine series" if p == ODD else "cosine series"
    parts = []
    R_coord = coordsys[wall]
    for c in coordsys.coords:
        q = -p if c == R_coord else p
        parts.append(f"{c.name}: {'sine' if q == ODD else 'cosine'}")
    return "cosine/sine by component (" + ", ".join(parts) + ")"


# ------------------------------------------------------------------------------------------------
# conditions -> parities
# ------------------------------------------------------------------------------------------------

def _strip(e, coord):
    """(field, m, factorless) for dz^m(f) (a constant factor allowed), else None."""
    e = as_expr(e)
    if isinstance(e, Mul):
        rest = [a for a in e.args if a.free_fields()]
        if len(rest) != 1:
            return None
        e = rest[0]
    m = 0
    while isinstance(e, Differentiate):
        if e.coord != coord:
            return None
        m += 1
        e = e.args[0]
    if isinstance(e, Field):
        return e, m
    return None


def _refuse_inhomogeneous(text, where, wall, a, b):
    raise BoundaryConditionError(
        f"'{text}': a non-homogeneous wall condition on a sine/cosine axis",
        where=where,
        why="a sine series vanishes at both walls and a cosine series has a zero derivative "
            "there: only homogeneous conditions (f = 0, d" + wall + "(f) = 0, free-slip) are "
            "representable",
        fix=f"write the equations for the perturbation about the profile that carries the "
            f"boundary values (e.g. T = T_cond({wall}) + theta with theta = 0 at the walls; for "
            f"Rayleigh-Benard: dt(theta) - lap(theta) - u·e{wall} = -u·grad(theta)), or use "
            f"{wall}=" + _CHEB.replace("a, b", f"{a:g}, {b:g}"), doc=_DOC)


def _analyse(comp, cond, side: str, ns, R) -> dict:
    """One condition at one wall -> dict(f, parity, text, where, kind)."""
    from ..ir.parse import parse_equation
    from .compiler import _text
    geo = comp.model.geometry
    wall = geo.sincos_coord
    a, b = geo._axes[wall].bounds
    coord = comp.coordsys[wall]
    unknowns = {f.name: f for f in comp._unknowns}
    if cond.unknown is not None:
        if cond.unknown not in unknowns:
            raise BoundaryConditionError(f"{cond.unknown}=...: not an unknown of the model",
                                         where=f"at({cond.boundary!r}, {cond.unknown}=...)",
                                         fix="unknowns: " + ", ".join(unknowns), doc=_DOC)
        f, v = unknowns[cond.unknown], cond.value
        where = f"at({cond.boundary!r}, {cond.unknown}={v!r})"
        if isinstance(v, WallCondition):
            if f.rank != 1:
                raise BoundaryConditionError(
                    f"{cond.unknown}={v!r}: {v.name} is a condition on a velocity (a vector)",
                    where=where, fix=f"declare {cond.unknown} with m.velocity(...)", doc=_DOC)
            if v is free_slip:
                return dict(f=f, parity=EVEN, kind="free-slip", where=where,
                            text=f"free-slip ({f.name}·n = 0, d{wall}({f.name}_t) = 0)")
            if v is not no_slip:                       # cn.navier_slip: a Robin condition
                raise BoundaryConditionError(
                    f"{cond.unknown}={v!r} on a sine/cosine axis",
                    why="the Navier slip mixes u_t and its derivative: no parity satisfies it",
                    fix="walls(a, b) (Chebyshev) for a Navier slip, or cn.free_slip", where=where,
                    doc=_DOC)
            m, text = 0, f"{f.name} = 0 (no-slip)"
        elif callable(v):
            _refuse_inhomogeneous(f"{cond.unknown} = {getattr(v, '__name__', 'data')}", where,
                                  wall, a, b)
        elif isinstance(v, (tuple, list)):
            if any(c != 0 for c in v):
                _refuse_inhomogeneous(f"{cond.unknown} = {tuple(v)}", where, wall, a, b)
            m, text = 0, f"{f.name} = 0"
        elif isinstance(v, numbers.Number):
            if v != 0:
                _refuse_inhomogeneous(f"{cond.unknown} = {v:g}", where, wall, a, b)
            m, text = 0, f"{f.name} = 0"
        else:
            raise BoundaryConditionError(f"{cond.unknown}={v!r}: not a wall value", where=where,
                                         fix="0, cn.free_slip, or a text condition", doc=_DOC)
    else:
        text = cond.text.strip()
        where = f"at({cond.boundary!r}, {text!r})"
        eq = parse_equation(_text(cond.text), ns)
        rhs = as_expr(eq.rhs)
        got = eq.lhs.free_fields() & comp._V
        if rhs.depends_on(comp._V) or not got:
            raise BoundaryConditionError(f"'{text}': write the unknown on the left and the value "
                                         "on the right", where=where, doc=_DOC)
        if not rhs.is_zero():
            _refuse_inhomogeneous(text, where, wall, a, b)
        if len(got) != 1:
            raise BoundaryConditionError(
                f"'{text}' couples {len(got)} unknowns ({', '.join(sorted(g.name for g in got))})",
                where=where,
                why="on a sine/cosine axis each condition only fixes the parity of one unknown; a "
                    "condition mixing unknowns is not a parity",
                fix="use " + _CHEB + f" for coupled conditions: {wall}=walls({a:g}, {b:g})",
                doc=_DOC)
        st = _strip(eq.lhs, coord)
        if st is None:
            kind = "Robin (a sum of terms)" if isinstance(eq.lhs, Add) else "not d" + wall + "^m(f)"
            raise BoundaryConditionError(
                f"'{text}': {kind} condition on a sine/cosine axis",
                where=where,
                why="only f = 0 (sine), d" + wall + "(f) = 0 (cosine) and d" + wall + "^m(f) = 0 "
                    "(parity (-1)^(m+1)) are exact parities; a Robin or mixed condition is not",
                fix=f"use {wall}=walls({a:g}, {b:g}) (Chebyshev, tau method) for this condition",
                doc=_DOC)
        f, m = st
    q = ODD if m % 2 == 0 else EVEN                  # dz^m f = 0 at both walls: parity (-1)^(m+1)
    if f.rank == 0:
        return dict(f=f, parity=q, kind=f"d{wall}^{m}" if m else "value", where=where, text=text)
    if f.rank > 1:
        raise UnsupportedError(f"'{text}': a condition on a tensor unknown on a sine/cosine axis",
                               where=where, fix=f"use {wall}=walls({a:g}, {b:g}) (Chebyshev)",
                               doc=_DOC)
    # vector: every component must have scalar parity q; component parity is p (tangential) or
    # -p (normal, along the reflected coordinate), ir/symmetry.py convention
    comps = comp.coordsys.coords
    need = {(-q if c == coord else q) for c in comps}
    if len(need) != 1:
        raise BoundaryConditionError(
            f"'{text}' is not representable on a sine/cosine axis",
            where=where,
            why=f"{f.name} = 0 at the walls needs every component in sines, but a reflection "
                f"gives the normal component ({f.name}_{wall}) the opposite parity of the "
                "tangential ones (a velocity is cosine tangentially and sine normally, or the "
                "reverse)",
            fix=f"u=cn.free_slip (stress-free walls: u·n = 0, d{wall}(u_t) = 0) is the velocity "
                f"condition of a sine/cosine axis; no-slip walls: {wall}=walls({a:g}, {b:g}) "
                "(Chebyshev)", doc=_DOC)
    return dict(f=f, parity=need.pop(), kind="vector", where=where, text=text)


def _equivariance_error(eqs, sub: SymmetrySubspace, R, labels) -> str | None:
    """None if every equation is equivariant under R with these parities, else the reason."""
    look = sub.lookup()
    for i, (lhs, rhs) in enumerate(eqs):
        from .compiler import _terms
        terms = _terms(lhs) + ([as_expr(rhs)] if not as_expr(rhs).is_zero() else [])
        ref = None
        for t in terms:
            try:
                p = parity_of(t, R, look)
            except (SymmetryBreakingError, SymmetryError) as err:
                return f"[{i}] {labels[i]}: {err}"
            if p is None:
                continue
            if ref is None:
                ref = (t, p)
            elif p != ref[1]:
                return (f"[{i}] {labels[i]}: term '{t}' is {p} but '{ref[0]}' is {ref[1]} under "
                        f"{R!r}")
    return None


def _sub(R, parities: dict) -> SymmetrySubspace:
    return SymmetrySubspace(R, **{f.name: p for f, p in parities.items()})


def place_parity(comp, eqs, unknowns, V, ns):
    """The sine/cosine path of the compiler: parities, gauge, no tau.
    Returns (variables, [(lhs, rhs)], SymmetrySubspace)."""
    from .compiler import _occurrences, _pressure_like
    model, geo = comp.model, comp.model.geometry
    comp._V, comp._unknowns = V, unknowns
    labels = comp.labels
    wall = geo.sincos_coord
    a, b = geo._axes[wall].bounds
    coord = comp.coordsys[wall]
    R = reflection(geo, comp.coordsys)
    eqs = list(eqs)
    # conditions -> parities (declaration order, each side of the named boundary)
    recs = []
    for c in model.conditions:
        for _cname, side in geo.boundary(c.boundary):
            r = _analyse(comp, c, side, ns, R)
            r["side"] = side
            recs.append(r)
    parity: dict = {}
    why: dict = {}
    for f in unknowns:
        mine = [r for r in recs if r["f"] == f]
        if not mine:
            continue
        ps = {r["parity"] for r in mine}
        if len(ps) > 1:
            listing = "; ".join(f"{r['text']} ({series_name(r['parity'], f.rank, comp.coordsys, wall)})"
                                for r in mine)
            raise BoundaryConditionError(
                f"the wall conditions of {f.name} ask for both a sine and a cosine series",
                where=", ".join(r["where"] for r in mine), why=listing,
                fix=f"on a sine/cosine axis every condition of {f.name} must give the same parity "
                    f"(e.g. {f.name} = 0 and d{wall}(d{wall}({f.name})) = 0 at both walls); "
                    f"clamped/no-slip-type conditions need {wall}=walls({a:g}, {b:g}) (Chebyshev)",
                doc=_DOC)
        parity[f] = ps.pop()
        texts = sorted({r["text"] for r in mine}, key=[r["text"] for r in mine].index)
        why[f] = "; ".join(texts) + " at walls"
    # counts: the same rule as a Chebyshev axis (order x 2 walls / 2), each wall alike
    for f in unknowns:
        q = max((lhs.order_in(coord, [f]) for lhs, _ in eqs), default=0)
        mine = [r for r in recs if r["f"] == f]
        if q == 0 and mine:
            raise BoundaryConditionError(
                f"{f.name} is not differentiated along {wall} in any implicit term: the "
                f"condition '{mine[0]['text']}' has nothing to constrain", where=mine[0]["where"],
                fix=f"remove the condition, or check the equations of {f.name}", doc=_DOC)
        if not mine:
            continue
        per = {s: sum(1 for r in mine if r["side"] == s) for s in ("left", "right")}
        if len(mine) != q or (q % 2 == 0 and per["left"] != per["right"]):
            raise BoundaryConditionError(
                f"{f.name}: {per['left']} condition(s) at {wall}={a:g} and {per['right']} at "
                f"{wall}={b:g} for order {q} along {wall} ({q // 2 if q % 2 == 0 else q} per wall "
                "needed)",
                where=", ".join(r["where"] for r in mine),
                why="the conditions are counted as on a Chebyshev axis (order x 2 walls / 2) so "
                    "that the same model runs with either basis; the sine/cosine basis imposes "
                    "each condition at both walls",
                fix=f"declare the conditions of {f.name} at 'walls' (both)", doc=_DOC)
    # pressure recognition (structural, as on a Chebyshev axis)
    pressures = [q for q in unknowns if q.rank == 0 and q not in parity and _pressure_like(q, eqs)]
    for h in model.unknowns:
        if h.role == "pressure" and comp.fields[h.name] not in pressures:
            raise BoundaryConditionError(
                f"{h.name} is declared as a pressure but does not have the structure of one",
                why="a pressure enters only through grad(p), has no boundary condition, and is "
                    "fixed by an algebraic constraint such as div(u) = 0",
                fix=f"check the equations, or declare it with m.scalar({h.name!r})", doc=_DOC)
    # unknowns without condition: the unique parity making every equation equivariant
    free = [f for f in unknowns if f not in parity]
    ok, reasons = [], []
    for combo in itertools.product((EVEN, ODD), repeat=len(free)):
        trial = {**parity, **dict(zip(free, combo, strict=True))}
        err = _equivariance_error(eqs, _sub(R, trial), R, labels)
        if err is None:
            ok.append(combo)
        else:
            reasons.append((combo, err))
    if len(ok) != 1:
        if not free:
            raise EquationError(
                f"the equations are not equivariant with the parities of the wall conditions: "
                f"{reasons[0][1]}",
                why="on a sine/cosine axis every term of an equation must map sines to sines (or "
                    "cosines to cosines); a source or coefficient without that parity (e.g. a "
                    "constant forcing of a sine field) is not representable",
                fix=f"give such terms the parity of their equation (e.g. sin(pi*({wall}-{a:g})/L)), "
                    f"or use {wall}=walls({a:g}, {b:g}) (Chebyshev)", doc=_DOC)
        names = ", ".join(f.name for f in free)
        if not ok:
            listing = "; ".join("(" + ", ".join(f"{f.name} {'cosine' if p == EVEN else 'sine'}"
                                                for f, p in zip(free, combo, strict=True)) + f"): {err}"
                                for combo, err in reasons)
            raise BoundaryConditionError(
                f"no sine/cosine parity of {names} makes the equations equivariant",
                why=listing,
                fix=f"check the equations and conditions, or use {wall}=walls({a:g}, {b:g}) "
                    "(Chebyshev)", doc=_DOC)
        listing = "; ".join("(" + ", ".join(f"{f.name} {'cosine' if p == EVEN else 'sine'}"
                                            for f, p in zip(free, combo, strict=True)) + ")" for combo in ok)
        raise BoundaryConditionError(
            f"several parities of {names} make the equations equivariant: {listing}",
            why="the parity of an unknown without wall condition is deduced from the equations; "
                "here they do not decide it, and it is never guessed",
            fix=f"add the wall conditions of {names}, or build the subspace in the IR "
                "(SymmetrySubspace(Reflection(...), ...))", doc=_DOC)
    for f, p in zip(free, ok[0], strict=True):
        parity[f] = p
        users = [labels[i] for i, (lhs, rhs) in enumerate(eqs)
                 if any(True for _ in _occurrences(lhs, f)) or any(True for _ in _occurrences(as_expr(rhs), f))]
        why[f] = "deduced from the equivariance of " + ", ".join(users)
    # pressure gauge, as in a periodic box: tau_p + integ(p) = 0 (no divergence tau, no wall tau)
    gauges = []
    for q in pressures:
        if parity[q] == ODD:
            comp.gauge.append(f"Pressure {q.name} is a sine series: its mean is zero by parity, no "
                              "gauge needed")
            continue
        mom = [i for i, (lhs, rhs) in enumerate(eqs)
               if any(True for _ in _occurrences(lhs, q)) or any(True for _ in _occurrences(as_expr(rhs), q))]
        ws = {w for i in mom for w in (eqs[i][0].free_fields() & V) if w.rank == 1}
        from ..ir import Divergence
        cons = [i for i, (lhs, _) in enumerate(eqs)
                if i not in mom and lhs.rank == 0 and lhs.time_order() == 0
                and not (comp.eigenvalue is not None and comp.eigenvalue in lhs.free_parameters())
                and {n.args[0] for n in lhs.walk() if isinstance(n, Divergence)} & ws]
        if len(cons) != 1:
            raise BoundaryConditionError(
                f"{q.name} enters only through grad({q.name}) but "
                + ("no" if not cons else "more than one") + " constraint equation pairs with it",
                fix=f"write the constraint (div(u) = 0), or give {q.name} its own equation",
                doc=_DOC)
        ci = cons[0]
        tq = Field(f"tau_{q.name}", Domain(comp.coordsys, []))
        lhs, rhs = eqs[ci]
        eqs[ci] = (lhs + tq, rhs)
        gauges.append((tq, q))
        parity[tq] = EVEN
        comp.gauge.append(
            f"Incompressibility recognized: {q.name} enters only through grad({q.name}) and has no "
            f"condition; [{ci}] {labels[ci]} is its constraint:")
        comp.gauge.append(f"  gauge tau {tq.name} (a constant) added to [{ci}]; integral constraint "
                          f"integ({q.name}) = 0 fixes the mean {q.name} (as in a periodic box: no "
                          "wall tau, no divergence tau)")
    # records for explain()
    comp.parity.append(f"Sine/cosine axis {wall} on [{a:g}, {b:g}]: no tau, no boundary row; each "
                       f"unknown is a pure series, so its wall conditions hold exactly. Parities "
                       f"(declared SymmetrySubspace(Reflection({wall}, about={a:g}))):")
    for f in unknowns:
        comp.parity.append(f"  {f.name}: {series_name(parity[f], f.rank, comp.coordsys, wall)} "
                           f"({why[f]})")
    comp.parity.append("  Solved in the reflection-reduced space (reduced=True): plan matrices, "
                       "time stepping and eigenvalues; symmetry-breaking states do not exist here "
                       "(they would violate the wall conditions).")
    variables = list(unknowns) + [tq for tq, _ in gauges]
    equations = list(eqs)
    for tq, q in gauges:
        equations.append((integ(q), 0))
        comp.labels.append(f"integ({q.name}) = 0 (gauge)")
        comp.fields[tq.name] = tq
    sub = _sub(R, parity)
    comp.symmetry = sub
    return variables, equations, sub


# ------------------------------------------------------------------------------------------------
# data on the sine/cosine axis
# ------------------------------------------------------------------------------------------------

def _doubled_grids(comp):
    from ..reference.transforms import grid as _grid
    names = comp.model.geometry.coord_names
    grids = []
    for i, c in enumerate(names):
        g = np.asarray(_grid(comp.bases[c], dealias=False))
        sh = [1] * len(names)
        sh[i] = g.size
        grids.append(g.reshape(sh))
    return grids


def data_parity(comp, data, rank: int, what: str, *, R=None):
    """Parity (EVEN/ODD) of a callable of all coordinate grids (IR convention) on the doubled
    period of the sine/cosine axis: f(2a - z) = p R_* f(z) checked on the grid, else refused."""
    geo = comp.model.geometry
    wall = geo.sincos_coord
    a, b = geo._axes[wall].bounds
    names = geo.coord_names
    iz = names.index(wall)
    R = R or reflection(geo, comp.coordsys)
    grids = _doubled_grids(comp)
    val = np.asarray(data(*grids), dtype=complex)
    dim = len(names)
    shape = (dim,) * rank + tuple(g.size for g in grids)
    val = np.broadcast_to(val, shape)
    n2 = grids[iz].size
    mirror = np.take(val, (-np.arange(n2)) % n2, axis=rank + iz)
    scale = max(1.0, float(np.abs(val).max()))
    sig = (comp.coordsys,) * rank
    for p in (EVEN, ODD):
        sgn = np.ones(shape[:rank]) if rank else np.ones(())
        for idx in np.ndindex(*shape[:rank]):
            sgn[idx] = component_parity(p, sig, idx, R)
        sgn = sgn.reshape(shape[:rank] + (1,) * dim)
        if np.abs(mirror - sgn * val).max() <= 1e-10 * scale:
            return p
    raise InputTypeError(
        f"{what} has no parity about {wall} = {a:g} on the sine/cosine axis",
        why=f"a coefficient on a sine/cosine axis must be an even or odd function of {wall} - "
            f"{a:g}, periodic over 2L = {2 * (b - a):g} (it multiplies sine and cosine series); it "
            f"was evaluated on the doubled grid [{a:g}, {2 * b - a:g}) and f(2a - {wall}) = "
            f"+-f({wall}) fails",
        fix=f"write it as a cosine or sine series in {wall} (e.g. cos(pi*({wall}-{a:g})/{b - a:g})), "
            f"or use {wall}=walls({a:g}, {b:g}) (Chebyshev) for a general profile such as 1 - {wall}",
        doc=_DOC)


def extend_to_period(comp, fn, rank: int, p, what: str):
    """Callable of the doubled grids that evaluates ``fn`` on [a, b] only and extends it to
    (b, 2b - a) by the parity p: f(z) = p R_* f(2b - z). Odd components must vanish at the walls
    (else the extension would jump: refused, never truncated)."""
    geo = comp.model.geometry
    wall = geo.sincos_coord
    a, b = geo._axes[wall].bounds
    names = geo.coord_names
    iz = names.index(wall)
    R = reflection(geo, comp.coordsys)
    dim = len(names)
    sig = (comp.coordsys,) * rank

    def ext(*grids):
        z = grids[iz]
        L = b - a
        beyond = z > b + 1e-12 * L
        zf = np.where(beyond, 2 * b - z, z)
        gs = list(grids)
        gs[iz] = zf
        val = np.asarray(fn(*gs))
        shape = (dim,) * rank + np.broadcast_shapes(*[g.shape for g in grids])
        val = np.broadcast_to(val, shape).copy()
        scale = max(1.0, float(np.abs(val).max()))
        at_wall = np.isclose(zf, a, rtol=0, atol=1e-12 * L) | np.isclose(zf, b, rtol=0, atol=1e-12 * L)
        for idx in np.ndindex(*shape[:rank]):
            s = component_parity(p, sig, idx, R)
            v = val[idx]
            if s == ODD:
                bad = np.abs(np.where(np.broadcast_to(at_wall, v.shape), v, 0)).max()
                if bad > 1e-10 * scale:
                    comp_txt = f" component {names[idx[0]]}" if rank else ""
                    raise InputTypeError(
                        f"{what}{comp_txt} is a sine series on the sine/cosine axis but is "
                        f"{bad:.3g} at a wall",
                        why="a sine series vanishes at both walls; the odd extension of these data "
                            "would jump (they are never truncated silently)",
                        fix=f"give initial data satisfying the wall conditions, e.g. "
                            f"sin(pi*({wall}-{a:g})/{b - a:g})*...", doc=_DOC)
            val[idx] = np.where(beyond, s * v, v)
        return val
    return ext


def physical_slice(comp) -> tuple[int, slice]:
    """(axis among the coordinates, slice) of the points of [a, b] on the doubled grid."""
    geo = comp.model.geometry
    wall = geo.sincos_coord
    n = comp.N[wall]
    return geo.coord_names.index(wall), slice(0, n + 1)
