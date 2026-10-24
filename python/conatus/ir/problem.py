"""Equations and problem declarations.

The IR declares and *counts*; it does not discretize. The LHS/RHS split is
checked: the LHS must be homogeneous linear in the problem variables (it is
treated implicitly), the RHS holds explicit terms. The rank report is an
IR-level pre-check; the structural rank of the discrete system is checked
by the discretization plan.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

# Public exception classes live in conatus.errors; ProblemError is EquationError.
from ..errors import (BoundaryConditionError, InputTypeError, NonlinearLHSError, ProblemError,
                      ResolutionError, TensorError, UndeclaredVariableError)
from .coords import Coordinate
from .domain import Domain
from .expr import (Add, Expr, Field, Integrate, Interpolate, Lift, Parameter, TimeDerivative, Zero,
                   _sig_str, as_expr, has_nonpolynomial, interp, mul)
from .symmetry import Parity, Reflection, SymmetryBreakingError, SymmetrySubspace, parity_of

__all__ = ["Equation", "Problem", "IVP", "LBVP", "EVP", "NonlinearLHSError", "RankReport",
           "ProblemError"]


class Equation:
    """``lhs = rhs`` (no check here; the Problem checks against its variables)."""

    def __init__(self, lhs: Any, rhs: Any = 0) -> None:
        lhs, rhs = as_expr(lhs), as_expr(rhs)
        if rhs.is_zero() and rhs.tensorsig != lhs.tensorsig:
            rhs = Zero(lhs.tensorsig)
        if lhs.tensorsig != rhs.tensorsig:
            raise TensorError(
                f"equation sides differ: {_sig_str(lhs.tensorsig)} vs {_sig_str(rhs.tensorsig)}",
                where=f"equation '{lhs} = {rhs}'",
                why="both sides of an equation must have the same tensor signature (a vector "
                    "equation has a vector on each side)",
                fix="compare the two sides: take a component (dot(ez, u)), a divergence "
                    "or a gradient so that both sides agree; scalar 0 is accepted on either side",
                doc="docs/site/guide/errors.md#tensorerror")
        self.lhs, self.rhs = lhs, rhs

    @property
    def domain(self) -> Domain | None:
        """Union of the domains of both sides."""
        return Domain.union(self.lhs.domain, self.rhs.domain)

    @property
    def n_components(self) -> int:
        """Number of tensor components of the equation."""
        n = 1
        for cs in self.lhs.tensorsig:
            n *= cs.dim
        return n

    def __str__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def __repr__(self) -> str:
        return f"Equation({self})"

    def explain(self) -> str:
        """The equation and its left-hand side tree."""
        return f"Equation: {self}\n  LHS:\n    " + self.lhs.explain().replace("\n", "\n    ")


def _terms(e: Expr) -> Sequence[Expr]:
    return e.args if isinstance(e, Add) else (e,)


def _n_comp(f: Expr) -> int:
    n = 1
    for cs in f.tensorsig:
        n *= cs.dim
    return n


@dataclass
class DirectionReport:
    """IR-level counts along one non-periodic coordinate: differential orders, lifted taus,
    boundary conditions and tau variables (``consistent`` when all four agree)."""

    coord: Coordinate
    order_sum: int                      # Σ over bulk equation components of the diff. order
    lifted_tau_components: int          # Σ lifted tau components in bulk equations
    boundary_components: int            # boundary-condition equation components
    tau_variable_components: int        # variables lifted along coord (tau variables)

    @property
    def consistent(self) -> bool:
        """True when the order sum, lifted taus, boundary components and tau variables agree."""
        return (self.order_sum == self.lifted_tau_components == self.boundary_components
                == self.tau_variable_components)


@dataclass
class RankReport:
    """IR-level count of a problem: unknown and equation components, gauges and one
    :class:`DirectionReport` per non-periodic coordinate; ``status`` and ``explain()``."""

    variable_components: int
    equation_components: int
    gauge_components: int
    directions: list[DirectionReport] = dc_field(default_factory=list)

    @property
    def square(self) -> bool:
        """True when there are as many equation components as variable components."""
        return self.variable_components == self.equation_components

    @property
    def status(self) -> str:
        """CONSISTENT, CHECK or FAIL of the count analysis."""
        if not self.square:
            return "FAIL"
        if all(d.consistent for d in self.directions):
            return "CONSISTENT"
        return "CHECK"

    def explain(self) -> str:
        """The count analysis: variable and equation components and the per-direction tau
        balance."""
        lines = [
            "System count analysis (IR level; the discrete rank is checked by discretize())",
            "-" * 72,
            f"Unknown field components   {self.variable_components:6d}",
            f"Equation components        {self.equation_components:6d}",
            f"Gauge/integral constraints {self.gauge_components:6d}",
        ]
        for d in self.directions:
            lines += [
                f"Direction {d.coord.name} (non-periodic):",
                f"  sum of differential orders (bulk)  {d.order_sum:4d}",
                f"  lifted tau components              {d.lifted_tau_components:4d}",
                f"  boundary-condition components      {d.boundary_components:4d}",
                f"  tau variable components            {d.tau_variable_components:4d}",
                f"  -> {'consistent' if d.consistent else 'MISMATCH (check gauge / first-order form)'}",
            ]
        lines.append(f"Status: {self.status}")
        if self.status == "FAIL":
            lines.append("  the system is not square: "
                         f"{self.equation_components} equation vs "
                         f"{self.variable_components} unknown components")
        elif self.status == "CHECK":
            lines.append("  counts differ in some direction: legitimate for gauges (pressure) "
                         "or mixed-order systems, otherwise a missing/extra BC or tau")
        return "\n".join(lines)

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import rank_report_html
        return rank_report_html(self)


class Problem:
    """Common machinery: variables, equations, LHS/RHS checks, counting, explain()."""

    kind = "abstract"

    def __init__(self, variables: Iterable[Field], *, name: str | None = None,
                 symmetry: SymmetrySubspace | None = None, namespace: Any = None) -> None:
        self.variables: tuple[Field, ...] = tuple(variables)
        # names that string equations may use besides the variables (conatus.ir.parse)
        self.namespace = namespace
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ProblemError(f"duplicate variable names: {names}",
                               where=f"{self.kind}([...]) variables {dup}",
                               why="variables are matched by name in data, checkpoints and reports",
                               fix="give each Field a distinct name (Field('u', D), Field('v', D))")
        for v in self.variables:
            if not isinstance(v, Field):
                raise InputTypeError(
                    f"variables must be Fields, got {v!r}", where=f"{self.kind}([...])",
                    why="the unknowns of a problem are Field / VectorField objects",
                    fix="build them with Field('name', domain) or VectorField('name', domain) "
                        "and pass the objects, not their names")
        self.name = name or self.kind
        self.equations: list[Equation] = []
        self.symmetry = symmetry
        self.equation_parities: list[dict[Reflection, Parity | None]] = []
        self.redundant_bcs: list[tuple[int, int, Reflection]] = []
        self._auto_taus: dict[int, list] = {}
        self._tau_log: list[str] = []
        self._bc_index: set[int] = set()
        if symmetry is not None:
            for v in self.variables:
                for R in symmetry.generators:
                    symmetry.parity(v, R)  # every unknown must declare its parity

    @property
    def _vars(self) -> frozenset[Field]:
        return frozenset(self.variables)

    def add_equation(self, eq: Equation | tuple | str, namespace: Any = None) -> Equation:
        """Add ``Equation``, ``(lhs, rhs)`` or a string ``"lhs = rhs"`` (parsed by
        ``conatus.ir.parse`` in the declared namespace: the variables, ``self.namespace`` and
        ``namespace``; no eval)."""
        if isinstance(eq, str):
            eq = self._parse(eq, namespace)
        if isinstance(eq, tuple):
            eq = Equation(*eq)
        self._check_lhs(eq)
        self._check_rhs(eq)
        if self.symmetry is not None:
            self._check_symmetry(eq)
        self.equations.append(eq)
        return eq

    # boundary conditions with declared automatic taus -------------------------
    def add_bc(self, *conditions: Any, into: int | None = None) -> list[Equation]:
        """Add boundary conditions and their tau terms (standard tau), declared in explain().

        Each condition is an expression meaning ``expr = 0`` or a tuple ``(lhs, rhs)``. All
        conditions of one call act along one coordinate (read off their ``interp`` nodes). For
        each condition one tau variable (constant along that coordinate) is created and lifted
        into the next highest mode of the target equation's basis: N-1, N-2, ... in the order
        of the conditions. The target is the only bulk equation along the coordinate, or
        ``into`` (equation index) when there are several: never guessed. With a symmetry
        subspace the taus and their parities must be declared by hand (refused here).
        """
        from .expr import Interpolate
        from .expr import lift as _lift
        if self.symmetry is not None:
            raise BoundaryConditionError(
                "add_bc: with a symmetry subspace, declare the tau variables and their parities "
                "explicitly (automatic taus would hide a choice)",
                where=f"{self.kind} '{self.name}' (symmetry={self.symmetry!r})",
                why="each tau variable needs a declared parity in the subspace",
                fix="declare tau fields, add their lift(tau, basis, -k) terms to the bulk equation, "
                    "give their parities in SymmetrySubspace(..., tau=EVEN) and add the conditions "
                    "with add_equation (docs/site/guide/taus.md)",
                doc="docs/site/guide/errors.md#boundaryconditionerror")
        conditions = tuple(self._parse(c, None) if isinstance(c, str) else c for c in conditions)
        eqs = [c if isinstance(c, Equation) else Equation(*c) if isinstance(c, tuple)
               else Equation(c, 0) for c in conditions]
        coords = {n.coord for eq in eqs for n in eq.lhs.walk() if isinstance(n, Interpolate)}
        if len(coords) != 1:
            raise BoundaryConditionError(
                f"add_bc: conditions must act along exactly one coordinate "
                f"(found {sorted(c.name for c in coords)}); call add_bc once per coordinate",
                where="add_bc(" + ", ".join(str(e) for e in eqs) + ")",
                why="each condition is a trace interp(f, x, position); the coordinate of the "
                    "trace says which boundary it acts on",
                fix="write every condition with interp(field, coord, 'left'|'right'|value) and "
                    "group them by coordinate, one add_bc call per coordinate")
        (coord,) = coords
        bulk = [i for i, eq in enumerate(self.equations)
                if eq.domain is not None and eq.domain.has(coord) and i not in self._bc_index]
        if into is None:
            if len(bulk) != 1:
                listing = "; ".join(f"[{i}] {self.equations[i]}" for i in bulk) or "none"
                raise BoundaryConditionError(
                    f"add_bc: {len(bulk)} bulk equations along {coord.name}; "
                    "say which one receives the taus with into=<index>",
                    where=f"bulk equations along {coord.name}: {listing}",
                    why="the tau terms of a condition go into one equation; with several "
                        "equations the choice changes the discrete system, so it is never guessed",
                    fix="add_bc(..., into=i) with i the equation that the condition closes "
                        "(usually the one where the constrained field has its highest derivative)"
                    if bulk else f"add a bulk equation along {coord.name} before its conditions")
            into = bulk[0]
        target = self.equations[into]
        basis = target.domain.basis(coord)
        self._check_bc_count(coord, basis, into, eqs)
        taken = self._auto_taus.setdefault(into, [])
        new_taus = []
        for _eq in eqs:
            n = len(taken) + len(new_taus) + 1
            new_taus.append(Field(f"tau_{coord.name}{into}_{n}", target.domain.without(coord),
                                  target.lhs.tensorsig))
        lifts = [_lift(t, basis, -(len(taken) + j + 1)) for j, t in enumerate(new_taus)]
        new_target = Equation(target.lhs + sum(lifts[1:], lifts[0]), target.rhs)
        self.variables = self.variables + tuple(new_taus)
        self._check_lhs(new_target)
        self.equations[into] = new_target
        taken += new_taus
        self._tau_log.append(
            f"equation [{into}]: {len(taken)} tau variable(s) {[t.name for t in taken]} lifted into "
            f"modes N-1..N-{len(taken)} of {type(basis).__name__}({coord.name}, N={basis.size}) "
            "(standard tau, placed by add_bc)")
        out = []
        for eq in eqs:
            self._bc_index.add(len(self.equations))
            out.append(self.add_equation(eq))
        return out

    def _check_bc_count(self, coord, basis, into: int, eqs: list[Equation]) -> None:
        """Refuse more boundary conditions than the differential order along ``coord`` (otherwise
        an extra condition gets an extra tau, the square system solved to a 1e-15 discrete
        residual and the PDE itself was no longer satisfied), a periodic direction, and a basis
        with fewer modes than tau terms."""
        V = self._vars
        where = "add_bc(" + ", ".join(str(e.lhs) + ("" if e.rhs.is_zero() else f" = {e.rhs}")
                                      for e in eqs) + ")"
        if basis.periodic:
            raise BoundaryConditionError(
                f"boundary condition along the periodic coordinate {coord.name}",
                where=where,
                why=f"{type(basis).__name__}({coord.name}) is periodic: there is no boundary, "
                    "periodicity is built into the basis and no tau can be lifted",
                fix=f"remove the condition along {coord.name}; to fix a mean value (gauge), add "
                    f"an integral constraint integ(u) = 0 with a constant tau, or use a "
                    f"non-periodic basis (Chebyshev) if {coord.name} is bounded by walls")
        bulk = [i for i, eq in enumerate(self.equations)
                if eq.domain is not None and eq.domain.has(coord) and i not in self._bc_index]
        order = sum(self.equations[i].n_components * self.equations[i].lhs.order_in(coord, V)
                    for i in bulk)
        have = sum(eq.n_components for i, eq in enumerate(self.equations) if i not in bulk
                   and any(isinstance(n, Interpolate) and n.coord == coord for n in eq.lhs.walk()))
        new = sum(eq.n_components for eq in eqs)
        if have + new > order:
            raise BoundaryConditionError(
                f"too many boundary conditions along {coord.name}: {have + new} condition "
                f"component(s) for a total differential order {order}",
                where=f"{where} (already declared along {coord.name}: {have})",
                why="each condition adds one tau term; beyond the differential order the extra "
                    "taus make the discrete system square again but the PDE is no longer the one "
                    "written (the solution satisfies a perturbed equation to rounding error)",
                fix=f"keep {order - have} condition component(s) along {coord.name} (e.g. u at "
                    "both walls for a second-order equation, u and du/dx at both walls for a "
                    "fourth-order one); pb.rank_report().explain() shows the counts")
        taken = len(self._auto_taus.get(into, []))
        if taken + len(eqs) > basis.size:
            raise ResolutionError(
                f"{type(basis).__name__}({coord.name}) has N={basis.size} modes but "
                f"{taken + len(eqs)} tau terms are requested",
                where=f"{where} into equation [{into}]",
                why="each condition is lifted into one of the highest modes N-1, N-2, ...; the "
                    "basis must have more modes than tau terms, and many more to resolve the "
                    "solution",
                fix=f"increase the resolution of {coord.name} (N >= {4 * (taken + len(eqs))} "
                    "as a starting point) and check convergence in N",
                doc="docs/site/guide/errors.md#resolutionerror")

    def _parse(self, text: str, namespace: Any) -> Equation:
        """String equation -> Equation in the namespace: variables, then ``self.namespace``,
        then the call's ``namespace`` (later entries win; see conatus.ir.parse.Namespace)."""
        from .parse import Namespace
        ns = Namespace.of({v.name: v for v in self.variables}, self.namespace, namespace)
        from .parse import parse_equation
        return parse_equation(text, ns)

    def _tau_explain(self) -> list[str]:
        return ["Tau terms (declared):"] + ["  " + s for s in self._tau_log] if self._tau_log else []

    # symmetry --------------------------------------------------------------
    def _reference_term(self, terms: Sequence[Expr]) -> Expr:
        V = self._vars
        dep = [t for t in terms if t.depends_on(V)]
        return max(dep or list(terms), key=lambda t: t.time_order())

    def _check_symmetry(self, eq: Equation) -> None:
        sym = self.symmetry
        look = sym.lookup()
        parities: dict[Reflection, Parity | None] = {}
        for R in sym.generators:
            terms = list(_terms(eq.lhs))
            ref = self._reference_term(terms)
            try:
                p_ref = parity_of(ref, R, look)
                sides = [(t, parity_of(t, R, look)) for t in terms]
                sides.append((eq.rhs, parity_of(eq.rhs, R, look)))
            except SymmetryBreakingError as err:
                raise SymmetryBreakingError(f"equation '{eq}': {err}") from None
            for t, p in sides:
                if p is not None and p_ref is not None and p != p_ref:
                    raise SymmetryBreakingError(
                        f"equation '{eq}' is not equivariant under {R!r}: term '{t}' is {p} "
                        f"but '{ref}' is {p_ref}. The term breaks the declared symmetry "
                        "(fix the parities or drop the symmetry).")
            if p_ref is None:
                p_ref = next((p for _, p in sides if p is not None), None)
            parities[R] = p_ref
            self._check_bc_mirror(eq, R, p_ref)
        self.equation_parities.append(parities)

    def _mirror(self, e: Expr, R: Reflection) -> Expr | None:
        """Replace traces along R.coord by traces at the mirror points; None if none."""
        mapping = {}
        for n in e.walk():
            if isinstance(n, Interpolate) and n.coord == R.coord:
                basis = n.args[0].domain.basis(n.coord)
                if not R.is_fixed(n.position, basis):
                    mapping[n] = interp(n.args[0], n.coord, R.mirror(n.position, basis))
        return e.substitute(mapping) if mapping else None

    def _check_bc_mirror(self, eq: Equation, R: Reflection, p: Parity | None) -> None:
        mirrored = self._mirror(eq.lhs, R)
        if mirrored is None:
            return
        j = len(self.equations)
        for i, other in enumerate(self.equations):
            if other.lhs != mirrored:
                continue
            if eq.lhs.rank == 0 and p is not None:
                ok = mul(p, other.rhs) == eq.rhs
            else:
                ok = other.rhs.is_zero() and eq.rhs.is_zero()
            if not ok:
                raise SymmetryBreakingError(
                    f"boundary conditions [{i}] '{other}' and [{j}] '{eq}' are mirror images under "
                    f"{R!r} but their right-hand sides are incompatible with parity {p}")
            self.redundant_bcs.append((i, j, R))

    def symmetry_report(self) -> str:
        """Parities of the equations and unknowns under each declared reflection."""
        if self.symmetry is None:
            return "No symmetry subspace declared."
        sym = self.symmetry
        lines = [sym.explain(), "Equivariance (parity of each equation):"]
        for k, (_eq, ps) in enumerate(zip(self.equations, self.equation_parities, strict=True)):
            desc = ", ".join(f"{'zero' if p is None else p} under {R!r}" for R, p in ps.items())
            lines.append(f"  [{k}] {desc}")
        for R in sym.generators:
            lines.append(f"Under {R!r}:")
            for v in self.variables:
                p = sym.parity(v, R)
                extra = ""
                if v.rank == 1:
                    from .symmetry import component_parity
                    extra = " (" + ", ".join(
                        f"{v.name}_{c.name} {component_parity(p, v.tensorsig, (i,), R)}"
                        for i, c in enumerate(v.tensorsig[0].coords)) + ")"
                if not v.domain.has(R.coord) and p != 1 and v.rank == 0:
                    extra += " -> constant along the mirror direction and odd: vanishes identically"
                lines.append(f"  {v.name:10s} {p}{extra}")
        for i, j, R in self.redundant_bcs:
            lines.append(f"  BC [{i}] and [{j}] are mirror images under {R!r}: redundant in the "
                         "subspace, one of them (and its tau) is dropped at discretization")
        return "\n".join(lines)

    def _check_lhs(self, eq: Equation) -> None:
        V = self._vars
        bad = []
        for t in _terms(eq.lhs):
            d = t.degrees(V)
            if d != frozenset({1}):
                bad.append((t, d))
        for t, d in bad:
            names = [v.name for v in self.variables]
            rest = " + ".join(str(o) for o in _terms(eq.lhs) if o is not t) or "0"
            moved = f"-({t})" if eq.rhs.is_zero() else f"{eq.rhs} - ({t})"
            if d == frozenset({0}):
                stray = sorted({f.name for f in t.free_fields()} - set(names))
                msg = (f"LHS term '{t}' does not depend on the variables "
                       f"({', '.join(names)}); move it to the RHS")
                if stray and any(isinstance(n, TimeDerivative) for n in t.walk()):
                    raise UndeclaredVariableError(
                        msg, where=f"equation '{eq}', term '{t}'",
                        why=f"{', '.join(repr(n) for n in stray)} is not a variable of the "
                            "problem; a field that is not a variable is fixed data, and fixed "
                            "data has no time derivative to solve for",
                        fix=f"declare it as an unknown: {self.kind}([{', '.join(names + stray)}]"
                            + (", eigenvalue=...)" if self.kind == "EVP" else ")"),
                        doc="docs/site/guide/errors.md#undeclaredvariableerror")
                raise ProblemError(
                    msg, where=f"equation '{eq}', term '{t}'",
                    why="the LHS is the implicit linear operator acting on the unknowns; a term "
                        "without any unknown is a source (forcing) term",
                    fix=(f"if {', '.join(stray)} should be unknown(s), declare them in "
                         f"{self.kind}([...]); if it is given data, " if stray else "")
                        + f"write the term on the RHS with the opposite sign: "
                          f"add_equation(({rest}, {moved}))")
            if self.kind == "IVP":
                fix = (f"write it on the RHS with the opposite sign (explicit in time): "
                       f"add_equation(({rest}, {moved}))")
            else:
                fix = (f"a {self.kind} is linear: linearize the nonlinear term around a base "
                       "state (IVP.linearize(...)/conatus.linearize) or declare an NLBVP "
                       "(solved by Newton)")
            raise NonlinearLHSError(
                f"LHS term '{t}' is not linear in the variables (degrees {sorted(d)}); "
                "nonlinear terms belong on the RHS (treated explicitly).",
                where=f"equation '{eq}', term '{t}'",
                why="the LHS is assembled once into sparse matrices (implicit part); a product "
                    "of unknowns has no matrix",
                fix=fix)
        # Non-variable fields left in the LHS are non-constant coefficients (see ncc_fields).

    def _check_rhs(self, eq: Equation) -> None:
        """Hook for subclasses; the base problem accepts any RHS."""

    def check(self) -> None:
        """Whole-problem consistency checks that single equations cannot see (hook)."""

    def ncc_fields(self) -> frozenset[Field]:
        """Non-variable fields appearing in LHS terms (non-constant coefficients)."""
        out: frozenset = frozenset()
        for eq in self.equations:
            out |= eq.lhs.free_fields() - self._vars
        return out

    # counting -------------------------------------------------------------
    def rank_report(self) -> RankReport:
        """IR-level counts (before discretization; the discrete rank is ``Plan.check``)."""
        V = self._vars
        var_comp = sum(_n_comp(v) for v in self.variables)
        eq_comp = sum(eq.n_components for eq in self.equations)
        gauge = sum(eq.n_components for eq in self.equations
                    if eq.domain is not None and not eq.domain.coords
                    and any(isinstance(e, Integrate) for e in eq.lhs.walk()))
        dom = None
        for v in self.variables:
            dom = Domain.union(dom, v.domain)
        dirs = []
        for c in (dom.nonperiodic_coords if dom else ()):
            order_sum = lifted = bcs = 0
            tau_vars: set = set()
            for eq in self.equations:
                ed = eq.domain
                if ed is not None and ed.has(c):
                    order_sum += eq.n_components * eq.lhs.order_in(c, V)
                    lifts = {e for e in eq.lhs.walk()
                             if isinstance(e, Lift) and e.basis.coord == c}
                    lifted += sum(_n_comp(e) for e in lifts)
                    for e in lifts:
                        tau_vars |= e.free_fields() & V
                elif any(isinstance(e, Interpolate) and e.coord == c for e in eq.lhs.walk()):
                    bcs += eq.n_components
            taus = sum(_n_comp(v) for v in tau_vars)
            dirs.append(DirectionReport(c, order_sum, lifted, bcs, taus))
        return RankReport(var_comp, eq_comp, gauge, dirs)

    def explain(self) -> str:
        """Variables, coefficients, parameters, equations, taus and the rank report."""
        lines = [f"{self.kind} problem '{self.name}'", "Variables:"]
        for v in self.variables:
            lines.append(f"  {v.name:10s} {_sig_str(v.tensorsig):28s} on {v.domain!r}")
        ncc = self.ncc_fields()
        if ncc:
            lines.append("Non-constant coefficients (LHS, fixed): "
                         + ", ".join(sorted(f.name for f in ncc)))
        params = set()
        for eq in self.equations:
            params |= eq.lhs.free_parameters() | eq.rhs.free_parameters()
        if params:
            lines.append("Parameters: " + ", ".join(
                f"{p.name}={p.value}" for p in sorted(params, key=lambda p: p.name)))
        lines.append("Equations (LHS implicit/linear | RHS explicit):")
        for i, eq in enumerate(self.equations):
            lines.append(f"  [{i}] {eq}")
        lines += self._tau_explain()
        nonpoly = sorted({f for eq in self.equations
                          for f in has_nonpolynomial(eq.lhs) + has_nonpolynomial(eq.rhs)})
        if nonpoly:
            lines.append("Non-polynomial pointwise functions: " + ", ".join(nonpoly)
                         + " -- pseudospectral evaluation is NOT alias-free for any dealias "
                         "factor (aliasing error nonzero; check convergence in N)")
        lines.append(self._extra_explain())
        lines.append(self.rank_report().explain())
        if self.symmetry is not None:
            lines.append(self.symmetry_report())
        return "\n".join(l for l in lines if l)

    def _extra_explain(self) -> str:
        return ""

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import problem_html
        return problem_html(self)


def _has_dt(e: Expr) -> bool:
    return any(isinstance(n, TimeDerivative) for n in e.walk())


class IVP(Problem):
    """Initial value problem: M dt(X) + L X = F(X, t). RHS explicit, no dt on the RHS."""

    kind = "IVP"

    def _check_rhs(self, eq: Equation) -> None:
        if _has_dt(eq.rhs):
            raise ProblemError(f"time derivative on the RHS: '{eq.rhs}'",
                               where=f"equation '{eq}'",
                               why="the RHS is evaluated explicitly from the current state; "
                                   "dt() of an unknown is what the time stepper solves for",
                               fix="move every dt() term to the LHS (M dt(X) + L X = F)")

    def _extra_explain(self) -> str:
        orders = [eq.lhs.time_order() for eq in self.equations]
        return f"Time orders per equation: {orders} (0 = algebraic constraint)"

    def linearize(self, around, eigenvalue, **kw):
        """Perturbation EVP around a base state (see conatus.linearize.linearize)."""
        from ..linearize import linearize
        return linearize(self, around, eigenvalue, **kw)


class LBVP(Problem):
    """Linear boundary value problem: L X = F, F independent of X."""

    kind = "LBVP"

    def _check_lhs(self, eq: Equation) -> None:
        if _has_dt(eq.lhs):
            raise ProblemError("LBVP equations cannot contain dt()", where=f"equation '{eq}'",
                               why="an LBVP is steady (L X = F)",
                               fix="drop the dt() terms for a steady state, or declare an IVP")
        super()._check_lhs(eq)

    def _check_rhs(self, eq: Equation) -> None:
        if eq.rhs.degrees(self._vars) != frozenset({0}):
            raise ProblemError(f"LBVP RHS '{eq.rhs}' depends on the variables",
                               where=f"equation '{eq}'",
                               why="an LBVP RHS is data F independent of the unknowns",
                               fix="move linear terms in the unknowns to the LHS; if a term is "
                                   "nonlinear, declare an NLBVP (Newton)")


class EVP(Problem):
    """Generalized eigenvalue problem L X = -sigma M X written with sigma in the LHS; RHS = 0."""

    kind = "EVP"

    def __init__(self, variables: Iterable[Field], eigenvalue: Parameter, *,
                 name: str | None = None, symmetry: SymmetrySubspace | None = None,
                 namespace: Any = None) -> None:
        super().__init__(variables, name=name, symmetry=symmetry, namespace=namespace)
        if not isinstance(eigenvalue, Parameter):
            raise InputTypeError(
                "eigenvalue must be a Parameter", where=f"EVP(..., eigenvalue={eigenvalue!r})",
                why="the eigenvalue is a symbol of the equations, read off to build the pencil "
                    "L + s M",
                fix="s = Parameter('s', dtype=complex); EVP([u, ...], eigenvalue=s) and use s "
                    "in the equations")
        self.eigenvalue = eigenvalue

    def _check_lhs(self, eq: Equation) -> None:
        if _has_dt(eq.lhs):
            raise ProblemError("EVP equations cannot contain dt(); use the eigenvalue parameter",
                               where=f"equation '{eq}'",
                               why="an EVP is the normal-mode form X(t) = X exp(s t), where dt "
                                   "becomes multiplication by the eigenvalue",
                               fix=f"replace dt(f) by {self.eigenvalue.name}*f, or declare the IVP "
                                   "and call pb.linearize(base_state, eigenvalue) to derive it")
        super()._check_lhs(eq)

    def _check_rhs(self, eq: Equation) -> None:
        if not eq.rhs.is_zero():
            raise ProblemError("EVP right-hand sides must be zero", where=f"equation '{eq}'",
                               why="an EVP is homogeneous: L X + s M X = 0",
                               fix="move linear terms to the LHS; a source term (independent of "
                                   "the unknowns) has no place in an EVP (it is an LBVP)")

    def check(self) -> None:
        """Refuse an EVP whose eigenvalue appears in no equation (M = 0, no spectrum at all)."""
        super().check()
        if not any(self.eigenvalue in eq.lhs.free_parameters() for eq in self.equations):
            raise ProblemError(f"eigenvalue '{self.eigenvalue.name}' appears in no equation: "
                               "the mass matrix M would be identically zero",
                               where=f"EVP '{self.name}'",
                               why="the eigenvalue multiplies the time-derivative terms (M)",
                               fix=f"multiply the time-derivative terms by the eigenvalue: "
                                   f"{self.eigenvalue.name}*u where the IVP has dt(u)")

    def _extra_explain(self) -> str:
        n = sum(1 for eq in self.equations if self.eigenvalue in eq.lhs.free_parameters())
        return (f"Eigenvalue: {self.eigenvalue.name} (appears in {n} of "
                f"{len(self.equations)} equations; M is read off its coefficient at discretization)")


class NLBVP(Problem):
    """Nonlinear boundary value problem F(X) = 0, F = LHS - RHS; any term may be nonlinear.

    There is no implicit/explicit split: Newton (``conatus.solvers.nlbvp``) linearizes every
    term with the IR's Fréchet derivative. dt() is refused (steady problems only).
    """

    kind = "NLBVP"

    def _check_lhs(self, eq: Equation) -> None:
        if _has_dt(eq.lhs):
            raise ProblemError("NLBVP equations cannot contain dt()", where=f"equation '{eq}'",
                               why="an NLBVP is steady: F(X) = 0 solved by Newton",
                               fix="drop dt() for a steady state, or declare an IVP")
        if not eq.lhs.depends_on(self._vars) and not eq.rhs.depends_on(self._vars):
            raise ProblemError(f"equation '{eq}' does not depend on the variables",
                               where=f"NLBVP '{self.name}'",
                               fix="every equation of an NLBVP involves at least one unknown "
                                   "(declare the missing variable, or drop the equation)")

    def _check_rhs(self, eq: Equation) -> None:
        if _has_dt(eq.rhs):
            raise ProblemError("NLBVP equations cannot contain dt()", where=f"equation '{eq}'",
                               why="an NLBVP is steady: F(X) = 0 solved by Newton",
                               fix="drop dt() for a steady state, or declare an IVP")

    def nonlinear_variables(self) -> tuple[Field, ...]:
        """Variables entering at least one term that is not linear homogeneous."""
        V = self._vars
        out: set = set()
        for eq in self.equations:
            for side in (eq.lhs, eq.rhs):
                for t in _terms(side):
                    if t.degrees(V) - frozenset({0, 1}) or len(t.degrees(V)) > 1:
                        out |= t.free_fields() & V
        return tuple(v for v in self.variables if v in out)

    def _extra_explain(self) -> str:
        nl = [v.name for v in self.nonlinear_variables()]
        return ("Nonlinear in: " + (", ".join(nl) if nl else "(none: this is an LBVP)")
                + "\nSolved by Newton with the exact Fréchet Jacobian of the IR "
                  "(conatus.solvers.nlbvp.NewtonSolver)")
