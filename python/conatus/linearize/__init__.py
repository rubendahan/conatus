"""Linearization of a problem around a base state, and inner products.

``linearize(ivp, around={u: U, ...}, eigenvalue=sigma)`` builds the EVP of the
perturbation equations from the IR's symbolic Fréchet derivative:

    M dt(u) + L u = F(u)     ->     (L - F'(U)) u' + sigma M u' = 0,

every term of every equation (bulk, boundary, gauge) linearized; dt(x) becomes
sigma * x. Base states are fields (non-constant coefficients of the EVP, data
passed at discretization). Perturbation unknowns are new fields named
``prefix + name`` (du, dT, dp...). A symmetry of the nonlinear problem is **not**
inherited: perturbations may break it; pass ``symmetry=`` to restrict explicitly.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..ir import expr as E
from ..ir.problem import EVP, IVP, Problem, ProblemError
from ..ir.symmetry import SymmetrySubspace
from .inner_product import InnerProduct, chebyshev_gram

__all__ = ["linearize", "LinearizedEVP", "InnerProduct", "chebyshev_gram"]


class LinearizedEVP(EVP):
    """EVP of perturbations, with the map original variable -> perturbation variable."""

    kind = "EVP"

    def __init__(self, *args: Any, perturbation: Mapping[E.Field, E.Field], base: Mapping,
                 source: Problem, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.perturbation = dict(perturbation)
        self.base = dict(base)
        self.source = source

    def _extra_explain(self) -> str:
        lines = [super()._extra_explain(),
                 f"Linearization of {self.source.kind} '{self.source.name}' around:"]
        for f, b in self.base.items():
            lines.append(f"  {f.name} = {b}  (perturbation {self.perturbation[f].name})")
        held = [v.name for v in self.source.variables if v not in self.base]
        if held:
            lines.append("  linear in (no base state needed): " + ", ".join(held))
        return "\n".join(lines)


def _dt_to_sigma(e: E.Expr, sigma: E.Parameter) -> E.Expr:
    mapping = {}
    for n in e.walk():
        if isinstance(n, E.TimeDerivative):
            inner = _dt_to_sigma(n.args[0], sigma)
            if inner.time_order():
                raise ProblemError(f"'{n}': second-order time derivatives are not supported")
            mapping[n] = E.mul(sigma, inner)
    return e.substitute(mapping) if mapping else e


def linearize(problem: IVP, around: Mapping[E.Field, Any], eigenvalue: E.Parameter, *,
              prefix: str = "d", symmetry: SymmetrySubspace | None = None,
              name: str | None = None) -> LinearizedEVP:
    """Perturbation EVP of ``problem`` around the base state ``around`` (variable -> field)."""
    if not isinstance(problem, IVP):
        raise TypeError("linearize() takes an IVP (time-dependent problem)")
    for f in around:
        if f not in problem.variables:
            raise ProblemError(f"'{f.name}' is not a variable of the problem")
    pert = {v: E.Field(prefix + v.name, v.domain, v.tensorsig, v.dtype) for v in problem.variables}
    clash = {p.name for p in pert.values()} & {v.name for v in problem.variables}
    if clash:
        raise ProblemError(f"perturbation names clash with variables: {sorted(clash)}; "
                           "choose another prefix")
    evp = LinearizedEVP(list(pert.values()), eigenvalue, perturbation=pert, base=dict(around),
                        source=problem, name=name or f"linearized {problem.name}",
                        symmetry=symmetry)
    V = frozenset(problem.variables)
    for eq in problem.equations:
        F = E.add(eq.lhs, E.neg(eq.rhs))
        lin = F.linearize(around, pert)
        lin = _dt_to_sigma(lin, eigenvalue)
        left = lin.free_fields() & V
        if left:
            raise ProblemError(
                f"linearized equation still depends on {sorted(f.name for f in left)}: give a "
                "base state for every variable that enters nonlinearly")
        if lin.is_zero():
            # Dropping it would silently change the equation count.
            raise ProblemError(f"equation '{eq}' linearizes to 0 around the given base state; "
                               "remove it from the problem or check the base state")
        evp.add_equation((lin, 0))
    # growth convention fixed by the construction (review of only the
    # native Linearization declared it): dt -> eigenvalue, normal modes exp(eigenvalue t),
    # whatever the eigenvalue's name ("c" was guessed as a phase speed, ranked by Im c)
    _n = getattr(eigenvalue, "name", eigenvalue)
    evp.growth = "real"
    evp.growth_why = (f"normal modes exp({_n} t) of the linearization: growth = "
                      f"Re {_n}")
    return evp

# The package called is its main function: cn.linearize(...) == conatus.expert.linearize(...).
from .._callable import make_callable  # noqa: E402

make_callable(__name__, linearize)
