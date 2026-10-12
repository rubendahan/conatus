"""``Ready``: what every function of :mod:`conatus.problems` returns.

A ready problem is an ordinary IR problem (``.problem``: an ``IVP`` or an ``EVP``) plus the
handles a user needs next: the unknowns by name, the physical parameters, the bases by
coordinate name, the non-constant-coefficient data to pass to ``discretize(problem, ncc=...)``,
and the text of the equations, units, boundary conditions and validation reference.
Nothing is hidden: ``explain()`` prints all of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir import Field, Parameter, Problem

__all__ = ["Ready"]


@dataclass
class Ready:
    """A ready-to-discretize problem and its handles.

    ``problem``
        the plain IR problem (``IVP`` or ``EVP``), built with the public API;
    ``fields``
        its unknowns by name (taus included: they are unknowns too);
    ``parameters``
        the physical parameters (``Parameter`` objects carrying the values given);
    ``bases``
        the bases by coordinate name;
    ``ncc``
        NCC data ``{Field: callable on the grid}`` for ``discretize(problem, ncc=ncc)``
        (empty dict when the problem has no non-constant coefficient);
    ``notes``
        equations, nondimensionalization, boundary conditions, structural choices;
    ``reference``
        the published value this problem is validated against (text);
    ``extras``
        problem-specific handles (e.g. base-flow harmonics of a Floquet problem).
    """

    problem: Problem
    fields: dict[str, Field]
    parameters: dict[str, Parameter]
    bases: dict[str, Any]
    ncc: dict
    notes: str
    reference: str
    extras: dict = field(default_factory=dict)

    def explain(self) -> str:
        """Problem, parameters, bases, NCC data, notes and validation reference, in plain text."""
        pars = ", ".join(f"{n} = {p.value}" for n, p in self.parameters.items()) or "none"
        bases = "\n".join(f"  {c}: {b.explain() if hasattr(b, 'explain') else b}"
                          for c, b in self.bases.items())
        ncc = ", ".join(f.name for f in self.ncc) or "none"
        extras = ", ".join(sorted(self.extras)) or "none"
        return "\n".join([
            f"Ready problem '{self.problem.name}' ({self.problem.kind}), expert layer "
            "(cn.problems.<name>: resolution in the call, taus written out); the native model "
            "of the same name is cn.problems.native.<name> (a cn.Model, resolution at simulate)",
            f"parameters: {pars}",
            "bases:", bases,
            f"unknowns: {', '.join(self.fields)}",
            f"NCC data (pass ncc=ready.ncc to discretize): {ncc}",
            f"extras: {extras}",
            "notes:", self.notes.strip(),
            "validation reference:", self.reference.strip(),
            "", self.problem.explain()])


def ready(problem: Problem, parameters: list[Parameter], bases: list[Any], ncc: dict | None,
          notes: str, reference: str, extras: dict | None = None) -> Ready:
    """Assemble a ``Ready`` (fields read off the problem, bases keyed by coordinate name)."""
    return Ready(problem=problem, fields={v.name: v for v in problem.variables},
                 parameters={p.name: p for p in parameters},
                 bases={b.coord.name: b for b in bases},
                 ncc=dict(ncc or {}), notes=notes, reference=reference, extras=dict(extras or {}))
