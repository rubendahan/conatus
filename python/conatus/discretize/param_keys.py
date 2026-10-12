"""Check the keys of a ``params`` mapping against the parameters a problem actually uses.

A stray key is almost always a typo (``"Ra_"`` for ``"Ra"``) or a stale name; ignoring it would
make the solver use the value declared on the ``Parameter`` instead of the one intended, i.e. a
silent change of the physics (audit R1 / C7 / R11). The message lists the valid keys and
suggests the closest one; nothing is ever applied for the user.
"""
from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping

from .compiler import DiscretizationError

__all__ = ["problem_parameter_names", "check_param_keys"]


def problem_parameter_names(problem) -> set[str]:
    """Names of every Parameter appearing in the equations (LHS and RHS, boundary rows included)."""
    return {p.name for eq in problem.equations for side in (eq.lhs, eq.rhs)
            for p in side.free_parameters()}


def check_param_keys(params: Mapping[str, object], valid: Iterable[str], *, where: str) -> None:
    """Raise DiscretizationError if ``params`` has a key outside ``valid`` (typo or stale name)."""
    valid = sorted(set(valid))
    stray = sorted(set(params) - set(valid))
    if not stray:
        return
    hints = []
    for k in stray:
        close = difflib.get_close_matches(str(k), valid, n=1, cutoff=0.5)
        if not close:                       # case-only typos ("K" for "k") are below the ratio
            close = [v for v in valid if v.lower() == str(k).lower()][:1]
        if close:
            hints.append(f"{k!r}: did you mean {close[0]!r}?")
    raise DiscretizationError(
        f"{where}: params given for parameter(s) not used by the problem: "
        f"{', '.join(repr(k) for k in stray)}; valid keys: {valid}"
        + (f" ({'; '.join(hints)})" if hints else "")
        + ". An unknown key would otherwise be ignored and the value declared on the Parameter "
          "used instead (nothing is renamed for you).")
