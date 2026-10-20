"""Check the keys of a ``params`` mapping against the parameters a problem actually uses.

A stray key is almost always a typo (``"Ra_"`` for ``"Ra"``) or a stale name; ignoring it would
make the solver use the value declared on the ``Parameter`` instead of the one intended, i.e. a
silent change of the physics. The message lists the valid keys and
suggests the closest one; nothing is ever applied for the user.
"""
from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping

from .compiler import DiscretizationError

__all__ = ["problem_parameter_names", "check_param_keys", "check_param_values", "PlanParams",
           "check_assembled_params"]


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


def check_param_values(params: Mapping[str, object], *, where: str) -> None:
    """Raise InputTypeError if a value of ``params`` is not a finite number (audit inputs: ``discretize(pb, params={"Re": float("nan")})`` built NaN matrices, ``"1"`` failed
    deep inside the assembly, ``True`` was read as 1)."""
    import math
    import numbers

    from ..errors import InputTypeError
    for k, v in params.items():
        ok = isinstance(v, numbers.Number) and not isinstance(v, bool) and type(v).__name__ != "bool_"
        if ok:
            c = complex(v)
            ok = math.isfinite(c.real) and math.isfinite(c.imag)
        if not ok:
            raise InputTypeError(f"{where}: params[{k!r}] = {v!r} is not a finite number",
                                 why="a parameter value enters every matrix; NaN, text or a flag "
                                     "is never converted for you",
                                 fix=f"params={{{k!r}: 1.0}}")


class PlanParams(dict):
    """``plan.params`` of a flat, polar or spherical plan: a dict whose writes are checked like
    ``discretize(problem, params=...)`` (a parameter of the problem, a finite number).

    Review of the audit "inputs": ``plan.params`` was a plain dict, so
    ``plan.params["Re"] = float("nan")`` (or ``"1"``, or a typo ``"re"``) went through, and the
    group matrices already assembled kept the old value while the evaluator of F read the new one.
    Writes are checked here; a change after an assembly is refused by
    :func:`check_assembled_params` (the IVP solvers refuse it at the next step,
    ``timestep.integrator.check_params_unchanged``)."""

    def __init__(self, values: Mapping[str, object] | None, valid: Iterable[str], where: str):
        super().__init__(values or {})
        self._valid = frozenset(valid)
        self._where = where

    def __setitem__(self, key: str, value: object) -> None:
        if "_valid" in self.__dict__:            # absent while unpickling (items come first)
            check_param_keys({key: value}, self._valid, where=self._where)
            check_param_values({key: value}, where=self._where)
        super().__setitem__(key, value)

    def update(self, *args: object, **kw: object) -> None:  # type: ignore[override]
        """Each item written through the checked ``__setitem__``."""
        for k, v in dict(*args, **kw).items():
            self[k] = v

    def setdefault(self, key: str, default: object = None) -> object:  # type: ignore[override]
        """A missing key written through the checked ``__setitem__``."""
        if key not in self:
            self[key] = default
        return self[key]


def check_assembled_params(plan) -> None:
    """Refuse an assembly after ``plan.params`` changed since the first one: the cached group
    matrices hold the old values (review of the audit "inputs": the change was taken
    by groups assembled later and by F, not by the groups already assembled, silently)."""
    box = getattr(plan, "_params_assembled", None)   # filled in place: the plan's attributes
    if box is None:                                  # keep their identity (resolution tests)
        box = plan._params_assembled = {}
    now = dict(plan.params)
    if "values" not in box:
        box["values"] = now
        return
    snap = box["values"]
    if now != snap:
        from ..errors import OptionError
        diff = ", ".join(f"{k}: {snap.get(k)!r} -> {now.get(k)!r}"
                         for k in sorted(set(snap) | set(now)) if snap.get(k) != now.get(k))
        raise OptionError(f"plan.params changed after group matrices were assembled ({diff})",
                          why="the assembled (cached) groups hold the previous values; the new "
                              "ones would reach only the groups assembled from now on",
                          fix="discretize again: discretize(problem, params={...}) (or "
                              "m.discretize(N, Re=...) on the native layer)")
