"""Helpers shared by the native compiler (:mod:`conatus.model.compiler`) and the tau placement
(:mod:`conatus.model.taus`): the declared Unicode rewrite of equation text, the coordinate
arguments of data callables, the record of one automatic tau and number formatting. The compiler
re-exports every name."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from ..errors import InputTypeError
from .geometry import _g  # noqa: F401 (re-export: one definition)

_DOC = "docs/site/guide/boundary_conditions.md"   # public guide page (never an internal design note)


# Unicode accepted in equation text, declared rewrite before parsing (never anything else).
_UNICODE = {"·": "@", "−": "-"}


def _text(s: str) -> str:
    for a, b in _UNICODE.items():
        s = s.replace(a, b)
    return s


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
    # the grids are passed by position (review finding 32: keyword-only arguments were accepted
    # here, then the call failed with a TypeError far from the declaration)
    kwonly = [p.name for p in params if p.kind is p.KEYWORD_ONLY]
    if kwonly:
        raise InputTypeError(f"{what}: keyword-only argument(s) {kwonly}",
                             why="the coordinate grids are passed by position, in the order of "
                                 "the arguments",
                             fix=f"write lambda {', '.join(names)}: ... (no '*' before them)",
                             doc=_DOC)
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
