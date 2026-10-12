"""Unbounded axes of the native layer: ``line(...)`` (Hermite) and ``half_line(...)`` (Laguerre).

. Like ``periodic`` and
``walls``, an axis only says *where* the problem lives; the scale is part of that declaration
because it is the one numerical length of these bases (width of the Gaussian envelope of the
Hermite functions, decay length of the Laguerre functions): it is never chosen for the user and
``explain()`` prints it with the basis.

    cn.Box(x=cn.line(scale=2.0))                 # x in R, Hermite functions psi_n(x/2)
    cn.Box(x=cn.periodic(6.0), y=cn.half_line(scale=0.5))   # y >= 0, wall "wall" (y = 0)

Rules of thumb (printed by ``explain()``): the functions represented decay like exp(-(x/s)^2/2)
on the line and like exp(-x/(2s)) on the half line, so ``scale`` should be of the order of the
width of the solution; a solution wider than about s*sqrt(2N) on the line (s*4N on the half
line) is not resolved at N modes.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from ..errors import InputTypeError

__all__ = ["Line", "HalfLine", "line", "half_line"]

_DOC = "docs/site/guide/geometries.md"   # public guide page (never an internal design note)


@dataclass(frozen=True)
class Line:
    """The whole real line, Hermite functions of declared ``scale`` centred at ``center``."""
    center: float = 0.0
    scale: float = 1.0

    def explain(self) -> str:
        """What this axis is."""
        return (f"whole line centred at {self.center:g}: Hermite functions of declared scale "
                f"{self.scale:g} (the width of their Gaussian envelope); decay at infinity built in")


@dataclass(frozen=True)
class HalfLine:
    """The half line x >= start (a wall at ``start``), Laguerre functions of declared ``scale``."""
    start: float = 0.0
    scale: float = 1.0

    def explain(self) -> str:
        """What this axis is."""
        return (f"half line x >= {self.start:g} (a wall at {self.start:g}): Laguerre functions of "
                f"declared scale {self.scale:g} (the width of their exponential envelope)")


def _scale(scale, where: str) -> float:
    if scale is None:
        raise InputTypeError(f"{where} needs scale=...",
                             why="the scale is the width of the basis functions' envelope, the one "
                                 "length of these bases; it changes the discretization like N and "
                                 "is never chosen for you",
                             fix=f"{where.split('(')[0]}(scale=1.0), of the order of the width of "
                                 "the solution (explain() prints it)", doc=_DOC)
    if isinstance(scale, bool) or not isinstance(scale, numbers.Real) or not (
            math.isfinite(scale) and scale > 0):
        raise InputTypeError(f"{where}: scale={scale!r} must be a finite number > 0",
                             why="the scale is the width of the basis functions' envelope (a "
                                 "numerical choice, like N); it is declared, never guessed",
                             fix=f"{where.split('(')[0]}(scale=1.0), of the order of the width of "
                                 "the solution", doc=_DOC)
    return float(scale)


def line(center: float = 0.0, *, scale: float | None = None) -> Line:
    """The whole line, discretized with Hermite functions psi_n((x - center)/scale) (no boundary,
    no condition: the functions decay like a Gaussian). ``scale`` is required."""
    if not isinstance(center, numbers.Real) or not math.isfinite(center):
        raise InputTypeError(f"line(center={center!r}): the center must be a finite number",
                             fix="line(0.0, scale=1.0)", doc=_DOC)
    return Line(float(center), _scale(scale, "line(...)"))


def half_line(start: float = 0.0, *, scale: float | None = None) -> HalfLine:
    """The half line x >= start, discretized with Laguerre functions of (x - start)/scale; the wall
    at ``start`` takes boundary conditions (named "wall", "x=start" and "bottom"). ``scale`` is
    required."""
    if not isinstance(start, numbers.Real) or not math.isfinite(start):
        raise InputTypeError(f"half_line(start={start!r}): the start must be a finite number",
                             fix="half_line(0.0, scale=1.0)", doc=_DOC)
    return HalfLine(float(start), _scale(scale, "half_line(...)"))
