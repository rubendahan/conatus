"""SymPy input of the public entry points: ``sympy.sympify`` evaluates a string as Python, so a
string is refused here and an expression or a number goes through."""
from __future__ import annotations

from typing import Any

import sympy

from .errors import InputTypeError


def expr(value: Any, what: str = "this argument") -> Any:
    """``sympy.sympify(value)`` for a number or an expression; InputTypeError for a string."""
    if isinstance(value, (str, bytes)):
        raise InputTypeError(f"{what} is the string {str(value)[:60]!r}",
                             why="a string given to SymPy is evaluated as Python code",
                             fix="pass a SymPy expression, a number or a callable, e.g. "
                                 "1 - Y**2 with Y = sympy.Symbol('y')")
    return sympy.sympify(value)
