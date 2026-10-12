"""The complete ``explain()`` text of an object, for records and machine reports.

Key objects print a short summary by default and the complete record with ``explain(full=True)``. Tools that keep the text (the run registry)
want the complete one; objects without the split have one text."""
from __future__ import annotations

import inspect
from typing import Any


def full_explain(obj: Any, *args: Any) -> str:
    """``obj.explain(*args, full=True)`` when the object has the short/full split, else
    ``obj.explain(*args)``."""
    try:
        has_full = "full" in inspect.signature(obj.explain).parameters
    except (TypeError, ValueError):
        has_full = False
    return obj.explain(*args, full=True) if has_full else obj.explain(*args)
