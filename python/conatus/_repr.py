"""One-line ``repr`` for result dataclasses that hold arrays.

The dataclass ``repr`` prints every array in full: ``print(result)`` in a log gave tens of lines
for a travelling wave, thousands for a POD. :func:`compact_repr` keeps the scalar fields and
summarises each array by its shape and dtype; ``explain()`` (where the class has one) is named at
the end. Assigned as ``__repr__ = compact_repr`` in the class body, before ``@dataclass`` runs,
so the dataclass keeps it. Nothing else changes: the arrays are still the attributes.
"""
from __future__ import annotations

import dataclasses
from typing import Any

#: Longest text kept for one field in a compact repr (longer ones are cut with "...").
FIELD_CHARS = 60
#: Sequences and mappings longer than this are summarised by their length.
SHORT_ITEMS = 6


def _short(v: Any) -> str:
    """One field as short text: arrays by shape and dtype, long containers by their length."""
    if hasattr(v, "shape") and hasattr(v, "dtype"):          # NumPy / CuPy array
        if getattr(v, "ndim", 1) == 0:
            return _short(v.item())
        return f"<array {tuple(v.shape)} {v.dtype}>"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, complex):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)) and (len(v) > SHORT_ITEMS or any(
            hasattr(x, "shape") for x in v)):
        return f"<{type(v).__name__} of {len(v)}>"
    if isinstance(v, dict) and (len(v) > SHORT_ITEMS or any(
            hasattr(x, "shape") for x in v.values())):
        return f"<dict of {len(v)}: {', '.join(map(str, list(v)[:SHORT_ITEMS]))}" + (
            ", ...>" if len(v) > SHORT_ITEMS else ">")
    if callable(v) and not isinstance(v, type):
        return getattr(v, "__qualname__", None) or type(v).__name__
    text = repr(v)
    if "\n" in text or len(text) > FIELD_CHARS:
        if dataclasses.is_dataclass(v) or "object at 0x" in text:
            return f"<{type(v).__name__}>"
        text = text.replace("\n", " ")[:FIELD_CHARS - 3] + "..."
    return text


def compact_repr(self: Any) -> str:
    """``Name(field=value, ...)`` on one line, arrays as ``<array shape dtype>``."""
    parts = [f"{f.name}={_short(getattr(self, f.name, None))}"
             for f in dataclasses.fields(self) if f.repr and not f.name.startswith("_")]
    hint = "; .explain()" if callable(getattr(self, "explain", None)) else ""
    return f"{type(self).__name__}({', '.join(parts)}{hint})"
