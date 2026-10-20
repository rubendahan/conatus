"""F205 (security review 2026-10-20): a forged inline array in a run record is refused, never built.

The record format stores small arrays inline ({"ndarray": [...], "shape": [...], "dtype": ...});
a record written by another account could claim any shape or an object dtype. Expected: a
refusal (ValueError) for a shape that is not that of the values, an array longer than the
library ever writes inline, a negative dimension, an object dtype.
"""
from _acceptance import check, passed, start

start(205, "forged inline arrays in records are refused")
from conatus.runs._util import from_jsonable  # noqa: E402

bad = []
for forged in ({"ndarray": [1.0], "shape": [10**12, 10**12], "dtype": "float64"},
               {"ndarray": [1.0] * 65, "shape": [65], "dtype": "float64"},
               {"ndarray": [1.0, 2.0], "shape": [-1], "dtype": "float64"},
               {"ndarray": ["x"], "shape": [1], "dtype": "object"}):
    try:
        from_jsonable(forged)
        bad.append(forged["shape"])
    except ValueError:
        pass
check(not bad, f"accepted: {bad}")
passed("all refused")
