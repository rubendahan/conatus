"""F716 (audit of 0.2.1, N55): replay differences on booleans and narrow integers.

Oracle: the exact integer differences."""
from _acceptance import check, passed, start

start(716, "replay differences on booleans and narrow integers")
import numpy as np  # noqa: E402

from conatus.runs.integrations import _same  # noqa: E402

for a, b, want in ((True, False, 1.), (np.uint8(0), np.uint8(255), 255.),
                   (np.int8(-128), np.int8(127), 255.), (np.int64(2**62), np.int64(2**62 + 1), 1.)):
    same, d = _same(a, b)
    check(not same and d == want, f"{a!r}, {b!r}: {same}, {d}, exact {want}")
check(_same(True, True) == (True, 0.), "equal booleans")
passed("differences exact")
