"""F361: expert bases refuse non-integer sizes and malformed bounds (audit inputs, 2026-10-20).

Chebyshev(z, 16.0, ...) and size=True were accepted, NaN/None were bare errors of int(); bounds
(0, 1, 2) dropped the third entry, ("0", "1") was parsed from text, (0,) a bare IndexError;
dealias="1.5" was parsed and None a bare TypeError. Fields of dtype int were accepted.
"""
from _acceptance import check, passed, refused, start

from conatus.expert import CartesianCoordinates, Chebyshev, Domain, Field

start(361, "basis sizes, bounds, dealias and field dtypes checked")
z = CartesianCoordinates("x", "z")["z"]
for size, bounds, dealias in ((16.0, (0, 1), 1), (True, (0, 1), 1), (float("nan"), (0, 1), 1),
                              (16, (0, 1, 2), 1), (16, ("0", "1"), 1), (16, (0,), 1),
                              (16, (0, 1), "1.5"), (16, (0, 1), None)):
    r, _ = refused(Chebyshev, z, size, bounds, dealias=dealias)
    check(r, f"Chebyshev(z, {size!r}, {bounds!r}, dealias={dealias!r}) accepted")
D = Domain(z.system, [Chebyshev(z, 16, (0, 1), dealias=1)])
r, _ = refused(Field, "u", D, dtype=int)
check(r, "Field(dtype=int) accepted")
passed("refused")
