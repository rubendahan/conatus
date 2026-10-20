"""F408 (invariance review 2026-10-20): a genuine imaginary part of small-amplitude data survives.

Real-part checks compared |Im| with tol x max(1, |Re|): an iterate (or data) of amplitude 1e-12
with a genuine imaginary part 1e-14 passed as round-off and the part was dropped. Expected: a
typed refusal for it, the real part for genuine round-off (1e-17 on 1).
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(408, "imaginary parts judged against the data's own size")
from conatus.solvers.nlbvp import _real_part  # noqa: E402

was_refused, _ = refused(_real_part, 1e-12 * np.array([1.0 + 1e-2j, 2.0]), "u")
check(was_refused, "a genuine imaginary part of an amplitude-1e-12 iterate was dropped")
ok_refused, r = refused(_real_part, np.array([1.0 + 1e-17j, 2.0]), "u")
check(not ok_refused and np.isrealobj(r), "round-off imaginary part refused")
passed("genuine kept (refused), round-off dropped")
