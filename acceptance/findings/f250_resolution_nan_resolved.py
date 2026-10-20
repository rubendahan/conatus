"""F250 (audit "assurances", 2026-10-20): a NaN state is never "resolved".

estimate_modes / resolution_report built the spectral envelope with Python's max, and
max(0.0, nan) is 0.0: an all-NaN Chebyshev series came out "zero" ("all resolved"), a NaN top
coefficient was dropped from the tail ("marginal").
"""
import numpy as np
from _acceptance import check, passed, start

start(250, "resolution verdict of non-finite coefficients")
import conatus.resolution as R  # noqa: E402
from conatus.expert import CartesianCoordinates, Chebyshev  # noqa: E402

b = Chebyshev(CartesianCoordinates("x")["x"], 16, (-1, 1), dealias=1.5)
v1 = R.estimate_modes(np.full(16, np.nan), b).direction.verdict
c = 0.5 ** np.arange(16)
c[-1] = np.nan
v2 = R.estimate_modes(c, b).direction.verdict
check(v1 not in ("zero", "resolved") and v2 not in ("resolved", "marginal"),
      f"verdicts {v1!r} (all NaN), {v2!r} (NaN top coefficient)")
passed(f"verdicts {v1!r}, {v2!r}")
