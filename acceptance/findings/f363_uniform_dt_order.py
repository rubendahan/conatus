"""F363: uniform_dt refuses decreasing or 2D times (audit inputs, 2026-10-20).

Decreasing snapshot times returned a negative dt (DMD frequencies of the wrong sign downstream);
a 2D array of times was flattened through np.diff; rtol < 0 was accepted.
"""
import numpy as np
from _acceptance import check, passed, refused, start

from conatus.expert import snapshots_from_arrays, uniform_dt

start(363, "snapshot times increase, one per snapshot")
for t in (np.linspace(1, 0, 11), np.linspace(0, 1, 12).reshape(3, 4)):
    r, _ = refused(uniform_dt, t, rtol=1e-9)
    check(r, f"uniform_dt(shape {t.shape}) accepted")
r, _ = refused(uniform_dt, np.linspace(0, 1, 11), rtol=-1.0)
check(r, "rtol < 0 accepted")
r, _ = refused(snapshots_from_arrays, {"u": np.zeros((3, 4))}, [0.2, 0.1, 0.0])
check(r, "decreasing snapshot times accepted")
check(abs(uniform_dt(np.linspace(0, 1, 11), rtol=1e-9) - 0.1) < 1e-15, "dt of a valid series")
passed("refused; a valid series gives dt = 0.1")
