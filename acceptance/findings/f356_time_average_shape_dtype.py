"""F356: cn.time_average must not flatten or cast its series (audit inputs, 2026-10-20).

A (samples, quantities) array was ravelled into one interleaved series, a complex series lost
its imaginary part (ComplexWarning only), the text "1234" was one sample 1234.0, and dt <= 0 or
NaN converted tau into negative or NaN time units.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(356, "time_average refuses 2D, complex, text series and a bad dt")
x = np.random.default_rng(1).standard_normal(400)
for bad, what in ((x.reshape(200, 2), "2D"), (x + 1j, "complex"), ("1234", "text")):
    r, _ = refused(cn.time_average, bad, discard=0)
    check(r, f"{what} series accepted")
for dt in (-1.0, 0.0, float("nan"), "0.1"):
    r, _ = refused(cn.time_average, x, discard=0, dt=dt)
    check(r, f"dt={dt!r} accepted")
rep = cn.time_average(x, discard=0, dt=0.1)
check(abs(rep.mean - x.mean()) < 1e-14, "mean of a valid series")
passed("refused; a valid series unchanged")
