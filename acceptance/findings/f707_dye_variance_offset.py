"""F707 (audit of 0.2.1, N27): native dye variance under a large mean.

Oracle: the integral of (eps sin(2 pi x/L))^2 over a period is eps^2 L/2, for any offset."""
from _acceptance import check, passed, start

start(707, "native dye variance under a large mean")
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

L, eps = 3., 1e-3
m = cn.Model(cn.Box(x=cn.periodic(L)))
m.scalar("c")
m.equations("dt(c) = 0")
s = m.simulate(16, scheme="RK222", dt=.01)
s.initial(c=lambda x: -1e6 + eps * np.sin(2 * np.pi * x / L))
p = Path(tempfile.mkdtemp()) / "d.h5"
s.dye("c", p, every=1, field=False)
s.run(steps=1)
got = float(cn.io.read(p).tasks["variance"][0])
check(abs(got / (eps**2 * L / 2) - 1) < 1e-5, f"variance {got} vs {eps**2 * L / 2}")
passed(f"variance {got:.6g}")
