"""F713 (audit of 0.2.1, N40): xarray export keeps user names.

Oracle: the task values written by hand: the constant 2 for u = 2."""
from _acceptance import check, passed, start

start(713, "xarray export keeps user names")
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

try:
    import xarray  # noqa: F401
except ImportError:
    passed("xarray not installed: nothing to export")
    raise SystemExit(0) from None
d = Path(tempfile.mkdtemp())
m = cn.Model(cn.Box(x=cn.periodic(1.)), name="f713")
m.scalar("u")
m.scalar("u_coeffs")
m.equations("dt(u) = 0", "dt(u_coeffs) = 0")
s = m.simulate(8, scheme="RK222", dt=.1)
s.initial(u=2., u_coeffs=3.)
o = s.output(d / "o.h5", {"t": "u"}, every=1)
s.run(steps=1)
check(np.allclose(cn.io.read(o.path).to_xarray()["t"].values, 2.), "task named t")
ds = cn.io.read(s.save(d / "c.h5", grids=True)).to_xarray()
check(np.allclose(ds["u_coeffs"].values, 3.) and len(ds.data_vars) == 4, "unknown named u_coeffs")
passed("names kept")
