"""F036: the mapped path must not drop a deliberate small imaginary part in grid().

q*u = 1 with a complex coefficient q = 1 + 1e-10j: u = 1/q has imaginary part -1e-10. 0.1.0
returned a float64 grid (the imaginary part below 1e-8 relative was taken as rounding because
the variable is real-declared and no *parameter* is complex). Expected: a complex grid with
Im u = -1e-10, or a typed refusal of complex data in a real problem.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(36, "mapped grid() keeps a genuine imaginary part")
geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0.0, top=1.0))
m = cn.Model(geo)
m.scalar("u")
q = 1 + 1e-10j
m.given("q", lambda x, z: q + 0 * x + 0 * z)
m.equations("q*u = 1")
was_refused, r = refused(lambda: m.compile((8, 8)).discretize().solve())
if was_refused:
    passed("complex coefficient in a real problem refused")
g = np.asarray(r.grid("u"))
err = float(np.max(np.abs(g - 1 / q)))
check(np.iscomplexobj(g) and err < 1e-14, f"grid dtype {g.dtype}, Im u = {np.imag(g).ravel()[0]:.3g} "
      f"(expected {np.imag(1 / q):.3g})")
passed(f"Im u = {np.imag(g).ravel()[0]:.3g}")
