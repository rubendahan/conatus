"""F004 (blocking): changing a coefficient of a mapped problem must change the solution.

q*u = q with q = 2 gives u = 1; solving again with q = 4 must still give u = 1, with a residual
measured against the UPDATED problem. 0.1.0 reused the q = 2 factorisation: u = 2, residual 0.
A typed refusal of the coefficient change is accepted.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(4, "a coefficient change on a mapped problem invalidates its caches")
geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0.0, top=1.0))
m = cn.Model(geo)
m.scalar("u")
m.given("q", lambda x, z: 2.0 + 0 * x + 0 * z)
m.equations("q*u = q")
c = m.compile((8, 8))
p = c.discretize()
first = float(np.mean(p.solve().grid("u")))
check(abs(first - 1.0) < 1e-10, f"first solve: mean u = {first}, expected 1")
was_refused, r = refused(p.solve, {c.fields["q"]: lambda x, z: 4.0 + 0 * x + 0 * z})
if was_refused:
    passed("coefficient change refused")
second = float(np.mean(r.grid("u")))
check(abs(second - 1.0) < 1e-10, f"after q -> 4: mean u = {second} (residual reported "
      f"{getattr(r, 'max_residual', '?')}), expected 1")
passed(f"mean u = {second:.12g}, residual {getattr(r, 'max_residual', '?')}")
