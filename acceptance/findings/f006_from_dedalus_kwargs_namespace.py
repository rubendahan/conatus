"""F006 (blocking): from_dedalus must not report a lossy translation as complete.

The source passes dealias=1.5 through **opts and gives the problem its own namespace (nu = 0.4
while the module-level nu = 0.05). 0.1.0 said complete=True, 0 unsupported, with dealias 1.0 and
the equation at nu = 0.05. Expected: either the semantics are kept (dealias 1.5, nu = 0.4), or
complete is False / the loss is counted as unsupported.
"""
from _acceptance import check, passed, refused, start

import conatus as cn

start(6, "from_dedalus keeps **kwargs and the problem namespace, or says it did not")
SOURCE = """
import numpy as np
import dedalus.public as d3
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
opts = {'dealias': 1.5}
basis = d3.RealFourier(coords['x'], 8, bounds=(0, 2*np.pi), **opts)
u = dist.Field(name='u', bases=basis)
nu = 0.05
problem = d3.IVP([u], namespace={'u': u, 'nu': 0.4})
problem.add_equation('dt(u) + nu*u = 0')
solver = problem.build_solver(d3.RK443)
solver.step(0.002)
"""
was_refused, r = refused(cn.from_dedalus, SOURCE)
if was_refused:
    passed("translation refused")
unsupported = r.counts().get("unsupported", 0)
dealias = getattr(r.namespace.get("basis"), "dealias", None)
equation = str(r.problem.equations[0]) if r.problem is not None and r.problem.equations else ""
honest = (not r.complete) or unsupported > 0
kept = dealias in (1.5, (1.5,)) and "0.4" in equation and "0.05" not in equation
check(honest or kept, f"complete={r.complete}, unsupported={unsupported}, dealias={dealias}, "
      f"equation {equation!r}: semantics lost and reported complete")
passed(f"complete={r.complete}, unsupported={unsupported}, dealias={dealias}, equation {equation!r}")
