"""F014 (important): splitting a run must not change the particle integration.

A particle in the uniform decaying flow u = exp(-t) moves to x(1) = 1 - exp(-1). Running n steps
at once or n one-step runs must give the same position (same scheme, same history); 0.1.0
cleared the fluid history at every run() call: order 3 became about 2 (error 7x larger at dt=0.1,
27x at dt=0.025).
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(14, "particle history survives contiguous run() calls")
rows = []
for dt in (0.1, 0.05, 0.025):
    errors = []
    for split in (False, True):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
        m.velocity("u")
        m.equations("dt(u)+u=0")
        s = m.simulate(8, scheme="RK443", dt=dt).initial(u=lambda x: (np.ones_like(x),))
        p = s.particles([[0.0]])
        n = round(1 / dt)
        if split:
            for _ in range(n):
                s.run(steps=1)
        else:
            s.run(steps=n)
        x = p.positions(unwrapped=True)[0, 0]
        errors.append(abs(x - (1 - np.exp(-1))))
    rows.append((dt, *errors))
for dt, single, split in rows:
    check(abs(split - single) <= 1e-12 + 1e-6 * single,
          f"dt={dt}: error {single:.3g} in one run, {split:.3g} in one-step runs")
passed("split = single: " + ", ".join(f"dt={dt}: {e:.2g}" for dt, e, _ in rows))
