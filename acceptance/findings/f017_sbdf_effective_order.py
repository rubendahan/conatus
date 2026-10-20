"""F017: SBDF3 / SBDF4 report the order they actually achieve.

du/dt + u = 0 to t = 1 at dt = 0.1, 0.05, 0.025 with the default startup="lower_order": the
observed global order is about 2 (the first steps are SBDF1/2), while 0.1.0's explain() says
"global order 3/4". Expected: the observed order matches the scheme order, or explain() states
the effective order 2 of the default startup.
"""
import re
import warnings

import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(17, "SBDF3/4 effective order stated or achieved")
rows = []
for scheme, q in (("SBDF3", 3), ("SBDF4", 4)):
    errs = []
    for dt in (0.1, 0.05, 0.025):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
        m.scalar("u")
        m.equations("dt(u) + u = 0")
        s = m.simulate(8, scheme=scheme, dt=dt).initial(u=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.run(steps=round(1 / dt))
        errs.append(abs(float(np.mean(s.grid("u"))) - np.exp(-1)))
    order = float(np.log2(errs[1] / errs[2]))
    text = s.solver.explain()
    stated = [int(k) for k in re.findall(r"(?:effective|global)[^.;\n]{0,40}?order (\d)", text)]
    rows.append((scheme, order, stated))
    achieved = order > q - 0.4
    honest = bool(stated) and all(abs(k - order) < 0.4 for k in stated)
    check(achieved or honest, f"{scheme}: observed order {order:.2f}, explain() states {stated or 'nothing'}")
passed("; ".join(f"{s}: observed {o:.2f}, stated {k}" for s, o, k in rows))
