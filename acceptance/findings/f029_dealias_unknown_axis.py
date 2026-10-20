"""F029: dealias={...} naming an axis the box does not have must be refused.

0.1.0 silently ignored the y entry of dealias={"x": 2, "y": 2} on a box with axis x only (a typo of the padding is a
silent change of the numerics).
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(29, "unknown axis in dealias is refused")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.equations("dt(u) = -u*u")
was_refused, s = refused(m.simulate, 16, scheme="RK222", dt=0.01, dealias={"x": 2, "y": 2})
check(was_refused, f"dealias={{'x': 2, 'y': 2}} accepted on a box with axis x only (dealias used: "
      f"{getattr(getattr(s, 'compiled', None), 'dealias', '?')})")
passed("refused")
