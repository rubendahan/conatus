"""F526 (external review of 0.1.1, F16): uncertainty().explain() prints tuple resolutions.

``N=[(8, 8), (12, 12), (16, 16)]`` is documented; the report was computed but explain() raised
TypeError (``f"{p:g}"`` on a tuple). Expected: the text, with the resolutions as tuples.
"""
import math

import numpy as np
from _acceptance import check, passed, start

start(526, "uncertainty report with tuple resolutions prints")
import conatus as cn  # noqa: E402

r = cn.uncertainty(lambda N, dt: 1 + np.exp(-N[0]) + dt ** 2,
                   N=[(8, 8), (12, 12), (16, 16)], dt=[0.04, 0.02, 0.01], scheme_order=2)
check(math.isclose(r.value, 1 + math.exp(-16) + 1e-4, rel_tol=1e-15), f"value {r.value}")
text = r.explain()
check("(8, 8), (12, 12), (16, 16)" in text, "the tuple resolutions are not printed")
passed("explain() prints (8, 8), (12, 12), (16, 16)")
