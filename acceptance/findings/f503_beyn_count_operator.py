"""F503 (external review of 0.1.1, F9): Beyn accepted a count computed for a different operator.

The count of the pencil z + 2 (root -2, outside the box: 0) passed as counted= for the pencil z
(root 0 inside) returned no root and "count Z = 0". Correct behaviour: a typed refusal.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(503, "beyn(counted=) refuses the count of another operator")
b = cn.box(re=(-1, 1), im=(-1, 1))
other = cn.contour.count((np.array([[2.0]]), np.eye(1)), b)
was_refused, r = refused(cn.contour.beyn, (np.array([[0.0]]), np.eye(1)), b, counted=other)
check(was_refused, f"accepted: values {getattr(r, 'values', r)}, count {getattr(r, 'count', '?')} (exact root 0)")
state = {"p": 0.3}
c = cn.contour.count(lambda z: z - state["p"], b, poles=0)
state["p"] = 2.3
was_refused, r = refused(cn.contour.beyn, lambda z: z - state["p"], b, poles=0, counted=c)
check(was_refused, f"callable after a parameter change: accepted, values {getattr(r, 'values', r)}")
passed("both counts of another operator refused")
