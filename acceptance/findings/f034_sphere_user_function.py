"""F034: a user function (m.function) works on the sphere as on the disk, or is refused.

dt(u) - lap(u) = g(u) with g = tanh on cn.Sphere: 0.1.0 crashed with KeyError: 'g' in the
sphere evaluator (the polar evaluator handles it). Expected: the run completes with a finite
state, or a typed refusal at simulate().
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(34, "user functions on the sphere")
m = cn.Model(cn.Sphere(1.0))
m.scalar("u")
m.function("g", np.tanh)
m.equations("dt(u) - lap(u) = g(u)")


def go():
    s = m.simulate((16, 8), scheme="RK222", dt=1e-3).initial(u=lambda phi, theta: np.cos(theta))
    s.run(steps=2)
    return s


was_refused, s = refused(go)
if was_refused:
    passed("refused")
g = np.asarray(s.grid("u"))
check(np.isfinite(g).all(), "non-finite state")
passed(f"ran, max|u| = {np.max(np.abs(g)):.4g}")
