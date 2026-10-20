"""F007 (blocking): integrals on a sin/cos (parity) axis use the physical domain [0, 1].

T = 1 (Neumann, cosine series): integ(T) = 1, integ(T*T) = 1.
T = sin(pi z) (Dirichlet, sine series): integ(T) = 2/pi = 0.63662, integ(T*T) = 0.5.
0.1.0 integrated over the extended domain: 2, 2 and 0, 1.
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(7, "sin/cos-axis integrals over the physical domain")
cases = [("dz(T)=0", lambda z: 1 + 0 * z, 1.0, 1.0),
         ("T=0", lambda z: np.sin(np.pi * z), 2 / np.pi, 0.5)]
got = []
for bc, initial, i1, i2 in cases:
    m = cn.Model(cn.Box(z=cn.walls(0, 1, basis="sincos")))
    m.scalar("T")
    m.equations("dt(T)-lap(T)=0")
    m.at("walls", bc)
    s = m.simulate(16, scheme="RK222", dt=0.1).initial(T=initial)
    a, b = float(np.real(s.evaluate("integ(T)"))), float(np.real(s.evaluate("integ(T*T)")))
    got.append((bc, a, b))
    check(abs(a - i1) < 1e-8 and abs(b - i2) < 1e-8,
          f"{bc}: integ(T) = {a:.6g}, integ(T*T) = {b:.6g}; expected {i1:.6g}, {i2:.6g}")
passed("; ".join(f"{bc}: {a:.5g}, {b:.5g}" for bc, a, b in got))
