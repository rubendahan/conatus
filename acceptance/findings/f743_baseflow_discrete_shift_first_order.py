"""F743 (audit of 0.2.1, N25): discrete_shift is a derivative.

Oracle: u_t = U^2 u + u_yy, Dirichlet on [-1, 1], U constant: sigma = U^2 - (pi/2)^2, d sigma = 2 U <dU> with the weight cos^2(pi y / 2).
"""
from _acceptance import check, passed, start

start(743, "base-flow discrete shift is first order")
import numpy as np  # noqa: E402


import sympy  # noqa: E402

from conatus.nonmodal import Y  # noqa: E402
from conatus.nonmodal.system import ModalSystem  # noqa: E402
from conatus.sensitivity import base_flow_sensitivity  # noqa: E402


def factory(U):
    return ModalSystem("square", (-1.0, 1.0), ("u",), {"u": 1.0}, (), {("u", "u"): {0: U ** 2, 2: sympy.Integer(1)}},
                       {}, {}, (("u", 0, "left"), ("u", 0, "right")))


s = base_flow_sensitivity(factory, sympy.Integer(1), N=32, target=1 - (np.pi / 2) ** 2,
                          adjoint="discrete", infinite_tol=1e-10)
check(abs(s.sigma - (1 - (np.pi / 2) ** 2)) < 1e-9, f"sigma {s.sigma}")
for d in (0.1, 1.0):
    got = s.discrete_shift(sympy.Float(d))
    check(abs(got - 2 * d) < 1e-9, f"dU = {d}: {got}, expected {2 * d}")
check(abs(s.predict(0.1) - 0.2) < 1e-8, f"density-based {s.predict(0.1)}")
got = s.discrete_shift(sympy.Float(0.1) * (1 + Y ** 2))
check(abs(got - 2 * .1 * (4 / 3 - 2 / np.pi ** 2)) < 1e-8, f"y-dependent {got}")
passed("2 U dU")
