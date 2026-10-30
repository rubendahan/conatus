"""F745 (audit of 0.2.1, N28, F12 code side): a system with no algebraic row is valid, and G is the squared norm.

Oracle: scipy.linalg.expm, G(T) = sigma_max(exp(A T))^2, G'(0) = 2 omega(A).
"""
from _acceptance import check, passed, start

start(745, "nonmodal growth with an empty constraint set")
import numpy as np  # noqa: E402


import scipy.linalg as sla  # noqa: E402
import sympy  # noqa: E402

from conatus.nonmodal.system import ModalSystem  # noqa: E402
from conatus.nonmodal.transient import propagator_growth  # noqa: E402

a = np.array([[-1.0, 10.0], [0.0, -2.0]])
one = sympy.Float
sys2 = ModalSystem("local", (-1.0, 1.0), ("u", "v"), {"u": 1.0, "v": 1.0}, (),
                   {("u", "u"): {0: one(a[0, 0])}, ("u", "v"): {0: one(a[0, 1])},
                    ("v", "v"): {0: one(a[1, 1])}}, {}, {}, ())
d = sys2.discretize(12)
for T in (0.05, 0.5):
    G = propagator_growth(d, T, 400, "RK443")
    exact = np.linalg.norm(sla.expm(a * T), 2) ** 2
    check(abs(G - exact) < 1e-5 * exact, f"T = {T}: {G} against {exact}")
omega = np.linalg.eigvalsh((a + a.T) / 2).max()
h = 1e-3
G1 = propagator_growth(d, h, 20, "RK443")
check(abs((G1 - 1) / h - 2 * omega) < .2, f"slope {(G1 - 1) / h} against {2 * omega}")
passed("G from the matrix exponential")
