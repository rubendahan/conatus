"""F885 (hunt 5D of 0.3.0, 2026-11-10, Q2): Floquet winding with a double exponent near the circle.
Decoupled Hill system, det M(mu) = prod_n prod_k (mu + i n - a_k): the winding on |mu - c| = 0.3 is
#{k : |a_k - c| < 0.3}. A double exponent 1e-3 inside the circle was counted 2 for 3 at 82 of 180
positions. Oracle: the exponents are given.
"""
from _acceptance import check, passed, start
start(885, "Floquet winding: a double exponent near the circle")
import numpy as np  # noqa: E402
from conatus.errors import ConvergenceError  # noqa: E402
from conatus.floquet.hill_general import hill_from_blocks  # noqa: E402

c, r = 0.05 + 0.02j, 0.3
wrong = refused = 0
for d in (1e-3, -1e-3):
    for th in np.linspace(0, 2 * np.pi, 181)[:-1]:
        base = c + r * (1 - d) * np.exp(1j * th)
        a = [base, base + 1e-7, c + 0.05]
        hb = hill_from_blocks(np.eye(3), np.diag(-np.asarray(a, complex)), {}, Nh=2, omega=1.0,
                              label="t", note="t")
        try:
            wrong += hb.winding(c, r, 16) != sum(abs(x - c) < r for x in a)
        except ConvergenceError:
            refused += 1
check(wrong == 0, f"{wrong} wrong windings of 360")
passed(f"360 positions: none wrong, {refused} refused")
