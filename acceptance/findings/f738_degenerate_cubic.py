"""F738 (audit of 0.2.1, N18): a3 = 0 has no sign.

Oracle: u_t = mu u - u^5 has a3 = 0 (branch mu^(1/4), stable); u_t = mu u - u^3 has a3 = -1 and amplitude 0.1 at mu = 0.01.
"""
from _acceptance import check, passed, start

start(738, "degenerate pitchfork is not subcritical")
import numpy as np  # noqa: E402


from conatus.bifurcation import ir_system, normal_form, ode_problem  # noqa: E402
from conatus.errors import ConvergenceError  # noqa: E402

pb = ode_problem(["u"], {"mu": 0.}, lambda X, P: [P["mu"] * X["u"] - X["u"] ** 5])
s = ir_system(pb, ["mu"])
nf = normal_form(s, np.zeros(s.n), kind="zero", parameter="mu")
check(nf.a3 == 0 and nf.degenerate, f"a3 = {nf.a3}")
check("subcritical" not in nf.explain() and "degenerate" in nf.explain(), nf.explain())
try:
    nf.amplitude(.01)
    check(False, "amplitude from a3 = 0")
except ConvergenceError:
    pass
pb = ode_problem(["u"], {"mu": 0.}, lambda X, P: [P["mu"] * X["u"] - X["u"] ** 3])
s = ir_system(pb, ["mu"])
nf = normal_form(s, np.zeros(s.n), kind="zero", parameter="mu")
check(not nf.degenerate and abs(nf.amplitude(.01) - .1) < 1e-10 * .1 + 1e-12, f"{nf.amplitude(.01)}")
passed("degenerate reported, cubic control kept")
