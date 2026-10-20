"""F091: normal_form() must refuse a point that is not an equilibrium.

u' = p + u^2 at p = 1, u = 0: G = 1 (no equilibrium exists), but J = 0, so 0.1.0 classified a
fold there. Expected: a typed refusal naming the residual of G at x0.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(91, "normal forms only at equilibria")
from conatus.bifurcation import ir_system, normal_form, ode_problem  # noqa: E402

pb = ode_problem(["u"], {"p": 1.0}, lambda X, P: [P["p"] + X["u"] * X["u"]])
system = ir_system(pb, ["p"])
x0 = np.zeros(system.n)
was_refused, nf = refused(normal_form, system, x0, kind="zero", parameter="p")
check(was_refused, f"normal form returned at a non-equilibrium (max|G| = 1): {type(nf).__name__}")
passed("refused")
