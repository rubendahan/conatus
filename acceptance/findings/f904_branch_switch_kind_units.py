"""F904 (review G9): branch_switch does not call a transcritical point a pitchfork when the
quadratic coefficient is small in the state's units.

Oracle: x' = p x - c x^2 has the branches x = 0 and x = p / c, a transcritical point for every
c != 0; x' = p x - x^3 is a pitchfork. For c = 1e-9 the classification is "transcritical" or
"undetermined", never "pitchfork". Controls: c = 1 transcritical; the cubic case pitchfork.
"""
import numpy as np
from _acceptance import check, passed, start

start(904, "branch_switch: transcritical never classified pitchfork under state units")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import branch_switch, continue_equilibria  # noqa: E402

cfg = C.ContinuationSettings(ds=0.06, ds_min=1e-8, ds_max=0.1, tolerance=1e-12, max_points=3)


def kind(rhs):
    ds = C.ir_system(C.ode_problem(["x"], {"p": -0.1}, rhs), ["p"])
    cv = continue_equilibria(ds, np.zeros(1), "p", cfg, marginal_tol=1e-10)
    (ev,) = cv.events_of("branch point")
    return branch_switch(cv, ev, cfg, step=0.03).switch["classification"]


for c in (1.0, 1e-3, 1e-9, 1e-12):
    k = kind(lambda X, P, c=c: [P["p"] * X["x"] - c * X["x"] ** 2])
    check(k in ("transcritical", "undetermined") and (c < 1e-6 or k == "transcritical"),
          f"c = {c:g}: {k}")
k = kind(lambda X, P: [P["p"] * X["x"] - X["x"] ** 3])
check(k.startswith("pitchfork"), f"cubic: {k}")
passed("transcritical for c = 1, 1e-3; transcritical or undetermined for 1e-9, 1e-12; pitchfork for the cubic")
