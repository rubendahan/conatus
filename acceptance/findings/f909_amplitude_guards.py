"""F909 (external synthesis of 0.2.3, P2): amplitude_equation refuses a degenerate pitchfork and
an observable blind to the critical mode, with a typed error instead of ZeroDivisionError.

Oracles. u' = p u - u^3: pitchfork, amplitude sqrt(dp) (control). u' = p u - u^5: a3 = 0, the
cubic Landau law does not exist (the true amplitude dp^(1/4) comes from fifth order). Stuart-
Landau x, y with z' = -z: the critical mode has no z component, so observable z cannot measure
its amplitude. Control: observable x gives a finite positive amplitude.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(909, "amplitude_equation: typed refusals for a degenerate pitchfork and a blind observable")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import amplitude_equation  # noqa: E402

for power in (3, 5):
    ds = C.ir_system(C.ode_problem(["u"], {"p": 0.0}, lambda U, P, k=power: [P["p"] * U["u"] - U["u"] ** k]),
                     ["p"])
    hit, le = refused(lambda ds=ds: amplitude_equation(ds, np.zeros(1), "p", kind="stationary",
                                                        observable=lambda v: v[0]))
    if power == 3:
        check(not hit and abs(le.amplitude(0.01) - 0.1) <= 1e-8, f"cubic: {le}")
    else:
        check(hit, f"quintic: not refused ({le})")


def rhs(U, P):
    x, y, z = U["x"], U["y"], U["z"]
    r = x * x + y * y
    return [P["p"] * x - y - x * r, x + P["p"] * y - y * r, -z]


ds = C.ir_system(C.ode_problem(["x", "y", "z"], {"p": 0.0}, rhs), ["p"])
hit, le = refused(lambda: amplitude_equation(ds, np.zeros(3), "p", kind="oscillatory",
                                             observable=lambda v: v[0]))
check(not hit and np.isfinite(le.amplitude(0.01)) and le.amplitude(0.01) > 0, f"observable x: {le}")
hit, le = refused(lambda: amplitude_equation(ds, np.zeros(3), "p", kind="oscillatory",
                                             observable=lambda v: v[2]))
check(hit, f"observable z: not refused ({le})")
passed("quintic pitchfork and blind observable refused with typed errors; cubic and observable x work")
