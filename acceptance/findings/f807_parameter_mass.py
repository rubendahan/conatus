"""F807 (external review of 0.2.3, finding 5): no parameter term of the mass is dropped as roundoff.

Oracle: divide (1 + c r) x' = -x by its coefficient: the rate is -1 / (1 + c r), exact by hand.
Each case is refused at construction, or gives the exact rate. A constant mass stays accepted
and an evaluation beyond the range it was checked in is refused.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(807, "a small parameter-dependent mass is not silently removed")
import conatus.expert as C  # noqa: E402
from conatus.errors import UnsupportedError  # noqa: E402
from conatus.bifurcation.equilibria import generalized_eigenvalues  # noqa: E402

seed = C.ode_problem(["x"], {}, lambda x, p: [-x["x"]])
x = seed.variables[0]
r = C.Parameter("r", 0.0)


def system_of(left, rhs=None):
    pb = C.IVP([x])
    pb.add_equation((left, -x if rhs is None else rhs))
    return C.ir_system(pb, ["r"])


for c in (1e-6, 1e-12, 1e-16, 1e-25):
    hit, out = refused(lambda c=c: system_of((1 + c * r) * C.dt(x)))
    check(hit and isinstance(out, UnsupportedError), f"mass (1 + {c:g} r) accepted")
hit, out = refused(lambda: system_of((1e-13 * r + 1e-3) * C.dt(x)))
check(hit, "small mass with a relatively larger parameter term accepted")
# constant mass with a parameter on the LHS: accepted, exact rate -(1 + r), bounded evaluation
ds = system_of(C.dt(x) + r * x)
for value in (0.0, 3.0, 1e6):
    rate = generalized_eigenvalues(ds.J(np.zeros(1), {"r": value}), ds.M)
    check(abs(rate[0].real + 1 + value) < 1e-9 * (1 + value), f"r = {value:g}: rate {rate}")
hit, out = refused(lambda: ds.J(np.zeros(1), {"r": 1e20}))
check(hit, "evaluation far outside the range in which the mass was checked")
passed("parameter-dependent masses refused from 1e-6 to 1e-25; a constant mass is accepted in range")
