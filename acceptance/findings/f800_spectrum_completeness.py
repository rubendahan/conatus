"""F800 (external review of 0.2.3, finding 1): a finite root of a nonsingular M survives.

Oracles: scipy.linalg.eigvals and the diagonal equations solved by hand. J = diag(1, -1),
M = diag(1e-12, 1) has roots 1e12 and -1. A row of M at rounding level (1e-17) stays
indistinguishable from a zero row: its root is removed and counted. A real algebraic
row keeps one finite root and counts nothing as ambiguous.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(800, "stability counts keep every finite root and count the ambiguous ones")
import scipy.linalg as sla  # noqa: E402

import conatus.expert as C  # noqa: E402
from conatus.bifurcation.equilibria import generalized_eigenvalues, pencil_spectrum  # noqa: E402

for eps in (1e-6, 1e-10, 1e-12, 1e-14):
    J, M = np.diag([1.0, -1.0]), np.diag([eps, 1.0])
    want = np.sort(sla.eigvals(J, M).real)
    got = np.sort(generalized_eigenvalues(J, M).real)
    check(got.size == 2 and np.allclose(got, want, rtol=1e-10, atol=0),
          f"M = diag({eps:g}, 1): {got}, SciPy {want}")
    for a, b in ((1e-9, 1.0), (1.0, 1e-9), (1e6, 1e6)):
        scaled = np.sort(generalized_eigenvalues(a * J, b * M).real)
        check(scaled.size == 2 and np.allclose(scaled, want * a / b, rtol=1e-10, atol=0),
              f"scaled ({a:g}, {b:g}) gives {scaled}")
    # the same pencil with one equation multiplied by a constant: same roots
    row = np.sort(generalized_eigenvalues(np.diag([1e6, 1.0]) @ J, np.diag([1e6, 1.0]) @ M).real)
    check(np.allclose(row, want, rtol=1e-10, atol=0), f"row scaling gives {row}")

seed = C.ode_problem(["x", "y"], {"r": 0.0}, lambda x, p: [x["x"], p["r"] - x["y"]])
x, y = seed.variables
r = C.Parameter("r", 0.0)
pb = C.IVP([x, y])
pb.add_equation((1e-12 * C.dt(x), x))
pb.add_equation((C.dt(y), r - y))
cfg = C.ContinuationSettings(ds=0.1, ds_min=1e-5, ds_max=0.2, tolerance=1e-10, max_points=3)
with warnings.catch_warnings():
    warnings.simplefilter("error", C.CorrectnessWarning if hasattr(C, "CorrectnessWarning") else Warning)
    curve = C.continue_equilibria(C.ir_system(pb, ["r"]), np.zeros(2), "r", cfg, marginal_tol=1e-10)
for pt in curve.points:
    check(pt.data["unstable"] == 1 and pt.data["ambiguous"] == 0,
          f"exact unstable count 1: got {pt.data['unstable']}, ambiguous {pt.data['ambiguous']}")

noise = pencil_spectrum(-np.eye(2), np.diag([1.0, 1e-17]))     # as dense QZ
check(np.allclose(noise.values, [-1.0]) and noise.ambiguous == 1
      and np.allclose(noise.ambiguous_removed, [1e17], rtol=1e-6),
      f"noise row: {noise.values}, removed {noise.ambiguous_removed}")
dae = pencil_spectrum(np.diag([1.0, 2.0]), np.diag([1.0, 0.0]))
check(np.allclose(dae.values, [1.0]) and dae.ambiguous == 0 and dae.infinite == 1,
      f"DAE: {dae.values}, ambiguous {dae.ambiguous}, infinite {dae.infinite}")

# a continuation through a noise row says that its count covers the settled roots only
pb2 = C.IVP([x, y])
pb2.add_equation((C.dt(x), -x + r))
pb2.add_equation((1e-17 * C.dt(y), y))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    curve = C.continue_equilibria(C.ir_system(pb2, ["r"]), np.zeros(2), "r", cfg, marginal_tol=1e-10)
check(all(pt.data["ambiguous"] == 1 for pt in curve.points), "ambiguous root not counted on the curve")
check(any("cannot be told" in str(w.message) for w in caught), "no warning for the ambiguous root")
passed("finite roots kept at eps down to 1e-14; noise row 1e-17 removed and counted; DAE unchanged")
