"""F607 (audit of 0.1.2, F07): complex covariant-vector angles use the Hermitian product.

Oracle: u = (1, i)/sqrt(2), v = (1, 2i)/sqrt(5): |u^H v| = 3/sqrt(10) (numpy.vdot)."""
from _acceptance import check, passed, start

start(607, "complex CLV principal angle")
import numpy as np  # noqa: E402

from conatus.chaos.lyapunov import clv_subspace_angles  # noqa: E402

V = np.array([[1, 1j], [1, 2j]]) / np.sqrt([[2], [5]])
exact = float(np.arccos(abs(np.vdot(V[0], V[1]))))
got = float(np.ravel(clv_subspace_angles(V, 1))[0])
check(abs(got - exact) < 1e-12, f"angle {got} instead of arccos(3/sqrt(10)) = {exact}")
W = V.copy()
W[1] *= np.exp(0.7j)
got2 = float(np.ravel(clv_subspace_angles(W, 1))[0])
check(abs(got2 - exact) < 1e-12, f"angle changes with a phase: {got2}")
passed(f"angle {got:.12f} rad, phase invariant")
