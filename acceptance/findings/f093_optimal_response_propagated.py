"""F093: the low-level optimal response is the PROPAGATED optimal input, not a unit vector.

Phi = diag(3, 1), F = I: the optimal input is e1 and its response Phi e1 = 3 e1 (norm sqrt(G) = 3).
0.1.0's gain() returned the normalised left singular vector e1 as the response.
"""
import numpy as np
from _acceptance import check, need, passed, start

start(93, "gain() response = Phi v_opt")
need("conatus.growth.propagator")
from conatus.growth.propagator import gain  # noqa: E402

Phi = np.diag([3.0, 1.0])
r = gain(Phi, np.eye(2), None)
v, u = np.asarray(r.v_opt), np.asarray(r.u_resp)
err = float(np.max(np.abs(u - Phi @ v)))
check(abs(r.G - 9.0) < 1e-12, f"G = {r.G}, expected 9")
check(err < 1e-12, f"u_resp = {u}, Phi v_opt = {Phi @ v} (|response| {np.linalg.norm(u):.3g}, expected 3)")
passed("response = Phi v_opt")
