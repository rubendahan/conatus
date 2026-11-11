"""F905 (review G10): ginzburg_landau follows the critical mode, not the nearest eigenvalue.

Oracle: u' = (p - q) u - u^3, v' = (-g - 0.1 q) v with q = (1 - k^2)^2. The critical mode is u at
k_c = 1, p_c = 0; its growth p - q(k) gives the diffusion D = -sigma''(k_c) / 2 = 4. With
g = 4e-5 the decoupled v mode is nearer 0 than u at k_c +- h for h = 0.1 (0.2.4: D = 0.405).
Control: g = 1.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(905, "ginzburg_landau follows the critical mode's identity")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import ginzburg_landau  # noqa: E402

for gap in (1.0, 4e-5):
    def system_at(k, gap=gap):
        q = (1 - k * k) ** 2

        def rhs(X, P):
            return [(P["p"] - q) * X["u"] - X["u"] ** 3, (-gap - 0.1 * q) * X["v"]]
        return C.ir_system(C.ode_problem(["u", "v"], {"p": 0.0}, rhs), ["p"]), np.zeros(2)
    for h in (0.1, 0.01):
        hit, gl = refused(lambda h=h, f=system_at: ginzburg_landau(
            f, 1.0, "p", kind="stationary", observable=lambda v: v[0], h=h, kernel_tol=1e-12))
        check(hit or abs(gl.D - 4) <= 0.05, f"gap {gap:g}, h {h:g}: D = {getattr(gl, 'D', gl)}")
        check(not hit, f"gap {gap:g}, h {h:g}: refused {gl}")
passed("D = 4 within the FD error for gaps 1 and 4e-5 with h = 0.1 and 0.01")
