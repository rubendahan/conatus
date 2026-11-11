"""F912 (hunt 5D): ginzburg_landau(kind="oscillatory") gives the group velocity with its sign.

Oracle, closed form: advected Swift-Hohenberg u_t = r u - (1 + d_xx)^2 u - V u_x - u^3 on one
period 2 pi / k. The mode e^{ikx} has sigma(k) = r - (1 - k^2)^2 - i V k: at k_c = 1, r_c = 0,
v_g = d omega / dk = V and D = 4. V = +-0.5 and +-2 (0.3.0: v_g = -|V|). Mirror check: x -> -x
maps V to -V and v_g to -v_g.
"""
import numpy as np
from _acceptance import check, passed, start

start(912, "Ginzburg-Landau group velocity of a travelling instability has the sign of V")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import cos_mode_observable, ginzburg_landau, ir_system  # noqa: E402


def sh(V):
    def build(k, N=16):
        cs = C.CartesianCoordinates("x")
        x = cs["x"]
        D = C.Domain(cs, [C.RealFourier(x, N, (0, 2 * np.pi / k), dealias=2)])
        u, r = C.Field("u", D), C.Parameter("r", 0.0)
        pb = C.IVP([u], name="advected SH")

        def L(f):
            return f + C.diff(C.diff(f, x), x)
        pb.add_equation((C.dt(u) - r * u + L(L(u)) + V * C.diff(u, x), -u * u * u))
        s = ir_system(pb, ["r"])
        return s, np.zeros(s.n)
    return build


for V in (0.5, -0.5, 2.0, -2.0):
    b = sh(V)
    s, _ = b(1.0)
    gl = ginzburg_landau(b, 1.0, "r", kind="oscillatory", observable=cos_mode_observable(s, "u"), h=1e-2)
    check(abs(gl.v_g - V) <= 1e-6 and abs(gl.D - 4) <= 1e-3, f"V = {V:+g}: v_g {gl.v_g}, D {gl.D}")
passed("v_g = V for V = +-0.5, +-2; D = 4")
