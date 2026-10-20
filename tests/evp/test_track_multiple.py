"""track()/critical() on a double eigenvalue: the (cos, sin) pair of a RealFourier group of a
reflection-symmetric problem (gallery gap: LinAlgError in the bordered Newton,
Swift–Hohenberg had to switch to ComplexFourier). The eigenspace member is a declared choice."""
import numpy as np

from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter,
                            ParametricEVP, RealFourier, critical, diff, dt, interp, lap, linearize)
from conatus.ir import Zero


def _sh_real(q):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, 8, (0, 2 * np.pi / q), dealias=1)])
    u, r = Field("u", D), Parameter("r", 0.1)
    d2 = lambda f: diff(diff(f, x), x)                        # noqa: E731
    ivp = IVP([u])
    ivp.add_equation((dt(u) - r * u + (u + d2(u)) + d2(u + d2(u)), -u * u * u))
    evp = linearize(ivp, around={u: Zero(())}, eigenvalue=Parameter("s", dtype=complex))
    return ParametricEVP(evp, [r], group=lambda g: g[0].k > 0)


def test_swift_hohenberg_realfourier_threshold():
    cp = critical(_sh_real(0.9), "r", bracket=(-0.5, 1.0), xtol=1e-9)
    assert abs(cp.value - 0.19 ** 2) < 1e-12
    assert "multiplicity 2" in cp.mode.note and "declared choice" in cp.mode.note


def test_realfourier_times_chebyshev_diffusion_onset():
    """u_t = lap u + r u on (x periodic, z in [-1, 1], u = 0): neutral r = k^2 + pi^2/4, a double
    eigenvalue in each group k > 0; the tracked vector continues the given one."""
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    D = Domain(cs, [RealFourier(x, 8, (0, 2 * np.pi), dealias=1), Chebyshev(z, 24, (-1, 1), dealias=1)])
    u, r = Field("u", D), Parameter("r", 1.0)
    ivp = IVP([u])
    ivp.add_equation((dt(u) - lap(u) - r * u, 0))
    ivp.add_bc(interp(u, z, -1), interp(u, z, 1))
    evp = linearize(ivp, around={}, eigenvalue=Parameter("s", dtype=complex))
    P = ParametricEVP(evp, [r], group=lambda g: g[0].k == 2)
    cp = critical(P, "r", bracket=(0.5, 10.0), xtol=1e-9)
    assert abs(cp.value - (4 + np.pi ** 2 / 4)) < 1e-10, cp.value
    m = P.least_stable(r=3.0)
    m2 = P.track(m.sigma, m.x, r=3.5)
    assert abs(m2.sigma - (m.sigma + 0.5)) < 1e-12                     # ds/dr = 1
    assert abs(np.vdot(m.x, m2.x)) / (np.linalg.norm(m.x) * np.linalg.norm(m2.x)) > 1 - 1e-12
    assert abs(P.sensitivity(m2, "r") - 1) < 1e-10
