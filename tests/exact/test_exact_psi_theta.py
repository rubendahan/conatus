"""Rayleigh–Bénard 2D (psi-theta), linear, stress-free walls: growth rate against the exact dispersion
relation (audit hole 5: the psi-theta IVP was only compared with Conatus' own EVP).

Thermal-diffusion units, z in (0, 1), u = psi_z, w = -psi_x, stress-free isothermal walls
(psi = psi_zz = theta = 0), plus a Newtonian cooling gamma theta (written as a *bare* + theta and
a Parameter (gamma - 1) theta, so is visible):

    (1/Pr) dt(lap psi) - lap lap psi + Ra theta_x = 0
    c dt(theta) + theta + (gamma - 1) theta - lap theta + psi_x = 0

Normal mode psi = Psi sin(kx) sin(n pi z), theta = Theta cos(kx) sin(n pi z), q^2 = k^2 + n^2 pi^2:
    (sigma + Pr q^2) (c sigma + gamma + q^2) = Ra Pr k^2 / q^2        (Chandrasekhar 1961, ch. II,
with c = 1, gamma = 0 the classical stress-free result, Ra_c = 27 pi^4 / 4 at k = pi / sqrt 2).
The largest root sigma_+ of this quadratic is the reference: independent of Conatus.

The IVP measures sigma from the norm ratio of theta (RK443); the EVP writes the eigenvalue inside
a product, (1/Pr) (s lap psi - Pr lap lap psi) (the nested form).
"""
import numpy as np
import pytest

import conatus.expert as C
from conatus.discretize import discretize

RA, PR, GAMMA, CM, LX = 2000.0, 0.7, 0.5, 1.5, 2.0
KX = 2 * np.pi / LX                  # k = pi


def sigma_exact(n: int = 1, k: float = KX) -> float:
    """Largest root of (s + Pr q^2)(c s + gamma + q^2) = Ra Pr k^2 / q^2."""
    q2 = k * k + (n * np.pi) ** 2
    # c s^2 + (gamma + q^2 + c Pr q^2) s + Pr q^2 (gamma + q^2) - Ra Pr k^2 / q^2 = 0
    roots = np.roots([CM, GAMMA + q2 + CM * PR * q2, PR * q2 * (GAMMA + q2) - RA * PR * k * k / q2])
    return float(np.max(roots.real))


def _domain(Nx=8, Nz=24):
    cs = C.CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = C.RealFourier(x, Nx, (0, LX), dealias=3 / 2)
    bz = C.Chebyshev(z, Nz, (0, 1), dealias=3 / 2)
    return cs, x, z, bz, C.Domain(cs, [bx, bz])


def _fields(D, Dx):
    psi, th = C.Field("psi", D), C.Field("theta", D)
    tps = [C.Field(f"tpsi{i}", Dx) for i in range(1, 5)]
    tth = [C.Field(f"tth{i}", Dx) for i in range(1, 3)]
    return psi, th, tps, tth


def _bcs(pb, psi, th, z):
    """Stress-free isothermal walls: psi = psi_zz = theta = 0 at z = 0, 1."""
    for side in ("left", "right"):
        pb.add_equation((C.interp(psi, z, side), 0))
        pb.add_equation((C.interp(C.diff(C.diff(psi, z), z), z, side), 0))
        pb.add_equation((C.interp(th, z, side), 0))


def test_dispersion_relation_itself():
    """The closed form: stress-free onset Ra_c = 27 pi^4 / 4 at k = pi / sqrt(2) (c = 1, gamma = 0)."""
    q2 = 1.5 * np.pi ** 2
    ra_c = q2 ** 3 / (0.5 * np.pi ** 2)
    assert abs(ra_c - 27 * np.pi ** 4 / 4) < 1e-9
    assert sigma_exact() > 1.0            # clearly unstable at the test parameters


def test_ivp_growth_rate_matches_dispersion_relation():
    cs, x, z, bz, D = _domain()
    psi, th, tps, tth = _fields(D, D.without(z))
    pr, ra, gm1, c = (C.Parameter("Pr", PR), C.Parameter("Ra", RA), C.Parameter("gm1", GAMMA - 1),
                      C.Parameter("c", CM))
    pb = C.IVP([psi, th, *tps, *tth])
    pb.add_equation(((1 / pr) * C.dt(C.lap(psi)) - C.lap(C.lap(psi)) + ra * C.diff(th, x)
                     + sum(C.lift(t, bz, -i) for i, t in enumerate(tps, 1)), 0))
    pb.add_equation((c * C.dt(th) + th + gm1 * th - C.lap(th) + C.diff(psi, x)
                     + sum(C.lift(t, bz, -i) for i, t in enumerate(tth, 1)), 0))
    _bcs(pb, psi, th, z)
    s = C.IVPSolver(discretize(pb), "RK443", 1e-3)
    s.set_state({th: lambda X, Z: 1e-6 * np.cos(KX * X) * np.sin(np.pi * Z)})
    s.run(600)                            # the decaying root sigma_- has died out (ratio < 1e-8)
    a1, t1 = np.linalg.norm(s.coeffs("theta")), s.t
    s.run(400)
    sigma = np.log(np.linalg.norm(s.coeffs("theta")) / a1) / (s.t - t1)
    # measured: -8.3e-8 at dt = 1e-3, -1.0e-8 at dt = 5e-4 (RK443 time error, order 3); Nz = 32
    # changes the error by 3e-12 (spatially converged). moves sigma by O(1).
    assert abs(sigma - sigma_exact()) < 2e-7, (sigma, sigma_exact())


@pytest.mark.parametrize("n", [1, 2])
def test_evp_nested_eigenvalue_matches_dispersion_relation(n):
    """EVP with the eigenvalue inside a product: (1/Pr)(s lap psi - Pr lap lap psi) + Ra theta_x,
    and c s theta + theta + (gamma - 1) theta - lap theta + psi_x. Mode k = pi: the two largest
    eigenvalues are sigma_+(n = 1) = 7.95 and sigma_+(n = 2) = -20.2 (sigma_-(n = 1) is below -30)."""
    cs, x, z, bz, D = _domain(Nx=4, Nz=32)
    Dx = D.without(z)
    psi, th = C.Field("psi", D, dtype=complex), C.Field("theta", D, dtype=complex)
    tps = [C.Field(f"tpsi{i}", Dx, dtype=complex) for i in range(1, 5)]
    tth = [C.Field(f"tth{i}", Dx, dtype=complex) for i in range(1, 3)]
    s = C.Parameter("s", dtype=complex)
    pr, ra, gm1, c = (C.Parameter("Pr", PR), C.Parameter("Ra", RA), C.Parameter("gm1", GAMMA - 1),
                      C.Parameter("c", CM))
    pb = C.EVP([psi, th, *tps, *tth], eigenvalue=s)
    pb.add_equation(((1 / pr) * (s * C.lap(psi) - pr * C.lap(C.lap(psi))) + ra * C.diff(th, x)
                     + sum(C.lift(t, bz, -i) for i, t in enumerate(tps, 1)), 0))
    pb.add_equation((c * (s * th) + th + gm1 * th - C.lap(th) + C.diff(psi, x)
                     + sum(C.lift(t, bz, -i) for i, t in enumerate(tth, 1)), 0))
    _bcs(pb, psi, th, z)
    plan = discretize(pb)
    grp = next(g for g in plan.groups if abs(g[0].k - KX) < 1e-12)
    res = C.eigenvalues(plan, grp)
    w = res.values[np.isfinite(res.values)]
    lead = np.sort(w.real)[::-1]
    lead = lead[np.concatenate([[True], np.diff(lead) < -1e-8])]   # cos/sin copies of one mode
    # measured: 1.1e-12 (n = 1), 2.1e-12 (n = 2)
    assert abs(lead[n - 1] - sigma_exact(n)) < 1e-9 * max(1.0, abs(sigma_exact(n)))
