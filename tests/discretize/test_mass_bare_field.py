"""A bare unknown (coefficient 1) never enters the mass matrix (ERREURS_COURANTES: bare unknown in M).

Found by the Swift–Hohenberg gallery case: in an equation holding dt() or the eigenvalue, a bare
field u was compiled as the identity in M as well as in L, so dt(u) + u = 0 decayed as e^{-t/2}
and the EVP s*u + u = 0 returned s = -1/2 (neutral curves and steady states were unaffected).
"""
import numpy as np

from conatus.discretize import discretize
from conatus.ir import (EVP, IVP, CartesianCoordinates, Chebyshev, ComplexFourier, Domain, Field,
                        Parameter, dt)
from conatus.solvers import eigenvalues
from conatus.timestep import IVPSolver


def _fourier_field():
    cs = CartesianCoordinates("x")
    return Field("u", Domain(cs, [ComplexFourier(cs["x"], 8, (0, 2 * np.pi), dealias=1)]),
                 dtype=complex)


def test_evp_bare_unknown_not_in_mass():
    u = _fourier_field()
    s = Parameter("s", dtype=complex)
    pb = EVP([u], eigenvalue=s)
    pb.add_equation((s * u + u, 0))
    plan = discretize(pb)
    for g in plan.groups:
        gm = plan.matrices(g)
        assert np.allclose(gm.M.toarray(), np.eye(gm.M.shape[0]))
        assert np.allclose(eigenvalues(plan, g).values, -1.0)


def test_ivp_bare_unknown_decays_at_rate_one():
    cs = CartesianCoordinates("z")
    z = cs["z"]
    b = Chebyshev(z, 12, (-1, 1), dealias=1)
    u = Field("u", Domain(cs, [b]))
    pb = IVP([u])
    pb.add_equation((dt(u) + u, 0))                     # u' = -u, no spatial operator
    s = IVPSolver(discretize(pb), "RK443", 1e-3)
    s.set_state({u: lambda Z: 1.0 + 0 * Z})
    s.run(1000)
    assert abs(s.grid("u").max() - np.exp(-1.0)) < 1e-10
