"""IVP with an integral constraint in place of a wall condition.

u_t = u_zz on [0, 1], u(0) = 0, int_0^1 u dz = 0: the modes are sin(2 pi n z) (the integral of
sin(kz) vanishes for k = 2 pi n), so u(z, 0) = sin(2 pi z) decays as exp(-4 pi^2 t) exactly. Before
the fix the border-row pinning gave the integral row T_2 (its T_1 coefficient is zero) and the
banded part was singular ("BorderedBanded: banded part is singular" from the C++ solver); the
LBVP path was unaffected. Checked in the expert layer and in the native layer (the condition
written as text at one wall), and the ordinary Dirichlet problem keeps its row ordering.
"""
import numpy as np

import conatus as cn
from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, IVPSolver, discretize,
                            dt, integ, interp, lap, lift)
from conatus.reference.transforms import grid


def _expert(N, second="integ"):
    cs = CartesianCoordinates("z")
    z = cs["z"]
    b = Chebyshev(z, N, (0, 1), dealias=1)
    D = Domain(cs, [b])
    u = Field("u", D)
    t1, t2 = Field("t1", D.without(z)), Field("t2", D.without(z))
    pb = IVP([u, t1, t2])
    pb.add_equation((dt(u) - lap(u) + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(u, z, "left"), 0))
    pb.add_equation((integ(u), 0) if second == "integ" else (interp(u, z, "right"), 0))
    return discretize(pb), u, b


def test_expert_integral_constraint_exact_decay():
    plan, u, b = _expert(24)
    s = IVPSolver(plan, "RK443", 2e-4)
    s.set_state({u: lambda z: np.sin(2 * np.pi * z)})
    s.run(500)
    zg = grid(b, dealias=False)
    err = np.abs(s.grid("u") - np.exp(-4 * np.pi ** 2 * s.t) * np.sin(2 * np.pi * zg)).max()
    assert err < 5e-9, err          # RK443 time error (9.5e-8 at dt = 1e-3, order 3)


def test_dirichlet_ordering_unchanged():
    plan, _, _ = _expert(12, "right")
    slot = IVPSolver(plan, "RK222", 1e-3).system.slots[0]
    assert list(slot.cperm[:2]) == [0, 1]                 # T_0, T_1 pinned as before


def test_native_integral_condition():
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="heat with integral constraint")
    m.scalar("u")
    m.equations("dt(u) - dz(dz(u)) = 0")
    m.at("bottom", u=0)
    m.at("top", "integ(u) = 0")
    sim = m.simulate(N=24, scheme="RK443", dt=2e-4)
    sim.initial(u=lambda z: np.sin(2 * np.pi * z))
    sim.run(steps=500)
    z = sim.coordinate("z")
    assert np.abs(sim.grid("u") - np.exp(-4 * np.pi ** 2 * sim.t) * np.sin(2 * np.pi * z)).max() < 5e-9
