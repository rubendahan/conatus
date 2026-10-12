"""Tutorial 2a -- a first initial-value problem: Rayleigh--Benard convection in 2D, on the CPU.

Expert layer: the same tutorial written directly in the IR (bases, tau variables and lifts,
LHS/RHS by hand). The native version one directory up gives bit-identical matrices (tests/model).

Expected values (checked at the end of the script):
  growth rate of the onset roll at Ra = 2500, k = pi, Pr = 1, rigid walls:
  sigma = 5.5425059 (thermal-diffusion units), from the EVP of the same equations (also found in
  the streamfunction formulation, tests/ivp/test_psi_theta.py); the IVP must reproduce it from a
  small perturbation. Ra = 2500 is above onset (Ra_c = 1707.762, Reid & Harris 1958).
Run: python t02a_rayleigh_benard_cpu.py   (about 10 s of CPU)
"""
# [imports]
import numpy as np

from conatus.expert import (CFL, IVP, EmbeddedError, CartesianCoordinates, Chebyshev, Domain,
                            Field, IVPSolver, Parameter, RealFourier, VectorField, Zero, discretize,
                            div, dot, dt, eigenvalues, grad, integ, interp, lap, lift, linearize,
                            methods, unit_vector)
# [/imports]

# [problem]
Nx, Nz, Lx = 16, 24, 2.0                     # one roll pair of wavenumber k = 2 pi / Lx = pi
cs = CartesianCoordinates("x", "z")
x, z = cs.coords
bx = RealFourier(x, Nx, (0, Lx), dealias=1.5)       # dealiasing is always declared
bz = Chebyshev(z, Nz, (-0.5, 0.5), dealias=1.5)
D = Domain(cs, [bx, bz])
u, T, p = VectorField("u", D), Field("T", D), Field("p", D)
tp = Field("tp", Domain(cs, []))                    # pressure gauge
tT1, tT2 = Field("tT1", D.without(z)), Field("tT2", D.without(z))
tu1, tu2 = VectorField("tu1", D.without(z)), VectorField("tu2", D.without(z))
Ra, Pr = Parameter("Ra", 2500.0), Parameter("Pr", 1.0)
ez = unit_vector(z)

pb = IVP([u, T, p, tp, tT1, tT2, tu1, tu2], name="Rayleigh-Benard 2D")
pb.add_equation((div(u) + dot(ez, lift(tu2, bz, -1)) + tp, 0))
pb.add_equation((dt(T) - lap(T) + lift(tT1, bz, -1) + lift(tT2, bz, -2), -dot(u, grad(T))))
pb.add_equation((dt(u) - Pr * lap(u) + grad(p) - Pr * Ra * T * ez
                 + lift(tu1, bz, -1) + lift(tu2, bz, -2), -dot(u, grad(u))))
pb.add_equation((interp(T, z, "left"), 0.5))         # hot bottom
pb.add_equation((interp(T, z, "right"), -0.5))       # cold top
pb.add_equation((interp(u, z, "left"), 0))           # no-slip walls
pb.add_equation((interp(u, z, "right"), 0))
pb.add_equation((integ(p), 0))
plan = discretize(pb)
# [/problem]

# [evp]
T0, s = Field("T0", Domain(cs, [bz])), Parameter("s", dtype=complex)
evp = linearize(pb, around={u: Zero((cs,)), T: T0}, eigenvalue=s)   # same equations, linearized
eplan = discretize(evp, ncc={T0: lambda X, Z: -Z})                  # conduction profile T0 = -z
group = next(g for g in eplan.groups if abs(g[0].k - np.pi) < 1e-12)
sigma_evp = eigenvalues(eplan, group).values.real.max()
print(f"EVP: growth rate of the k = pi roll at Ra = 2500: sigma = {sigma_evp:.10f}")
# [/evp]

# [ivp]
def roll_amplitude(solver):
    """Max over the grid of T minus its horizontal mean (the conduction profile -z drops out)."""
    Tg = solver.grid("T")
    return np.abs(Tg - Tg.mean(axis=0)).max()


solver = IVPSolver(plan, "RK443", 5e-4)              # scheme and dt are explicit, never guessed
solver.set_state({T: lambda X, Z: -Z + 1e-6 * np.cos(np.pi * X) * np.cos(np.pi * Z)})
print(solver.explain())
cfl, err = CFL(solver, "u", safety=0.5), EmbeddedError(solver)
solver.run(600)                                      # t = 0.3: faster-decaying modes are gone
a1, t1 = roll_amplitude(solver), solver.t
solver.run(999)
print(f"one more step: embedded error estimate {err.step():.1e} (RK443 vs RK222)")
print(f"CFL time step at safety 0.5: {cfl.compute():.2e}  (a stability indicator, not an error)")
# [/ivp]

# [growth]
a2, t2 = roll_amplitude(solver), solver.t
sigma_ivp = np.log(a2 / a1) / (t2 - t1)
print(f"IVP: measured growth rate sigma = {sigma_ivp:.10f} over t in [{t1:.2f}, {t2:.2f}]")
print(methods(plan, solver).explain())               # the "Methods" paragraph for a paper
# [/growth]

# [check]
SIGMA = 5.5425059                            # EVP of the same equations (tests/ivp/test_psi_theta.py)
assert abs(sigma_evp - SIGMA) < 1e-6, sigma_evp
assert abs(sigma_ivp - sigma_evp) < 1e-5, (sigma_ivp, sigma_evp)
print("OK: the IVP grows at the EVP rate; the EVP matches the psi-theta formulation")
# [/check]
