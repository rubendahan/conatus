"""Tutorial 2a -- a first initial-value problem: Rayleigh--Benard convection in 2D, on the CPU.

Written with the native layer (cn.Model): equations as on paper, no bases, no tau, no LHS/RHS
split by hand; explain() lists what was placed automatically. The expert version (IR, hand-written
taus) is expert/t02a_rayleigh_benard_cpu.py; both give bit-identical matrices and right-hand
sides (tests/model).

Expected values (checked at the end of the script):
  growth rate of the onset roll at Ra = 2500, k = pi, Pr = 1, rigid walls:
  sigma = 5.5425059 (thermal-diffusion units), from the EVP of the same equations (also found in
  the streamfunction formulation, tests/ivp/test_psi_theta.py); the IVP must reproduce it from a
  small perturbation. Ra = 2500 is above onset (Ra_c = 1707.762, Reid & Harris 1958).
Run: python t02a_rayleigh_benard_cpu.py   (about 10 s of CPU)
"""
# [imports]
import numpy as np

import conatus as cn
# [/imports]

# [problem]
box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))   # one roll pair: k = 2 pi / 2 = pi
m = cn.Model(box, name="Rayleigh-Benard 2D", Ra=2500, Pr=1)
u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
m.equations("""
    momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
    continuity: div(u) = 0
    heat:       dt(T) - lap(T) = -u·grad(T)
""")
m.at("bottom", u=0, T=+0.5)                  # hot bottom, no slip
m.at("top", u=0, T=-0.5)                     # cold top, no slip
print(m.explain())                           # split, taus, pressure gauge, dealiasing
# [/problem]

# [evp]
lin = m.linearize(about={u: 0, T: lambda z: -z})          # conduction state T0 = -z
modes = lin.eigenvalues(N=24, k=np.pi)                     # 24 Chebyshev modes, wavenumber pi
sigma_evp = modes.values.real.max()
print(lin.explain())
print(f"EVP: growth rate of the k = pi roll at Ra = 2500: sigma = {sigma_evp:.10f}")
# [/evp]

# [ivp]
sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)   # resolution, scheme, dt: explicit
sim.initial(T=lambda x, z: -z + 1e-6 * np.cos(np.pi * x) * np.cos(np.pi * z))
print(sim.explain())


def roll_amplitude(sim):
    """Max over the grid of T minus its horizontal mean (the conduction profile -z drops out)."""
    Tg = sim.grid("T")
    return np.abs(Tg - Tg.mean(axis=0)).max()


sim.run(until=0.3)                           # faster-decaying modes are gone
a1, t1 = roll_amplitude(sim), sim.t
sim.run(until=0.8)
print(f"CFL time step at safety 0.5: {sim.cfl('u', 0.5):.2e}  (a stability indicator, not an error)")
# [/ivp]

# [growth]
a2, t2 = roll_amplitude(sim), sim.t
sigma_ivp = np.log(a2 / a1) / (t2 - t1)
print(f"IVP: measured growth rate sigma = {sigma_ivp:.10f} over t in [{t1:.2f}, {t2:.2f}]")
print(sim.methods().explain())               # the "Methods" paragraph for a paper
# [/growth]

# [jobs]
import tempfile
from pathlib import Path

work = Path(tempfile.mkdtemp(prefix="rb_chain_"))   # a real run: a directory of your project space


def job(until):
    """The script of every job of a chain: the same lines for the first job and the next ones."""
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
    # the latest checkpoint of chk/ if there is one, else this initial state at t = 0
    sim.resume_or_start(work / "chk",
                        T=lambda x, z: -z + 1e-6 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sim.checkpoints(work / "chk", every=200, keep=2)          # exact restart files
    sim.output(work / "rb.h5", every=100,                     # no mode=: follows the branch
               tasks={"T": "T", "KE": lambda s: s.evaluate("integ(u@u)/2")})
    sim.run(until=until)
    return sim


job(0.3)                                     # first job: from t = 0, stopped at t = 0.3
chain = job(0.8)                             # next job: from the checkpoint of t = 0.3
print("\n".join(s for s in chain.explain().splitlines() if "resume_or_start" in s))
d = cn.io.read(work / "rb.h5")               # also readable with h5py alone
print(f"{len(d.sim_time)} rows, t = {d.sim_time[0]:.2f} ... {d.sim_time[-1]:.2f}, "
      f"kinetic energy at the end {d.tasks['KE'][-1]:.3e}")
# [/jobs]

# [check]
SIGMA = 5.5425059                            # EVP of the same equations (tests/ivp/test_psi_theta.py)
assert abs(sigma_evp - SIGMA) < 1e-6, sigma_evp
assert abs(sigma_ivp - sigma_evp) < 1e-5, (sigma_ivp, sigma_evp)
assert np.array_equal(chain.solver.X, sim.solver.X) and len(d.sim_time) == 17   # two jobs = one run (t = 0 row included)
print("OK: the IVP grows at the EVP rate; the EVP matches the psi-theta formulation; the two-job "
      "chain is the uninterrupted run bit for bit")
# [/check]
