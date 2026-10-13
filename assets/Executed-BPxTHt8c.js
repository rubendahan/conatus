import{r as e}from"./index-Bc6XKBYc.js";var t=`"""A Dedalus 3 script, written for tutorial 14 of Conatus (not taken from the Dedalus examples).

Onset of Rayleigh-Benard convection between two no-slip walls, periodic sideways, in diffusive
units (length: the depth; time: depth^2 / thermal diffusivity). The temperature is the conduction
profile 1 - z plus a deviation th, which vanishes on both walls; the velocity is u, the pressure p.
With these units the equations are

    dt(u) - Pr lap(u) + grad(p) - Ra Pr th e_z = -u.grad(u),   div(u) = 0,
    dt(th) - lap(th) - u.e_z = -u.grad(th)          (u.e_z: advection of the conduction profile).

Dedalus wants first-order equations in z here: each gradient carries the tau term of the bottom
wall, each evolution equation the tau term of the top wall, and a constant tau closes the pressure
gauge. The time step is fixed and the start is one deterministic roll, so that another code can
repeat the run exactly.

Tutorial 14 never runs this file: it gives it to \`\`cn.from_dedalus\`\`, which reads it without
executing it. \`\`t14b_dedalus_check.py\`\` runs it with Dedalus (if installed) to compare.
Run with Dedalus: python rb_cell_dedalus.py
"""
import numpy as np
import dedalus.public as d3

# Physics and numerics
Ra = 3000.0          # above the no-slip onset, Ra_c = 1708
Pr = 7.0             # water
period = 2.0         # close to the critical wavelength 2.016
N_x, N_z = 24, 16
tstep = 5e-4
steps = 40

# Coordinates, bases (quadratic terms on a 3/2 grid) and unknowns
plane = d3.CartesianCoordinates('x', 'z')
mesh = d3.Distributor(plane, dtype=np.float64)
horiz = d3.RealFourier(plane['x'], size=N_x, bounds=(0, period), dealias=1.5)
vert = d3.ChebyshevT(plane['z'], size=N_z, bounds=(0, 1), dealias=1.5)
u = mesh.VectorField(plane, name='u', bases=(horiz, vert))
p = mesh.Field(name='p', bases=(horiz, vert))
th = mesh.Field(name='th', bases=(horiz, vert))
u_bot = mesh.VectorField(plane, name='u_bot', bases=horiz)    # tau terms, one per wall condition
u_top = mesh.VectorField(plane, name='u_top', bases=horiz)
th_bot = mesh.Field(name='th_bot', bases=horiz)
th_top = mesh.Field(name='th_top', bases=horiz)
gauge = mesh.Field(name='gauge')                               # constant tau of the pressure

# Tau terms live in the first-derivative basis of z; the bottom ones enter through the gradients
e_x, e_z = plane.unit_vector_fields(mesh)
dz_basis = vert.derivative_basis(1)
tau_z = lambda f: d3.Lift(f, dz_basis, -1)  # noqa: E731 (a Dedalus idiom)
Gu = d3.grad(u) + e_z*tau_z(u_bot)
Gth = d3.grad(th) + e_z*tau_z(th_bot)

onset = d3.IVP([u, p, th, u_bot, u_top, th_bot, th_top, gauge], namespace=locals())
onset.add_equation("dt(u) - Pr*div(Gu) + grad(p) - Ra*Pr*th*e_z + tau_z(u_top) = -u@grad(u)")
onset.add_equation("trace(Gu) + gauge = 0")
onset.add_equation("dt(th) - div(Gth) - u@e_z + tau_z(th_top) = -u@grad(th)")
onset.add_equation("u(z=0) = 0")
onset.add_equation("th(z=0) = 0")
onset.add_equation("u(z=1) = 0")
onset.add_equation("th(z=1) = 0")
onset.add_equation("integ(p) = 0")

# Fixed-step SBDF2 run from one roll of the deviation
run = onset.build_solver(d3.SBDF2)
run.stop_iteration = steps
x, z = mesh.local_grids(horiz, vert)
th['g'] = 1e-3*np.cos(2*np.pi*x/period)*np.sin(np.pi*z)
while run.proceed:
    run.step(tstep)
`,n=`"""Tutorial 0 -- getting started: convection between two plates, from the equations to the onset.

The guided path of the website (/tutorial) and of the documentation (tutorials/00_getting_started):
one model of Rayleigh--Benard convection written as on paper, then three questions asked of it.
  1. Is the fluid at rest stable?  The model, linearised about the conduction state.
  2. Where exactly does convection start?  The normal-mode eigenproblem and its critical point.
  3. What happens above onset?  A simulation that grows at the predicted rate and saturates into
     steady rolls, its Methods paragraph, and the run kept in a registry.

Expected values (checked at the end of the script):
  onset of convection between rigid plates: Ra_c = 1707.762 at k_c = 3.117 (Reid & Harris 1958;
  Chandrasekhar 1961, ch. II); the conduction state is stable at Ra = 1500 and unstable at
  Ra = 2500 for the roll k = pi; the simulation grows at the eigenvalue's rate (to 1e-5) and
  saturates into steady rolls whose heat flux is the same through both plates and through the
  layer (Nu, three ways, agree), and larger than conduction alone (Nu > 1).
Writes rolls.png (the temperature of the saturated rolls) into the working directory.
Run: python t00_getting_started.py   (about two minutes of CPU; needs matplotlib: pip install ".[tutorials]")
"""
# [imports]
import numpy as np

import conatus as cn
from conatus.expert import critical
# [/imports]


# [model]
def convection(Ra):
    """Rayleigh-Benard convection between two plates; one roll pair fits the box of length 2."""
    box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))   # x periodic, plates at z = -1/2, 1/2
    m = cn.Model(box, name="Rayleigh-Benard", Ra=Ra, Pr=1)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)                # hot plate below, no slip
    m.at("top", u=0, T=-0.5)                   # cold plate above, no slip
    return m


m = convection(Ra=2500)
print(m.explain())
# [/model]

# [stability]
growth = {}
for Ra in (1500, 2500):
    lin = convection(Ra).linearize(about={"u": 0, "T": lambda z: -z})   # fluid at rest, T = -z
    modes = lin.eigenvalues(N=24, k=np.pi)     # 24 Chebyshev modes across, roll wavenumber pi
    growth[Ra] = modes.values.real.max()
    print(f"Ra = {Ra}: largest growth rate {growth[Ra]:+.6f}")
# [/stability]

# [onset]
onset = cn.Model(cn.Box(z=cn.walls(-0.5, 0.5)), name="convection onset", Ra=1700.0, k=3.1, Pr=1)
onset.scalar("w"), onset.scalar("theta")      # w(z), theta(z) times exp(i k x + s t)
onset.eigenvalue("s", growth="real")
onset.define("L(f) = dz(dz(f)) - k**2*f")     # the Laplacian of one Fourier mode
onset.equations("""
    momentum: s*L(w) = Pr*L(L(w)) - Pr*Ra*k**2*theta
    heat:     s*theta = w + L(theta)
""")
onset.at("walls", "w = 0", "dz(w) = 0", "theta = 0")
# drift=48: a mode counts only if it is found again at 48 modes (spurious tau modes are not)
P = onset.parametric(N=32, over=("Ra", "k"), drift=48, drift_tol=1e-6)
crit = critical(P, "Ra", bracket=(1500, 2000), over={"k": (2.8, 3.4)}, xtol=1e-9)
print(f"Ra_c = {crit.value:.4f}   k_c = {crit.over['k']:.5f}")
# [/onset]

# [simulate]
sim = m.simulate(N=(32, 24), scheme="RK443", dt=5e-4)
sim.initial(T=lambda x, z: -z + 1e-6 * np.cos(np.pi * x) * np.cos(np.pi * z))


def roll_amplitude(sim):
    """Largest departure of T from its horizontal mean: the size of the rolls."""
    T = sim.grid("T")
    return np.abs(T - T.mean(axis=0)).max()


sim.run(until=0.3)
a1, t1 = roll_amplitude(sim), sim.t
sim.run(until=0.8)
a2, t2 = roll_amplitude(sim), sim.t
sigma = np.log(a2 / a1) / (t2 - t1)
print(f"measured growth rate {sigma:.6f}   eigenvalue {growth[2500]:.6f}")
# [/simulate]

# [nusselt]
from conatus.diagnostics import Chebyshev, Grid, Periodic

grid = Grid([Periodic(2.0, 32), Chebyshev(-0.5, 0.5, 24)])   # the simulation's grid points
profile = Grid([Chebyshev(-0.5, 0.5, 24)])                    # functions of z alone


def nusselt(sim):
    """Heat carried across the layer in units of conduction, measured three ways."""
    w, T = sim.grid("u")[1], sim.grid("T")
    layer = 1 + float(grid.mean(w * T))       # conduction + convection, averaged over the layer
    dTdz = profile.cheb_coeffs(profile.d(T.mean(axis=0), 0), 0)
    bottom = -np.polynomial.chebyshev.chebval(-1.0, dTdz)    # conduction through each plate
    top = -np.polynomial.chebyshev.chebval(+1.0, dTdz)
    return layer, bottom, top
# [/nusselt]

# [saturate]
with cn.runs.track("rb_Ra2500", tags=["getting-started"], root="runs") as run:
    sim.solver.run(round((6.0 - sim.t) / sim.dt), record=run)     # on to t = 6
    Nu_layer, Nu_bottom, Nu_top = nusselt(sim)
    run.result(Nu=Nu_layer)
print(f"t = {sim.t:.1f}: Nu = {Nu_layer:.6f} (layer)  {Nu_bottom:.6f} (bottom plate)  "
      f"{Nu_top:.6f} (top plate)")
print(run.explain().splitlines()[0])
print(sim.methods().explain())
# [/saturate]

# [figure]
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The figure draws the spectral interpolant of T (Fourier in x, Chebyshev in z) on a fine grid.
T = sim.grid("T")
cz = grid.cheb_coeffs(T, 1)                     # Chebyshev coefficients in z
zf = np.linspace(-0.5, 0.5, 201)
Tz = np.polynomial.chebyshev.chebval(2 * zf, cz.T)          # (x nodes, fine z)
Tf = np.fft.irfft(np.fft.rfft(Tz, axis=0), n=256, axis=0) * 256 / 32   # Fourier in x
xf = np.linspace(0, 2, 256, endpoint=False)
signed = LinearSegmentedColormap.from_list(   # the site's signed palette: glass, black, rubric
    "signed", ["#9DBBA6", "#3F5245", "#0B0B0B", "#6E2019", "#D4574A"])
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
theta = Tf + zf[None, :]                       # departure from conduction: warm red, cool green
ax.pcolormesh(xf, zf, theta.T, cmap=signed, shading="gouraud", vmin=-0.3, vmax=0.3)
ax.contour(xf, zf, Tf.T, levels=np.linspace(-0.45, 0.45, 10), colors="#EDEBE6",
           linewidths=0.5, linestyles="solid")
ax.set_aspect("equal")
ax.set_axis_off()
out = os.environ.get("CN_TUTORIAL_OUT", ".")
os.makedirs(out, exist_ok=True)                # a CN_TUTORIAL_OUT that does not exist yet
fig.savefig(os.path.join(out, "rolls.png"),
            bbox_inches="tight", pad_inches=0, facecolor="#0B0B0B")
# [/figure]

# [check]
RA_C, K_C = 1707.762, 3.117                    # Reid & Harris (1958); Chandrasekhar (1961)
assert abs(crit.value - RA_C) < 1e-3 and abs(crit.over["k"] - K_C) < 1e-3, crit
assert growth[1500] < 0 < growth[2500], growth
assert abs(sigma - growth[2500]) < 1e-5, (sigma, growth[2500])
assert Nu_layer > 1 and abs(Nu_bottom - Nu_layer) < 1e-3 * Nu_layer \\
    and abs(Nu_top - Nu_layer) < 1e-3 * Nu_layer, (Nu_layer, Nu_bottom, Nu_top)
print("OK: onset at Reid & Harris' Ra_c and k_c; the simulation grows at the eigenvalue's rate "
      "and saturates into rolls carrying the same heat through both plates")
# [/check]
`,r=`"""Tutorial 1 -- a first eigenvalue problem: Orr--Sommerfeld stability of plane Poiseuille flow.

Written with the native layer (cn.Model): the equation as on paper, conditions by boundary name;
the tau terms are placed automatically and listed by explain(). The expert version (IR with
add_bc) is expert/t01_orr_sommerfeld.py; both give bit-identical matrices (tests/model).

Expected values (checked at the end of the script):
  c(Re = 1e4, alpha = 1) = 0.23752649 + 0.00373967i   Orszag (1971), J. Fluid Mech. 50, 689
  Re_c = 5772.22, alpha_c = 1.02056                    Orszag (1971)
Run: python t01_orr_sommerfeld.py   (under a minute of CPU)
"""
# [imports]
import conatus as cn
from conatus.expert import critical, verify
# [/imports]

# [problem]
line = cn.Box(y=cn.walls(-1, 1))               # wall-normal coordinate y in [-1, 1]
m = cn.Model(line, name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
m.scalar("v")                                  # v(y) exp(i alpha (x - c t)), wall-normal velocity
m.given(U=lambda y: 1 - y**2)                  # plane Poiseuille base flow (depends on y)
m.eigenvalue("c", growth="imag")               # wave speed c: a mode grows with Im c
m.define("L2(f) = dy(dy(f)) - alpha**2*f")     # D^2 - alpha^2
m.equations("Orr_Sommerfeld: L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")            # rigid walls at y = -1 and y = 1
print(m.explain(full=True))                    # every automatic decision (taus, modes)
# [/problem]

# [solve]
res = m.eigenvalues(N=64, drift=96, drift_tol=1e-6)   # keep modes found again at N = 96
i = int(res.values.imag.argmax())                     # least stable surviving mode
print(res.explain(key="imag"))
print(f"c = {res.values[i]:.10f}   backward residual {res.residuals[i]:.1e}")
# [/solve]

# [mode]
mode = res.mode(i)                             # the eigenfunction on the grid, taus left out
print(mode.explain())                          # unknowns, grid, declared normalisation
y, v = mode.coordinate("y"), mode["v"]         # Chebyshev points, complex v(y)
# [/mode]

# [figure]
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
ax.plot(y, v.real, label="Re v")
ax.plot(y, v.imag, label="Im v")
ax.plot(y, abs(v), "k--", lw=0.8, label="|v|")
ax.set_xlabel("y")
ax.set_title(f"Orr-Sommerfeld mode, Re = 1e4, alpha = 1: c = {res.values[i]:.6f}")
ax.legend()
out = os.environ.get("CN_TUTORIAL_OUT", ".")
os.makedirs(out, exist_ok=True)                # a CN_TUTORIAL_OUT that does not exist yet
fig.savefig(os.path.join(out, "mode.png"),
            bbox_inches="tight")
# [/figure]

# [critical]
# L(Re, alpha) + c M(Re, alpha), exact in the parameters; the same pencil at N = 96 rejects the
# spurious tau modes (drift test, as for eigenvalues above); growth = Im c (the phase speed c)
P = m.parametric(N=64, over=("Re", "alpha"), drift=96, drift_tol=1e-6)
least = P.least_stable(Re=1e4, alpha=1.0)
crit = critical(P, "Re", bracket=(5000, 7000), over={"alpha": (0.98, 1.06)}, xtol=1e-9)
print(crit.explain())                          # growth convention, drift counts, verification
print(f"Re_c = {crit.value:.4f}   alpha_c = {crit.over['alpha']:.6f}")
print(f"dc/dRe = {P.sensitivity(least, 'Re'):.3e}")
print(verify(P, crit.mode, crit).explain())  # residuals, spectral tail, Taylor tests
# [/critical]

# [check]
C_ORSZAG = 0.23752649 + 0.00373967j          # Orszag (1971), Re = 1e4, alpha = 1
RE_C, ALPHA_C = 5772.22, 1.02056             # Orszag (1971); accurate value 1.020547
assert abs(res.values[i] - C_ORSZAG) < 1e-8, res.values[i]
assert res.residuals[i] < 1e-12
assert abs(abs(v).max() - 1) < 1e-14 and abs(v[0]) < 1e-3 and abs(v[-1]) < 1e-3   # normalised, v = 0 at the walls
assert abs(crit.value - RE_C) < 0.01 and abs(crit.over["alpha"] - ALPHA_C) < 2e-5
print("OK: c and Re_c agree with Orszag (1971) to the published digits; alpha_c within 2e-5")
# [/check]
`,i=`"""Tutorial 2a -- a first initial-value problem: Rayleigh--Benard convection in 2D, on the CPU.

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
print("\\n".join(s for s in chain.explain().splitlines() if "resume_or_start" in s))
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
`,a=`"""Tutorial 2b -- the same Rayleigh--Benard IVP on a GPU (requires: gpu).

The model of tutorial 2a (native layer), at Ra = 1e5 and 128 x 64; only the backend changes.
The expert version is expert/t02b_rayleigh_benard_gpu.py.

Expected values (checked at the end of the script):
  CPU and GPU states agree to rounding: max |T_cpu - T_gpu| <= 1e-11 max |T| after 20 RK443
  steps at Ra = 1e5 (cross-backend model; tests/ivp/test_gpu_backend.py). Later the difference grows
  with the flow (linear growth rates up to ~225 at Ra = 1e5, then nonlinear saturation): two CPU
  orderings of the same sums (evaluator vs fused) differ by ~4e-6 after 200 steps as well;
  the CUDA-graph replay gives the same numbers as plain GPU steps (<= 1e-13 relative).
Run on a GPU node: python t02b_rayleigh_benard_gpu.py   (about a minute, mostly start-up)
"""
# [imports]
import time

import numpy as np

import conatus as cn
# [/imports]


# [problem]
def rayleigh_benard(Ra):
    """The model of tutorial 2a (same equations, same conditions), at Rayleigh number Ra."""
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="Rayleigh-Benard 2D",
                 Ra=Ra, Pr=1)
    m.velocity("u")
    m.pressure("p")
    m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    return m


m = rayleigh_benard(Ra=1e5)
T_init = lambda x, z: -z + 1e-2 * np.cos(np.pi * x) * np.cos(np.pi * z)  # noqa: E731
# [/problem]

# [backends]
runs, early = {}, {}
for label, kw in [("cpu", dict(backend="cpu")),
                  ("gpu", dict(backend="cuda", f_backend="fused"))]:   # both choices explicit
    s = m.simulate(N=(128, 64), scheme="RK443", dt=2e-4, **kw)
    s.initial(T=T_init)
    s.run(steps=20)                                  # compare here: rounding not yet amplified
    early[label] = s.grid("T").copy()
    t0 = time.perf_counter()
    s.run(steps=180)                                 # timing (the flow becomes strongly nonlinear)
    runs[label] = (s, (time.perf_counter() - t0) / 180)
    print(f"{label}: {1e3 * runs[label][1]:.2f} ms per step")
print(runs["gpu"][0].explain())
# [/backends]

# [graph]
g = m.simulate(N=(128, 64), scheme="RK443", dt=2e-4, backend="cuda", f_backend="fused")
g.initial(T=T_init)
g.solver.run_graph(20)                               # expert: capture one step, replay it
graph_early = g.grid("T").copy()
t0 = time.perf_counter()
g.solver.run_graph(180)
print(f"gpu + CUDA graph: {1e3 * (time.perf_counter() - t0) / 180:.2f} ms per step")
# [/graph]

# [check]
Tc, Tg = early["cpu"], early["gpu"]
diff_gpu = np.abs(Tc - Tg).max() / np.abs(Tc).max()
diff_graph = np.abs(graph_early - Tg).max() / np.abs(Tg).max()
late = np.abs(runs["cpu"][0].grid("T") - runs["gpu"][0].grid("T")).max()
print(f"after 20 steps: relative max |T_cpu - T_gpu| = {diff_gpu:.1e}; graph vs plain = {diff_graph:.1e}")
print(f"after 200 steps: max |T_cpu - T_gpu| = {late:.1e} (rounding amplified by the instability)")
assert diff_gpu <= 1e-11 and diff_graph <= 1e-13
print("OK: CPU and GPU agree to rounding; the graph replay is the same computation")
# [/check]
`,o=`"""Tutorial 3 -- symmetries and symmetry subspaces: Rayleigh--Benard onset in half the unknowns.

Written with the native layer (cn.Model): the convection equations as on paper, conditions by
boundary name. The symmetry subspace itself is declared on the compiled problem (expert layer):
the native layer has no symmetry= option yet, so the parities of the boundary unknowns it created
are written out here.
Expected values (checked at the end of the script):
  rigid-rigid onset Ra_c = 1707.762 at k_c = 3.1163 (Reid & Harris 1958; Chandrasekhar 1961):
  the leading growth rate there is 0 to the published digits, in the full space and in the
  subspace even about mid-depth; both eigenvalues agree to rounding with half the unknowns.
Run: python t03_symmetries.py   (a few seconds of CPU)
"""
# [imports]
import time

import numpy as np

import conatus as cn
from conatus.expert import (EVEN, ODD, Reflection, SymmetryBreakingError, SymmetrySubspace,
                            discretize, eigenvalues, linearize)
# [/imports]

# [problem]
RA_C, K_C = 1707.762, 3.1163                  # Reid & Harris (1958)
box = cn.Box(x=cn.periodic(2 * np.pi / K_C), z=cn.walls(-0.5, 0.5))   # one wavelength at k_c
m = cn.Model(box, name="Boussinesq RB", Ra=RA_C, Pr=1)
u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
m.equations("""
    momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
    continuity: div(u) = 0
    heat:       dt(T) - lap(T) = -u·grad(T)
""")
m.at("bottom", u=0, T=+0.5)                   # rigid plates, fixed temperatures
m.at("top", u=0, T=-0.5)
onset_evp = m.linearize(about={u: 0, T: lambda z: -z})   # conduction state T0 = -z
# [/problem]

# [full]
plan = onset_evp.discretize(32, k=K_C)        # N = 32 across the layer, the Fourier size holds k_c
group = onset_evp.group(plan, K_C)
t0 = time.perf_counter()
res = eigenvalues(plan, group)
wall_full = time.perf_counter() - t0
i = int(res.values.real.argmax())
sf, nf = res.values[i], plan.matrices(group).L.shape[0]
print(f" full space: leading sigma = {sf.real:+.4e}   n = {nf:3d}   QZ {1e3 * wall_full:5.1f} ms   "
      f"residual {res.residuals[i]:.1e}")
# [/full]

# [subspace]
evp, ncc, comp = onset_evp.compile({"x": 4, "z": 32})   # the compiled problem (expert objects)
print("unknowns of the perturbation problem:", ", ".join(v.name for v in evp.variables))
Rz = Reflection(comp.coordsys["z"], about=0)  # z -> -z about mid-depth
# Perturbation unknowns carry the prefix d. Only the "scalar" parity is declared for a vector:
# du=ODD means u_x odd and u_z even (the component along z flips sign under the reflection).
# The boundary unknowns tau_* were created by the native layer (listed by m.explain()); their
# parities follow from the walls they enforce and are declared too.
onset = SymmetrySubspace(Rz, du=ODD, dT=EVEN, dp=ODD, T0=ODD,         # w, T even: the onset roll
                         dtau_u1=EVEN, dtau_u2=ODD, dtau_T1=ODD, dtau_T2=EVEN, dtau_p=ODD)
other = SymmetrySubspace(Rz, du=EVEN, dT=ODD, dp=EVEN, T0=ODD,       # the complementary class
                         dtau_u1=ODD, dtau_u2=EVEN, dtau_T1=EVEN, dtau_T2=ODD, dtau_p=EVEN)
print(onset.explain())
# [/subspace]

# [solve]
out = {}
for label, sym in (("onset class", onset), ("other class", other)):
    sub = linearize(comp.problem, evp.base, evp.eigenvalue, symmetry=sym)
    splan = discretize(sub, ncc=ncc)
    g = next(g for g in splan.groups if g[0].k > 0)
    t0 = time.perf_counter()
    r = eigenvalues(splan, g)
    wall = time.perf_counter() - t0
    j = int(r.values.real.argmax())
    out[label] = (r.values[j], splan.matrices(g).L.shape[0])
    print(f"{label:>11}: leading sigma = {r.values[j].real:+.4e}   n = {out[label][1]:3d}   "
          f"QZ {1e3 * wall:5.1f} ms   residual {r.residuals[j]:.1e}")
    if sym is onset:
        print(sub.symmetry_report())          # mirrored conditions dropped, what the subspace hides
# [/solve]

# [guard]
wrong = SymmetrySubspace(Rz, du=EVEN, dT=EVEN, dp=EVEN, T0=ODD,       # T with the parity of w:
                         dtau_u1=ODD, dtau_u2=EVEN, dtau_T1=EVEN, dtau_T2=ODD, dtau_p=EVEN)
try:                                                                   # not equivariant
    linearize(comp.problem, evp.base, evp.eigenvalue, symmetry=wrong)
except SymmetryBreakingError as e:
    print(f"refused: {e}")
# [/guard]

# [check]
(so, no), (sa, _) = out["onset class"], out["other class"]
assert abs(sf) < 1e-4, sf                     # neutral at (Ra_c, k_c) to the published digits
assert abs(so - sf) < 1e-10                   # the onset class carries the leading mode ...
assert sa.real < -1                           # ... the other class decays at onset
assert no <= 0.55 * nf                        # with about half the unknowns
print(f"OK: neutral mode at Reid & Harris' (Ra_c, k_c) in the onset class, n = {no} vs {nf}")
# [/check]
`,s=`"""Tutorial 4 -- a nonlinear boundary-value problem and its continuation: the Bratu problem.

u'' + lam exp(u) = 0 on [-1, 1], u(-1) = u(1) = 0.
Expected values (checked at the end of the script):
  closed form u = 2 ln(cosh a / cosh(a x)), lam = 2 a^2 / cosh(a)^2 (Bratu 1914; Gelfand 1963)
  fold at a tanh a = 1:  lam_c = 0.87845767978129...  (= 3.51383071912516 / 4, the classical
  [0, 1] value, e.g. Jacobsen & Schmitt, J. Differ. Equ. 184, 2002)
Written with the native layer (cn.Model): the equation as on paper, the condition by boundary
name; Newton and continuation act on the compiled problem.
Run: python t04_bratu.py   (under a minute of CPU)
"""
# [imports]
import numpy as np
from scipy.optimize import brentq

import conatus as cn
from conatus.expert import ContinuationSettings, NewtonSolver, NLBVPContinuation, grid
# [/imports]

# [problem]
m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
u = m.scalar("u")
m.equations("bratu: dx(dx(u)) + lam*exp(u) = 0")   # no dt(): a nonlinear boundary-value problem
m.at("walls", u=0)
print(m.explain())                               # kind, tau placement, dealiasing: all declared
pb = m.compile(64, dealias=1)                    # dealias declared: exp(u) is not alias-free anyway;
                                                 # 64 modes resolve exp(u) to 1e-12 up the upper branch
# [/problem]

# [newton]
newton = NewtonSolver(pb.problem, tol=1e-13, damping=1.0, max_iterations=20, params={"lam": 0.3})
sol = newton.solve({pb.fields["u"]: lambda x: 0 * x})   # initial guess u = 0
print(sol.explain())                             # residual per iteration, observed orders
# [/newton]

# [continuation]
settings = ContinuationSettings(ds=0.2, ds_min=1e-8, ds_max=0.6, adapt=1.5, fast_iterations=3,
                                tolerance=1e-12, max_newton=15, max_points=6, dp_fd=1e-6)
cont = NLBVPContinuation(newton, "lam", settings)
branch = cont.run(sol.coeffs, 0.3, +1, p_bounds=(0.3, 1.0))
print(branch.explain())
(fold,) = branch.folds()                          # located by bisection on the tangent
print(f"fold: lam_c = {fold.p:.14f}")
# [/continuation]

# [check]
a_star = brentq(lambda a: a * np.tanh(a) - 1.0, 0.5, 2.0, xtol=1e-15)
LAM_C = 2 * a_star ** 2 / np.cosh(a_star) ** 2   # 0.87845767978129...
a03 = brentq(lambda a: 2 * a * a / np.cosh(a) ** 2 - 0.3, 1e-12, a_star, xtol=1e-15)
exact = lambda X: 2 * np.log(np.cosh(a03) / np.cosh(a03 * X))   # noqa: E731  lower branch
X = grid(pb.bases["x"], dealias=False)          # the Chebyshev points of the solution
assert sol.converged and sol.residual < 1e-13
assert max(sol.observed_orders()) > 1.9          # quadratic Newton
assert np.max(abs(sol.grid("u") - exact(X))) < 1e-11
assert abs(fold.p - LAM_C) < 1e-10, (fold.p, LAM_C)
print(f"OK: lam_c within {abs(fold.p - LAM_C):.1e} of the closed form {LAM_C:.14f}")
# [/check]
`,c=`"""Tutorial 5 -- invariant solutions of Kuramoto--Sivashinsky, L = 22: equilibria, a travelling
wave and the shortest pre-periodic orbit, by Newton--Krylov--hookstep.

u_t + u u_x + u_xx + u_xxxx = 0 on (0, 22), periodic.
Expected values (checked at the end of the script):
  E1, E2 energies 0.2609, 0.4382; E1 leading eigenvalues 0.1308 +- 0.3341i, 0.0824 +- 0.3402i;
  TW1 speed c = 0.73699, energy 0.4649
      Cvitanovic, Davidchack & Siminos (2010), SIAM J. Appl. Dyn. Syst. 9, 1-33, Tables 5.1-5.2, Sec. 5
  shortest pre-periodic orbit ppo_10.25: leading Floquet exponent mu_1 = 0.033209, theta_1 = 2.0079
      Ding & Cvitanovic (2016), SIAM J. Appl. Dyn. Syst. 15, Table 1
Written with the native layer (cn.Model): the equation as on paper; the searches act on the time
stepper of the simulation (sim.solver). Dealiasing: the declared native default 3/2.
Run: python t05_ks_orbits.py   (about a minute of CPU: the orbit monodromy is the long part)
"""
# [imports]
import numpy as np

import conatus as cn
from conatus.expert import (NewtonKrylovHookstep, equilibrium_eigenvalues, find_equilibrium,
                            find_relative_periodic_orbit, find_travelling_wave, floquet_multipliers)
# [/imports]

# [problem]
L, N = 22.0, 32
m = cn.Model(cn.Box(x=cn.periodic(L)), name="Kuramoto-Sivashinsky L=22")
u = m.scalar("u")
m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")   # split: linear implicit, u u_x explicit
sim = m.simulate(N=N, scheme="RK443", dt=0.01)
ks = sim.solver                                           # the flow map Phi_T used by every search


def newton(max_it=40):
    """Trust-region Newton--Krylov: the accuracy target tol is required; the algorithm controls
    (GMRES tolerance and budget, trust radii) keep their declared defaults (nk.explain())."""
    return NewtonKrylovHookstep(tol=1e-12, max_iterations=max_it)


def state(u0):
    """Search seed: the state vector of the simulation started from u(x) = u0(x)."""
    sim.initial(u=u0)
    return ks.X.copy()


def sine(n, a):
    """u0 = a sin(2 pi n x / L)."""
    return lambda x: a * np.sin(2 * np.pi * n * x / L)
# [/problem]


# [equilibria]
E1 = find_equilibrium(ks, state(sine(1, 1.0)), newton())
E2 = find_equilibrium(ks, state(sine(2, 1.0)), newton())
ev1 = equilibrium_eigenvalues(ks, E1, n_modes=8, dense_max=500, arnoldi_tol=1e-12)
print(E1.explain())
print(f"E1: energy {E1.energy:.4f}, residual {E1.residual:.1e}; leading eigenvalues",
      np.round(ev1[:4], 4))
print(f"E2: energy {E2.energy:.4f}, residual {E2.residual:.1e}")
# [/equilibria]

# [wave]
# A seed stored by an earlier search (a draw of four random Fourier modes from which Newton
# converges to the wave; most draws fall back to an equilibrium): the cos and -sin amplitudes of
# modes 1 to 4, written here so that the tutorial does not depend on a random stream.
a = [0.34389095585778406, 0.5568341791702948, -0.9472943734057513, -0.529362057631228,
     -0.349148197714577, -0.9358415262182913, 1.3914943017041073, -0.3967285827537215]
k = 2 * np.pi * np.arange(1, 5) / L
TW1 = find_travelling_wave(ks, state(lambda x: sum(a[2 * i] * np.cos(k[i] * x) - a[2 * i + 1]
                                                   * np.sin(k[i] * x) for i in range(4))),
                           0.737, newton(60))
print(f"TW1: c = {TW1.params['c']:.5f}, energy {TW1.energy:.4f}, residual {TW1.residual:.1e}")
# [/wave]

# [orbit]
# A seed stored by an earlier search (found from a near-recurrence of a long run, then
# converged: residual 2.3e-14 at N = 32, RK443, 260 steps); written here so that the script runs
# from any copy of the tutorial: the real-Fourier coefficients (cos, -sin pairs) of u.
d = {"T": 10.252921849878513, "ell": 8.755862804801978, "nsteps": 260,
     "coeffs": [
    0.0, 0.0, -0.27444650100045515,
    -0.4549146953066752, 1.2816593495532775, -0.6597701693325723,
    -0.2821722812900809, -0.20428309227599858, -0.6751593555768368,
    -0.06667477767893365, 0.011104239323311504, 0.07859985341390308,
    0.06934384084271598, 0.12096395798957167, 0.01569319313275601,
    -0.020552713428854303, 0.006041366263581926, -0.020555504371149513,
    -0.004598584840336716, -5.2520567420378755e-05, -0.0027049597098856965,
    0.0013803228538428875, 0.0004941724838834005, 0.0005564464437612077,
    0.00036788525068114227, 0.00013058074788196403, 1.4913993864317824e-05,
    -0.00010871877873268142, -2.0527324424147996e-05, -4.320240118505132e-05,
    -1.2426042412902393e-05, 8.178542359785038e-06]}
nsteps = 2 * d["nsteps"]                          # dt ~ 0.02: the multipliers need the finer step
flow = m.simulate(N=N, scheme="RK443", dt=d["T"] / nsteps).solver
nk = NewtonKrylovHookstep(tol=1e-12, max_iterations=5, gmres_rtol=1e-6, gmres_max_iterations=80,
                          delta0=0.5, delta_min=1e-10, delta_max=5.0)
ppo = find_relative_periodic_orbit(flow, flow.system.gather_state([np.array(d["coeffs"])]),
                                   d["T"], d["ell"], nk, steps=nsteps, reflect=True)
fl = floquet_multipliers(flow, ppo, n_modes=8, dense_max=200, arnoldi_tol=1e-12)
T = ppo.params["T"]
# leave out the marginal multipliers (|Lambda| = 1: the time and translation directions)
lead = max((lam for lam in fl["multipliers"] if abs(abs(lam) - 1) > 1e-6), key=abs)
mu1, theta1 = np.log(abs(lead)) / T, abs(np.angle(lead))
print(f"ppo: T = {T:.6f}, residual {ppo.residual:.1e}; mu_1 = {mu1:.6f}, theta_1 = {theta1:.4f}")
# [/orbit]

# [check]
assert E1.residual < 1e-12 and abs(E1.energy - 0.2609) < 6e-5
assert E2.residual < 1e-12 and abs(E2.energy - 0.4382) < 6e-5
for lam in (0.1308 + 0.3341j, 0.0824 + 0.3402j):            # Table 5.1 (to 4 digits)
    assert np.min(np.abs(ev1 - lam)) < 6e-5 or np.min(np.abs(ev1 - np.conj(lam))) < 6e-5
assert TW1.residual < 1e-12 and abs(TW1.params["c"] - 0.73699) < 6e-6
assert abs(TW1.energy - 0.4649) < 6e-5
assert ppo.residual < 1e-10 and abs(mu1 - 0.033209) <= 1e-6 and abs(theta1 - 2.0079) <= 1e-4
print("OK: E1, E2, TW1 and ppo_10.25 agree with the published digits")
# [/check]
`,l=`"""Tutorial 8 -- curvilinear geometry: a drum, a pipe and two rotating cylinders.

The bases of the disk, the annulus and the cylinder are regular at the axis (Zernike polynomials,
spin components), never 1/(r + eps); the user names the geometry and its walls, nothing more.
Expected values (checked at the end of the script):
  Vibrating drum (Laplacian on the unit disk, T = 0 at the wall): eigenvalues -j_{m,n}^2, the
    squared zeros of the Bessel functions J_m (closed form; scipy.special.jn_zeros)
  Hagen--Poiseuille, R = 1000, (m, k) = (1, 1): centre mode -0.0708640053 + 0.8467498288i,
    wall mode -0.0911426036 + 0.4691428789i (conjugated, see the page);
    (m, k) = (1, 0): -0.0146819706          Meseguer & Trefethen (2003), J. Comput. Phys. 186, Table 1
  Taylor--Couette, eta = 0.5, outer cylinder at rest, axisymmetric: Re_c = 68.19 at k_c = 3.16
    Recktenwald, Luecke & Mueller (1993), Phys. Rev. E 48, 4444, Table I
Run: python t08_polar.py   (under a minute of CPU)
"""
# [imports]
import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros

import conatus as cn
from conatus.expert import eigenvalues
# [/imports]

# [disk]
drum = cn.Model(cn.Disk(radius=1.0), name="vibrating drum")
drum.scalar("T")
drum.eigenvalue("s", growth="real")
drum.equations("s*T = lap(T)")
drum.at("wall", T=0)
print(drum.explain())
plan = drum.discretize(N=(8, 32))                   # (phi, r): one group per azimuthal number m
drum_modes = {}
for g in plan.groups[:3]:                           # m = 0, 1, 2
    res = eigenvalues(plan, g)
    drum_modes[g.m] = np.sort(res.values.real)[::-1][:3]
    print(f"m = {g.m}: s = {np.round(drum_modes[g.m], 8)}  backward residual "
          f"{res.residuals.max():.1e}")
# [/disk]

# [pipe]
hp = cn.problems.pipe_evp(N=32, Re=1000.0, m=1, k=1.0)   # ready problem: equations and walls
print(hp.problem.name, "|", hp.notes.strip())   # hp.explain(): equations, bases, reference
plan = cn.discretize(hp.problem, ncc=hp.ncc)
res = eigenvalues(plan, plan.group(1, 1.0))         # one group per (m, k_z)
top = res.values[np.argsort(-res.values.real)[:3]]
print("rightmost eigenvalues (m, k) = (1, 1):", np.round(top, 10))
print(f"max backward residual {res.residuals.max():.1e}")
s10 = eigenvalues(plan, plan.group(1, 0.0)).values  # k = 0 is also a group of this plan
s10 = s10[np.argmax(s10.real)]
# [/pipe]

# [couette]
def couette_growth(Re, k, N=24):
    """Largest growth rate of axisymmetric (m = 0) perturbations with axial wavenumber k."""
    tc = cn.problems.taylor_couette_evp(N=N, Re=Re, eta=0.5, mu=0.0, m=0, k=k)
    plan = cn.discretize(tc.problem, ncc=tc.ncc)
    return eigenvalues(plan, plan.group(0, k)).values.real.max()


Re_c = {k: brentq(lambda Re, k=k: couette_growth(Re, k), 60.0, 80.0, xtol=1e-6) for k in (3.10, 3.16, 3.22)}
kc = min(Re_c, key=Re_c.get)
print("neutral Re(k):", {k: round(v, 4) for k, v in Re_c.items()})
# [/couette]

# [check]
for m in (0, 1, 2):                                 # -j_{m,n}^2, closed form
    assert np.allclose(drum_modes[m], -jn_zeros(m, 3) ** 2, rtol=1e-12, atol=0), m
MT_CENTRE, MT_WALL, MT_K0 = -0.0708640053 + 0.8467498288j, -0.0911426036 + 0.4691428789j, -0.0146819706
for ref in (MT_CENTRE, MT_WALL):                    # Table 1 lists the conjugate (see the page)
    assert np.min(np.abs(res.values - np.conj(ref))) < 1e-10
assert abs(s10 - MT_K0) < 1e-10 and res.residuals.max() < 1e-13
assert kc == 3.16 and abs(Re_c[kc] - 68.19) < 0.01   # published to 4 digits
print(f"OK: drum = Bessel zeros; pipe eigenvalues to 1e-10 of Meseguer & Trefethen; "
      f"Re_c(3.16) = {Re_c[kc]:.4f}")
# [/check]
`,u=`"""Tutorial 10 -- bifurcation analysis: branch switching, cycles, codimension 2, normal forms.

Lorenz (1963): dx/dt = sigma (y - x), dy/dt = r x - y - x z, dz/dt = x y - b z, sigma = 10, b = 8/3.
Expected values (checked at the end of the script):
  r = 1: symmetry-breaking pitchfork of the origin, C+- with x^2 = b (r - 1)            (exact)
  Hopf on C+- at r_H = sigma (sigma + b + 3) / (sigma - b - 1) = 470/19, omega^2 = b (sigma + r_H),
  subcritical (Marsden & McCracken 1976)
  Bogdanov-Takens normal form x' = y, y' = b1 + b2 x + x^2 - x y (Kuznetsov 2004, sec. 8.4):
  fold curve b1 = b2^2 / 4, Hopf curve b1 = 0, both ending at the BT point (0, 0)
  Swift-Hohenberg u_t = r u - (1 + d_xx)^2 u - u^3: rolls A^2 = 4 r / 3, xi0^2 = 4, g0 = 3
  (Cross & Hohenberg 1993)
The ODEs are written as Python right-hand sides (ode_problem); the PDE (Swift-Hohenberg) is a
native model (cn.Model) compiled to the dynamical-system interface (ir_system).
Run: python t10_bifurcations.py   (about 20 s of CPU)
"""
# [imports]
import numpy as np

import conatus as cn
from conatus.expert import (EVEN, IVP, CollocationMesh, ContinuationSettings, Reflection,
                            SymmetrySubspace, amplitude_equation, branch_switch, continue_codim2,
                            continue_equilibria, cos_mode_observable, ginzburg_landau, ir_system,
                            normal_form, ode_problem)
# [/imports]

# [system]
lorenz = ode_problem(["x", "y", "z"], {"sigma": 10.0, "r": 0.5, "b": 8 / 3},
                     lambda X, P: [P["sigma"] * (X["y"] - X["x"]),
                                   P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                   X["x"] * X["y"] - P["b"] * X["z"]], name="Lorenz")
L = ir_system(lorenz, ["r"])          # G(x; r), exact J, dG/dr, and the multilinear forms B, C
print(L.explain())
# [/system]

# [equilibria]
cfg = ContinuationSettings(ds=0.1, ds_min=1e-9, ds_max=2.0, adapt=1.3, fast_iterations=4,
                           tolerance=1e-11, max_newton=12, max_points=60, dp_fd=None)
origin = continue_equilibria(L, np.zeros(3), "r", cfg, marginal_tol=1e-9, bounds=(0, 3))
(bp,) = origin.events_of("branch point")
S = np.diag([-1.0, -1.0, 1.0])        # (x, y, z) -> (-x, -y, z): a symmetry of Lorenz, declared
cplus = branch_switch(origin, bp, cfg, step=0.05, symmetry=S, bounds=(0, 30))
print(cplus.explain())
(hopf,) = cplus.events_of("hopf")
# [/equilibria]

# [normal_forms]
pf = normal_form(L, bp.point.z[:-1], kind="zero", par={"r": bp.point.z[-1]}, parameter="r", symmetry=S)
hf = normal_form(L, hopf.point.z[:-1], kind="hopf", par={"r": hopf.point.z[-1]}, parameter="r")
print(pf.explain())
print(hf.explain())
# [/normal_forms]

# [cycles]
cfo = ContinuationSettings(ds=0.2, ds_min=1e-9, ds_max=1.0, adapt=1.3, fast_iterations=4,
                           tolerance=1e-9, max_newton=12, max_points=12, dp_fd=None)
cycles = branch_switch(cplus, hopf, cfo, amplitude=0.2, mesh=CollocationMesh.uniform(40, 4),
                       tests=("lpc",))
print(cycles.explain())
# [/cycles]

# [codim2]
bt = ir_system(ode_problem(["x", "y"], {"b1": -0.5, "b2": -1.0},
                           lambda X, P: [X["y"], P["b1"] + P["b2"] * X["x"] + X["x"] * X["x"]
                                         - X["x"] * X["y"]], name="Bogdanov-Takens"), ["b1", "b2"])
cfg2 = ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.3, adapt=1.3, fast_iterations=3,
                            tolerance=1e-11, max_newton=12, max_points=40, dp_fd=None)
eq = continue_equilibria(bt, np.array([(1 - np.sqrt(3)) / 2, 0.0]), "b1", cfg2, marginal_tol=1e-9,
                         bounds=(-2, 2))
fold_pt, hopf_pt = eq.events_of("fold")[0], eq.events_of("hopf")[0]
fold_curve = continue_codim2(bt, fold_pt.point.z[:-1], "fold", ("b1", "b2"), cfg2,
                             par={"b1": fold_pt.point.z[-1]}, bounds={"b2": (-1.5, 1.5)})
hopf_curve = continue_codim2(bt, hopf_pt.point.z[:-1], "hopf", ("b1", "b2"), cfg2,
                             par={"b1": hopf_pt.point.z[-1]}, bounds={"b2": (-1.5, 1.5)})
print(fold_curve.explain())
print(hopf_curve.explain())
# [/codim2]

# [landau]
def swift_hohenberg(k, N=16):
    """Swift-Hohenberg on one wavelength 2 pi / k, as a native model; r is the free parameter."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi / k)), name="Swift-Hohenberg", r=0.0)
    m.scalar("u")                                          # the unknown
    m.define("S(f) = f + dx(dx(f))")                       # (1 + d_xx), a named operator
    m.equations("dt(u) - r*u + S(S(u)) = -u*u*u")
    c = m.compile(N, dealias=2)                            # u^3: dealias 2, declared
    # The even subspace u(-x) = u(x) (pins the phase of the rolls) is declared on the compiled
    # problem: the native layer has no symmetry= option yet.
    even = IVP(c.problem.variables, name=c.problem.name,
               symmetry=SymmetrySubspace(Reflection(c.coordsys["x"], about=0), u=EVEN))
    for eq in c.problem.equations:
        even.add_equation(eq)
    s = ir_system(even, ["r"])
    return s, np.zeros(s.n)


sh, u0 = swift_hohenberg(1.0)
cos_coeff = cos_mode_observable(sh, "u")          # A = coefficient of cos x in u (linear functional)
landau = amplitude_equation(sh, u0, "r", kind="stationary", observable=cos_coeff)
gl = ginzburg_landau(swift_hohenberg, 1.0, "r", kind="stationary", observable=cos_coeff, h=1e-2)
print(landau.explain())
print(gl.explain())
# [/landau]

# [check]
sig, b = 10.0, 8 / 3
rH = sig * (sig + b + 3) / (sig - b - 1)                        # 470/19, Lorenz (1963)
assert abs(bp.point.z[-1] - 1) < 1e-12 and cplus.switch["classification"] == "pitchfork (symmetry-breaking)"
xs = np.array([p.z[0] for p in cplus.points])
assert np.abs(xs ** 2 - b * (cplus.column("r") - 1)).max() < 1e-9          # C+: x^2 = b (r - 1)
assert abs(hopf.point.z[-1] - rH) < 1e-9 and abs(hf.omega ** 2 - b * (sig + rH)) < 1e-9
assert hf.l1 > 0                                                          # subcritical
assert cycles.column("r").max() < rH                                      # cycles exist for r < r_H
assert np.abs(fold_curve.column("b1") - fold_curve.column("b2") ** 2 / 4).max() < 1e-10
assert np.abs(hopf_curve.column("b1")).max() < 1e-12
assert np.abs(hopf_curve.events_of("bt")[0].point.z[-2:]).max() < 1e-10
assert abs(landau.g3 + 0.75) < 1e-12 and abs(gl.xi0_sq - 4) < 1e-9 and abs(gl.g0 - 3) < 1e-12
print(f"CHECK r_H = {hopf.point.z[-1]:.12f} (470/19 = {rH:.12f}), l1 = {hf.l1:.6e} > 0; "
      f"SH: A^2/r = {landau.amplitude(1.0) ** 2:.12f} (4/3), xi0^2 = {gl.xi0_sq.real:.10f}, g0 = {gl.g0.real:.12f}")
# [/check]
`,d=`"""Tutorial 11 -- the run registry: record, query, compare, reproduce, and reuse (only when asked).

Heat equation u_t = kappa u_xx on [0, 2 pi), u(x, 0) = sin(3 x): u = exp(-9 kappa t) sin(3 x).
Expected values (checked at the end of the script):
  the measured decay rate of each recorded run equals 9 kappa (closed form) to < 1e-9 relative;
  find(kappa__gt=0.15, status="certified") returns exactly the two certified runs;
  reproduce() gives the recorded end state bit for bit; the cache is used only with cache=True,
  and then returns the recorded end state bit for bit without taking a step.
Run: python t11_runs.py   (a few seconds of CPU)
"""
# [imports]
import os
import shutil
import warnings

import numpy as np

import conatus as cn

os.environ["CN_RUNS"] = "registry"          # this tutorial's own registry, in the working directory
shutil.rmtree("registry", ignore_errors=True)   # (default: the directory named by $CN_RUNS)
# [/imports]

# [problem]
T_END, DT = 0.2, 1e-3
NSTEPS = round(T_END / DT)


def simulation(kappa: float):
    """Heat equation with diffusivity kappa (a model parameter: it becomes a run parameter)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat sin(3x)", kappa=kappa)
    m.scalar("u")
    m.equations("dt(u) = kappa*lap(u)")
    sim = m.simulate(N=32, scheme="RK443", dt=DT)
    return sim.initial(u=lambda x: np.sin(3 * x))


def amplitude(sim) -> float:
    """Coefficient of sin(3x) in u (projection on the grid, exact for this resolution)."""
    g = sim.grid("u")
    X = np.linspace(0, 2 * np.pi, g.size, endpoint=False)
    return 2 * float(np.mean(g * np.sin(3 * X)))
# [/problem]

# [record]
runs = {}
for kappa in (0.1, 0.2, 0.4):
    sim = simulation(kappa)
    with cn.runs.track(f"heat_k{kappa}", tags=["tutorial"]) as run:
        sim.solver.run(NSTEPS, record=run)             # numerics, problem hash, start/end state
        rate = -np.log(amplitude(sim)) / sim.t         # measured decay rate
        run.result(rate=rate, rel_error=abs(rate - 9 * kappa) / (9 * kappa))
    runs[kappa] = run
    print(run.explain().splitlines()[0])
for kappa in (0.2, 0.4):                               # certify with the evidence, never bare
    runs[kappa].certify("decay rate = 9 kappa (closed form)",
                        evidence={"rel_error": runs[kappa].results["rel_error"]})
# [/record]

# [query]
found = cn.runs.find(kappa__gt=0.15, status="certified", tags="tutorial", order="created")
print(found.explain())
cmp = cn.runs.compare(runs[0.2], runs[0.4])
print(cmp.explain())
# [/query]

# [reproduce]
rep = runs[0.2].reproduce(simulation(0.2).solver)      # same problem and numerics, checked
print(rep.explain())
# [/reproduce]

# [cache]
plain = simulation(0.4).solver
again = plain.run(NSTEPS, record="heat_k0.4_again")     # identical call, cache off: computed
cached = simulation(0.4).solver
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    hit = cached.run(NSTEPS, record="heat_k0.4_cached", cache=True)
print(caught[0].message)
print({k: hit.cache_report[k] for k in ("hit", "reused_from", "computed")})
print(cn.runs.lineage(again).explain())               # the cached run is a descendant
# [/cache]

# [check]
for kappa, run in runs.items():
    assert run.results["rel_error"] < 1e-9, (kappa, run.results)      # exp(-9 kappa t), closed form
assert [r.id for r in found] == [runs[0.2].id, runs[0.4].id]
assert rep.identical and rep.max_abs_diff == 0.0
assert again.rec["reused_from"] is None and again.cache_report is None
assert hit.cache_report["hit"] and hit.cache_report["reused_from"] in (runs[0.4].id, again.id)
assert np.array_equal(cached.X, plain.X) and cached.iteration == NSTEPS
print("OK: decay rates exact, query, bit-for-bit reproduce, cache only on request")
# [/check]
`,f=`"""Tutorial 12 -- chaos and ensembles: Lyapunov spectrum, covariant vectors, Poincaré map, basins.

Lorenz (1963): dx/dt = sigma (y - x), dy/dt = r x - y - x z, dz/dt = x y - b z, sigma = 10, b = 8/3.
Expected values (checked at the end of the script):
  r = 28: exponents (0.906, 0, -14.572), D_KY = 2.062 (Sprott, lorenzle.htm); sum of the exponents
  = -(sigma + 1 + b) = -13.667 exactly for the flow (constant divergence); the neutral covariant
  vector is the flow direction f(x)
  r = 28: successive maxima of z lie on a one-dimensional cusp map (Lorenz 1963, Fig. 4), z_max < 50
  r = 20 (pre-turbulent): initial states end on C+ or C- (no chaotic attractor; a few long transients
  stay undecided at the horizon and are reported as such), basins exchanged by (x, y) -> (-x, -y)
An ODE has no geometry, so the Lorenz system is written with ode_problem (expert layer): the same
kind of problem object a native model compiles to, time-stepped by the same IVPSolver.
Run: python t12_chaos_ensembles.py   (about 1 min of CPU)
"""
# [imports]
import os

import numpy as np

import conatus as cn
from conatus.chaos import lyapunov_spectrum, poincare_section
from conatus.ensembles import Ensemble, ensemble_fate
from conatus.expert import IVPSolver, ode_problem
# [/imports]

# [system]
SIGMA, B = 10.0, 8 / 3


def lorenz(r, dt):
    pb = ode_problem(["x", "y", "z"], {"sigma": SIGMA, "r": r, "b": B},
                     lambda X, P: [P["sigma"] * (X["y"] - X["x"]),
                                   P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                   X["x"] * X["y"] - P["b"] * X["z"]], name="Lorenz")
    return IVPSolver(cn.discretize(pb), "RK443", dt)   # scheme and dt: explicit choices


s = lorenz(28.0, 0.01)
# [/system]

# [lyapunov]
rng = np.random.default_rng(0)
X0 = np.column_stack([rng.uniform(-10, 10, 32), rng.uniform(-10, 10, 32), rng.uniform(10, 40, 32)])
res = lyapunov_spectrum(s, X0, n_vectors=3, transient_steps=1000, tangent_transient_steps=1000,
                        steps=10000, reortho_every=10, inner="euclidean", n_blocks=5, seed=1,
                        clv_window=50, clv_backward_transient=150, clv_keep_every=10)
print(res.explain())
# [/lyapunov]

# [clv]
x, y, z = (res.clv_states[..., i] for i in range(3))
f = np.stack([SIGMA * (y - x), 28.0 * x - y - x * z, x * y - B * z], axis=-1)   # vector field
g = res.clv[:, :, 1, :]                                                         # neutral CLV
cos_fg = np.abs((f * g).sum(-1)) / (np.linalg.norm(f, axis=-1) * np.linalg.norm(g, axis=-1))
print(f"neutral CLV vs f(x): min |cos| = {cos_fg.min():.8f} over {cos_fg.size} samples")
# [/clv]

# [poincare]
zmax = poincare_section(s, X0[:8], lambda X: X[:, 0] * X[:, 1] - B * X[:, 2],   # dz/dt = 0
                        direction=-1, steps=5000, g_tol=1e-10, max_iterations=30)
zk, zk1 = zmax.return_pairs(lambda p: p[2])
print(zmax.explain())
print(f"Lorenz map: {zk.size} pairs (z_k, z_k+1), z in [{zk.min():.2f}, {zk.max():.2f}]")
# [/poincare]

# [basins]
s20 = lorenz(20.0, 0.01)
c = np.sqrt(B * 19.0)
Cp, Cm = np.array([c, c, 19.0]), np.array([-c, -c, 19.0])
half = np.linspace(20 / 48, 20, 24)
grid = np.concatenate([-half[::-1], half])      # exactly symmetric (linspace(-20, 20) is not)
GX, GY = np.meshgrid(grid, grid)
starts = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, 19.0)])
ens = Ensemble(s20, len(starts), rhs="batched")          # 2304 members in one batch
ens.set_states(starts)


def attractor(e):
    X = e.states()
    near_p, near_m = np.linalg.norm(X - Cp, axis=1) < 1.0, np.linalg.norm(X - Cm, axis=1) < 1.0
    return np.where(near_p, 1, np.where(near_m, -1, 0))


fate = ensemble_fate(ens, attractor, check_every=100, min_steps=100, max_steps=10000,
                     classifier="|X - C+-| < 1")
basin = fate.labels.reshape(GX.shape)
print(fate.explain())
print(ens.explain())
# [/basins]

# [figures]
out = os.environ.get("CN_TUTORIAL_OUT")
if out:
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)            # a CN_TUTORIAL_OUT that does not exist yet
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(zk, zk1, ".", ms=2)
    ax[0].set(xlabel="$z_k$", ylabel="$z_{k+1}$", title="Lorenz map (maxima of z)")
    ax[1].imshow(basin, origin="lower", extent=[-20, 20, -20, 20], cmap="coolwarm")
    ax[1].set(xlabel="x", ylabel="y", title="basins of $C^\\\\pm$, r = 20, z = 19")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "t12_lorenz_map_basins.png"), dpi=120)
# [/figures]

# [check]
lam, err = res.exponents, res.stderr
D, dD = res.kaplan_yorke()
assert abs(lam.sum() + (SIGMA + 1 + B)) < 1e-2                 # RK443 truncation ~7e-3 at dt = 0.01
assert abs(lam[0] - 0.906) < 4 * err[0] + 0.01 and abs(lam[1]) < 4 * err[1] + 0.01
assert abs(lam[2] + 14.572) < 4 * err[2] + 0.02 and abs(D - 2.062) < 4 * dD + 0.01
assert cos_fg.min() > 1 - 1e-6
assert zk.size > 100 and 25 < zk.min() and zk.max() < 50
# r = 20: no chaotic attractor, but a start near the stable manifold of the origin can stay in
# transient chaos beyond the horizon: reported as undecided (0), never assigned
assert (fate.labels == 0).sum() <= 0.005 * fate.labels.size
assert np.array_equal(basin, -basin[::-1, ::-1])                # (x, y) -> (-x, -y) swaps C+ and C-
print(f"CHECK lambda = ({lam[0]:.4f} +- {err[0]:.4f}, {lam[1]:.4f} +- {err[1]:.4f}, "
      f"{lam[2]:.4f} +- {err[2]:.4f}), sum = {lam.sum():.5f} (-13.66667), D_KY = {D:.4f} +- {dD:.4f}; "
      f"basins: {int((basin == 1).sum())} / {int((basin == -1).sum())} members, "
      f"{int((basin == 0).sum())} undecided")
# [/check]
`,p=`"""Tutorial 13 -- where a global mode is born, and what the data say about it.

Expected values (checked at the end of the script):
  Ginzburg-Landau wavemaker S_0(x) = |exp(-chi^2 x^2)| |chi| / sqrt(pi)   Chomaz (2005), closed form
  d lambda_0 / d mu_2 = 1 / (4 chi^2)                                   closed form
  DMD of a transient: the four least stable eigenvalues, to 1e-9      the discrete spectrum
Run: python t13_sensitivity_data.py   (about 20 s of CPU)
"""
# [imports]
import numpy as np
import scipy.linalg as sla

from conatus.expert import (DataInnerProduct, Y, base_flow_sensitivity, dmd, ginzburg_landau_exact,
                            linear_ginzburg_landau, state_space,
                            structural_sensitivity)
from conatus.diagnostics import Chebyshev, Grid
# [/imports]

# [model]
NU, GAMMA, MU_C, MU_2 = 2 + 0.4j, 1 - 1j, 0.34, -0.01        # Bagheri et al. (2009)
BOUNDS = (-60.0, 60.0)                                     # the whole line, truncated (declared)


def gl(mu):
    """dq/dt = -nu q' + gamma q'' + mu(x) q, q = 0 at both ends; mu is a sympy expression in Y."""
    return linear_ginzburg_landau(nu=NU, gamma=GAMMA, mu=mu, bounds=BOUNDS)


MU = MU_C + MU_2 * Y ** 2 / 2
exact = ginzburg_landau_exact(nu=NU, gamma=GAMMA, mu_c=MU_C, mu2=MU_2, n=0)
print(exact.explain())
# [/model]

# [sensitivity]
bfs = base_flow_sensitivity(gl, MU, N=140, target=exact.eigenvalue, adjoint="continuous",
                            infinite_tol=1e-10, adjoint_bcs=[("q", 0, "left"), ("q", 0, "right")])
pair = bfs.pair
print(pair.explain())
st = structural_sensitivity(pair)
x_star, s_star = st.maximum(points=1201)
print(f"wavemaker maximum S = {s_star:.12f} at x = {x_star:.2e}")
dmu2 = bfs.predict(Y ** 2 / 2)                            # delta mu = x^2/2, i.e. d/d mu_2
print(f"d lambda/d mu_2 = {dmu2:.12f} (density)  {bfs.discrete_shift(Y ** 2 / 2):.12f} (pencil)")
# [/sensitivity]

# [data]
# The same model, smaller and more damped (mu_c = -0.2), in energy-orthonormal coordinates.
small = linear_ginzburg_landau(nu=NU, gamma=GAMMA, mu=-0.2 + MU_2 * Y ** 2 / 2, bounds=(-20, 20))
ss = state_space(small.discretize(32), infinite_tol=1e-10)
grid = Grid([Chebyshev(-20.0, 20.0, 64)])                 # values on 64 Gauss nodes
t_nodes = (2 * grid.axes[0].nodes() + 0.0) / 40.0
E = np.polynomial.chebyshev.chebvander(t_nodes, 31) @ ss.to_state      # q -> grid values
ip = DataInnerProduct.from_grid(grid, components=["q"], weights={"q": 1.0})

# (1) DMD of a free transient started in the span of the 4 least stable global modes
dt = 0.5
lam, V = np.linalg.eig(ss.A)
lead4 = np.argsort(-lam.real)[:4]
q0 = V[:, lead4] @ np.array([1.0, 0.5j, 0.3, 0.2 - 0.1j])
Phi = sla.expm(ss.A * dt)
X = E @ np.stack([np.linalg.matrix_power(Phi, j) @ q0 for j in range(40)], axis=1)
r = dmd(X, dt=dt, rank=4, inner_product=ip, amplitudes="optimal")
print(r.explain())

# [/data]

# [check]
S0 = abs(exact.chi) / np.sqrt(np.pi)                      # Chomaz (2005): S_0(0) = |chi|/sqrt(pi)
assert abs(s_star - S0) < 1e-9 * S0 and abs(x_star) < 1e-4
assert abs(dmu2 - 1 / (4 * exact.chi ** 2)) < 1e-9 * abs(dmu2)
for mu in lam[lead4]:
    assert np.min(np.abs(r.omega - mu)) < 1e-9, (mu, r.omega)
print("OK: wavemaker and d lambda/d mu_2 match the closed forms; DMD finds the four least stable "
      "eigenvalues")
# [/check]
`,m=`"""Tutorial 14 -- coming from Dedalus: a convection cell as a native model, then read from a Dedalus script.

Expected values (checked at the end of the script):
  the Dedalus script rb_cell_dedalus.py (written for this tutorial, next to this file) is
  translated by cn.from_dedalus with no unsupported line: SBDF2, fixed dt = 5e-4, 40 steps;
  the native model (equations as on paper, walls by name, no tau written) and the translation
  (the script's own tau terms and lifts) are two tau formulations of the same equations, so they
  differ by a truncation error that decays spectrally with nz: at nz = 32 the two agree to
  rounding (below 1e-12 relative), at least 1000 times closer than at nz = 12.
  Dedalus is not needed and not imported here; t14b_dedalus_check.py runs the same script with
  Dedalus and compares it with the translation (rounding).
Run: python t14_from_dedalus.py   (about 5 s of CPU)
"""
# [imports]
import os

import numpy as np

import conatus as cn
# [/imports]

# [native]
# Onset of convection in diffusive units: th is the temperature minus the conduction profile 1 - z,
# so it vanishes on both walls; u.ez carries the conduction profile into the heat equation.
RA, PR, PERIOD, NX, DT = 3000.0, 7.0, 2.0, 24, 5e-4


def start(x, z):
    """One roll of the temperature deviation."""
    return 1e-3 * np.cos(2 * np.pi * x / PERIOD) * np.sin(np.pi * z)


def native(nz):
    """The cell as a Conatus model: unknowns by role, equations as written on paper, conditions by
    wall name. No tau field and no lift: the native layer places one tau per condition."""
    m = cn.Model(cn.Box(x=cn.periodic(PERIOD), z=cn.walls(0, 1)), name="convection onset",
                 Ra=RA, Pr=PR)
    m.velocity("u"), m.pressure("p"), m.scalar("th")
    m.equations("""
        momentum:  dt(u) - Pr*lap(u) + grad(p) - Ra*Pr*th*ez = -u·grad(u)
        mass:      div(u) = 0
        heat:      dt(th) - lap(th) - u·ez = -u·grad(th)
    """)
    m.at("bottom", u=0, th=0)
    m.at("top", u=0, th=0)
    sim = m.simulate(N=(NX, nz), scheme="SBDF2", dt=DT)
    sim.initial(th=start)
    return sim.run(steps=40)


cell = native(16)
print(f"native model: t = {cell.t:.3f}, max|u| = {np.abs(cell.grid('u')).max():.3e}")
# [/native]

# [translate]
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rb_cell_dedalus.py")
tr = cn.from_dedalus(SCRIPT)          # read with \`ast\`, never executed; Dedalus is not imported
print(tr.explain())
counts = tr.counts()
# [/translate]

# [run]
solver = tr.ivp_solver(startup="lower_order")      # the script's SBDF2, dt and initial th
solver.run(tr.steps())                              # 40 steps, no Python time loop
print(f"translation: t = {solver.t:.3f} after {tr.steps()} steps, "
      f"max|u| = {np.abs(solver.grid('u')).max():.3e}")
# [/run]

# [compare]
def translated(nz):
    """The same script at another wall-normal resolution (a declared, reported override)."""
    t = cn.from_dedalus(SCRIPT, overrides={"N_z": nz})
    s = t.ivp_solver(startup="lower_order")
    s.run(t.steps())
    return s


def rel(a, b, name):
    return np.abs(a.grid(name) - b.grid(name)).max() / np.abs(b.grid(name)).max()


gap = {}
for nz in (12, 16, 32):
    a, d = (cell, solver) if nz == 16 else (native(nz), translated(nz))
    gap[nz] = (rel(a, d, "th"), rel(a, d, "u"))
    print(f"nz = {nz:2d}: native vs translated, relative difference th {gap[nz][0]:.1e}, "
          f"u {gap[nz][1]:.1e}")
# [/compare]

# [check]
P = tr.numbers()
assert counts["unsupported"] == 0 and tr.scheme == "SBDF2" and tr.steps() == 40
assert P["Ra"] == RA and P["Pr"] == PR and P["tstep"] == DT       # same physics, same step
assert max(gap[32]) < 1e-12 and max(gap[32]) < 1e-3 * max(gap[12])   # spectral convergence
print("PASS")
# [/check]
`,h=`"""Tutorial 14b -- check the translation of rb_cell_dedalus.py against Dedalus itself.

Expected values (checked at the end of the script):
  the Dedalus script rb_cell_dedalus.py is run twice: by Dedalus (imported here at run time, and
  only here), and by Conatus through cn.from_dedalus. Same bases, tau terms, lifts, scheme, fixed
  dt, start-up and dealiasing, so after 40 steps of SBDF2 the temperature deviation and the
  velocity agree to rounding: below 1e-10 relative.
  Without Dedalus the comparison is skipped and the script says so (exit 0).
Run: python t14b_dedalus_check.py   (under a minute of CPU, most of it importing Dedalus)
"""
# [imports]
import logging
import os
import runpy

import numpy as np

import conatus as cn
# [/imports]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rb_cell_dedalus.py")

# [conatus]
tr = cn.from_dedalus(SCRIPT)
solver = tr.ivp_solver(startup="lower_order")      # Dedalus's multistep start-up, said explicitly
solver.run(tr.steps())
th_cn, u_cn = solver.grid("th"), solver.grid("u")
print(f"Conatus: t = {solver.t:.3f} after {tr.steps()} steps")
# [/conatus]

# [dedalus]
os.environ.setdefault("OMP_NUM_THREADS", "1")      # Dedalus asks for it before its import
try:
    import dedalus  # noqa: F401  (comparison oracle only; the conatus package never imports it)
    have_dedalus = True
except ImportError:
    have_dedalus = False
    print("Dedalus is not installed here: the comparison with Dedalus is skipped (declared)")

if have_dedalus:
    logging.disable(logging.INFO)                   # Dedalus's per-step log lines
    ns = runpy.run_path(SCRIPT, run_name="__main__")   # the script, run by Dedalus as written
    out = {}
    for name in ("th", "u"):
        ns[name].change_scales(1)                   # back to the grid without 3/2 padding
        out[name] = np.array(ns[name]["g"])
    rel_th = np.abs(out["th"] - th_cn).max() / np.abs(out["th"]).max()
    rel_u = np.abs(out["u"] - u_cn).max() / np.abs(out["u"]).max()
    print(f"Dedalus: t = {ns['run'].sim_time:.3f}; relative difference th {rel_th:.1e}, "
          f"u {rel_u:.1e}")
# [/dedalus]

# [check]
assert tr.counts()["unsupported"] == 0 and tr.steps() == 40
if have_dedalus:
    assert rel_th < 1e-10 and rel_u < 1e-10, (rel_th, rel_u)
print("PASS")
# [/check]
`,g=`"""Tutorial 15 -- covariant Lyapunov vectors of Kuramoto--Sivashinsky, L = 22: exponents, the
covariance check, tangencies and the eight physical modes.

u_t + u u_x + u_xx + u_xxxx = 0 on (0, 22), periodic, zero mean.
Expected values (checked at the end of the script):
  leading exponents (0.048, 0, 0, -0.003, -0.189, -0.256, -0.290, -0.310, -1.963, -1.967, ...)
      Cvitanovic, Davidchack & Siminos (2010), SIAM J. Appl. Dyn. Syst. 9, p. 12 (zero mean)
  eight "physical" modes: the first eight covariant vectors are entangled (frequent violations of
  the ordering of their local exponents, near-tangencies), the others are isolated (in pairs)
      Ding, Chate, Cvitanovic, Siminos & Takeuchi (2016), Phys. Rev. Lett. 117, 024101
A short run (4 trajectories, 200 time units each): the exponents carry their standard errors and
are compared within them; the long runs of the validation table go to 2500 time units.
Run: python t15_ks_covariant.py   (about two minutes of CPU)
"""
# [imports]
import numpy as np

import conatus as cn
# [/imports]

# [model]
L = 22.0
m = cn.Model(cn.Box(x=cn.periodic(L)), name="Kuramoto-Sivashinsky L=22")
m.scalar("u")
m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
sim = m.simulate(N=32, scheme="RK443", dt=0.05)   # 32 modes, as in the references


def start(c):
    """A smooth zero-mean initial state from four random modes."""
    return {"u": lambda x: 0.3 * sum(c[j, 0] * np.cos(2 * np.pi * (j + 1) * x / L)
                                     + c[j, 1] * np.sin(2 * np.pi * (j + 1) * x / L)
                                     for j in range(4))}


rng = np.random.default_rng(7)
starts = [start(rng.standard_normal((4, 2))) for _ in range(4)]
# [/model]

# [lyapunov]
res = sim.lyapunov(12, transient=100.0, align=50.0, T=200.0, reortho_every=10,
                   inner="euclidean", n_blocks=4, seed=1, starts=starts, zero_mean="u",
                   covariant=True, clv_window=40.0, clv_backward=40.0, clv_checkpoints=8)
print(res.explain())
# [/lyapunov]

# [domination]
dom = res.domination(tau=5.0)
print(dom.explain())
theta8 = res.angles(8)                     # span(gamma_1..gamma_8) vs the rest, every sample
theta4 = res.angles(4)
print(f"smallest angle at the split 8 | 9: {theta8.min():.3f} rad; at 4 | 5: "
      f"{theta4.min():.4f} rad ({np.mean(theta4 < 0.1):.0%} of the samples below 0.1 rad)")
# [/domination]

# [checks]
cds = np.array([0.048, 0.0, 0.0, -0.003, -0.189, -0.256, -0.290, -0.310, -1.963, -1.967])
dev = np.abs(res.exponents[:10] - cds)
assert np.all(dev < 4 * res.stderr[:10] + 0.01), (res.exponents[:10], res.stderr[:10])
# J gamma_i(t) is parallel to gamma_i(t + dt_interval), to the rounding floor of each vector
assert np.all(res.clv_covariance_residual < 100 * res.clv_covariance_floor)
assert dom.n_physical == 8 and dom.violation(8) == 0.0
assert res.clv_split_convergence[7] < 1e-6            # the split 8 | 9 itself has converged
assert theta8.min() > 10 * theta4.min()               # isolated split versus tangencies
print("checks passed: exponents within their errors of CDS 2010, eight physical modes")
# [/checks]
`,_=`"""Tutorial 16 -- Lagrangian tracers and a numerical dye: exact velocity at the particles, the
order of the time integration, and a dye whose mass is measured.

1. Decaying Taylor--Green vortices, u = e^{-2 nu t} (sin x cos y, -cos x sin y), an exact solution
   of the Navier--Stokes equations: a tracer follows the streamline psi = sin x sin y at the time
   s(t) = (1 - e^{-2 nu t}) / (2 nu); the reference is that path integrated to 1e-13 (DOP853).
2. Plane Couette flow (walls moving at +-1, u = y): tracers move at their height's speed, tracers
   on a wall move with it; a dye blob is sheared and diffuses, its mass is conserved.
Expected values (checked at the end of the script): the tracer error falls as dt^3 in the decaying
flow (cubic interpolation in time of the flow states; dt^4 in a steady flow); Couette tracers
exact to rounding; dye mass conserved to the level reported (1e-8 bound with walls).
Run: python t16_particles_dye.py   (under a minute of CPU)
"""
# [imports]
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import conatus as cn
# [/imports]

# [taylor_green]
nu, T = 0.5, 2.0


def taylor_green(dt):
    box = cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi))
    m = cn.Model(box, name="decaying Taylor-Green", nu=nu)
    m.velocity("u"), m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    sim = m.simulate(N=(16, 16), scheme="RK443", dt=dt)
    sim.initial(u=lambda x, y: (np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)))
    return sim


X0 = np.random.default_rng(0).uniform(0, 2 * np.pi, (32, 2))
s = (1 - np.exp(-2 * nu * T)) / (2 * nu)


P = len(X0)


def path(_t, Z):
    x, y = Z[:P], Z[P:]
    return np.concatenate([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)])


ref = solve_ivp(path, (0, s), np.concatenate([X0[:, 0], X0[:, 1]]), method="DOP853",
                rtol=1e-13, atol=1e-13).y[:, -1]
ref = np.stack([ref[:len(X0)], ref[len(X0):]], axis=1)
errors = []
for dt in (0.1, 0.05, 0.025):
    sim = taylor_green(dt)
    tracers = sim.particles(X0)
    sim.run(until=T)
    errors.append(np.abs(tracers.positions(unwrapped=True) - ref).max())
    print(f"dt = {dt:<6}: largest tracer error {errors[-1]:.2e}")
orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
print("observed orders:", np.round(orders, 2))
print(tracers.explain())
# [/taylor_green]

# [couette]
box = cn.Box(x=cn.periodic(4.0), y=cn.walls(-1, 1))
m = cn.Model(box, name="plane Couette with dye", nu=0.1)
m.velocity("u"), m.pressure("p")
m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
m.at("bottom", u=(-1, 0))
m.at("top", u=(1, 0))
m.dye("c", velocity="u", kappa=0.01)      # dt(c) - 0.01*lap(c) = -div(c*u), dy(c) = 0 at walls
print(m.explain())
sim = m.simulate(N=(32, 32), scheme="RK443", dt=0.01)
sim.initial(u=lambda x, y: (y, 0 * x), c=lambda x, y: np.exp(-((x - 2) ** 2 + y ** 2) / 0.3))
out = Path(tempfile.mkdtemp())
starts = np.array([[0.0, -1.0], [1.0, -0.5], [2.0, 0.0], [3.0, 0.5], [0.5, 1.0]])
tracers = sim.particles(starts, path=out / "tracers.h5", every=50)
dye = sim.dye("c", out / "dye.h5", every=50)
sim.run(until=3.0)
print(dye.explain())
data = cn.io.read(out / "tracers.h5").to_xarray()
print(data["position_unwrapped"].isel(t=-1).values)
mass = cn.io.read(out / "dye.h5").tasks["mass"]
drift = np.abs(mass - mass[0]).max() / mass[0]
print(f"dye mass {mass[0]:.12f}, largest relative drift {drift:.1e}")
# [/couette]

# [checks]
assert np.all(orders > 2.6), orders                   # third order in an unsteady flow
exact = starts + np.column_stack([3.0 * starts[:, 1], 0 * starts[:, 1]])
assert np.abs(tracers.positions(unwrapped=True) - exact).max() < 1e-12
assert drift < 1e-8
print("checks passed: order 3 in time, Couette tracers exact, dye mass conserved")
# [/checks]
`,v=`# t00_getting_started: PASS (exit 0) in 75.6 s | 2026-10-09
Model 'Rayleigh-Benard': Box(x=periodic((0, 2)), z=walls(-0.5, 0.5)) (Cartesian box)
Unknowns: u (velocity), p (pressure), T (scalar); parameters: Ra = 2500, Pr = 1
Equations as written:
  momentum: dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
  continuity: div(u) = 0
  heat: dt(T) - lap(T) = -u·grad(T)
Problem: IVP (a time derivative dt() appears)
Time stepping: 7 linear term(s) implicit; nonlinear terms explicit (momentum, heat)
Boundary conditions: bottom: u = 0, T = 0.5; top: u = 0, T = -0.5 (tau method, placed automatically)
Gauge: the mean of p is fixed (integ(p) = 0)
Dealiasing: x: 3/2, z: 3/2 (declared default for products of degree 2)
Resolution: not chosen yet (simulate/eigenvalues/discretize(N=...))
(every decision, tau by tau and term by term: m.explain(full=True))
Ra = 1500: largest growth rate -1.637012
Ra = 2500: largest growth rate +5.542506
Ra_c = 1707.7618   k_c = 3.11632
measured growth rate 5.542508   eigenvalue 5.542506
t = 6.0: Nu = 1.474455 (layer)  1.474455 (bottom plate)  1.474455 (top plate)
Run (rb_Ra2500): state completed, status provisional, 47.5 s
Methods paragraph generated from the plan (edit before publishing):

The IVP 'Rayleigh-Benard' for (u, p, T) was solved with Conatus. The domain is discretized as follows: x ∈ [0, 2) is periodic and expanded in a real (cosine/sine) Fourier series of 32 modes; z ∈ [-0.5, 0.5] is expanded in 24 Chebyshev polynomials of the first kind. Derivatives, basis conversions and multiplications by non-constant coefficients use the sparse, banded ultraspherical operators of Olver & Townsend, and boundary conditions are imposed with the generalized tau method (4 tau variable(s), each entering through a lift polynomial), following Burns et al. The Fourier directions decouple, giving 16 independent systems, one per wavenumber. Nonlinear terms are evaluated pseudospectrally with dealiasing factors 3/2 in x, 3/2 in z. The structural rank of every system was checked before factorization. Time integration uses the third-order, four-stage IMEX Runge–Kutta scheme (4,4,3) of Ascher, Ruuth & Spiteri, whose implicit diagonal is constant so that a single factorization serves all stages, with a constant step Δt = 0.0005. The implicit systems are solved by banded LU factorization with a low-rank (Woodbury) correction for the tau rows, batched over wavenumbers on CPU, in float64 arithmetic. The equations were declared in the native layer of Conatus as written on paper; the implicit-explicit split, the 4 tau term(s) (each in the equation where its unknown has its highest wall-normal order) and the pressure gauge and its integral constraint were placed by fixed, declared rules (listed by explain()). The dealiasing factor 3/2 is the declared default of the native layer ((p+1)/2 for the highest polynomial degree p = 2 of the equations, at least 3/2).

Citations: olver2013, burns2020, ascher1997
Facts used: kind=IVP, groups=16, N_x=32, N_z=24, zbasis=Chebyshev, taus=4, dealias={'x': 1.5, 'z': 1.5}, scheme=RK443, dt=0.0005, backend=cpu, native=True, dealias_default=True
OK: onset at Reid & Harris' Ra_c and k_c; the simulation grows at the eigenvalue's rate and saturates into rolls carrying the same heat through both plates
`,y=`# t01_orr_sommerfeld: PASS (exit 0) in 3.6 s | 2026-10-09
Model 'Orr-Sommerfeld'
Geometry: Box(y=walls(-1, 1)) (Cartesian box)
  bases (fixed per geometry, sizes given at discretization):
    y: walls at -1, 1 -> Chebyshev T(N_y)
  named boundaries: bottom = y left; top = y right; y=-1 = y left; y=1 = y right; walls = y left, y right
Unknowns: v (scalar)
Parameters: Re = 10000, alpha = 1 (change: m.<name> = value; for one call: simulate(..., name=value))
Given fields: U
Eigenvalue: c (complex); growth = Im c, declared with the eigenvalue c
Operator: L2(f) = dy(dy(f)) - alpha**2*f
Equations as written:
  [0] Orr_Sommerfeld: L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0
Conditions as written:
  at walls: v = 0
  at walls: dy(v) = 0
Problem kind: EVP (eigenvalue c declared)
Term placement:
  [0] Orr_Sommerfeld
      operator  (1j*alpha*Re)**-1*(dy(dy(-alpha**2*v + dy(dy(v)))) - (-alpha**2*v + dy(dy(v)))*alpha**2)   (linear, constant coefficients)
      operator  c*(-alpha**2*v + dy(dy(v)))   (linear, constant coefficients)
      operator  dy(dy(U))*v   (linear, non-constant coefficient U)
      operator  -(-alpha**2*v + dy(dy(v)))*U   (linear, non-constant coefficient U)
Boundary conditions -> tau terms (automatic, deterministic):
  v = 0 at walls (left: y=-1): tau_v1 lifted into mode N_y-1 of [0] Orr_Sommerfeld -- v has its highest y-order (4) there; Chebyshev lift into the last modes
  v = 0 at walls (right: y=1): tau_v2 lifted into mode N_y-2 of [0] Orr_Sommerfeld -- v has its highest y-order (4) there; Chebyshev lift into the last modes
  dy(v) = 0 at walls (left: y=-1): tau_v3 lifted into mode N_y-3 of [0] Orr_Sommerfeld -- v has its highest y-order (4) there; Chebyshev lift into the last modes
  dy(v) = 0 at walls (right: y=1): tau_v4 lifted into mode N_y-4 of [0] Orr_Sommerfeld -- v has its highest y-order (4) there; Chebyshev lift into the last modes
Dealiasing: y: 3/2 (declared default 3/2 of the native layer: max(3/2, (p+1)/2) for the highest polynomial degree p = 1 of the equations in the unknowns, exact (alias-free) products on Fourier and polynomial bases; override with dealias=)
Bases: Chebyshev(y, size=8, bounds=(-1.0, 1.0), dealias=3/2)
Resolution: not chosen yet (simulate/eigenvalues/discretize(N=...))
EVP result [single group (no Fourier direction)] (dense QZ (scipy.linalg.eig)): 8 finite modes kept by the filters below, 8 infinite (M singular on tau/boundary rows)
  spectral-tail test (default, spurious='tail'): 0 of 8 modes flagged as spurious tau or unresolved modes (relative energy in the last 0.1 of the wall-normal coefficients above SPURIOUS_TAIL_TOL = 0.001); they are kept and ranked last by order()
  kept 8 of 60 modes: resolution drift <= 1e-06 against 92 modes at the finer resolution; 52 unresolved or spurious (tau) modes removed
  growth: growth = Im c, declared with the eigenvalue c
  modes sorted by decreasing imag part:
  c =  2.375264888194e-01+3.739670621163e-03j  residual 4.2e-16, drift 8.6e-12
  c =  9.646309154726e-01-3.516727756365e-02j  residual 5.7e-16, drift 7.1e-11
  c =  9.646425099829e-01-3.518658363628e-02j  residual 2.7e-16, drift 1.7e-10
  c =  2.772043438891e-01-5.089872753275e-02j  residual 1.6e-15, drift 2.9e-10
  c =  9.363165149228e-01-6.320147641683e-02j  residual 6.3e-16, drift 2.9e-08
  c =  9.363517179151e-01-6.325156402150e-02j  residual 3.0e-16, drift 6.3e-08
  (2 more log line(s): pencil and infinite-eigenvalue checks, solver settings: explain(full=True))
c = 0.2375264888+0.0037396706j   backward residual 4.2e-16
EVPMode 0 [single group (no Fourier direction)]: eigenvalue 0.237526488819+0.00373967062116j, residual 4.2e-16
  unknowns: v (complex grid values on the non-dealiased grid, y: 64; tau variables left out)
  normalisation: max |q| = 1 for the first non-zero unknown (declaration order), real and positive at its largest grid value; the other unknowns scaled by the same complex factor (here: v, max |v| = 1 at its largest grid value)
CriticalPoint Re_c = 5772.221816 at alpha=1.020547442 (11 neutral-curve evaluations)
  start: Re = 6000 (bracket midpoint (default))
  tolerance: |dRe| <= 1e-09 |Re| + 2e-09
  growth = Im c, declared with the eigenvalue c
  spectral-tail test passed 35 of 60 (SPURIOUS_TAIL_TOL = 0.001); the spectral-tail test set aside 2 mode(s) of larger growth than every mode it kept (largest: sigma = 0.01709590363+2424.532428j, growth 2.425e+03, tail 1.0e+00): spurious tau or unresolved modes; raise N, or spurious='none' to select them; drift test kept 13 of 60 (resolution drift <= 1e-06 against the finer pencil [single group (no Fourier direction)] size (100, 100)) finite modes (8 infinite)
  at the critical point: spectral-tail test passed 35 of 60 (SPURIOUS_TAIL_TOL = 0.001); the spectral-tail test set aside 2 mode(s) of larger growth than every mode it kept (largest: sigma = 0.01678229327+2495.520313j, growth 2.496e+03, tail 1.0e+00): spurious tau or unresolved modes; raise N, or spurious='none' to select them; drift test kept 14 of 60 (resolution drift <= 1e-06 against the finer pencil [single group (no Fourier direction)] size (100, 100)) finite modes (8 infinite)
  ParametricEVP [single group (no Fourier direction)] size (68, 68): Re: exponents -1..0 (nominal 10000.0), alpha: exponents -1..3 (nominal 1.0); 5 coefficient matrices from 10 plans at roots of unity (5 identically zero dropped); reconstruction error at an independent real point 1.5e-16 (refused above 1e-10); pencil L + sM: structural rank 68/68, regular at a generic s (equilibrated sigma_min/sigma_max = 1.1e-07); mode tracking: bordered Newton, declared defaults TRACK_TOL = 1e-13, TRACK_MAX_ITERATIONS = 30
  verification: full QZ at the critical point, tracked mode is the least stable admissible one (max growth -2.1e-13), residual 5.0e-20
  TrackedMode sigma = 2.640002595765e-01-3.080497937910e-17j at {'alpha': np.float64(1.0205474415539748), 'Re': np.float64(5772.221816217812)}: residual 5.0e-20 after 1 bordered-Newton iterations; left vector available (sensitivities)
Re_c = 5772.2218   alpha_c = 1.020547
dc/dRe = -4.213e-06+3.428e-07j
Verification report: 9 checks, 0 to investigate by hand
  [PASS] pencil reconstruction: 1.54e-16 (threshold 1e-10) — parametric L(p), M(p) vs a directly assembled plan
  [PASS] pencil regular: 0.00e+00 (threshold 5e-01) — pencil L + sM: structural rank 68/68, regular at a generic s (equilibrated sigma_min/sigma_max = 1.1e-07)
  [PASS] right residual: 5.02e-20 (threshold 1e-12) — ||(L + sigma M) x|| backward error
  [PASS] left residual: 1.28e-20 (threshold 1e-12) — ||y^H (L + sigma M)|| backward error
  [PASS] spectral tail: 1.72e-10 (threshold 1e-08) — last 4 Chebyshev coefficients / max (mode resolved in N)
  [PASS] d sigma/d Re Taylor test: 4.45e-09 (threshold 1e-05) — exact sensitivity vs central difference of tracked sigma
  [PASS] d sigma/d alpha Taylor test: 7.60e-09 (threshold 1e-05) — exact sensitivity vs central difference of tracked sigma
  [PASS] critical point mode: 0.00e+00 (threshold 5e-01) — full QZ at the critical point confirms the tracked mode is the least stable
  [PASS] critical growth: 3.08e-17 (threshold 1e-10) — growth of the tracked mode at the neutral value
OK: c and Re_c agree with Orszag (1971) to the published digits; alpha_c within 2e-5
`,b=`# t02a_rayleigh_benard_cpu: PASS (exit 0) in 14.8 s | 2026-10-09
Model 'Rayleigh-Benard 2D': Box(x=periodic((0, 2)), z=walls(-0.5, 0.5)) (Cartesian box)
Unknowns: u (velocity), p (pressure), T (scalar); parameters: Ra = 2500, Pr = 1
Equations as written:
  momentum: dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
  continuity: div(u) = 0
  heat: dt(T) - lap(T) = -u·grad(T)
Problem: IVP (a time derivative dt() appears)
Time stepping: 7 linear term(s) implicit; nonlinear terms explicit (momentum, heat)
Boundary conditions: bottom: u = 0, T = 0.5; top: u = 0, T = -0.5 (tau method, placed automatically)
Gauge: the mean of p is fixed (integ(p) = 0)
Dealiasing: x: 3/2, z: 3/2 (declared default for products of degree 2)
Resolution: not chosen yet (simulate/eigenvalues/discretize(N=...))
(every decision, tau by tau and term by term: m.explain(full=True))
Linearization of 'Rayleigh-Benard 2D' about a base state, eigenvalue s (normal modes exp(s t)):
  u = 0 about the zero field
  T = T0(z), a non-constant coefficient
  Fourier sizes chosen to hold k: N_x = 4 (the matrices of one wavenumber do not depend on the others)
EVP: growth rate of the k = pi roll at Ra = 2500: sigma = 5.5425059231
Simulation of 'Rayleigh-Benard 2D' at N = x: 16, z: 24: RK443, dt = 0.0005, backend cpu
  operators assembled (kept for reuse by the next identical simulate(); cache of 2 per model)
Parameters in use (compiled into this simulation): Ra = 2500, Pr = 1
Problem: IVP (a time derivative dt() appears)
Time stepping: 7 linear term(s) implicit; nonlinear terms explicit (momentum, heat)
Boundary conditions: bottom: u = 0, T = 0.5; top: u = 0, T = -0.5 (tau method, placed automatically)
Gauge: the mean of p is fixed (integ(p) = 0)
Dealiasing: x: 3/2, z: 3/2 (declared default for products of degree 2)
Initial state: T given; u, p start at zero
IVPSolver(scheme=RK443, dt=0.0005, constant step, backend=cpu)
  scheme: RK443: IMEX Runge–Kutta (4 stages), implicit L-stable SDIRK3 on L, explicit 4-stage Runge–Kutta on F; global order 3 (Ascher, Ruuth & Spiteri 1997, (4,4,3))
  finite-state check: every 100 steps (default), at the end of each run call and before each checkpoint/output write (NaN/inf -> NumericalError naming step, time and unknown; check_finite=False turns it off)
  resolution check: with the finite-state check, the relative spectral tail of each unknown and direction above 0.001 warns once (none so far)
(every decision, tau by tau and term by term, and the solver layout: sim.explain(full=True))
CFL time step at safety 0.5: 2.18e+01  (a stability indicator, not an error)
IVP: measured growth rate sigma = 5.5425080071 over t in [0.30, 0.80]
Methods paragraph generated from the plan (edit before publishing):

The IVP 'Rayleigh-Benard 2D' for (u, p, T) was solved with Conatus. The domain is discretized as follows: x ∈ [0, 2) is periodic and expanded in a real (cosine/sine) Fourier series of 16 modes; z ∈ [-0.5, 0.5] is expanded in 24 Chebyshev polynomials of the first kind. Derivatives, basis conversions and multiplications by non-constant coefficients use the sparse, banded ultraspherical operators of Olver & Townsend, and boundary conditions are imposed with the generalized tau method (4 tau variable(s), each entering through a lift polynomial), following Burns et al. The Fourier directions decouple, giving 8 independent systems, one per wavenumber. Nonlinear terms are evaluated pseudospectrally with dealiasing factors 3/2 in x, 3/2 in z. The structural rank of every system was checked before factorization. Time integration uses the third-order, four-stage IMEX Runge–Kutta scheme (4,4,3) of Ascher, Ruuth & Spiteri, whose implicit diagonal is constant so that a single factorization serves all stages, with a constant step Δt = 0.0005. The implicit systems are solved by banded LU factorization with a low-rank (Woodbury) correction for the tau rows, batched over wavenumbers on CPU, in float64 arithmetic. The equations were declared in the native layer of Conatus as written on paper; the implicit-explicit split, the 4 tau term(s) (each in the equation where its unknown has its highest wall-normal order) and the pressure gauge and its integral constraint were placed by fixed, declared rules (listed by explain()). The dealiasing factor 3/2 is the declared default of the native layer ((p+1)/2 for the highest polynomial degree p = 2 of the equations, at least 3/2).

Citations: olver2013, burns2020, ascher1997
Facts used: kind=IVP, groups=8, N_x=16, N_z=24, zbasis=Chebyshev, taus=4, dealias={'x': 1.5, 'z': 1.5}, scheme=RK443, dt=0.0005, backend=cpu, native=True, dealias_default=True
resume_or_start('<path>/rb_chain_5e4oi7tr/chk'): continued from checkpoint_000000600.h5 (iteration 600, t = 0.3); outputs without mode= continue their file (mode='resume')
AnalysisFile <path>/rb_chain_5e4oi7tr/rb.h5 (HDF5, conatus.analysis/1): every 100 steps (= 0.05 time units), 17 write(s) so far [mode 'resume' chosen by resume_or_start (restart)]
17 rows, t = 0.00 ... 0.80, kinetic energy at the end 1.028e-06
OK: the IVP grows at the EVP rate; the EVP matches the psi-theta formulation; the two-job chain is the uninterrupted run bit for bit
`,x=`# t02b_rayleigh_benard_gpu: PASS (exit 0) in 14.5 s | 2026-10-02
cpu: 27.98 ms per step
gpu: 0.61 ms per step
Simulation of 'Rayleigh-Benard 2D' (native model) at N = x: 128, z: 64
Problem kind: IVP (a time derivative dt() appears)
Term placement (IMEX split, automatic; override with implicit(...) / explicit(...)):
  [0] momentum
      implicit  grad(p)   (linear, constant coefficients)
      implicit  -Pr*lap(u)   (linear, constant coefficients)
      implicit  dt(u)   (linear, constant coefficients)
      explicit  -dot(u, grad(u))   (nonlinear, degree(s) [2] in the unknowns)
      implicit  -Ra*Pr*T*ez   (linear, constant coefficients; moved from the right side)
  [1] continuity
      implicit  div(u)   (linear, constant coefficients)
  [2] heat
      implicit  dt(T)   (linear, constant coefficients)
      implicit  -lap(T)   (linear, constant coefficients)
      explicit  dot(-u, grad(T))   (nonlinear, degree(s) [2] in the unknowns)
Boundary conditions -> tau terms (automatic, deterministic):
  u = 0 at bottom: tau_u1 lifted into mode 63 (N_z-1) of [0] momentum -- u has its highest z-order (2) there; Chebyshev lift into the last modes
  u = 0 at top: tau_u2 lifted into mode 62 (N_z-2) of [0] momentum -- u has its highest z-order (2) there; Chebyshev lift into the last modes
  T = 0.5 at bottom: tau_T1 lifted into mode 63 (N_z-1) of [2] heat -- T has its highest z-order (2) there; Chebyshev lift into the last modes
  T = -0.5 at top: tau_T2 lifted into mode 62 (N_z-2) of [2] heat -- T has its highest z-order (2) there; Chebyshev lift into the last modes
Incompressibility recognized: p enters only through grad(p) and has no condition; [1] continuity is its constraint (div(u), no time derivative):
  gauge tau tau_p (a constant) added to [1]; integral constraint integ(p) = 0 fixes the mean p
  divergence tau dot(ez, lift(tau_u2, z, -1)) (the velocity tau of the lowest lifted mode)
Dealiasing: x: 3/2, z: 3/2 (declared default 3/2 of the native layer; override with dealias=)
Bases: RealFourier(x, size=128, bounds=(0.0, 2.0), dealias=3/2); Chebyshev(z, size=64, bounds=(-0.5, 0.5), dealias=3/2)
Initial state: T given; u, p start at zero
IVPSolver(scheme=RK443, dt=0.0002, constant step, backend=cuda)
  scheme: RK443: IMEX Runge–Kutta (4 stages), implicit L-stable SDIRK3 on L, explicit 4-stage Runge–Kutta on F; global order 3 (Ascher, Ruuth & Spiteri 1997, (4,4,3))
GlobalSystem: 64 group(s), 33275 unknowns (full space), dtype float64, ordering=interleaved, complexify=False
  2 solver bucket(s) (groups sharing one sparsity pattern, batched GPU bordered banded solves, one warp per mode)
  [x: k=0] n=263, border rows=7, band kl=12 ku=18
  [x: k=3.14159] n=524, border rows=12, band kl=24 ku=37
  0 complexified bucket(s)
  state resident on the device (CuPy), pivot=partial
  solves: gather/scatter fused in the solve kernel, independent buckets on concurrent streams, level-scheduled sparse LU factors (same factors and per-row arithmetic as the band kernel)
  F: pseudospectral CuPy (GPU) evaluator, products on each basis' declared dealiased grid; linear solves: C++ bordered banded, batched per bucket
FusedRHS (polynomial form): 8 factors -> 3 outputs, real, one batched transform each way, one fused kernel
  factor u[0]
  factor dx^1u[0]
  factor dz^1u[0]
  factor u[1]
  factor dx^1u[1]
  factor dz^1u[1]
  factor dx^1T
  factor dz^1T
  eq 0[0]: 2 monomials
  eq 0[1]: 2 monomials
  eq 2: 2 monomials
  real projection on RealFourier grids checked: imaginary part <= 1e-13 relative, error otherwise (GPU: accumulated on the device, checked after each step / replayed block)
  derivatives of products are refused (no Leibniz rewrite); parameters frozen, recompiled when they change
  F: 2D RealFourier x Chebyshev pipeline (8 factors -> 3 outputs, grid 192 x 96): factors from X (one block per Fourier pair), cuFFT (Chebyshev DCT by length-M real FFTs (Makhoul), then x), fused products, cuFFT back, C^(lam) conversion into the stacked rows; ~10 kernels per evaluation; same numbers as the fused evaluator to FFT rounding
  step loop: Python loop over steps; step() launches every kernel from the host, run_graph(n) captures one step as a CUDA graph and replays it once per step (same kernels, same arithmetic; constant dt, F independent of t)
  per stage: RK stage right-hand side MX0 + sum dt a F - dt h L X in one kernel for real float64 states (M X and L X applied inside it, roundings identical to the array operations; timestep/lincomb.py), then the batched solve
gpu + CUDA graph: 0.67 ms per step
after 20 steps: relative max |T_cpu - T_gpu| = 8.6e-15; graph vs plain = 0.0e+00
after 200 steps: max |T_cpu - T_gpu| = 9.1e-07 (rounding amplified by the instability)
OK: CPU and GPU agree to rounding; the graph replay is the same computation
`,S=`# t03_symmetries: PASS (exit 0) in 1.8 s | 2026-10-09
 full space: leading sigma = +1.6955e-06   n = 268   QZ 167.1 ms   residual 4.4e-18
unknowns of the perturbation problem: du, dp, dT, dtau_u1, dtau_u2, dtau_T1, dtau_T2, dtau_p
Symmetry subspace: Z2 (order 2)
  generator: Reflection(z -> -z): reflection of z about z = 0; vector components along it change sign, others do not
  T0         odd under Reflection(z -> -z)
  dT         even under Reflection(z -> -z)
  dp         odd under Reflection(z -> -z)
  dtau_T1    odd under Reflection(z -> -z)
  dtau_T2    even under Reflection(z -> -z)
  dtau_p     odd under Reflection(z -> -z)
  dtau_u1    even under Reflection(z -> -z)
  dtau_u2    odd under Reflection(z -> -z)
  du         odd under Reflection(z -> -z)
  expected reduction: up to x2 per reflection, x m0 per m0-fold rotation (adapted bases: parity blocks, kept azimuthal classes, docs/site/tutorials/03_symmetries.md)
  WARNING: computing in the invariant subspace hides symmetry-breaking instabilities; a symmetric solution is not declared stable before symmetry_breaking_check().
onset class: leading sigma = +1.6955e-06   n = 134   QZ  67.1 ms   residual 4.2e-18
Symmetry subspace: Z2 (order 2)
  generator: Reflection(z -> -z): reflection of z about z = 0; vector components along it change sign, others do not
  T0         odd under Reflection(z -> -z)
  dT         even under Reflection(z -> -z)
  dp         odd under Reflection(z -> -z)
  dtau_T1    odd under Reflection(z -> -z)
  dtau_T2    even under Reflection(z -> -z)
  dtau_p     odd under Reflection(z -> -z)
  dtau_u1    even under Reflection(z -> -z)
  dtau_u2    odd under Reflection(z -> -z)
  du         odd under Reflection(z -> -z)
  expected reduction: up to x2 per reflection, x m0 per m0-fold rotation (adapted bases: parity blocks, kept azimuthal classes, docs/site/tutorials/03_symmetries.md)
  WARNING: computing in the invariant subspace hides symmetry-breaking instabilities; a symmetric solution is not declared stable before symmetry_breaking_check().
Equivariance (parity of each equation):
  [0] odd under Reflection(z -> -z)
  [1] odd under Reflection(z -> -z)
  [2] even under Reflection(z -> -z)
  [3] odd under Reflection(z -> -z)
  [4] even under Reflection(z -> -z)
  [5] odd under Reflection(z -> -z)
  [6] even under Reflection(z -> -z)
  [7] zero under Reflection(z -> -z)
Under Reflection(z -> -z):
  du         odd (du_x odd, du_z even)
  dp         odd
  dT         even
  dtau_u1    even (dtau_u1_x even, dtau_u1_z odd)
  dtau_u2    odd (dtau_u2_x odd, dtau_u2_z even)
  dtau_T1    odd -> constant along the mirror direction and odd: vanishes identically
  dtau_T2    even
  dtau_p     odd -> constant along the mirror direction and odd: vanishes identically
  BC [3] and [5] are mirror images under Reflection(z -> -z): redundant in the subspace, one of them (and its tau) is dropped at discretization
  BC [4] and [6] are mirror images under Reflection(z -> -z): redundant in the subspace, one of them (and its tau) is dropped at discretization
other class: leading sigma = -4.2627e+01   n = 134   QZ  66.0 ms   residual 4.4e-18
refused: equation 's*du - Pr*lap(du) + lift(dtau_u1, z, -1) + grad(dp) + lift(dtau_u2, z, -2) - Ra*dT*Pr*ez = 0' is not equivariant under Reflection(z -> -z): term '-Ra*dT*Pr*ez' is odd but 's*du' is even. The term breaks the declared symmetry (fix the parities or drop the symmetry).
  docs:  guide/errors#symmetrybreakingerror  (python -m conatus docs guide/errors)
OK: neutral mode at Reid & Harris' (Ra_c, k_c) in the onset class, n = 134 vs 268
`,C=`# t04_bratu: PASS (exit 0) in 12.8 s | 2026-10-09
Model 'Bratu': Box(x=walls(-1, 1)) (Cartesian box)
Unknowns: u (scalar); parameters: lam = 0.3
Equations as written:
  bratu: dx(dx(u)) + lam*exp(u) = 0
Problem: NLBVP (no dt(), nonlinear in the unknowns)
Boundary conditions: walls (left: x=-1): u = 0; walls (right: x=1): u = 0 (tau method, placed automatically)
Dealiasing: x: 3/2 (declared default for products of degree 1)
  warning: not polynomial in the unknowns (no dealiasing factor removes their aliasing): exp(u)*lam
Resolution: not chosen yet (simulate/eigenvalues/discretize(N=...))
(every decision, tau by tau and term by term: m.explain(full=True))
Newton on NLBVP 'Newton step of Bratu': converged in 3 iteration(s)
  settings: tolerance=1e-13, damping=1.0, max_iterations=20
  k= 0  |F|=3.000e-01  |F|/scale=1.000e+00  |dx|=8.568e-02
  k= 1  |F|=3.366e-03  |F|/scale=1.135e-02  |dx|=9.786e-04
  k= 2  |F|=4.634e-07  |F|/scale=1.564e-06  |dx|=1.347e-07
  k= 3  |F|=8.715e-15  |F|/scale=2.941e-14  |dx|=-
  observed orders: 1.99
  note: trivial-state check: max |u| = 8.665e-02 vs 1e-08 x the guess (0.000e+00)
Pseudo-arclength branch: 7 points, p in [0.3, 0.87704], stop: max_points = 6 reached
  ds = 0.2 in [1e-08, 0.6], x1.5 when the corrector takes <= 3 (default) its, /2 on failure
  corrector: tolerance 1e-12, budget 15 its; G_p: central FD, dp = 1e-06 (given)
  max points 6
  events: secant budget event_max_iterations = 30 (default), fold_tol = 1e-12 (default) (|t_p|), crossing_tol = 1e-11 (default) (|Re λ| / max(1, |λ|)), hopf_imag_tol = 1e-08 (default) (|Im λ| / max(1, |λ|))
  corrector failures (step halved): 0
  fold at p = 0.878457679781 between points (2, 3): |t_p| = 4.4e-18 at the located point, |G|/scale = 1.6e-15 [located by |t_p| <= fold_tol = 1e-12 (default)]
fold: lam_c = 0.87845767978129
OK: lam_c within 0.0e+00 of the closed form 0.87845767978129
`,w=`# t05_ks_orbits: PASS (exit 0) in 41.3 s | 2026-10-09
equilibrium: space=antisymmetric subspace R u = u (cos components frozen at 0), mean frozen
  relative residual 1.385e-16 (7 Newton iterations), energy E = 0.260873, status provisional
  note: residual = ‖v(u)‖/‖u‖ with v = M⁻¹(F − L u), evaluated in the full space
  note: trivial-state threshold: ‖u‖ < 1e-08 × 1 ⇒ retracted
E1: energy 0.2609, residual 1.4e-16; leading eigenvalues [0.1308-0.3341j 0.1308+0.3341j 0.0824-0.3402j 0.0824+0.3402j]
E2: energy 0.4382, residual 7.1e-16
TW1: c = 0.73699, energy 0.4649, residual 2.5e-16
ppo: T = 10.252881, residual 9.5e-14; mu_1 = 0.033208, theta_1 = 2.0079
OK: E1, E2, TW1 and ppo_10.25 agree with the published digits
`,T=`# t08_polar: PASS (exit 0) in 3.7 s | 2026-10-09
Model 'vibrating drum': Disk(radius=1) (disk)
Unknowns: T (scalar)
Eigenvalue: s; growth = Re s, declared with the eigenvalue s
Equations as written:
  eq0: s*T = lap(T)
Problem: EVP (eigenvalue s declared)
Boundary conditions: wall: T = 0 (tau method, placed automatically)
Dealiasing: phi: 3/2, r: 3/2 (declared default for products of degree 1)
Resolution: not chosen yet (simulate/eigenvalues/discretize(N=...))
(every decision, tau by tau and term by term: m.explain(full=True))
m = 0: s = [ -5.78318596 -30.47126234 -74.88700679]  backward residual 5.8e-16
m = 1: s = [ -14.68197064  -49.21845632 -103.4994539 ]  backward residual 4.5e-16
m = 2: s = [ -26.37461643  -70.84999892 -135.02070887]  backward residual 5.1e-16
Hagen-Poiseuille stability | Use plan.group(1, 1.0). Eigenvalue s = sigma + i omega.
rightmost eigenvalues (m, k) = (1, 1): [-0.07086401-0.84674983j -0.0911426 -0.46914288j -0.15121606-0.74819401j]
max backward residual 1.1e-15
neutral Re(k): {3.1: 68.2054, 3.16: 68.1863, 3.22: 68.2021}
OK: drum = Bessel zeros; pipe eigenvalues to 1e-10 of Meseguer & Trefethen; Re_c(3.16) = 68.1863
`,E=`# t10_bifurcations: PASS (exit 0) in 4.3 s | 2026-10-09
DynamicalSystem: M dx/dt = G(x; p), n = 3, origin: IR: Lorenz (0-D: one constant Fourier mode)
  parameters: r = 0.5
  mass matrix: identity
  origin: IR (discretized IVP; F by the spectral evaluator, derivatives symbolic)
  IVPSolver used as evaluator/matrix container only: no time step is taken
  pointwise=True: 0-D system, IR expressions compiled to vectorised numpy (same expressions as the spectral evaluator)
  dG/dr (RHS): exact Laurent reconstruction, exponents [0, 1]
equilibrium curve: 32 points, r in [1.00047, 31.1039]; stop: r = 31.1039 left (0, 30)
  branch switch at r = 1: pitchfork (symmetry-breaking); ABE c = (0.000e+00, -7.036e-01, 0.000e+00), known-branch alignment 1.000000000000, step 0.05 (direction +1); first corrected point residual 2.2e-19; S φ = −φ on an S-invariant branch: the other half-branch is the S-image
  equilibria of IR: Lorenz (0-D: one constant Fourier mode) in r
  stability: #Re λ > 1e-09 (generalized eigenvalues of (J, M), |λ| > 1e+10(1 + max|J|) removed as infinite)
  tests: fold, branch point, hopf; Hopf test over 3 eigenvalue(s)
  ds = 0.1 in [1e-09, 2], x1.3 (default) when the corrector takes <= 4 its, /2 on failure
  corrector: tolerance 1e-11, budget 12 (default) its; G_p: analytic G_p (dp_fd=None)
  max points 60
  events: secant budget event_max_iterations = 30 (default), fold_tol = 1e-12 (default) (|t_p|), crossing_tol = 1e-11 (default) (|Re λ| / max(1, |λ|)), hopf_imag_tol = 1e-08 (default) (|Im λ| / max(1, |λ|))
  corrector failures (step halved): 0; max residual on the curve 8.4e-12
  hopf at r = 24.7368421053 between points (26, 27), |test| = 5.3e-13, residual 2.8e-14: λ = -3.00e-13 ± 9.62453006371 i [located by |test hopf| <= crossing_tol = 1e-11, secant budget 30]
Normal form (pitchfork (supercritical)) at r = 1: da/dt = μ Δp a + a3 a³
  β = 0, μ = 0.9090909091, a2 = 0, a3 = -0.1704545455
  x ≈ x0 + a q + a²/2 h2, ‖q‖ = 1 (declared norm), pᵀM q = 1
  check max|G(x0)|: 0.0e+00
  check |λ_0|: 0.0e+00
  check ‖J q‖: 2.2e-16
  check ‖pᵀJ‖: 2.2e-16
  check residual h2: 0.0e+00
  check residual x_p: 0.0e+00
  declared symmetry: S q = −q
Hopf normal form at r = 24.7368421053: ω = 9.62453006371
  c1 = 0.002555495334 -0.02765775717 i, first Lyapunov coefficient l1 = Re c1 / ω = 0.0002655189726 (subcritical)
  normalization: ‖q‖ = 1 (declared norm), pᴴM q = 1 (Kuznetsov); Re c1 / ‖q‖² is invariant
  λ'(r) = 0.03022248268 +0.1814503065 i -> Landau: dz/dt = λ' Δr z + c1 z|z|²
  check max|G(x0)|: 2.8e-14
  check |Re λ|: 3.0e-13
  check ‖(J − iωM) q‖: 2.2e-13
  check ‖pᴴ(J − iωM)‖: 2.3e-13
  check |pᴴ M q − 1|: 6.4e-17
  check Re h11 imag part: 0.0e+00
periodic orbit curve: 13 points, r in [22.3351, 24.7352], T in [0.652887, 0.748573]; stop: max_points = 12 reached
  Hopf → cycles at r = 24.7368421053 (ω = 9.624530064, l1 = 0.000265519, subcritical); predictor: amplitude 0.2 (declared), Δr = -1.6370e-03, T = 0.65288682; corrected at fixed r: T = 0.65288682
  periodic orbits of IR: Lorenz (0-D: one constant Fourier mode) in r: orthogonal collocation, ntst = 40, ncol = 4 (Gauss points), integral phase condition
  mesh: uniform, fixed
  arclength norm: nodes weighted 1/npts (discrete L2 in τ), T and p weight 1
  Floquet multipliers from the collocation monodromy (condensation); |μ_triv − 1| stored on every point; multipliers below 1e-08 x max(1, |μ|max) are unresolved (count stored) and left out of the tests
  stability: 'unstable' counts nontrivial |μ| > 1 + 1e-06; ||μ| − 1| <= 1e-06 is near-marginal (stored as 'marginal', listed below), not counted
  ds = 0.2 in [1e-09, 1], x1.3 (default) when the corrector takes <= 4 its, /2 on failure
  corrector: tolerance 1e-09, budget 12 (default) its; G_p: analytic G_p (dp_fd=None)
  max points 12
  events: secant budget event_max_iterations = 30 (default), fold_tol = 1e-12 (default) (|t_p|), crossing_tol = 1e-11 (default) (|Re λ| / max(1, |λ|)), hopf_imag_tol = 1e-08 (default) (|Im λ| / max(1, |λ|))
  corrector failures (step halved): 0; max residual on the curve 5.9e-10
  no event detected
fold (codim 2) curve: 15 points, b1 in [0.000137149, 0.58141], b2 in [-1, 1.52501]; stop: b2 = 1.52501 left (-1.5, 1.5)
  fold curve of IR: Bogdanov-Takens (0-D: one constant Fourier mode) in (b1, b2), minimally augmented system (Govaerts 2000); bordering vectors refreshed after every accepted point
  tests: cusp, bt, zh
  ds = 0.05 in [1e-09, 0.3], x1.3 (default) when the corrector takes <= 3 (default) its, /2 on failure
  corrector: tolerance 1e-11, budget 12 (default) its; G_p: analytic G_p (dp_fd=None)
  max points 40
  events: secant budget event_max_iterations = 30 (default), fold_tol = 1e-12 (default) (|t_p|), crossing_tol = 1e-11 (default) (|Re λ| / max(1, |λ|)), hopf_imag_tol = 1e-08 (default) (|Im λ| / max(1, |λ|))
  corrector failures (step halved): 0; max residual on the curve 1.4e-12
  bt at b1 = -6.7114253235e-16, b2 = 1.11699588465e-12 between points (7, 8), |test| = 5.6e-13, residual 6.7e-16 [located by |test bt| <= crossing_tol = 1e-11, secant budget 30]
hopf (codim 2) curve: 17 points, b1 in [1.90168e-14, 2.38022e-14], b2 in [-1, 1.53084], kappa in [-1.53084, 1]; stop: b2 = 1.53084 left (-1.5, 1.5)
  hopf curve of IR: Bogdanov-Takens (0-D: one constant Fourier mode) in (b1, b2), minimally augmented system (Govaerts 2000); bordering vectors refreshed after every accepted point
  tests: bt, gh, zh, hh
  Hopf conditions: entries (2, 3) of the 2x2 bordered G (row-major), chosen once at the start for the best-conditioned augmented Jacobian
  ds = 0.05 in [1e-09, 0.3], x1.3 (default) when the corrector takes <= 3 (default) its, /2 on failure
  corrector: tolerance 1e-11, budget 12 (default) its; G_p: analytic G_p (dp_fd=None)
  max points 40
  events: secant budget event_max_iterations = 30 (default), fold_tol = 1e-12 (default) (|t_p|), crossing_tol = 1e-11 (default) (|Re λ| / max(1, |λ|)), hopf_imag_tol = 1e-08 (default) (|Im λ| / max(1, |λ|))
  corrector failures (step halved): 0; max residual on the curve 1.4e-13
  bt at b1 = 1.90259931987e-14, b2 = -2.65065747129e-13, kappa = -5.55111512313e-17 between points (8, 9), |test| = 5.6e-17, residual 1.9e-14: κ = ω² = -5.55e-17 [located by |test bt| <= crossing_tol = 1e-11, secant budget 30]
  (rejected zh between (8, 9): det J = 0 with κ = ω² = -4.7e-16: the Bogdanov–Takens point)
Landau equation (stationary) at r_c = 0: dA/dt = μ Δr A + g3 A³
  μ = 1+0j, g3 = -0.75+0j (observable of q: 1+0j)
  A² = −Re μ Δp / Re g3 = 1.333333333 Δr
  from the pitchfork normal form: a3 / ℓ(q)², A = a ℓ(q)
Ginzburg–Landau at k_c = 1, r_c = 0:
  ∂t A = μ Δr A + v_g-term + D ∂X² A + g |A|² A, u = A e^(i k_c x) q + c.c.
  μ = 1+0j, D = −σ_kk/2 = 4-0j, v_g = -0, g = -3+0j
  Cross–Hohenberg form (ε = Δr): τ0 = 1, ξ0² = 4+0j, g0 = 3+0j
  σ_kk: 4th-order central differences, h = 0.01 (declared); |4th − 2nd order| = 1.0e-04
  g = 4 g3 (u = A e^{ikx} + c.c. = 2|A| cos kx)
CHECK r_H = 24.736842105253 (470/19 = 24.736842105263), l1 = 2.655190e-04 > 0; SH: A^2/r = 1.333333333333 (4/3), xi0^2 = 4.0000000000, g0 = 3.000000000000
`,D=`# t11_runs: PASS (exit 0) in 4.0 s | 2026-10-09
Run (heat_k0.1): state completed, status provisional, 0.0359 s
Run (heat_k0.2): state completed, status provisional, 0.034 s
Run (heat_k0.4): state completed, status provisional, 0.033 s
  completed certified   rate=1.8000000002184402, rel_error=1.2135564158515738e-10
  completed certified   rate=3.600000003494189, rel_error=9.706080383864446e-10
compare (a) vs (b)
  problem: same structural hash
  params kappa: 0.2 -> 0.4
  numerics problem.parameters.kappa: 0.2 -> 0.4
  result rate: a=1.8000000002184402 b=3.600000003494189 (rel. diff 5.000e-01)
  result rel_error: a=1.2135564158515738e-10 b=9.706080383864446e-10 (rel. diff 8.750e-01)
  code: identical executed tree; git b38dcba9a5 -> b38dcba9a5
reproduce -> bit-for-bit identical
  code: identical to the recorded code
  steps: 200
  compared: end state X (all unknowns)
conatus.runs cache hit: nothing computed; state after 200 steps read from run (took 0.032884 s); key 63006cd4b163, report in run.cache_report
{'hit': True, 'reused_from': '', 'computed': 'nothing: 0 steps taken'}
 (this run)
  └─ (cache reuse)
OK: decay rates exact, query, bit-for-bit reproduce, cache only on request
`,O=`# t12_chaos_ensembles: PASS (exit 0) in 133.2 s | 2026-10-09
Lyapunov spectrum of the discrete map (RK443, dt = 0.01, backend = cpu): 3 vector(s), 32 trajectory(ies)
  transient 1000 steps, tangent transient 1000 steps, average over T = 100 per trajectory (1000 QR every 10 steps = 0.1)
  inner product: Euclidean on the coefficient vector; projection: none; seed 1
  error: standard error of 160 block means (5 per trajectory), max |lag-1 corr| = 0.43
  λ1   = +0.90848 ± 0.00681   (CLV growth +0.95967)
  λ2   = -0.00041 ± 0.00332   (CLV growth +0.04044)
  λ3   = -14.58248 ± 0.00477   (CLV growth -14.64957)
  sum = -13.67441
  Kaplan–Yorke dimension = 2.0623 ± 0.0003
neutral CLV vs f(x): min |cos| = 0.99999985 over 160 samples
Poincaré section (g decreasing through 0), 8 member(s), 530 crossing(s) over 5000 steps of RK443 dt = 0.01
  location: τ* in (0, dt] with g(Φ_τ*(X_n)) = 0 on the scheme (Illinois regula falsi), |g| <= 1e-10, max 30 iterations; observed max |g| 8.2e-11, mean iterations 4.9
Lorenz map: 522 pairs (z_k, z_k+1), z in [29.29, 45.38]
EnsembleFate: 2304 member(s); checks every 100 steps (dt = 0.01), no verdict before step 100, horizon 10000 steps; classifier: |X - C+-| < 1
  outcomes {-1: 1152, 1: 1152} (0 = undecided, reported, never assigned)
Ensemble(2304 member(s), 1 parameter set(s), scheme=RK443, dt=0.01, backend=cpu)
  step: IVPSolver arithmetic on (B, n) arrays (same stages, same order)
  BatchedRHS(polynomial form, CPU/NumPy): 7 term(s), 0 with per-member coefficients; transforms, derivatives and products vectorized over a leading member axis (same operations as the fused/linearized forms; equal to IVPSolver.F to rounding, not bit for bit)
  BatchedSystem: 2304 member(s), n = 3, M and L shared by all members (one sparse product over the batch); CPU stacked bordered-banded solves, one call per bucket for the whole batch; factors of each member stacked (members with the same L share the same matrices, replicated in the stack: memory B x one factorization); 1 (a, b) pair(s) factored
CHECK lambda = (0.9085 +- 0.0068, -0.0004 +- 0.0033, -14.5825 +- 0.0048), sum = -13.67441 (-13.66667), D_KY = 2.0623 +- 0.0003; basins: 1152 / 1152 members, 0 undecided
`,k=`# t13_sensitivity_data: PASS (exit 0) in 101.1 s | 2026-10-12
Ginzburg-Landau (whole line) mode n = 0: λ = -0.0176886987015018-0.647820287354721j, χ = 0.239159529837+0.0475717882612j (Re χ² > 0); ν = (2+0.4j), γ = (1-1j), μ_c = 0.34, μ₂ = -0.01. Closed-form direct/adjoint modes, wavemaker, gradient w.r.t. μ(x) and dλ/d(ν, μ_c, μ₂) (Chomaz 2005; Bagheri et al. 2009).
ModePair of 'linear Ginzburg-Landau: nu = (2+0.4j), gamma = (1-1j), mu = 0.34 - 0.005*y**2' (Chebyshev N = 140): σ = -0.0176886987015045-0.647820287354706j; pencil residuals right 1.4e-16, left 1.1e-16; physical adjoint = continuous adjoint (eigenvalue -0.0176886987015126+0.647820287354736j, residual 2.4e-17); product ⟨a,b⟩ = Σ w_i ∫ ā_i b_i dy, w = {'q': 1.0}; normalization ⟨u,u⟩ = 1, ⟨u†,u⟩ = 1; κ = 19.1473. nearest other finite eigenvalue at distance 1.682e-01
wavemaker maximum S = 0.137574776291 at x = 4.00e-11
d lambda/d mu_2 = 3.884434935074-1.608985632263j (density)  3.884434935072-1.608985632266j (pencil)
DMD (exact, rank 4, Δt = 0.5, amplitudes 'optimal', quadrature product): ω = -0.5576897846-0.6478195408j (|b| 1.41, res 1.2e-14); ω = -0.7130915004-0.5834606995j (|b| 0.707, res 8.2e-14); ω = -0.8686490565-0.5192373431j (|b| 0.424, res 2.4e-13); ω = -1.024295602-0.4562651356j (|b| 0.316, res 1.4e-13); misfit 3.5e-14; principal log (|Im ω| < π/Δt = 6.28319). 
OK: wavemaker and d lambda/d mu_2 match the closed forms; DMD finds the four least stable eigenvalues
`,A=`# t14_from_dedalus: PASS (exit 0) in 2.8 s | 2026-10-10
native model: t = 0.020, max|u| = 4.225e-02
Dedalus -> Conatus: rb_cell_dedalus.py
  39 translated, 0 approximated, 0 unsupported, 0 skipped (output only)
  problem: IVP with 8 equations, variables u, p, th, u_bot, u_top, th_bot, th_top, gauge
  time stepping: SBDF2, dt = 0.0005, stop_sim_time = None; ivp_solver() imposes time-dependent boundary data as Dedalus does (bc_time='step'; bc_time='auto' for the Conatus default, which keeps the order of the scheme)
  data: th (translated)
Line-by-line report:
  20  translated   import numpy as np
  21  translated   import dedalus.public as d3
  24  translated   Ra = 3000.0
  25  translated   Pr = 7.0
  26  translated   period = 2.0
  27  translated   N_x, N_z = 24, 16
  28  translated   tstep = 5e-4
  29  translated   steps = 40
  32  translated   plane = d3.CartesianCoordinates('x', 'z')
  33  translated   mesh = d3.Distributor(plane, dtype=np.float64)
  34  translated   horiz = d3.RealFourier(plane['x'], size=N_x, bounds=(0, period), de...
  35  translated   vert = d3.ChebyshevT(plane['z'], size=N_z, bounds=(0, 1), dealias=1.5)
  36  translated   u = mesh.VectorField(plane, name='u', bases=(horiz, vert))
  37  translated   p = mesh.Field(name='p', bases=(horiz, vert))
  38  translated   th = mesh.Field(name='th', bases=(horiz, vert))
  39  translated   u_bot = mesh.VectorField(plane, name='u_bot', bases=horiz)
  40  translated   u_top = mesh.VectorField(plane, name='u_top', bases=horiz)
  41  translated   th_bot = mesh.Field(name='th_bot', bases=horiz)
  42  translated   th_top = mesh.Field(name='th_top', bases=horiz)
  43  translated   gauge = mesh.Field(name='gauge')
  46  translated   e_x, e_z = plane.unit_vector_fields(mesh)
  47  translated   dz_basis = vert.derivative_basis(1)
  48  translated   tau_z = lambda f: d3.Lift(f, dz_basis, -1)
  49  translated   Gu = d3.grad(u) + e_z*tau_z(u_bot)
                  -> 'ez*lift(u_bot, z, -1, lam=1)' of two tensors -> outer (Dedalus convention)
  50  translated   Gth = d3.grad(th) + e_z*tau_z(th_bot)
  52  translated   onset = d3.IVP([u, p, th, u_bot, u_top, th_bot, th_top, gauge], nam...
  53  translated   onset.add_equation("dt(u) - Pr*div(Gu) + grad(p) - Ra*Pr*th*e_z + t...
                  -> div(outer(ez, lift(u_bot, z, -1, lam=1))) -> dz(lift(u_bot, z, -1, lam=1)) (exact identity, constant unit vector)
  54  translated   onset.add_equation("trace(Gu) + gauge = 0")
                  -> trace(outer(ez, lift(u_bot, z, -1, lam=1))) -> dot(...) (exact identity); trace(grad(u)) -> div(u) (exact identity); namespace functions used: trace
  55  translated   onset.add_equation("dt(th) - div(Gth) - u@e_z + tau_z(th_top) = -u@...
  56  translated   onset.add_equation("u(z=0) = 0")
  57  translated   onset.add_equation("th(z=0) = 0")
  58  translated   onset.add_equation("u(z=1) = 0")
  59  translated   onset.add_equation("th(z=1) = 0")
  60  translated   onset.add_equation("integ(p) = 0")
  63  translated   run = onset.build_solver(d3.SBDF2)
                  -> solver: IVPSolver(plan, 'SBDF2', dt) (dt from the time loop)
  64  translated   run.stop_iteration = steps
                  -> number of steps: solver.run(n)
  65  translated   x, z = mesh.local_grids(horiz, vert)
                  -> grid coordinates: used for initial / given data only
  66  translated   th['g'] = 1e-3*np.cos(2*np.pi*x/period)*np.sin(np.pi*z)
                  -> grid data of th: initial condition / given field (tr.data)
  67  translated   while run.proceed:
                  -> time loop -> solver.run(n) with a fixed dt = 0.0005 (no Python loop); the loop body (logging, analysis) is not translated
translation: t = 0.020 after 40 steps, max|u| = 4.225e-02
nz = 12: native vs translated, relative difference th 1.2e-06, u 3.0e-07
nz = 16: native vs translated, relative difference th 7.2e-09, u 2.2e-10
nz = 32: native vs translated, relative difference th 3.8e-15, u 2.8e-15
PASS
`,j=`# t14b_dedalus_check: PASS (exit 0) in 5.8 s | 2026-10-10
Conatus: t = 0.020 after 40 steps
2026-10-10 08:13:05,007 dedalus 0/1 WARNING :: Threading has not been disabled. This may massively degrade Dedalus performance.
2026-10-10 08:13:05,007 dedalus 0/1 WARNING :: We strongly suggest setting the "OMP_NUM_THREADS" environment variable to "1".
Dedalus: t = 0.020; relative difference th 3.0e-14, u 3.8e-15
PASS
`,M=`# t15_ks_covariant: PASS (exit 0) in 47.1 s | 2026-10-10
Lyapunov spectrum of the discrete map (RK443, dt = 0.05, backend = cpu): 12 vector(s), 4 trajectory(ies)
  transient 2000 steps, tangent transient 1000 steps, average over T = 200 per trajectory (400 QR every 10 steps = 0.5)
  inner product: Euclidean on the coefficient vector; projection: zero mean: the wavenumber-zero coefficient of u set to 0 in every tangent vector (1 component(s) of the state vector); seed 1
  error: standard error of 16 block means (4 per trajectory), max |lag-1 corr| = 0.58
  λ1   = +0.04621 ± 0.00764   (CLV growth +0.05199)
  λ2   = +0.01029 ± 0.00604   (CLV growth -0.00273)
  λ3   = -0.00203 ± 0.00469   (CLV growth -0.01073)
  λ4   = -0.01404 ± 0.00657   (CLV growth -0.03727)
  λ5   = -0.19905 ± 0.01423   (CLV growth -0.17382)
  λ6   = -0.24069 ± 0.00884   (CLV growth -0.29031)
  λ7   = -0.29359 ± 0.01117   (CLV growth -0.29710)
  λ8   = -0.31376 ± 0.00784   (CLV growth -0.31540)
  λ9   = -1.96142 ± 0.00567   (CLV growth -1.95551)
  λ10  = -1.96677 ± 0.00378   (CLV growth -1.96138)
  λ11  = -5.60755 ± 0.00210   (CLV growth -5.60392)
  λ12  = -5.60776 ± 0.00202   (CLV growth -5.60537)
  sum = -16.15014
  Kaplan–Yorke dimension = 4.2031 ± 0.0787
  covariant vectors (Ginelli et al. 2007): 80 kept time(s) x 4 trajectory(ies), window 80 interval(s) after a backward transient of 80, kept every 1
  memory: checkpointed: 8 checkpoint(s) (Q, X) every 20 interval(s), 160 R matrices, one segment of at most 20 Q_j recomputed at a time: 1.05 MiB instead of 2.67 MiB, for one extra forward pass over the 160 stored intervals
  recomputed R vs stored R: max relative difference 0.00e+00
  backward convergence (sine of the angle between the CLVs of two independent random backward starts): max 1.00e+00 (worst vector 8); slow for nearly equal exponents (rate λ_i-1 - λ_i): lengthen clv_backward if it matters
  subspace convergence per split j (span(γ_j+1..γ_k), two starts, largest sine; rate λ_j - λ_j+1): 1.0e+00, 1.0e+00, 8.6e-01, 1.3e-01, 4.9e-01, 1.0e+00, 1.0e+00, 6.4e-15, 9.9e-01, 5.4e-15, 8.9e-01
  covariance residual (sine of the angle between J γ_i(t_j) and γ_i(t_j+1) over one interval; consistency of the stored factors with the tangent): max 5.57e-15 (vector 11); largest ratio to its rounding floor eps exp((λ_1 - λ_i) Δt): 5.43 (vector 7)
  smallest principal angle between span(γ_1..γ_j) and the rest, j = 1..11: 0.00483, 0.000164, 0.000597, 0.000841, 0.00916, 0.0106, 0.0278, 0.448, 0.0881, 0.94, 0.0641 rad
  note: 12 of 31 exponents computed: sum and Kaplan–Yorke use these only
  note: native: sim.lyapunov on 'Kuramoto-Sivashinsky L=22' (transient 100, align 50, T 200 per trajectory, CLV window 40 ending 40 before the end); the simulation's own state is not advanced
Domination of the Oseledets splitting at tau = 5: 32 samples (windows x trajectories), 12 covariant vectors
  violation fraction across each split j | j+1 (largest nu_ij, i <= j < l): 1:0.406, 2:0.406, 3:0.469, 4:0.219, 5:0.125, 6:0.406, 7:0.344, 8:0, 9:0.406, 10:0, 11:0.344
  clusters of entangled modes (any observed violation joins them): 1-8, 9-10, 11-12
  physical modes (leading entangled cluster): 8; the split after it has violation 0 and smallest principal angle 0.448 rad (its subspaces converged to 6.4e-15)
  a violation fraction is a sample estimate: 0 means none observed in 32 samples; modes near k may entangle with modes not computed
smallest angle at the split 8 | 9: 0.448 rad; at 4 | 5: 0.0008 rad (47% of the samples below 0.1 rad)
checks passed: exponents within their errors of CDS 2010, eight physical modes
`,N=`# t16_particles_dye: PASS (exit 0) in 70.5 s | 2026-10-10
dt = 0.1   : largest tracer error 6.46e-05
dt = 0.05  : largest tracer error 8.40e-06
dt = 0.025 : largest tracer error 1.07e-06
observed orders: [2.94 2.97]
Particles: 32 passive tracer(s) advected by u, backend cpu, 80 step(s) taken; no file (particles.positions())
  velocity at a particle: exact evaluation of the spectral series (x: cos/sin series (16 modes), y: cos/sin series (16 modes))
  time integration: classical RK4 per flow step; the midpoint velocity is the cubic interpolant in time of the last four flow states (2 and 3 states on the first two steps of a run): order 4 in a steady flow, 3 in an unsteady flow (the time error of the flow states themselves, third order for RK443, and the linear midpoint of the first step of a run)
  periodic axes: positions unwrapped (written also wrapped into [a, a + L))
  walls: at_walls='refuse' (a step ending outside the walls raises); stage positions are not clipped (the series continues there)
  not in the simulation's checkpoints: after a restart, declare the particles again (e.g. from the last row of the file)
Model 'plane Couette with dye': Box(x=periodic((0, 4)), y=walls(-1, 1)) (Cartesian box)
Unknowns: u (velocity), p (pressure), c (dye); parameters: nu = 0.1
Equations as written:
  eq0: dt(u) + grad(p) - nu*lap(u) = -u@grad(u)
  eq1: div(u) = 0
  dye_c: dt(c) - 0.01*lap(c) = -div(c*u)
Problem: IVP (a time derivative dt() appears)
Time stepping: 6 linear term(s) implicit; nonlinear terms explicit (eq0, dye_c)
Boundary conditions: bottom: u = (-1, 0); top: u = (1, 0); walls (left: y=-1): dy(c) = 0; walls (right: y=1): dy(c) = 0 (tau method, placed automatically)
Gauge: the mean of p is fixed (integ(p) = 0)
Dealiasing: x: 3/2, y: 3/2 (declared default for products of degree 2)
Resolution: not chosen yet (simulate/eigenvalues/discretize(N=...))
(every decision, tau by tau and term by term: m.explain(full=True))
Dye file <path>/tmpv2xmfnu8/dye.h5 (HDF5, conatus.analysis/1): every 50 steps (= 0.5 time units), 7 write(s) so far
  dye c (m.dye): dt(c) - 0.01*lap(c) = -div(c*u); no flux: dy(c) = 0 at the walls; advected by u, same scheme and dt as the flow
  c: grid values of c
  mass: integ(c) (integral of the spectral series)
  variance: integ(c*c) - integ(c)^2 / V, V = 8
  minimum: min of the grid values of c
  maximum: max of the grid values of c
  tail: largest |coefficient| among the last two modes of each axis, relative to the largest
  the state at the start (iteration 0, t = 0) written as a row (initial=True, the default of sim.output)
  scales: sim_time, iteration, write_number, x, y; every array labelled with its dimensions (HDF5 dimension scales)
  XDMF sidecar (ParaView): <path>/tmpv2xmfnu8/dye.xmf
[[-3.  -1. ]
 [-0.5 -0.5]
 [ 2.   0. ]
 [ 4.5  0.5]
 [ 3.5  1. ]]
dye mass 0.933219300417, largest relative drift 1.1e-11
checks passed: order 3 in time, Couette tracers exact, dye mass conserved
`,P={generated:`2026-10-13T06:43:17.632Z`,tutorials:{t00_getting_started:{fresh:!0,by:`commit`,reason:null},t01_orr_sommerfeld:{fresh:!0,by:`commit`,reason:null},t02a_rayleigh_benard_cpu:{fresh:!0,by:`commit`,reason:null},t02b_rayleigh_benard_gpu:{fresh:!0,by:`commit`,reason:null},t03_symmetries:{fresh:!0,by:`commit`,reason:null},t04_bratu:{fresh:!0,by:`commit`,reason:null},t05_ks_orbits:{fresh:!0,by:`commit`,reason:null},t08_polar:{fresh:!0,by:`commit`,reason:null},t10_bifurcations:{fresh:!0,by:`commit`,reason:null},t11_runs:{fresh:!0,by:`commit`,reason:null},t12_chaos_ensembles:{fresh:!0,by:`commit`,reason:null},t13_sensitivity_data:{fresh:!0,by:`commit`,reason:null},t14_from_dedalus:{fresh:!0,by:`commit`,reason:null},t14b_dedalus_check:{fresh:!0,by:`commit`,reason:null},t15_ks_covariant:{fresh:!0,by:`commit`,reason:null},t16_particles_dye:{fresh:!0,by:`commit`,reason:null}}},F=Object.assign({"../../../docs/site/tutorials/code/rb_cell_dedalus.py":t,"../../../docs/site/tutorials/code/t00_getting_started.py":n,"../../../docs/site/tutorials/code/t01_orr_sommerfeld.py":r,"../../../docs/site/tutorials/code/t02a_rayleigh_benard_cpu.py":i,"../../../docs/site/tutorials/code/t02b_rayleigh_benard_gpu.py":a,"../../../docs/site/tutorials/code/t03_symmetries.py":o,"../../../docs/site/tutorials/code/t04_bratu.py":s,"../../../docs/site/tutorials/code/t05_ks_orbits.py":c,"../../../docs/site/tutorials/code/t08_polar.py":l,"../../../docs/site/tutorials/code/t10_bifurcations.py":u,"../../../docs/site/tutorials/code/t11_runs.py":d,"../../../docs/site/tutorials/code/t12_chaos_ensembles.py":f,"../../../docs/site/tutorials/code/t13_sensitivity_data.py":p,"../../../docs/site/tutorials/code/t14_from_dedalus.py":m,"../../../docs/site/tutorials/code/t14b_dedalus_check.py":h,"../../../docs/site/tutorials/code/t15_ks_covariant.py":g,"../../../docs/site/tutorials/code/t16_particles_dye.py":_}),I=Object.assign({"../generated/outputs/t00_getting_started.txt":v,"../generated/outputs/t01_orr_sommerfeld.txt":y,"../generated/outputs/t02a_rayleigh_benard_cpu.txt":b,"../generated/outputs/t02b_rayleigh_benard_gpu.txt":x,"../generated/outputs/t03_symmetries.txt":S,"../generated/outputs/t04_bratu.txt":C,"../generated/outputs/t05_ks_orbits.txt":w,"../generated/outputs/t08_polar.txt":T,"../generated/outputs/t10_bifurcations.txt":E,"../generated/outputs/t11_runs.txt":D,"../generated/outputs/t12_chaos_ensembles.txt":O,"../generated/outputs/t13_sensitivity_data.txt":k,"../generated/outputs/t14_from_dedalus.txt":A,"../generated/outputs/t14b_dedalus_check.txt":j,"../generated/outputs/t15_ks_covariant.txt":M,"../generated/outputs/t16_particles_dye.txt":N}),L=P.tutorials;function R(e,t,n){throw Error(`[web/snippet] ${e.path}: ${t}.\n  The site cuts this from the executed tutorial; the tutorial changed. Update the cut`+(n?` in ${n}`:``)+` (web/src/lib/snippet.ts explains the markers).`)}function z(e){let t=Object.entries(F).find(([t])=>t.endsWith(`/code/${e}.py`)),n=Object.entries(I).find(([t])=>t.endsWith(`/outputs/${e}.txt`));if(!t)throw Error(`[web/snippet] docs/site/tutorials/code/${e}.py not found (renamed or removed tutorial?)`);let r=!!n&&L[e]?.fresh===!0;return{code:{path:`docs/site/tutorials/code/${e}.py`,text:t[1]},out:r?{path:`docs/site/tutorials/outputs/${e}.txt`,text:n[1]}:null}}function B(e,t,n){let r=e.text.split(`
`),i=r.findIndex(e=>e.trim()===`# [${t}]`),a=r.findIndex((e,n)=>n>i&&e.trim()===`# [/${t}]`);(i<0||a<0)&&R(e,`region "# [${t}]" … "# [/${t}]" not found`,n);let o=r.slice(i+1,a).filter(e=>!/^\s*# \[\/?[\w-]+\]\s*$/.test(e)).join(`
`).replace(/^\n+|\s+$/g,``);return o||R(e,`region "${t}" is empty`,n),o}function V(e,t,n){let r=e.text.split(`
`);for(let i of t)r.some(e=>i.test(e))||R(e,`no line matches ${i}`,n);return r.filter(e=>t.some(t=>t.test(e))).join(`
`)}function H(e){let t=e.text.match(/^# (\S+): (\w+) \(exit \d+\) in ([\d.]+) s \| (?:run_id (\S+) \| commit \S+ \| \S+ \| )?(\S+)/);return t||R(e,`unexpected output header (expected '# name: PASS (exit 0) in … s | run_id …')`),t[2]!==`PASS`&&R(e,`the tutorial did not pass (status ${t[2]}): the site shows only passing executions`),{script:t[1],status:t[2],seconds:Number(t[3]),runId:t[4]??``,date:t[5]}}var U=e();function W({out:e,patterns:t,page:n,label:r,note:i}){if(!e)return null;H(e);let a=V(e,t,n).replace(/\s+$/,``).split(`
`);return(0,U.jsxs)(`figure`,{className:`replay`,children:[(0,U.jsx)(`figcaption`,{className:`replay-head`,children:(0,U.jsx)(`span`,{className:`replay-label`,children:r??`Output`})}),(0,U.jsx)(`pre`,{tabIndex:0,children:(0,U.jsx)(`code`,{children:a.map((e,t)=>(0,U.jsxs)(`span`,{className:/^OK:|\[PASS\]/.test(e.trim())?`replay-ok`:void 0,children:[e,`
`]},t))})}),i&&(0,U.jsx)(`div`,{className:`replay-note`,children:i})]})}export{z as i,V as n,B as r,W as t};