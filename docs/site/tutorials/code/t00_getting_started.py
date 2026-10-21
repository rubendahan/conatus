"""Tutorial 0 -- getting started: convection between two plates, from the equations to the onset.

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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The figure draws the spectral interpolant of T (Fourier in x, Chebyshev in z) on a fine grid.
T = sim.grid("T")
cz = grid.cheb_coeffs(T, 1)                     # Chebyshev coefficients in z
zf = np.linspace(-0.5, 0.5, 201)
Tz = np.polynomial.chebyshev.chebval(2 * zf, cz.T)          # (x nodes, fine z)
Tf = np.fft.irfft(np.fft.rfft(Tz, axis=0), n=256, axis=0) * 256 / 32   # Fourier in x
xf = np.linspace(0, 2, 256, endpoint=False)
signed = LinearSegmentedColormap.from_list(   # the site's signed palette: blue, black, red
    "signed", ["#8FB0E0", "#2E4A78", "#0B0B0B", "#6E2019", "#D4574A"])
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
theta = Tf + zf[None, :]                       # departure from conduction: warm red, cool blue
im = ax.pcolormesh(xf, zf, theta.T, cmap=signed, shading="gouraud", vmin=-0.3, vmax=0.3)
ax.contour(xf, zf, Tf.T, levels=np.linspace(-0.45, 0.45, 10), colors="#77736B",
           linewidths=0.5, linestyles="solid")
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("z")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
cb.set_label("T + z (departure from conduction)")
ax.set_title(f"temperature at t = {sim.t:g} (units of the plate temperature difference and depth)",
             fontsize=9)
fig.savefig("rolls.png", bbox_inches="tight", transparent=True)
# [/figure]

# [check]
RA_C, K_C = 1707.762, 3.117                    # Reid & Harris (1958); Chandrasekhar (1961)
assert abs(crit.value - RA_C) < 1e-3 and abs(crit.over["k"] - K_C) < 1e-3, crit
assert growth[1500] < 0 < growth[2500], growth
assert abs(sigma - growth[2500]) < 1e-5, (sigma, growth[2500])
assert Nu_layer > 1 and abs(Nu_bottom - Nu_layer) < 1e-3 * Nu_layer \
    and abs(Nu_top - Nu_layer) < 1e-3 * Nu_layer, (Nu_layer, Nu_bottom, Nu_top)
print("OK: onset at Reid & Harris' Ra_c and k_c; the simulation grows at the eigenvalue's rate "
      "and saturates into rolls carrying the same heat through both plates")
# [/check]
