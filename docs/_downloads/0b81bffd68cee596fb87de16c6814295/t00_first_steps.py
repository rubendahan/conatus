"""First steps -- the viscous Burgers equation: simulate, plot, measure the energy (about ten minutes).

Viscous Burgers equation u_t + u u_x = nu u_xx on [0, 2 pi), periodic, from u(x, 0) = sin(x): the
wave steepens into a front that viscosity holds at a finite width. The physical number is the
kinetic energy E(t) = 1/2 int u^2 dx, which viscosity dissipates.
Expected value (checked at the end of the script): E at t = 2 equals the exact Cole-Hopf solution
(Cole 1951; Hopf 1950) to 1e-9 relative.
Writes burgers.png (the profiles of u) into the working directory.
Run: python t00_first_steps.py   (a few seconds of CPU; needs matplotlib: pip install "conatus[tutorials]")
"""
# [model]
import numpy as np

import conatus as cn

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="Burgers", nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*lap(u) = -u*dx(u)")
print(m.explain())
# [/model]

# [simulate]
sim = m.simulate(N=128, scheme="RK443", dt=1e-3)    # resolution, scheme, time step
sim.initial(u=lambda x: np.sin(x))
x = sim.coordinate("x")
profiles = {}
for t in (0.0, 0.5, 1.0, 1.5, 2.0):
    if t > sim.t:
        sim.run(until=t)
    profiles[t] = sim.grid("u").copy()
# [/simulate]

# [plot]
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
styles = ["-", "--", ":", "-."]                # the site's figure palette, one line style per curve
colours = ["#6C7FA8", "#C8463A", "#A8864F", "#77736B"]
for k, (t, ut) in enumerate(profiles.items()):
    ax.plot(x, ut, styles[k % 4], color=colours[k % 4], label=f"t = {t:g}")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.savefig("burgers.png", bbox_inches="tight", transparent=True)
# [/plot]

# [energy]
u = sim.grid("u")
energy = np.pi * np.mean(u**2)                     # E = 1/2 int u^2 dx (grid mean, exact here)
print(f"kinetic energy at t = {sim.t:g}: E = {energy:.6f}")
# [/energy]

# [check]
from scipy.special import ive


def cole_hopf_energy(nu, t, n=4096, kmax=400):
    """E(t) of the exact solution u = -2 nu (log theta)_x, theta solving the heat equation from
    theta(x, 0) = exp(cos(x) / (2 nu)) = I_0 + 2 sum I_k cos(kx) (scaled Bessel functions)."""
    xs = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = np.arange(1, kmax)
    c = ive(k, 1 / (2 * nu)) / ive(0, 1 / (2 * nu)) * np.exp(-nu * k**2 * t)
    theta = 1 + 2 * (c[:, None] * np.cos(np.outer(k, xs))).sum(axis=0)
    theta_x = -2 * ((k * c)[:, None] * np.sin(np.outer(k, xs))).sum(axis=0)
    return np.pi * np.mean((-2 * nu * theta_x / theta) ** 2)


exact = cole_hopf_energy(0.1, sim.t)
print(f"exact (Cole-Hopf): E = {exact:.6f}, relative error {abs(energy - exact) / exact:.1e}")
assert abs(energy - exact) < 1e-9 * exact, (energy, exact)
# [/check]
